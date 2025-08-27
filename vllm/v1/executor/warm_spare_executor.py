# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""
Warm Spare Executor for resilient worker management with IPC loading.

This executor maintains "warm spare" workers that are partially initialized
(model loaded via IPC but no KV cache) alongside primary workers. On failure
or reinitialize request, it switches from primary to warm spare workers.

Architecture:
- Port allocation uses consistent offsets:
  - Primary workers: base_port + 0, base_port + 1 (two-phase init)
  - Warm spares: base_port + 100, base_port + 101 (two-phase init with offset)
  - Companions: base_port + 200, base_port + 201, etc.
- Warm spares share companion processes with primary workers
  - This ensures only one copy of model weights on GPU (via IPC)
  - Both primary and warm spare workers connect to same companions
- Each DP rank creates its own warm spare worker(s)
- Proper synchronization ensures ports can be safely reused:
  - Barriers and RPC sync ensure old processes release ports before new ones use them
  - Port counter is reset after promotion to allow port reuse
"""
import os
import signal
import time
import threading
import weakref
from concurrent.futures import TimeoutError as FutureTimeoutError
from threading import Thread
from typing import Any, Callable, Optional, Union
from dataclasses import dataclass

from vllm.config import VllmConfig
from vllm.logger import init_logger
from vllm.v1.executor.multiproc_executor import (
    MultiprocExecutor, WorkerProc, WorkerProcHandle, UnreadyWorkerProcHandle
)
from vllm.distributed.device_communicators.shm_broadcast import MessageQueue
import vllm.envs as envs

logger = init_logger(__name__)

# Default timeout for collective RPC calls (seconds)
DEFAULT_RPC_TIMEOUT = 300  # 5 minutes

@dataclass
class WorkerSet:
    """Container for primary and warm spare workers."""
    primary: list[WorkerProcHandle]
    warm_spare: list[WorkerProcHandle]
    
    def all_workers(self) -> list[WorkerProcHandle]:
        return self.primary + self.warm_spare


class WarmSpareExecutor(MultiprocExecutor):
    """
    Executor with warm spare workers for resilience and fast recovery.
    
    Features:
    - Maintains warm spare workers alongside primary workers
    - Warm spares are initialized to phase 1 (model loaded, fake distributed)
    - On failure detection or manual switch, promotes warm spares to primary
    - Automatically starts new warm spares after switchover
    - Supports timeouts on collective RPC calls for hung worker detection
    """
    
    def __init__(self, vllm_config: VllmConfig):
        # Check prerequisites
        if not vllm_config.load_config.enable_companion_process:
            raise ValueError(
                "WarmSpareExecutor requires IPC loading to be enabled. "
                "Set --enable-ipc-loading or load_config.enable_companion_process=True"
            )
        
        # Enable two-phase initialization
        os.environ["VLLM_TWO_PHASE_INIT"] = "1"
        
        # Track initialization state - MUST be set before calling parent init
        self.warm_spares_initialized = False
        self.switching_to_warm_spare = False
        self.warm_spare_recovery_occurred = False
        
        # Track the primary worker port for reuse when promoting warm spares
        self.primary_distributed_init_port = None
        
        # Worker monitoring  
        self.rpc_timeout = DEFAULT_RPC_TIMEOUT
        
        # DP coordination callback (set by EngineCore if in DP mode)
        self.dp_warm_spare_callback = None
        
        super().__init__(vllm_config)
        
    def _init_executor(self) -> None:
        """Override to initialize both primary and warm spare workers."""
        # Call parent initialization (sets up basic structures)
        super()._init_executor()
        
        # Create warm spare workers in background - don't block primary startup
        # This allows the engine to start serving immediately while warm spares
        # are being initialized in the background
        logger.info("Starting warm spare creation in background")
        threading.Thread(
            target=self._create_warm_spare_workers,
            daemon=True,
            name="InitialWarmSpareCreator"
        ).start()
    
    def initialize_from_config(self,
                               kv_cache_configs: list) -> None:
        """Override to store KV cache config for warm spares."""
        # Store the primary KV cache configuration for potential reuse
        # Store the full config objects, not just the num_blocks
        self.primary_kv_cache_configs = kv_cache_configs.copy()
        logger.info(f"Storing primary KV cache config: "
                   f"{kv_cache_configs[0].num_blocks} GPU blocks")
        
        # Call parent implementation to actually initialize workers
        super().initialize_from_config(kv_cache_configs)
    
    def start_worker_monitor(self):
        """Override parent's worker monitor to intercept failures for warm spare recovery.
        
        Instead of immediately shutting down and calling the failure callback,
        we first attempt to recover using warm spares. Only if recovery fails
        do we propagate the failure upward.
        
        This method is called by parent's _init_executor before warm spares exist,
        so we only start monitoring after warm spares are initialized.
        """
        # Only start monitoring if warm spares are initialized
        # This prevents starting the monitor from parent's _init_executor
        if not self.warm_spares_initialized:
            logger.debug("Skipping monitor start - warm spares not yet initialized")
            return
        
        # Prevent duplicate monitors during recovery
        if getattr(self, '_monitor_active', False):
            logger.debug("Monitor already active, skipping duplicate start")
            return
        
        self._monitor_active = True
        workers = self.workers
        self_ref = weakref.ref(self)

        def monitor_workers():
            import multiprocessing.connection
            sentinels = [h.proc.sentinel for h in workers]
            died = multiprocessing.connection.wait(sentinels)
            _self = self_ref()
            if not _self or getattr(_self, 'shutting_down', False):
                _self._monitor_active = False
                return
                
            # Find which worker died
            failed_idx = next(i for i, h in enumerate(workers)
                            if h.proc.sentinel == died[0])
            proc_name = workers[failed_idx].proc.name
            
            logger.error("Worker proc %s (idx %d) died unexpectedly, "
                        "attempting warm spare recovery...", proc_name, failed_idx)
            
            # Mark monitor as inactive before recovery attempt
            _self._monitor_active = False
            
            # Attempt warm spare recovery
            recovery_successful = _self._attempt_warm_spare_recovery(failed_idx)
            
            if not recovery_successful:
                # Recovery failed - follow parent's failure path
                _self.is_failed = True
                logger.error("Warm spare recovery failed, shutting down executor")
                _self.shutdown()
                callback = _self.failure_callback
                if callback is not None:
                    _self.failure_callback = None
                    callback()
            # Note: Don't restart monitor here for DP case
            # The monitor will be restarted after the actual switch completes

        Thread(target=monitor_workers,
               daemon=True,
               name="WarmSpareMonitor").start()
        
    def _create_warm_spare_workers(self) -> None:
        """Create and partially initialize warm spare workers.
        
        This method can be called from a background thread to avoid blocking
        the main engine operations. The engine can continue serving while
        warm spares are being created and initialized.
        """
        # Don't create warm spares if we're shutting down
        if getattr(self, 'shutting_down', False):
            logger.info("System is shutting down, skipping warm spare creation")
            return
            
        logger.info("Creating warm spare workers in background...")
        
        try:
            # Warm spares share companion processes with primary workers
            # This ensures only one copy of model weights exists on GPU
            if self.vllm_config.load_config.enable_companion_process:
                use_multiproc = os.environ.get("VLLM_IPC_USE_MULTIPROC", "1") == "1"
                
                if use_multiproc:
                    # Verify companion coordinator address is set
                    primary_coordinator_address = self.vllm_config.companion_config.coordinator_address
                    
                    if primary_coordinator_address:
                        logger.info("Warm spares will share companion processes at %s", 
                                   primary_coordinator_address)
                        # The address will be copied to warm spare configs via deep copy
                    else:
                        logger.error("Companion coordinator address not found in config!")
                        raise RuntimeError("Cannot create warm spares without companion coordinator")
            
            # Warm spares shadow primary workers with a port offset
            # The distributed_init_method will be set automatically in parallel_state.py
            # when it detects process_type == "warm_spare_worker"
            distributed_init_method = None
            
            # Create scheduler output handle for warm spares
            max_chunk_bytes = envs.VLLM_MQ_MAX_CHUNK_BYTES_MB * 1024 * 1024
            self.warm_spare_rpc_mq = MessageQueue(
                self.world_size, self.world_size, max_chunk_bytes=max_chunk_bytes)
            warm_spare_scheduler_handle = self.warm_spare_rpc_mq.export_handle()
            
            # Create warm spare workers with proper DP configuration
            # Each engine process only creates its own warm spare worker(s)
            unready_warm_spares: list[UnreadyWorkerProcHandle] = []
            
            import copy
            
            # Determine which warm spare workers this engine process should create
            # In DP mode, each engine process creates warm spares for its DP rank only
            dp_size = self.vllm_config.parallel_config.data_parallel_size
            if dp_size > 1:
                # Each engine process creates warm spares for its own DP rank
                # The warm spare will have the same dp_rank as this engine
                my_dp_rank = self.vllm_config.parallel_config.data_parallel_rank
                logger.info("Engine DP rank %d creating its warm spare workers",
                           my_dp_rank)
            else:
                # No DP, single engine creates all warm spares
                my_dp_rank = 0
            
            # Create warm spare workers for all ranks in this TP/PP group
            # but with the dp_rank of this engine process
            for rank_within_tp_pp in range(self.world_size):
                # Create a copy of vllm_config with proper settings for warm spares
                warm_spare_config = copy.deepcopy(self.vllm_config)
                
                # Set process type to warm_spare_worker
                warm_spare_config.parallel_config.process_type = "warm_spare_worker"
                
                # Warm spare inherits the dp_rank from this engine process
                warm_spare_config.parallel_config.data_parallel_rank = my_dp_rank
                
                # Verify the dp_rank is set correctly
                assert warm_spare_config.parallel_config.data_parallel_rank == my_dp_rank, \
                    f"DP rank mismatch: expected {my_dp_rank}, got {warm_spare_config.parallel_config.data_parallel_rank}"
                
                logger.info("Engine DP rank %d creating warm spare: rank_within_tp_pp=%d, dp_rank=%d, dp_size=%d",
                           my_dp_rank, rank_within_tp_pp, my_dp_rank,
                           warm_spare_config.parallel_config.data_parallel_size)
                
                # Warm spares share companion processes with primary workers
                
                # Pass the rank within TP/PP group, not global rank
                unready_warm_spares.append(
                    WorkerProc.make_worker_process(
                        vllm_config=warm_spare_config,
                        local_rank=rank_within_tp_pp,  # Local rank within the node
                        rank=rank_within_tp_pp,  # Rank within TP/PP group (DP handled by parallel_state)
                        distributed_init_method=distributed_init_method,
                        input_shm_handle=warm_spare_scheduler_handle,
                    ))
            
            # Wait for warm spares to be ready
            self.warm_spare_workers = WorkerProc.wait_for_ready(unready_warm_spares)
            # Ensure message queues are ready (must mirror primary path)
            self.warm_spare_rpc_mq.wait_until_ready()
            for w in self.warm_spare_workers:
                w.worker_response_mq.wait_until_ready()
            
            # Initialize warm spares to phase 1 only
            self._initialize_warm_spares_phase1()
            
            logger.info("Warm spare workers created and initialized to phase 1")
            logger.info("Warm spares ready for fast failover")
            
            # Now start the worker monitor (was skipped when parent called it)
            self.start_worker_monitor()
        except Exception as e:
            logger.error(f"Failed to create warm spare workers: {e}")
            logger.warning("Continuing without warm spares - failover will not be available")
            # Clean up any partially created workers
            if hasattr(self, 'warm_spare_workers'):
                for worker in self.warm_spare_workers:
                    try:
                        worker.proc.terminate()
                    except:
                        pass
            self.warm_spare_workers = []
            self.warm_spares_initialized = False
        
    def _initialize_warm_spares_phase1(self) -> None:
        """Initialize warm spare workers through phase 1 only."""
        try:
            # Phase 1: Device + fake distributed setup
            logger.info("Initializing warm spare devices...")
            self._collective_rpc_warm_spares("init_device", timeout=60)
            logger.info("Warm spare devices initialized")
            
            # Phase 2: Load model weights (via IPC)
            logger.info("Loading warm spare model weights via IPC...")
            self._collective_rpc_warm_spares("load_model", timeout=120)
            logger.info("Warm spare model weights loaded")
            
            # Phase 3: Pre-compile model
            logger.info("Pre-compiling warm spare models...")
            self._collective_rpc_warm_spares("precompile_model", timeout=180)
            logger.info("Warm spare models pre-compiled")
            
            # Stop here - don't finalize two-phase init or allocate KV cache
            self.warm_spares_initialized = True
            logger.info("Warm spares initialized to phase 1 (model loaded, fake distributed)")
        except TimeoutError as e:
            logger.error(f"Warm spare initialization timed out: {e}")
            # Kill warm spare workers that may be hung
            for worker in self.warm_spare_workers:
                try:
                    worker.proc.terminate()
                except:
                    pass
            raise
        except Exception as e:
            logger.error(f"Failed to initialize warm spares: {e}")
            raise
        
    def _collective_rpc_warm_spares(
        self,
        method: Union[str, Callable],
        timeout: Optional[float] = None,
        args: tuple = (),
        kwargs: Optional[dict] = None
    ) -> list[Any]:
        """Execute collective RPC on warm spare workers."""
        # Similar to collective_rpc but for warm spares
        deadline = None if timeout is None else time.monotonic() + timeout
        kwargs = kwargs or {}
        
        # Send to warm spare workers
        self.warm_spare_rpc_mq.enqueue((method, args, kwargs, None))
        
        responses = []
        for w in self.warm_spare_workers:
            if deadline:
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    raise TimeoutError(f"Warm spare RPC {method} timed out")
                dequeue_timeout = remaining
            else:
                dequeue_timeout = None
                
            status, result = w.worker_response_mq.dequeue(timeout=dequeue_timeout)
            if status != WorkerProc.ResponseStatus.SUCCESS:
                raise RuntimeError(f"Warm spare worker failed: {result}")
            responses.append(result)
            
        return responses
        
    def set_dp_warm_spare_callback(self, callback: Callable[[], bool]):
        """Set callback for DP-wide warm spare coordination.
        
        Args:
            callback: Function to call when warm spare switch is needed in DP setup
        """
        self.dp_warm_spare_callback = callback
        logger.info("DP warm spare callback registered")
    
    def execute_warm_spare_switch(self) -> bool:
        """Execute warm spare switch when commanded by EngineCore.
        
        This is called when EngineCore coordinates DP-wide switch.
        """
        logger.info("Executing warm spare switch on EngineCore command")
        
        # Clear the queued flag now that we're actually executing
        self._warm_spare_switch_queued = False
        
        # Don't attempt if we're shutting down
        if getattr(self, 'shutting_down', False):
            logger.info("System is shutting down, skipping warm spare switch")
            return False
            
        if self.switching_to_warm_spare:
            logger.warning("Already switching to warm spare")
            return True  # Consider it success if already switching
            
        if not self.warm_spares_initialized:
            logger.warning("Warm spares not ready yet")
            # Wait for warm spares to be ready
            logger.info("Waiting for warm spares to initialize...")
            for i in range(60):  # Wait up to 60 seconds
                time.sleep(1)
                if self.warm_spares_initialized:
                    logger.info(f"Warm spares ready after {i+1} seconds")
                    break
            else:
                logger.error("Warm spares failed to initialize in time")
                return False
        
        # Terminate primary workers and switch to warm spares
        self._terminate_all_primary_workers()
        
        try:
            success = self._switch_to_warm_spares_internal()
            if success:
                # Restart monitor after successful switch
                logger.info("Warm spare switch successful, restarting worker monitor")
                self.start_worker_monitor()
            return success
        except Exception as e:
            logger.error(f"Exception during warm spare switch: {e}")
            logger.exception("Full traceback:")
            return False
    
    def _attempt_warm_spare_recovery(self, failed_worker_idx: int) -> bool:
        """Attempt to recover from worker failure using warm spares.
        
        For DP setups, coordinates with all DP ranks to switch together.
        
        Returns:
            bool: True if recovery successful (or queued), False otherwise
        """
        # Don't attempt recovery if we're shutting down
        if getattr(self, 'shutting_down', False):
            logger.info("System is shutting down, skipping warm spare recovery")
            return False
            
        if self.switching_to_warm_spare:
            logger.warning("Already switching to warm spare")
            return False
            
        # Prevent monitor from restarting while switch is pending
        if getattr(self, '_warm_spare_switch_queued', False):
            logger.debug("Warm spare switch already queued, waiting for execution")
            return False  # Return false to prevent monitor restart
            
        # Check if we need DP-wide coordination
        if self.dp_warm_spare_callback is not None:
            # This is a DP setup - coordinate with all ranks
            logger.info(f"Worker {failed_worker_idx} failed in DP setup - "
                       f"initiating DP-wide warm spare switch across all ranks")
            try:
                # The callback now queues the switch and returns immediately
                result = self.dp_warm_spare_callback()
                if result:
                    # Mark that switch is queued to prevent monitor restarts
                    self._warm_spare_switch_queued = True
                    logger.info("Warm spare switch queued for next sync point")
                    # Return True to indicate recovery is in progress
                    # Monitor won't restart because we removed that in monitor_workers
                    return True
                else:
                    logger.error("Failed to queue warm spare switch")
                    return False
            except Exception as e:
                logger.error(f"DP warm spare coordination failed: {e}")
                # Fall back to local recovery if DP coordination fails
                logger.info("Falling back to local warm spare recovery")
                return self.execute_warm_spare_switch()
        else:
            # Non-DP case: proceed with local recovery immediately
            logger.info(f"Worker {failed_worker_idx} failed (non-DP) - "
                       f"switching ALL {len(self.workers)} workers to warm spares locally")
            success = self.execute_warm_spare_switch()
            if success:
                # Restart monitor for non-DP case
                self.start_worker_monitor()
            return success
        
    def _switch_to_warm_spares_internal(self) -> bool:
        """Internal method to perform the warm spare switch.
        
        Returns:
            bool: True if successful, False otherwise
        """
        if self.switching_to_warm_spare:
            logger.warning("Switch already in progress")
            return False
            
        self.switching_to_warm_spare = True
        # Set recovery flag so EngineCore knows to clean up
        self.warm_spare_recovery_occurred = True
        logger.info("Starting switch to warm spare workers...")

        # NOTE: New warm spares are created asynchronously in background
        try:
            # Signal warm spares that primary workers are terminated
            # This ensures they can safely proceed with memory profiling
            logger.info("Signaling warm spares that primaries are terminated...")
            self._collective_rpc_warm_spares("acknowledge_primaries_terminated", timeout=10)
            
            # Complete warm spare initialization
            try:
                self._complete_warm_spare_initialization()
            except Exception as e:
                logger.error(f"Failed to complete warm spare initialization: {e}")
                return False
            
            # Swap primary and warm spare workers
            old_rpc_mq = self.rpc_broadcast_mq
            
            self.workers = self.warm_spare_workers
            self.rpc_broadcast_mq = self.warm_spare_rpc_mq
            
            # Clear warm spare references
            self.warm_spare_workers = []
            self.warm_spares_initialized = False
            
            # Keep the warm spare companion coordinator environment variable
            # since we're reusing the primary coordinator and future warm spares may need it
            
            # MessageQueue doesn't have a close method - cleanup happens on GC
            # Just let the old queue be garbage collected
            del old_rpc_mq
            
            # Reset is_failed flag since we recovered
            self.is_failed = False
            
            # CRITICAL: Reset switching flag IMMEDIATELY after successful switch
            # This allows the engine to resume serving requests right away
            self.switching_to_warm_spare = False
            logger.info("Successfully switched to warm spare workers - engine can now serve")
            
            # No need to reset port counter anymore - we use fixed ports:
            # - Promoted workers now use base+1 for their phase 2
            # - New warm spares will use base+100 for their phase 1
            # The port allocation is handled by parallel_state.py based on process_type
            
            # Create new warm spares in background - non-blocking
            # The engine continues serving while new warm spares are created
            # IMPORTANT: Ensure promoted workers have fully released their old ports
            # before creating new warm spares that will use those same ports
            def create_new_warm_spares_with_sync():
                try:
                    # Use barrier to ensure all promoted workers have completed cleanup
                    # The promoted workers just destroyed their phase 1 distributed group
                    logger.info("Ensuring promoted workers have released old ports...")
                    
                    # Call a synchronization method on promoted workers to ensure cleanup
                    # This will block until all workers confirm they've released resources
                    self.collective_rpc("synchronize_after_promotion", timeout=10)
                    logger.info("All promoted workers confirmed port cleanup complete")
                    
                    logger.info("Creating new warm spare workers...")
                    self._create_warm_spare_workers()
                except Exception as e:
                    logger.error(f"Failed to create new warm spares: {e}")
                    logger.warning("Continuing without new warm spares - failover will not be available")
            
            logger.info("Starting new warm spare creation in background (with synchronization)")
            threading.Thread(
                target=create_new_warm_spares_with_sync,
                daemon=True,
                name="WarmSpareCreator"
            ).start()
            
            return True
            
        except Exception as e:
            logger.error(f"Unexpected error during warm spare switch: {e}")
            self.switching_to_warm_spare = False
            return False
        
    def _terminate_all_primary_workers(self) -> None:
        """Terminate all primary workers and ensure complete cleanup.
        
        This is called when ANY worker fails, to prevent NCCL communication
        hangs in the remaining workers. Uses proper process synchronization.
        """
        # Check if we're shutting down normally vs recovery
        is_normal_shutdown = getattr(self, 'shutting_down', False)
        
        if is_normal_shutdown:
            logger.info("Terminating ALL primary workers for shutdown...")
        else:
            logger.info("Terminating ALL primary workers to prevent NCCL hangs...")
        
        # First, send SIGKILL to all workers for immediate termination
        for i, worker in enumerate(self.workers):
            try:
                if worker.proc.is_alive():
                    logger.debug(f"Killing primary worker {i} (pid={worker.proc.pid})")
                    worker.proc.kill()
            except Exception as e:
                logger.warning(f"Error killing worker {i}: {e}")
        
        # Now join all processes to ensure they're completely terminated
        # This blocks until each process is fully cleaned up by the OS
        for i, worker in enumerate(self.workers):
            try:
                logger.debug(f"Waiting for worker {i} termination...")
                worker.proc.join(timeout=5.0)
                if worker.proc.is_alive():
                    # This shouldn't happen after SIGKILL, but handle it
                    logger.error(f"Worker {i} survived SIGKILL, trying again...")
                    os.kill(worker.proc.pid, signal.SIGKILL)
                    worker.proc.join(timeout=2.0)
            except (ProcessLookupError, OSError):
                pass  # Process already dead
            except TimeoutError:
                logger.error(f"Worker {i} failed to terminate after 7 seconds")
        
        logger.info("All primary workers terminated and joined")
    
    def switch_to_warm_spares(self) -> None:
        """Public API to manually trigger warm spare switch.
        
        This allows external callers (e.g., EngineCore) to trigger a switch.
        """
        if not self.warm_spares_initialized:
            logger.error("Cannot switch: Warm spares not yet initialized")
            raise RuntimeError("Warm spares not initialized")
            
        # Terminate all primary workers first
        self._terminate_all_primary_workers()
        
        # Attempt the switch
        if not self._switch_to_warm_spares_internal():
            raise RuntimeError("Failed to switch to warm spares")
            
    def _complete_warm_spare_initialization(self) -> None:
        """Complete initialization of warm spare workers (phase 2)."""
        logger.info("Completing warm spare initialization...")
        
        try:
            # Phase 4: Finalize two-phase init (switch to real distributed)
            # Promoted warm spares keep their process_type as "warm_spare_worker"
            # but will use base+1 for phase 2 (detected by torch.distributed.is_initialized())
            
            logger.info("Finalizing two-phase init for warm spares...")
            logger.info("Promoted warm spares will use base+1 for phase 2 (same as primary workers)")
            self._collective_rpc_warm_spares(
                "finalize_two_phase_init", 
                timeout=60
            )
            
            # Check if we should skip memory profiling for faster failover
            skip_memory_profiling = os.environ.get(
                "VLLM_SKIP_WARM_SPARE_MEMORY_PROFILING", "0") == "1"
            
            if skip_memory_profiling:
                # Verify that parallel configuration matches exactly
                logger.info("Skipping memory profiling (using primary config)")
                # Get the existing KV cache config from primary initialization
                # Use the stored configs from when primary workers were initialized
                if not hasattr(self, 'primary_kv_cache_configs'):
                    raise RuntimeError(
                        "Cannot skip memory profiling: primary KV cache configs not stored")
                
                kv_cache_configs = self.primary_kv_cache_configs
                logger.info(f"Using cached KV config with {kv_cache_configs[0].num_blocks} GPU blocks")
            else:
                # Do normal memory profiling
                # CRITICAL: Refresh memory snapshot after primary workers are gone
                # The warm spares' init_snapshot was taken during phase 1 when primary
                # workers were still using GPU memory. We need a fresh snapshot now.
                logger.info("Refreshing memory snapshot for accurate profiling...")
                self._collective_rpc_warm_spares("refresh_memory_snapshot", timeout=30)
                
                # Phase 5: Determine available memory
                logger.info("Determining available memory for warm spares...")
                available_memory = self._collective_rpc_warm_spares(
                    "determine_available_memory", timeout=120)
                
                # Phase 6: Initialize KV cache
                # Use the same logic as EngineCore to determine cache config
                from vllm.v1.core.kv_cache_utils import get_kv_cache_config, unify_kv_cache_configs
                
                logger.info("Getting KV cache specs from warm spares...")
                kv_cache_specs = self._collective_rpc_warm_spares("get_kv_cache_spec", timeout=30)
                kv_cache_configs = [
                    get_kv_cache_config(self.vllm_config, spec, mem)
                    for spec, mem in zip(kv_cache_specs, available_memory)
                ]
                unify_kv_cache_configs(kv_cache_configs)
            
            logger.info(f"Initializing KV cache with {kv_cache_configs[0].num_blocks} GPU blocks...")
            self._collective_rpc_warm_spares(
                "initialize_from_config", args=(kv_cache_configs,), timeout=60)
            
            # Phase 7: Compile/warm up model
            logger.info("Compiling/warming up models on warm spares...")
            self._collective_rpc_warm_spares("compile_or_warm_up_model", timeout=180)
            
            logger.info("Warm spare initialization completed successfully")
            
        except TimeoutError as e:
            logger.error(f"Timeout during warm spare initialization: {e}")
            raise
        except Exception as e:
            logger.error(f"Error during warm spare initialization: {e}")
            logger.exception("Full traceback:")
            raise
        
    def collective_rpc(
        self,
        method: Union[str, Callable],
        timeout: Optional[float] = None,
        args: tuple = (),
        kwargs: Optional[dict] = None,
        non_block: bool = False,
        unique_reply_rank: Optional[int] = None
    ) -> list[Any]:
        """
        Execute collective RPC with timeout support for hung worker detection.
        
        If timeout is specified and exceeded, attempts to switch to warm spares.
        """
        # Use default timeout if not specified
        if timeout is None:
            timeout = self.rpc_timeout
            
        try:
            # Try to execute on primary workers with timeout
            return super().collective_rpc(
                method=method,
                timeout=timeout,
                args=args,
                kwargs=kwargs,
                non_block=non_block,
                unique_reply_rank=unique_reply_rank
            )
        except (TimeoutError, FutureTimeoutError):
            logger.error(f"Collective RPC {method} timed out after {timeout}s")
            
            # Attempt to switch to warm spares
            if not self.switching_to_warm_spare and self.warm_spares_initialized:
                logger.info("Attempting to switch to warm spare workers...")
                self.switch_to_warm_spares()
                
                # Retry the RPC on the new primary workers (former warm spares)
                return super().collective_rpc(
                    method=method,
                    timeout=timeout,
                    args=args,
                    kwargs=kwargs,
                    non_block=non_block,
                    unique_reply_rank=unique_reply_rank
                )
            else:
                raise
                
    def shutdown(self) -> None:
        """Shutdown both primary and warm spare workers."""
        # Set the shutting_down flag to prevent monitor from trying recovery
        if not getattr(self, 'shutting_down', False):
            self.shutting_down = True
            logger.info("WarmSpareExecutor shutting down...")
        
        if hasattr(self, 'monitoring_enabled'):
            self.monitoring_enabled = False
        
        # No need to shutdown companion coordinator as warm spares reuse the primary one
        # The primary executor handles companion coordinator lifecycle
        
        # Shutdown warm spare workers
        if hasattr(self, 'warm_spare_workers'):
            logger.info("Shutting down %d warm spare workers...", 
                       len(self.warm_spare_workers))
            for worker in self.warm_spare_workers:
                try:
                    worker.proc.terminate()
                    worker.proc.join(timeout=5)
                    if worker.proc.is_alive():
                        worker.proc.kill()
                except Exception as e:
                    logger.warning(f"Error shutting down warm spare: {e}")
                    
        # Shutdown primary workers
        super().shutdown()
