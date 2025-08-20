# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""
Warm Spare Executor for resilient worker management with IPC loading.

This executor maintains "warm spare" workers that are partially initialized
(model loaded via IPC but no KV cache) alongside primary workers. On failure
or reinitialize request, it switches from primary to warm spare workers.
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
from vllm.utils import get_distributed_init_method, get_loopback_ip, get_open_port
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
        if not vllm_config.load_config.enable_ipc_loading:
            raise ValueError(
                "WarmSpareExecutor requires IPC loading to be enabled. "
                "Set --enable-ipc-loading or load_config.enable_ipc_loading=True"
            )
        
        # Enable two-phase initialization
        os.environ["VLLM_TWO_PHASE_INIT"] = "1"
        
        # Track initialization state - MUST be set before calling parent init
        self.warm_spares_initialized = False
        self.switching_to_warm_spare = False
        self.warm_spare_recovery_occurred = False
        
        # Worker monitoring  
        self.rpc_timeout = DEFAULT_RPC_TIMEOUT
        
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
            
        workers = self.workers
        self_ref = weakref.ref(self)

        def monitor_workers():
            import multiprocessing.connection
            sentinels = [h.proc.sentinel for h in workers]
            died = multiprocessing.connection.wait(sentinels)
            _self = self_ref()
            if not _self or getattr(_self, 'shutting_down', False):
                return
                
            # Find which worker died
            failed_idx = next(i for i, h in enumerate(workers)
                            if h.proc.sentinel == died[0])
            proc_name = workers[failed_idx].proc.name
            
            logger.error("Worker proc %s (idx %d) died unexpectedly, "
                        "attempting warm spare recovery...", proc_name, failed_idx)
            
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
            else:
                logger.info("Successfully recovered using warm spares")
                # Start monitoring the new workers
                _self.start_worker_monitor()

        Thread(target=monitor_workers,
               daemon=True,
               name="WarmSpareMonitor").start()
        
    def _create_warm_spare_workers(self) -> None:
        """Create and partially initialize warm spare workers.
        
        This method can be called from a background thread to avoid blocking
        the main engine operations. The engine can continue serving while
        warm spares are being created and initialized.
        """
        logger.info("Creating warm spare workers in background...")
        
        try:
            # Get new distributed init method for warm spares
            distributed_init_method = get_distributed_init_method(
                get_loopback_ip(), get_open_port())
            
            # Create scheduler output handle for warm spares
            max_chunk_bytes = envs.VLLM_MQ_MAX_CHUNK_BYTES_MB * 1024 * 1024
            self.warm_spare_rpc_mq = MessageQueue(
                self.world_size, self.world_size, max_chunk_bytes=max_chunk_bytes)
            warm_spare_scheduler_handle = self.warm_spare_rpc_mq.export_handle()
            
            # Create warm spare workers
            # Set environment variable to indicate these are warm spares
            os.environ["VLLM_WARM_SPARE_WORKER"] = "1"
            unready_warm_spares: list[UnreadyWorkerProcHandle] = []
            for rank in range(self.world_size):
                unready_warm_spares.append(
                    WorkerProc.make_worker_process(
                        vllm_config=self.vllm_config,
                        local_rank=rank,
                        rank=rank,
                        distributed_init_method=distributed_init_method,
                        input_shm_handle=warm_spare_scheduler_handle,
                    ))
            # Clear the environment variable after creating workers
            os.environ.pop("VLLM_WARM_SPARE_WORKER", None)
            
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
        
    def _attempt_warm_spare_recovery(self, failed_worker_idx: int) -> bool:
        """Attempt to recover from worker failure using warm spares.
        
        Returns:
            bool: True if recovery successful, False otherwise
        """
        if self.switching_to_warm_spare:
            logger.warning("Already switching to warm spare")
            return False
            
        if not self.warm_spares_initialized:
            logger.warning(f"Worker {failed_worker_idx} failed but warm spares not ready yet")
            # Wait for warm spares to be ready - we need them for failover
            logger.info("Waiting for warm spares to initialize...")
            for i in range(60):  # Wait up to 60 seconds
                time.sleep(1)
                if self.warm_spares_initialized:
                    logger.info(f"Warm spares ready after {i+1} seconds")
                    break
            else:
                logger.error("Warm spares failed to initialize in time")
                return False
                
        logger.info(f"Worker {failed_worker_idx} failure detected - "
                   f"switching ALL {len(self.workers)} workers to warm spares")
        
        # Immediately terminate ALL primary workers to prevent NCCL hangs
        self._terminate_all_primary_workers()
        
        # Now switch to warm spares
        try:
            return self._switch_to_warm_spares_internal()
        except Exception as e:
            logger.error(f"Exception during warm spare recovery: {e}")
            logger.exception("Full traceback:")
            return False
        
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
            
            # MessageQueue doesn't have a close method - cleanup happens on GC
            # Just let the old queue be garbage collected
            del old_rpc_mq
            
            # Reset is_failed flag since we recovered
            self.is_failed = False
            
            # CRITICAL: Reset switching flag IMMEDIATELY after successful switch
            # This allows the engine to resume serving requests right away
            self.switching_to_warm_spare = False
            logger.info("Successfully switched to warm spare workers - engine can now serve")
            
            # Create new warm spares in background - non-blocking
            # The engine continues serving while new warm spares are created
            logger.info("Starting new warm spare creation in background")
            threading.Thread(
                target=self._create_warm_spare_workers,
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
            # This also calls recreate_persistent_buffers() to ensure fresh CPU/GPU
            # buffers with correct initialization - critical for position encoding
            logger.info("Finalizing two-phase init for warm spares...")
            self._collective_rpc_warm_spares("finalize_two_phase_init", timeout=60)
            
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
        except (TimeoutError, FutureTimeoutError) as e:
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
        if hasattr(self, 'monitoring_enabled'):
            self.monitoring_enabled = False
        
        # Shutdown warm spare workers
        if hasattr(self, 'warm_spare_workers'):
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
