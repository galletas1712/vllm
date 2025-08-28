# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""A GPU worker class."""
import copy
import gc
import os
from contextlib import AbstractContextManager, nullcontext
from typing import TYPE_CHECKING, Any, Optional

import torch
import torch.distributed
import torch.nn as nn

import vllm.envs as envs
from vllm.config import VllmConfig
from vllm.distributed import (ensure_model_parallel_initialized,
                              init_distributed_environment,
                              set_custom_all_reduce)
from vllm.distributed.kv_transfer import ensure_kv_transfer_initialized
from vllm.distributed.parallel_state import (
    destroy_model_parallel,
    destroy_distributed_environment,
    model_parallel_is_initialized,
    FAKE_DISTRIBUTED_BACKEND,
)
from vllm.distributed.parallel_state import get_pp_group, get_tp_group
from vllm.logger import init_logger
from vllm.lora.request import LoRARequest
from vllm.model_executor import set_random_seed
from vllm.model_executor.warmup.kernel_warmup import kernel_warmup
from vllm.platforms import current_platform
from vllm.sequence import IntermediateTensors
from vllm.tasks import SupportedTask
from vllm.utils import GiB_bytes, MemorySnapshot, memory_profiling
from vllm.v1.engine import ReconfigureDistributedRequest, ReconfigureRankType
from vllm.v1.kv_cache_interface import KVCacheConfig, KVCacheSpec
from vllm.v1.outputs import EMPTY_MODEL_RUNNER_OUTPUT, ModelRunnerOutput
from vllm.v1.utils import report_usage_stats
from vllm.v1.worker.gpu_model_runner import GPUModelRunner
from vllm.v1.worker.worker_base import WorkerBase

logger = init_logger(__name__)

def get_model_weights_size_bytes(model: torch.nn.Module) -> int:
    """Calculate total unique bytes of all parameters & buffers on CUDA.
    
    This accounts for tied weights by tracking unique storage pointers.
    """
    seen_storages: set[int] = set()
    total = 0
    
    # Count parameters
    for param in model.parameters():
        if not param.is_cuda:
            continue
        data_ptr = param.storage().data_ptr()
        if data_ptr in seen_storages:
            continue  # Skip tied weights / shared storage
        seen_storages.add(data_ptr)
        total += param.storage().nbytes()
    
    # Count buffers (e.g., layer norm weights)
    for buffer in model.buffers():
        if not buffer.is_cuda:
            continue
        data_ptr = buffer.storage().data_ptr()
        if data_ptr in seen_storages:
            continue
        seen_storages.add(data_ptr)
        total += buffer.storage().nbytes()
    
    return total


if TYPE_CHECKING:
    from vllm.model_executor.model_loader.tensorizer import TensorizerConfig
    from vllm.v1.core.sched.output import SchedulerOutput


class Worker(WorkerBase):

    def __init__(
        self,
        vllm_config: VllmConfig,
        local_rank: int,
        rank: int,
        distributed_init_method: str,
        is_driver_worker: bool = False,
    ):

        super().__init__(vllm_config=vllm_config,
                         local_rank=local_rank,
                         rank=rank,
                         distributed_init_method=distributed_init_method,
                         is_driver_worker=is_driver_worker)

        if self.model_config.trust_remote_code:
            # note: lazy import to avoid importing torch before initializing
            from vllm.utils import init_cached_hf_modules
            init_cached_hf_modules()

        # Buffers saved before sleep
        self._sleep_saved_buffers: dict[str, torch.Tensor] = {}

        # Torch profiler. Enabled and configured through env vars:
        # VLLM_TORCH_PROFILER_DIR=/path/to/save/trace
        if envs.VLLM_TORCH_PROFILER_DIR:
            torch_profiler_trace_dir = envs.VLLM_TORCH_PROFILER_DIR
            logger.info("Profiling enabled. Traces will be saved to: %s",
                        torch_profiler_trace_dir)
            logger.debug(
                "Profiler config: record_shapes=%s,"
                "profile_memory=%s,with_stack=%s,with_flops=%s",
                envs.VLLM_TORCH_PROFILER_RECORD_SHAPES,
                envs.VLLM_TORCH_PROFILER_WITH_PROFILE_MEMORY,
                envs.VLLM_TORCH_PROFILER_WITH_STACK,
                envs.VLLM_TORCH_PROFILER_WITH_FLOPS,
            )
            self.profiler = torch.profiler.profile(
                activities=[
                    torch.profiler.ProfilerActivity.CPU,
                    torch.profiler.ProfilerActivity.CUDA,
                ],
                record_shapes=envs.VLLM_TORCH_PROFILER_RECORD_SHAPES,
                profile_memory=envs.VLLM_TORCH_PROFILER_WITH_PROFILE_MEMORY,
                with_stack=envs.VLLM_TORCH_PROFILER_WITH_STACK,
                with_flops=envs.VLLM_TORCH_PROFILER_WITH_FLOPS,
                on_trace_ready=torch.profiler.tensorboard_trace_handler(
                    torch_profiler_trace_dir, use_gzip=True))
        else:
            self.profiler = None

    def sleep(self, level: int = 1) -> None:
        from vllm.device_allocator.cumem import CuMemAllocator

        free_bytes_before_sleep = torch.cuda.mem_get_info()[0]

        # Save the buffers before level 2 sleep
        if level == 2:
            model = self.model_runner.model
            self._sleep_saved_buffers = {
                name: buffer.cpu().clone()
                for name, buffer in model.named_buffers()
            }

        allocator = CuMemAllocator.get_instance()
        allocator.sleep(offload_tags=("weights", ) if level == 1 else tuple())
        free_bytes_after_sleep, total = torch.cuda.mem_get_info()
        freed_bytes = free_bytes_after_sleep - free_bytes_before_sleep
        used_bytes = total - free_bytes_after_sleep
        assert freed_bytes >= 0, "Memory usage increased after sleeping."
        logger.info(
            "Sleep mode freed %.2f GiB memory, "
            "%.2f GiB memory is still in use.", freed_bytes / GiB_bytes,
            used_bytes / GiB_bytes)

    def wake_up(self, tags: Optional[list[str]] = None) -> None:
        from vllm.device_allocator.cumem import CuMemAllocator

        allocator = CuMemAllocator.get_instance()
        allocator.wake_up(tags)

        # Restore the buffers after level 2 sleep
        if len(self._sleep_saved_buffers):
            model = self.model_runner.model
            for name, buffer in model.named_buffers():
                if name in self._sleep_saved_buffers:
                    buffer.data.copy_(self._sleep_saved_buffers[name].data)
            self._sleep_saved_buffers = {}

    def _maybe_get_memory_pool_context(self,
                                       tag: str) -> AbstractContextManager:
        if self.vllm_config.model_config.enable_sleep_mode:
            from vllm.device_allocator.cumem import CuMemAllocator

            allocator = CuMemAllocator.get_instance()
            if tag == "weights":
                assert allocator.get_current_usage() == 0, (
                    "Sleep mode can only be "
                    "used for one instance per process.")
            context = allocator.use_memory_pool(tag=tag)
        else:
            context = nullcontext()
        return context

    def initialize_cache(self, num_gpu_blocks: int,
                         num_cpu_blocks: int) -> None:
        self.cache_config.num_gpu_blocks = num_gpu_blocks
        self.cache_config.num_cpu_blocks = num_cpu_blocks

    def init_device(self):
        # Check if two-phase initialization is enabled
        use_two_phase = os.environ.get("VLLM_TWO_PHASE_INIT", "0") == "1"
        
        if self.device_config.device.type == "cuda":
            # torch.distributed.all_reduce does not free the input tensor until
            # the synchronization point. This causes the memory usage to grow
            # as the number of all_reduce calls increases. This env var disables
            # this behavior.
            # Related issue:
            # https://discuss.pytorch.org/t/cuda-allocation-lifetime-for-inputs-to-distributed-all-reduce/191573
            os.environ["TORCH_NCCL_AVOID_RECORD_STREAMS"] = "1"

            # This env var set by Ray causes exceptions with graph building.
            os.environ.pop("NCCL_ASYNC_ERROR_HANDLING", None)
            self.device = torch.device(f"cuda:{self.local_rank}")
            current_platform.set_device(self.device)

            _check_if_gpu_supports_dtype(self.model_config.dtype)
            gc.collect()
            torch.cuda.empty_cache()

            # take current memory snapshot
            self.init_snapshot = MemorySnapshot()
            self.requested_memory = (self.init_snapshot.total_memory *
                                     self.cache_config.gpu_memory_utilization)
            GiB = lambda b: round(b / GiB_bytes, 2)
            
            # When using IPC loading, we don't need to allocate memory for model weights
            # as they are already loaded by the model server. Skip the strict check
            # since we'll do proper accounting after model loading.
            if self.load_config.enable_companion_process:
                logger.info(
                    "IPC loading enabled: Skipping initial memory check. "
                    "Free memory: %.2f GiB, Total: %.2f GiB, Requested: %.2f GiB",
                    GiB(self.init_snapshot.free_memory), 
                    GiB(self.init_snapshot.total_memory), 
                    GiB(self.requested_memory)
                )
            elif self.init_snapshot.free_memory < self.requested_memory:
                raise ValueError(
                    f"Free memory on device "
                    f"({GiB(self.init_snapshot.free_memory)}/"
                    f"{GiB(self.init_snapshot.total_memory)} GiB) on startup "
                    f"is less than desired GPU memory utilization "
                    f"({self.cache_config.gpu_memory_utilization}, "
                    f"{GiB(self.requested_memory)} GiB). Decrease GPU memory "
                    f"utilization or reduce GPU memory used by other processes."
                )
        else:
            raise RuntimeError(
                f"Not support device type: {self.device_config.device}")
        
        if use_two_phase:
            # Initialize with FakeGroupCoordinator
            logger.info(
                "Init: creating fake distributed groups "
                "(two-phase enabled)")
            init_worker_distributed_environment(
                self.vllm_config, self.rank,
                self.distributed_init_method,
                self.local_rank,
                backend=FAKE_DISTRIBUTED_BACKEND  # Use fake backend
            )
            # Store flag to complete initialization later
            self._two_phase_init_pending = True
        else:
            # Standard initialization
            init_worker_distributed_environment(
                self.vllm_config, self.rank,
                self.distributed_init_method,
                self.local_rank,
                current_platform.dist_backend
            )
            self._two_phase_init_pending = False
        
        # Set random seed.
        set_random_seed(self.model_config.seed)

        # Construct the model runner
        self.model_runner: GPUModelRunner = GPUModelRunner(
            self.vllm_config, self.device)

        if self.rank == 0:
            # If usage stat is enabled, collect relevant info.
            report_usage_stats(self.vllm_config)

    # FIXME(youkaichao & ywang96): Use TorchDispatchMode instead of memory pool
    # to hijack tensor allocation.
    def load_model(self) -> None:
        if getattr(self, '_two_phase_init_pending', False):
            logger.info("Init: loading model weights")
        
        # Validate companion server availability if using IPC loading
        if self.load_config.enable_companion_process:
            self._validate_companion_configuration()
        
        eep_scale_up = os.environ.get("VLLM_ELASTIC_EP_SCALE_UP_LAUNCH") == "1"
        with self._maybe_get_memory_pool_context(tag="weights"):
            self.model_runner.load_model(eep_scale_up=eep_scale_up)
    
    def _validate_companion_configuration(self) -> None:
        """Validate that companion configuration is correct for this worker.
        
        This provides early validation and helpful error messages before
        attempting to load the model via IPC.
        """
        import os
        
        # Get environment info
        cuda_visible = os.environ.get('CUDA_VISIBLE_DEVICES', '')
        logical_device = torch.cuda.current_device()
        
        # Log configuration for debugging
        logger.info(
            "[GPU-WORKER] Companion configuration validation:\n"
            "  CUDA_VISIBLE_DEVICES: %s\n"
            "  Logical device (current): %d\n"
            "  DP rank: %d, DP size: %d\n"
            "  TP rank: %d, TP size: %d\n"
            "  PP rank: %d, PP size: %d",
            cuda_visible if cuda_visible else "(not set - all GPUs visible)",
            logical_device,
            self.parallel_config.data_parallel_rank if hasattr(self.parallel_config, 'data_parallel_rank') else 0,
            self.parallel_config.data_parallel_size if hasattr(self.parallel_config, 'data_parallel_size') else 1,
            self.parallel_config.rank % self.parallel_config.tensor_parallel_size,
            self.parallel_config.tensor_parallel_size,
            (self.parallel_config.rank // self.parallel_config.tensor_parallel_size) % self.parallel_config.pipeline_parallel_size,
            self.parallel_config.pipeline_parallel_size
        )
        
        # Warn if configuration looks problematic
        if cuda_visible and self.parallel_config.data_parallel_size > 1:
            visible_gpus = len(cuda_visible.split(','))
            if visible_gpus != 1:
                logger.warning(
                    "[GPU-WORKER] Worker sees %d GPUs but expected 1 for DP. "
                    "CUDA_VISIBLE_DEVICES=%s. This may cause companion routing issues.",
                    visible_gpus, cuda_visible
                )
        
        # Check if companion config is present
        if not self.vllm_config.companion_config:
            logger.warning(
                "[GPU-WORKER] No companion config found but IPC loading is enabled. "
                "This will likely fail during model loading."
            )
        elif not self.vllm_config.companion_config.coordinator_address:
            logger.warning(
                "[GPU-WORKER] Companion coordinator address not set. "
                "IPC loading will fail."
            )

    def update_config(self, overrides: dict[str, Any]) -> None:
        self.model_runner.update_config(overrides)

    def reload_weights(self) -> None:
        with self._maybe_get_memory_pool_context(tag="weights"):
            self.model_runner.reload_weights()

    def precompile_model(self) -> None:
        """Pre-compile (torch.compile) in fake env during two-phase.

        Runs dummy forwards to trigger compilation for configured sizes only.
        Does not capture CUDA graphs or finalize distributed state.
        """
        compile_cfg = self.vllm_config.compilation_config

        # Collect sizes to compile: explicit compile sizes plus cudagraph
        # capture sizes (to avoid recompile later) when not forcing eager.
        sizes: set[int] = set()
        for s in getattr(compile_cfg, "compile_sizes", []) or []:
            if isinstance(s, int) and s > 0:
                sizes.add(int(s))
        if not self.model_config.enforce_eager:
            for s in getattr(compile_cfg, "cudagraph_capture_sizes", []) or []:
                if isinstance(s, int) and s > 0:
                    sizes.add(int(s))

        max_tokens = self.scheduler_config.max_num_batched_tokens
        bounded = sorted([s for s in sizes if s <= max_tokens], reverse=True)
        if not bounded:
            # Fallback: compile at a reasonable default size to ensure graph
            # exists before finalize.
            fallback = min(getattr(compile_cfg, "max_capture_size",
                                   max_tokens) or max_tokens, max_tokens)
            fallback = int(fallback) if fallback and fallback > 0 else 1
            bounded = [fallback]

        logger.info("Init: precompile model (sizes=%s)", bounded)
        for size in bounded:
            # Ensure this only triggers compile; no CUDA graph capture here.
            self.model_runner._dummy_run(size, skip_eplb=True)

    def acknowledge_primaries_terminated(self) -> None:
        """Acknowledge that primary workers have been terminated.
        
        This serves as a synchronization barrier to ensure warm spares
        know that primary workers are fully terminated before proceeding
        with memory-sensitive operations like profiling.
        """
        logger.info("Acknowledged primary workers termination")
        # Synchronize to ensure GPU operations are complete
        torch.cuda.synchronize()
        
        # Use a distributed barrier if available to ensure all warm spares
        # are at the same point
        if torch.distributed.is_initialized():
            try:
                torch.distributed.barrier()
                logger.info("All warm spare workers synchronized")
            except Exception as e:
                logger.warning(f"Barrier synchronization failed: {e}")
    
    def refresh_memory_snapshot(self) -> None:
        """Refresh the initial memory snapshot.
        
        This is used by warm spares after primary workers are terminated
        to get an accurate baseline for memory profiling.
        """
        logger.info("Refreshing memory snapshot for accurate profiling")
        torch.cuda.empty_cache()
        torch.cuda.synchronize()
        
        # Take a new snapshot of current memory state
        self.init_snapshot = MemorySnapshot()
        logger.info(f"New memory snapshot: {self.init_snapshot.free_memory / GiB_bytes:.2f} GiB free")
    
    @torch.inference_mode()
    def determine_available_memory(self) -> int:
        """Profiles the peak memory usage of the model to determine how much
        memory can be used for KV cache without OOMs.

        The engine will first conduct a profiling of the existing memory usage.
        Then, it calculate the free memory that can be used for KV cache in
        bytes.

        Tip:
            You may limit the usage of GPU memory
            by adjusting the `gpu_memory_utilization` parameter.
        """
        # If we're in two-phase init mode, skip profiling for now
        # It will be done after compilation in compile_or_warm_up_model
        from vllm.distributed.parallel_state import IS_FAKE_DISTRIBUTED
        assert not IS_FAKE_DISTRIBUTED(), "Memory profiling should not be called in fake mode"
        
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
        GiB = lambda b: b / GiB_bytes

        # Execute a forward pass with dummy inputs to profile the memory usage
        # of the model.
        with memory_profiling(
                self.init_snapshot,
                weights_memory=int(
                    self.model_runner.model_memory_usage)) as profile_result:
            self.model_runner.profile_run()

        free_gpu_memory = profile_result.after_profile.free_memory
        # NOTE(woosuk): Here we assume that the other processes using the same
        # GPU did not change their memory usage during the profiling.
        assert self.init_snapshot.free_memory > free_gpu_memory, (
            "Error in memory profiling. "
            f"Initial free memory {GiB(self.init_snapshot.free_memory)} GiB, "
            f"current free memory {GiB(free_gpu_memory)} GiB. "
            "This happens when other processes sharing the same container "
            "release GPU memory while vLLM is profiling during initialization. "
            "To fix this, ensure consistent GPU memory allocation or "
            "isolate vLLM in its own container.")
        # Calculate available memory for KV cache
        if self.load_config.enable_companion_process:
            # With IPC loading, the model server has already allocated memory
            # for weights. We should only subtract the torch memory increase
            # from this process, not the non_torch_increase which includes
            # the model server's memory.
            weights_bytes = get_model_weights_size_bytes(self.model_runner.model)
            logger.info(
                "IPC loading enabled: Model weights (%.2f GiB) are managed by "
                "model server process", 
                weights_bytes / GiB_bytes
            )
            logger.info(
                "Memory profiling details:\n"
                "- Free GPU memory: %.2f GiB\n"
                "- GPU utilization: %.2f\n"
                "- torch_peak_increase: %.2f GiB\n"
                "- non_torch_increase: %.2f GiB (includes model server memory)\n"
                "- weights_memory: %.2f GiB",
                free_gpu_memory / GiB_bytes,
                self.cache_config.gpu_memory_utilization,
                profile_result.torch_peak_increase / GiB_bytes,
                profile_result.non_torch_increase / GiB_bytes,
                profile_result.weights_memory / GiB_bytes
            )
            # Only subtract PyTorch memory from this process
            available_kv_cache_memory = int(free_gpu_memory * 
                                          self.cache_config.gpu_memory_utilization) \
                - profile_result.torch_peak_increase
        else:
            # Normal loading: use standard calculation
            available_kv_cache_memory = self.requested_memory \
                - profile_result.non_kv_cache_memory

        if self.load_config.enable_companion_process:
            logger.debug(
                "Initial free memory: %.2f GiB, free memory: %.2f GiB, "
                "using free memory * %.2f = %.2f GiB for KV cache calculation",
                GiB(self.init_snapshot.free_memory), GiB(free_gpu_memory),
                self.cache_config.gpu_memory_utilization,
                GiB(free_gpu_memory * self.cache_config.gpu_memory_utilization))
        else:
            logger.debug(
                "Initial free memory: %.2f GiB, free memory: %.2f GiB, "
                "requested GPU memory: %.2f GiB",
                GiB(self.init_snapshot.free_memory), GiB(free_gpu_memory),
                GiB(self.requested_memory))
        logger.debug(profile_result)
        logger.info("Available KV cache memory: %.2f GiB",
                    GiB(available_kv_cache_memory))
        gc.collect()

        return int(available_kv_cache_memory)

    def get_kv_cache_spec(self) -> dict[str, KVCacheSpec]:
        return self.model_runner.get_kv_cache_spec()

    def initialize_from_config(self, kv_cache_config: KVCacheConfig) -> None:
        """Allocate GPU KV cache with the specified kv_cache_config."""

        if self.vllm_config.model_config.enable_sleep_mode:
            from vllm.device_allocator.cumem import CuMemAllocator

            allocator = CuMemAllocator.get_instance()
            context = allocator.use_memory_pool(tag="kv_cache")
        else:
            from contextlib import nullcontext
            context = nullcontext()
        with context:
            self.model_runner.initialize_kv_cache(kv_cache_config)


    def synchronize_after_promotion(self) -> int:
        """Synchronize after being promoted from warm spare to primary.
        
        This ensures all promoted workers have completed their cleanup
        (destroyed old distributed groups, released ports) before the
        executor creates new warm spares that will use those same ports.

        Returns:
            int: The current DP base port used by primaries on this rank.
        """
        # Ensure distributed cleanup is complete  
        if torch.distributed.is_initialized():
            try:
                # Barrier to ensure all promoted workers reach this point
                torch.distributed.barrier()
            except Exception as e:
                logger.warning(f"Barrier failed during promotion sync: {e}")
        
        # Wait for the warm spare ports to be actually available
        # Only rank 0 tests the port to avoid conflicts
        import socket
        import time
        
        # Default return value in case logic below is skipped
        base_port_return = self.vllm_config.parallel_config.data_parallel_master_port

        if hasattr(self.vllm_config, 'parallel_config'):
            base_port = self.vllm_config.parallel_config.data_parallel_master_port
            warm_spare_offset = self.vllm_config.parallel_config.warm_spare_port_offset
            warm_spare_port = base_port + warm_spare_offset  # Port that new warm spares will use
            # Record for return to executor
            base_port_return = base_port
            
            # Only rank 0 checks port availability to avoid conflicts
            if torch.distributed.get_rank() == 0:
                # Check if warm spare phase 1 port (base+100) is available for new warm spares
                # This port should be free after promoted workers switched to base+1 for phase 2
                ports_to_check = [
                    (warm_spare_port, "new warm spare phase 1"),
                ]
                
                for port, description in ports_to_check:
                    max_retries = 50  # 5 seconds max (100ms per retry)
                    for retry in range(max_retries):
                        try:
                            # Try to bind to the port to check if it's free
                            test_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
                            test_socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
                            test_socket.bind(('', port))
                            test_socket.close()
                            # Port is available!
                            logger.info(f"Port {port} confirmed available for {description}")
                            break
                        except OSError as e:
                            # Port still in use or TIME_WAIT, retry
                            if retry == max_retries - 1:
                                logger.warning(f"Port {port} still unavailable for {description}: {e}")
                            else:
                                time.sleep(0.1)  # Wait 100ms before retry
            
            # All ranks synchronize after port check
            if torch.distributed.is_initialized():
                try:
                    torch.distributed.barrier()
                except Exception as e:
                    logger.warning(f"Post-port-check barrier failed: {e}")
        
        logger.info("Promotion synchronization complete")
        # Return the current base DP port so the executor can align its config
        return base_port_return
        
    def finalize_two_phase_init(
            self,
            new_distributed_init_method: Optional[str] = None) -> None:
        """Finalize two-phase by switching to real distributed groups.
        EngineCore will handle memory profiling afterwards.
        Optionally overrides the distributed_init_method.
        """
        if getattr(self, '_two_phase_init_pending', False):
            # Ensure all ranks reach this point before tearing down the old PG
            if torch.distributed.is_initialized():
                from contextlib import suppress
                with suppress(Exception):
                    torch.distributed.barrier()

            logger.info("Init: switching to real distributed groups")
            
            # For warm spares doing phase 2, temporarily clear process_type
            # so they use primary worker ports (base+1) instead of warm spare ports
            original_process_type = self.vllm_config.parallel_config.process_type
            if original_process_type == "warm_spare_worker":
                logger.info("Warm spare phase 2: using primary worker port allocation")
                self.vllm_config.parallel_config.process_type = None

            # Override init method if provided
            if new_distributed_init_method:
                self.distributed_init_method = (
                    new_distributed_init_method)

            # Re-init under current vLLM config context
            from vllm.config import set_current_vllm_config
            with set_current_vllm_config(self.vllm_config):
                # Destroy fake groups
                if model_parallel_is_initialized():
                    destroy_model_parallel()
                destroy_distributed_environment()
                
                # Re-initialize with actual backend
                init_worker_distributed_environment(
                    self.vllm_config, self.rank,
                    self.distributed_init_method,
                    self.local_rank,
                    current_platform.dist_backend
                )
                
                self._two_phase_init_pending = False
                # Track that we did two-phase init
                self._did_two_phase_init = True
                
                # Restore original process_type if we temporarily cleared it
                if 'original_process_type' in locals() and original_process_type == "warm_spare_worker":
                    self.vllm_config.parallel_config.process_type = original_process_type
                    logger.info("Restored warm spare process type after phase 2")
                
                logger.info("Init: two-phase complete")

            # Ensure all ranks have re-initialized before proceeding
            if torch.distributed.is_initialized():
                torch.distributed.barrier()

            # Disable custom all-reduce after two-phase init
            # The custom all-reduce has cached buffer addresses that become
            # invalid after recreating persistent buffers. It's safer to
            # disable it and fall back to NCCL for CUDA graph capture.
            from vllm.distributed.parallel_state import get_tp_group
            tp_group = get_tp_group()
            if hasattr(tp_group, 'device_communicator') and tp_group.device_communicator:
                # Force disable custom all-reduce
                tp_group.device_communicator.ca_comm = None
                logger.info("Disabled TP custom all-reduce after two-phase init")
            
            # Disable custom all-reduce for DP
            from vllm.distributed.parallel_state import get_ep_group
            ep_group = get_ep_group()
            if hasattr(ep_group, 'device_communicator') and ep_group.device_communicator:
                # Force disable custom all-reduce
                ep_group.device_communicator.ca_comm = None
                logger.info("Disabled EP custom all-reduce after two-phase init")

            # Recreate persistent buffers that may have been corrupted
            # during fake mode operations
            if hasattr(self, 'model_runner'):
                self.model_runner.recreate_persistent_buffers()
            
            # Prepare communication buffer for model
            from vllm.distributed.parallel_state import prepare_communication_buffer_for_model
            prepare_communication_buffer_for_model(self.model_runner.model)


    def compile_or_warm_up_model(self) -> None:
        """Capture CUDA graphs and runtime warmups only.
        No torch.compile or finalize here.
        
        Note: If two-phase init was used, persistent buffers were recreated
        in finalize_two_phase_init() to avoid illegal memory access errors
        during CUDA graph capture.
        """
        if not self.model_config.enforce_eager:
            self.model_runner.capture_model()

        # Warm up sampler and preallocate memory buffer for logits and other
        # sampling related tensors of max possible shape to avoid memory
        # fragmentation issue.
        # NOTE: This is called after `capture_model` on purpose to prevent
        # memory buffers from being cleared by `torch.cuda.empty_cache`.
        if get_pp_group().is_last_rank:
            max_num_reqs = min(self.scheduler_config.max_num_seqs,
                               self.scheduler_config.max_num_batched_tokens)
            # activate building attn_metadata for this dummy run to avoid
            # potential illegal memory access for full cudagraph relay.
            attn_cudagraph = self.compilation_config.full_cuda_graph and\
                not self.model_config.enforce_eager

            # We skip EPLB here since we don't want to record dummy metrics
            hidden_states, last_hidden_states = \
                self.model_runner._dummy_run(
                    num_tokens=max_num_reqs,
                    capture_attn_cudagraph=attn_cudagraph,
                    skip_eplb=True,
                )
            
            if self.model_runner.is_pooling_model:
                self.model_runner._dummy_pooler_run(hidden_states)
            else:
                self.model_runner._dummy_sampler_run(
                    hidden_states=last_hidden_states)
            
            # CRITICAL: Reset InputBatch state after ALL dummy runs!
            # These dummy runs happen during warm spare activation and leave
            # stale num_computed_tokens values that cause position corruption
            # Must be after _dummy_run AND _dummy_sampler_run/_dummy_pooler_run
            self.model_runner.reset_input_batch_state()

        # Warmup kernels used during model execution
        kernel_warmup(self)

        # Reset the seed to ensure that the random state is not affected by
        # the model initialization and profiling.
        set_random_seed(self.model_config.seed)
        
        # CRITICAL: Reset InputBatch state after dummy runs
        # Dummy runs during initialization leave stale num_computed_tokens
        # values that cause position encoding corruption for warm spares
        self.model_runner.reset_input_batch_state()

    def get_model(self) -> nn.Module:
        return self.model_runner.get_model()

    def get_supported_tasks(self) -> tuple[SupportedTask, ...]:
        return self.model_runner.get_supported_tasks()

    @torch.inference_mode()
    def execute_model(
        self,
        scheduler_output: "SchedulerOutput",
    ) -> Optional[ModelRunnerOutput]:
        intermediate_tensors = None
        if not get_pp_group().is_first_rank:
            intermediate_tensors = IntermediateTensors(
                get_pp_group().recv_tensor_dict(
                    all_gather_group=get_tp_group()))

        output = self.model_runner.execute_model(scheduler_output,
                                                 intermediate_tensors)

        parallel_config = self.vllm_config.parallel_config
        if parallel_config.distributed_executor_backend != "external_launcher" \
            and not get_pp_group().is_last_rank:
            assert isinstance(output, IntermediateTensors)
            get_pp_group().send_tensor_dict(output.tensors,
                                            all_gather_group=get_tp_group())

            kv_connector_output = output.kv_connector_output
            if not kv_connector_output:
                return None

            # In case of PP with kv transfer, we need to pass through the
            # kv_connector_output
            if (not kv_connector_output.finished_sending
                    and not kv_connector_output.finished_recving):
                return EMPTY_MODEL_RUNNER_OUTPUT

            output = copy.copy(EMPTY_MODEL_RUNNER_OUTPUT)
            output.kv_connector_output = kv_connector_output
            return output

        assert isinstance(output, ModelRunnerOutput)
        return output

    def profile(self, is_start: bool = True):
        if self.profiler is None:
            raise RuntimeError("Profiler is not enabled.")
        if is_start:
            self.profiler.start()
        else:
            self.profiler.stop()
            print(self.profiler.key_averages().table(
                sort_by="self_cuda_time_total"))

    def execute_dummy_batch(self) -> None:
        self.model_runner._dummy_run(1)

    def add_lora(self, lora_request: LoRARequest) -> bool:
        return self.model_runner.add_lora(lora_request)

    def remove_lora(self, lora_id: int) -> bool:
        return self.model_runner.remove_lora(lora_id)

    def list_loras(self) -> set[int]:
        return self.model_runner.list_loras()

    def pin_lora(self, lora_id: int) -> bool:
        return self.model_runner.pin_lora(lora_id)

    def check_health(self) -> None:
        # worker will always be healthy as long as it's running.
        return

    def _eplb_before_scale_down(self, old_ep_size: int,
                                new_ep_size: int) -> None:
        from vllm.distributed.parallel_state import get_ep_group
        if get_ep_group().rank == 0:
            logger.info("[Elastic EP] Starting expert resharding "
                        "before scaling down...")
        rank_mapping = {
            old_ep_rank: old_ep_rank if old_ep_rank < new_ep_size else -1
            for old_ep_rank in range(old_ep_size)
        }
        assert self.model_runner.eplb_state is not None
        self.model_runner.eplb_state.rearrange(self.model_runner.model,
                                               execute_shuffle=True,
                                               global_expert_load=None,
                                               rank_mapping=rank_mapping)
        torch.cuda.synchronize()
        if get_ep_group().rank == 0:
            logger.info("[Elastic EP] Expert resharding completed!")

    def _eplb_after_scale_up(
            self, old_ep_size: int, new_ep_size: int,
            global_expert_load: Optional[torch.Tensor]) -> None:
        from vllm.distributed.parallel_state import get_ep_group
        if get_ep_group().rank == 0:
            logger.info("[Elastic EP] Starting expert resharding "
                        "after scaling up...")
        rank_mapping = {
            old_ep_rank: old_ep_rank
            for old_ep_rank in range(old_ep_size)
        }
        assert self.model_runner.eplb_state is not None
        self.model_runner.eplb_state.rearrange(
            self.model_runner.model,
            execute_shuffle=True,
            global_expert_load=global_expert_load,
            rank_mapping=rank_mapping)
        if get_ep_group().rank == 0:
            logger.info("[Elastic EP] Expert resharding completed!")

    def _reconfigure_parallel_config(
            self, reconfig_request: ReconfigureDistributedRequest) -> None:
        """
        Update parallel config with provided reconfig_request
        """
        parallel_config = self.vllm_config.parallel_config
        parallel_config.data_parallel_size = \
            reconfig_request.new_data_parallel_size
        if reconfig_request.new_data_parallel_rank != \
        ReconfigureRankType.KEEP_CURRENT_RANK:
            parallel_config.data_parallel_rank = \
                reconfig_request.new_data_parallel_rank
        if reconfig_request.new_data_parallel_rank_local != \
        ReconfigureRankType.KEEP_CURRENT_RANK:
            parallel_config.data_parallel_rank_local = \
                reconfig_request.new_data_parallel_rank_local
        parallel_config.data_parallel_master_ip = \
            reconfig_request.new_data_parallel_master_ip
        parallel_config.data_parallel_master_port = \
            reconfig_request.new_data_parallel_master_port

    def _reconfigure_moe(self, old_ep_size: int,
                         new_ep_size: int) -> Optional[torch.Tensor]:
        """
        Reconfigure MoE modules with provided reconfig_request

        Return the global expert load if new_ep_size > old_ep_size,
        otherwise None
        """
        from vllm.distributed.parallel_state import (
            get_dp_group, get_ep_group, prepare_communication_buffer_for_model)
        from vllm.model_executor.layers.fused_moe.layer import (
            FusedMoEParallelConfig)

        parallel_config = self.vllm_config.parallel_config
        moe_modules = [
            module for module in self.model_runner.model.modules()
            if module.__class__.__name__ == "FusedMoE"
        ]
        num_local_experts = moe_modules[0].moe_config.num_local_experts
        assert all(module.moe_config.num_local_experts == num_local_experts
                   for module in moe_modules), (
                       "All MoE modules must have the same number of experts")
        for module in moe_modules:
            module.moe_config.num_experts = num_local_experts * new_ep_size
            module.global_num_experts = module.moe_config.num_experts
            module.moe_parallel_config = FusedMoEParallelConfig.make(
                tp_size_=get_tp_group().world_size,
                dp_size_=get_dp_group().world_size,
                vllm_parallel_config=parallel_config,
            )
            module.moe_config.moe_parallel_config = module.moe_parallel_config
        if new_ep_size < old_ep_size:
            num_local_physical_experts = num_local_experts
            assert self.model_runner.eplb_state is not None
            new_physical_experts = \
                self.model_runner.eplb_state.physical_to_logical_map.shape[1]
            parallel_config.num_redundant_experts = (
                new_physical_experts -
                self.model_runner.eplb_state.logical_replica_count.shape[1])
            global_expert_load = None
        else:
            num_local_physical_experts = torch.tensor([num_local_experts],
                                                      dtype=torch.int32,
                                                      device="cpu")
            torch.distributed.broadcast(num_local_physical_experts,
                                        group=get_ep_group().cpu_group,
                                        group_src=0)
            num_local_physical_experts = num_local_physical_experts.item()
            new_physical_experts = num_local_physical_experts * new_ep_size
            assert self.model_runner.eplb_state is not None
            global_expert_load = self.model_runner.eplb_state.rearrange(
                self.model_runner.model, execute_shuffle=False)
            parallel_config.num_redundant_experts = (
                new_physical_experts - global_expert_load.shape[1])
        prepare_communication_buffer_for_model(self.model_runner.model)
        self.model_runner.model.update_physical_experts_metadata(
            num_physical_experts=new_physical_experts,
            num_local_physical_experts=num_local_physical_experts)
        return global_expert_load

    def reinitialize_distributed(
            self, reconfig_request: ReconfigureDistributedRequest) -> None:
        from vllm.config import set_current_vllm_config
        from vllm.distributed.parallel_state import (
            cleanup_dist_env_and_memory, get_ep_group)

        old_ep_size = get_ep_group().world_size
        old_ep_rank = get_ep_group().rank
        new_ep_size = reconfig_request.new_data_parallel_size * get_tp_group(
        ).world_size * get_pp_group().world_size
        if new_ep_size < old_ep_size:
            self._eplb_before_scale_down(old_ep_size, new_ep_size)

        cleanup_dist_env_and_memory()

        if reconfig_request.new_data_parallel_rank == \
        ReconfigureRankType.SHUTDOWN_CURRENT_RANK:
            assert old_ep_rank >= new_ep_size
            # shutdown
            return

        self._reconfigure_parallel_config(reconfig_request)

        with set_current_vllm_config(self.vllm_config):
            init_worker_distributed_environment(self.vllm_config, self.rank,
                                                self.distributed_init_method,
                                                self.local_rank)

        global_expert_load = self._reconfigure_moe(old_ep_size, new_ep_size)

        if new_ep_size > old_ep_size:
            assert global_expert_load is not None
            self._eplb_after_scale_up(old_ep_size, new_ep_size,
                                      global_expert_load)

    def save_sharded_state(
        self,
        path: str,
        pattern: Optional[str] = None,
        max_size: Optional[int] = None,
    ) -> None:
        from vllm.model_executor.model_loader import ShardedStateLoader
        ShardedStateLoader.save_model(
            self.model_runner.model,
            path,
            pattern=pattern,
            max_size=max_size,
        )

    def save_tensorized_model(
        self,
        tensorizer_config: "TensorizerConfig",
    ) -> None:
        self.model_runner.save_tensorized_model(
            tensorizer_config=tensorizer_config, )


def init_worker_distributed_environment(
    vllm_config: VllmConfig,
    rank: int,
    distributed_init_method: Optional[str] = None,
    local_rank: int = -1,
    backend: str = "nccl",
) -> None:
    """Initialize the distributed environment.
    
    If backend is FAKE_DISTRIBUTED_BACKEND, initializes with FakeGroupCoordinator
    for two-phase initialization support.
    """
    parallel_config = vllm_config.parallel_config
    set_custom_all_reduce(not parallel_config.disable_custom_all_reduce)

    init_distributed_environment(parallel_config.world_size, rank,
                                 distributed_init_method, local_rank, backend)

    ensure_model_parallel_initialized(parallel_config.tensor_parallel_size,
                                      parallel_config.pipeline_parallel_size)

    # Only initialize KV transfer for real device backends
    if backend != FAKE_DISTRIBUTED_BACKEND:
        ensure_kv_transfer_initialized(vllm_config)


def _check_if_gpu_supports_dtype(torch_dtype: torch.dtype):
    # Check if the GPU supports the dtype.
    if torch_dtype == torch.bfloat16:  # noqa: SIM102
        if not current_platform.has_device_capability(80):
            capability = current_platform.get_device_capability()
            gpu_name = current_platform.get_device_name()

            if capability is None:
                compute_str = "does not have a compute capability"
            else:
                version_str = capability.as_version_str()
                compute_str = f"has compute capability {version_str}"

            raise ValueError(
                "Bfloat16 is only supported on GPUs with compute capability "
                f"of at least 8.0. Your {gpu_name} GPU {compute_str}. "
                "You can use float16 instead by explicitly setting the "
                "`dtype` flag in CLI, for example: --dtype=half.")
