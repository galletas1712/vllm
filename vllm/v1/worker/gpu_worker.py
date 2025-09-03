# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""A GPU worker class."""
import copy
from enum import Enum
import gc
import os
from contextlib import AbstractContextManager, nullcontext
from typing import TYPE_CHECKING, Any, Iterable, Optional

import torch
import torch.distributed
import torch.nn as nn

import vllm.envs as envs
from vllm.config import VllmConfig, set_current_vllm_config
from vllm.distributed import (ensure_model_parallel_initialized,
                              init_distributed_environment,
                              set_custom_all_reduce)
from vllm.distributed.kv_transfer import ensure_kv_transfer_initialized
from vllm.distributed.parallel_state import (
    cleanup_dist_env_and_memory,
    FAKE_DISTRIBUTED_BACKEND,
)
from vllm.distributed.parallel_state import get_pp_group, get_tp_group, get_ep_group
from vllm.logger import init_logger
from vllm.lora.request import LoRARequest
from vllm.model_executor import set_random_seed
from vllm.model_executor.model_loader.utils import process_weights_after_loading
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
from vllm.device_allocator.cumem import CuMemAllocator

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


class WorkerInitPhase(Enum):
    PHASE_1 = "phase_1"
    PHASE_2 = "phase_2"
    INITIALIZING_CACHE = "initializing_cache"
    PHASE_3 = "phase_3"
    DONE = "done"


class Worker(WorkerBase):

    def __init__(
        self,
        vllm_config: VllmConfig,
        local_rank: int,
        rank: int,
        distributed_init_method: str,
        fake_distributed_init_method: str,
        is_driver_worker: bool = False,
    ):

        super().__init__(vllm_config=vllm_config,
                         local_rank=local_rank,
                         rank=rank,
                         distributed_init_method=distributed_init_method,
                         is_driver_worker=is_driver_worker)
        
        self.fake_distributed_init_method = fake_distributed_init_method

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
        
        self.init_phase = WorkerInitPhase.PHASE_1
        # Record the original load format so we can defer real weights and
        # switch back after phase 2 init if needed.
        self._original_load_format: Optional[str] = self.load_config.load_format
    
    def phase_1_init(self) -> None:
        """
        Initialize the common worker components that can be reused and don't depend on the real distributed environment:
        1. init_device (device + fake distributed setup)
        2. init_model_runner (which includes buffers)
        3. load_model (model weights)
        4. precompile_model (pre-compile model)
        5. shut down the fake distributed environment
        """
        assert self.init_phase == WorkerInitPhase.PHASE_1, "Worker must be in phase 1"
        self.init_device(fake_distributed_env=True)
        self.init_model_runner()
        # Defer real weights: load dummy weights in phase 1 so that we can
        # precompile without committing real weight memory. We'll switch
        # to the real loader and reload weights in phase 2.
        if self._original_load_format != "dummy":
            self.model_runner.update_config({
                "load_config": {
                    "load_format": "dummy",
                }
            })
        self.load_model()
        self.precompile_model()

        self._phase_1_ep_size = get_ep_group().world_size
        self._phase_1_ep_rank = get_ep_group().rank_in_group
        logger.info(f"Phase 1 - EP size: {self._phase_1_ep_size}, EP rank: {self._phase_1_ep_rank}")

        # Clean up the model runner and the fake distributed environment
        self.model_runner.reset_input_batch_state()
        cleanup_dist_env_and_memory()

        # Offload weight pool to free physical backing while retaining virtual
        # mappings so phase 2 can measure GPU free memory without weights.
        self.sleep(level=2)

        self.init_phase = WorkerInitPhase.PHASE_2
    
    def phase_2_init(self) -> None:
        """
        1. init_device (refreshes memory snapshot and reinitializes distributed environment)
        2. Reinitialize buffers in model runner
        3. Calculate available memory
        Since the EngineCore calculates the minimum available memory across all workers (DP incl.),
        phase 2 and 3 need to be split.
        """
        assert self.init_phase == WorkerInitPhase.PHASE_2, "Worker must be in phase 2"

        self.wake_up(tags=("weights", ))
        # This only reinitializes the distributed environment, but doesn't take a memory snapshot
        # since we want to use the memory snapshot without weights loaded
        self.reinit_device()
        self.model_runner.reset_input_batch_state()
       
        # Switch back to the original loader and load real weights inplace.
        # The weights are already mapped in GPU memory from wake_up.
        if self._original_load_format and self._original_load_format != "dummy":
            self.model_runner.update_config({
                "load_config": {
                    "load_format": self._original_load_format,
                }
            })

            # NOTE: must update expert map before reloading weights to ensure expert_map is valid before loading
            if self.vllm_config.parallel_config.enable_expert_parallel:
                moe_modules = [
                    module for module in self.model_runner.model.modules()
                    if module.__class__.__name__ == "FusedMoE"
                ]
                for module in moe_modules:
                    if hasattr(module, 'update_expert_map'):
                        logger.debug(f"Updating expert map for module: {module}")
                        module.update_expert_map()

            # Allocate actual weights into the existing parameter storages.
            logger.info("Reloading weights")
            self.model_runner.reload_weights()  # NOTE: don't use self.reload_weights() since this wraps context with memory pool

            if self.vllm_config.parallel_config.enable_expert_parallel:
                current_ep_size = get_ep_group().world_size
                current_ep_rank = get_ep_group().rank_in_group
                assert current_ep_size == self._phase_1_ep_size, "EP size should be the same"
                assert current_ep_rank == self._phase_1_ep_rank, "EP rank should be the same"
                self._reconfigure_moe(old_ep_size=self._phase_1_ep_size, new_ep_size=current_ep_size)
        
        self.model_runner.recreate_persistent_buffers()

        available_memory = self.determine_available_memory()
        self.init_phase = WorkerInitPhase.INITIALIZING_CACHE
        return available_memory

    def phase_3_init(self) -> None:
        """
        Captures CUDA graphs for the model.
        This assumes the KV cache has already been initialized.
        """
        assert self.init_phase == WorkerInitPhase.PHASE_3, "Worker must be in phase 3"
        self.compile_or_warm_up_model()
        self.init_phase = WorkerInitPhase.DONE
    
    def sleep_tensors(self, tensors: Iterable[torch.Tensor], offload: bool = False) -> None:
        free_bytes_before_sleep = torch.cuda.mem_get_info()[0]
        from vllm.device_allocator.cumem import CuMemAllocator
        allocator = CuMemAllocator.get_instance()
        allocator.sleep_tensors(tensors)
        free_bytes_after_sleep, total = torch.cuda.mem_get_info()
        freed_bytes = free_bytes_after_sleep - free_bytes_before_sleep
        used_bytes = total - free_bytes_after_sleep
        assert freed_bytes >= 0, "Memory usage increased after sleeping."
        logger.info(
            "Sleep mode freed %.2f GiB memory, "
            "%.2f GiB memory is still in use.", freed_bytes / GiB_bytes,
            used_bytes / GiB_bytes)

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
    
    def init_model_runner(self) -> None:
        self.model_runner: GPUModelRunner = GPUModelRunner(
            self.vllm_config, self.device)

    def initialize_cache(self, num_gpu_blocks: int,
                         num_cpu_blocks: int) -> None:
        assert self.init_phase == WorkerInitPhase.INITIALIZING_CACHE, "Worker must be in cache initialization phase"
        self.cache_config.num_gpu_blocks = num_gpu_blocks
        self.cache_config.num_cpu_blocks = num_cpu_blocks
        self.init_phase = WorkerInitPhase.PHASE_3
    
    def _take_memory_snapshot(self):
        gc.collect()
        torch.cuda.empty_cache()

        # take current memory snapshot
        self.init_snapshot = MemorySnapshot()
        self.requested_memory = (self.init_snapshot.total_memory *
                                    self.cache_config.gpu_memory_utilization)
        GiB = lambda b: round(b / GiB_bytes, 2)
        
        if self.init_snapshot.free_memory < self.requested_memory:
            raise ValueError(
                f"Free memory on device "
                f"({GiB(self.init_snapshot.free_memory)}/"
                f"{GiB(self.init_snapshot.total_memory)} GiB) on startup "
                f"is less than desired GPU memory utilization "
                f"({self.cache_config.gpu_memory_utilization}, "
                f"{GiB(self.requested_memory)} GiB). Decrease GPU memory "
                f"utilization or reduce GPU memory used by other processes."
            )
    
    def _init_distributed_environment(self, fake_distributed_env: bool = False):
        logger.info("Initializing worker distributed environment...")
        with set_current_vllm_config(self.vllm_config):
            init_worker_distributed_environment(
                self.vllm_config, self.rank,
                self.fake_distributed_init_method if fake_distributed_env else self.distributed_init_method,
                self.local_rank,
                backend=(
                    FAKE_DISTRIBUTED_BACKEND if fake_distributed_env
                    else current_platform.dist_backend
                )
            )
        logger.info("Worker distributed environment initialized")
    
    def init_device(self, fake_distributed_env: bool = False):
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
        else:
            raise RuntimeError(
                f"Not support device type: {self.device_config.device}")
        
        self._take_memory_snapshot()
        self._init_distributed_environment(fake_distributed_env)
        
        # Set random seed.
        set_random_seed(self.model_config.seed)

        if self.rank == 0:
            # If usage stat is enabled, collect relevant info.
            report_usage_stats(self.vllm_config)
        
        logger.info("Worker init_device has completed")
    
    def reinit_device(self):
        logger.info("Reinitializing device and distributed environment...")
        torch.cuda.synchronize()
        self._init_distributed_environment(fake_distributed_env=False)
        set_random_seed(self.model_config.seed)
        
        if self.rank == 0:
            report_usage_stats(self.vllm_config)
        logger.info("Worker reinit_device has completed")
    
    # FIXME(youkaichao & ywang96): Use TorchDispatchMode instead of memory pool
    # to hijack tensor allocation.
    def load_model(self) -> None:
        eep_scale_up = os.environ.get("VLLM_ELASTIC_EP_SCALE_UP_LAUNCH") == "1"
        with self._maybe_get_memory_pool_context(tag="weights"):
            self.model_runner.load_model(eep_scale_up=eep_scale_up)
    
    def update_config(self, overrides: dict[str, Any]) -> None:
        self.model_runner.update_config(overrides)

    def reload_weights(self) -> None:
        with self._maybe_get_memory_pool_context(tag="weights"):
            self.model_runner.reload_weights()

    def precompile_model(self) -> None:
        """Pre-compile (torch.compile) in fake distributed environment..

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
        from vllm.distributed.parallel_state import IS_FAKE_DISTRIBUTED
        assert not IS_FAKE_DISTRIBUTED(), "Memory profiling should not be called in fake mode"
        
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
        GiB = lambda b: b / GiB_bytes

        # Execute a forward pass with dummy inputs to profile the memory usage
        # of the model.
        # NOTE: since our initial memory snapshot was taken prior to weights loaded, we need to pass the weights size to the profiler
        weights_bytes = get_model_weights_size_bytes(self.model_runner.model)
        with memory_profiling(
                self.init_snapshot,
                weights_memory=weights_bytes) as profile_result:
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
        available_kv_cache_memory = self.requested_memory \
            - profile_result.non_kv_cache_memory

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

    def compile_or_warm_up_model(self) -> None:
        """Capture CUDA graphs and runtime warmups only.
        No torch.compile or finalize here.
        
        Note: Persistent buffers were recreated to avoid illegal memory access errors during CUDA graph capture.
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
            # Reinitialize MoE kernel after updating expert_map
            if hasattr(module.quant_method, 'init_prepare_finalize'):
                module.quant_method.init_prepare_finalize(module.moe_config)
        global_expert_load = None
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
            if self.model_runner.eplb_state is not None:
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
    
    If backend is FAKE_DISTRIBUTED_BACKEND, initializes with FakeGroupCoordinator.
    """
    parallel_config = vllm_config.parallel_config
    # TODO (schwinns): remove this once custom all-reduce is supported
    assert parallel_config.disable_custom_all_reduce, "Custom all-reduce is not supported."
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
