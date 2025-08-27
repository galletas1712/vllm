# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""Model instance manager for companion servers (shared by MultiProc and Dynamo)."""

import torch
from torch.multiprocessing.reductions import reduce_tensor

from vllm.config import VllmConfig, set_current_vllm_config
from vllm.distributed import parallel_state
from vllm.logger import init_logger
from vllm.model_executor.model_loader.default_loader import DefaultModelLoader
from vllm.distributed.parallel_state import (
    FAKE_DISTRIBUTED_BACKEND,
    init_distributed_environment
)
from vllm.model_executor.parameter import UninitializedParameterFromTensor
from vllm.companion.messages import CUDATensorRebuildInfo
from vllm.utils import get_distributed_init_method


logger = init_logger(__name__)


def override_vllm_config(
    vllm_config: VllmConfig,
    device_id: int,
) -> VllmConfig:
    # NOTE: If we deepcopy, there's a bunch of other stuff that gets copied that we don't want
    new_vllm_config = VllmConfig(
        model_config=vllm_config.model_config,
        parallel_config=vllm_config.parallel_config,
        cache_config=vllm_config.cache_config,
        device_config=vllm_config.device_config,
        load_config=vllm_config.load_config,
    )
    # Override load config for fake run on SPECIFIED device
    # NOTE: we use device_id and not local_rank because we assume CUDA_VISIBLE_DEVICES is not set
    # local_rank would come from the client view which may not be correct
    new_vllm_config.load_config.device = torch.device(f"cuda:{device_id}")

    # We want to load the model for real here
    new_vllm_config.load_config.enable_companion_process = False
    
    return new_vllm_config


class ModelInstanceManager:
    def __init__(
        self,
        vllm_config: VllmConfig,
        device_id: int,
        local_rank: int,
        global_rank: int,
        world_size: int,
        companion_master_port: int,
    ):
        self.device_id = device_id
        self.local_rank = local_rank
        self.global_rank = global_rank
        self.world_size = world_size
        self.companion_master_port = companion_master_port
        
        self.vllm_config = override_vllm_config(
            vllm_config,
            device_id,
        )

        torch.cuda.set_device(self.vllm_config.load_config.device)

        self._model = None
        self._model_parameters_ipc_info: dict[str, CUDATensorRebuildInfo] = {}
        self._distributed_initialized = False
    
    def initialize_distributed(self):
        """Initialize distributed environment. Must be called from main thread."""
        if self._distributed_initialized:
            logger.warning("[DIST-INIT] Distributed already initialized, skipping")
            return
            
        logger.info("[DIST-INIT] Initializing distributed environment from main thread")
        self._initialize_fake_distributed_environment()
        self._distributed_initialized = True
        logger.info("[DIST-INIT] ✓ Distributed environment initialized successfully")
    
    def load_model_weights(self):
        """Load model weights. Can be called from thread pool."""
        if not self._distributed_initialized:
            raise RuntimeError("Must call initialize_distributed() before load_model_weights()")
        
        logger.info("[MODEL-LOAD] Starting model weight loading in thread pool")
        
        # Create model loader
        assert self.vllm_config.load_config is not None
        assert self.vllm_config.load_config.enable_companion_process is False
        
        logger.info(
            "[MODEL-LOAD] Creating DefaultModelLoader with load_config:\n"
            "  device=%s\n"
            "  load_format=%s\n"
            "  download_dir=%s",
            self.vllm_config.load_config.device,
            self.vllm_config.load_config.load_format,
            self.vllm_config.load_config.download_dir,
        )
        
        default_loader = DefaultModelLoader(self.vllm_config.load_config)

        architectures = getattr(self.vllm_config.model_config, 'architectures', [])
        architecture_name = architectures[0] if architectures else "Unknown"
        logger.info(
            "[MODEL-LOAD] Starting model load for %s\n"
            "  Model architecture: %s\n"
            "  Current device: %s",
            self.vllm_config.model_config.model,
            architecture_name,
            self.vllm_config.load_config.device,
        )

        # Load model with vllm config context
        with set_current_vllm_config(self.vllm_config):
            self._model = default_loader.load_model(
                self.vllm_config, self.vllm_config.model_config
            )

        logger.info("[MODEL-LOAD] ✓ Model loaded successfully!")
        
        # Make sure model is in eval mode
        assert self._model.training is False

        logger.info("[MODEL-LOAD] Getting IPC rebuild info for model parameters...")
        
        # Extract parameters and create IPC rebuild info
        self._model_parameters_ipc_info = {}
        for name, param in self._model.named_parameters():
            if not isinstance(param, torch.nn.Parameter):
                continue

            # Check if parameter is still uninitialized (should not happen after load_weights)
            if isinstance(param, UninitializedParameterFromTensor):
                logger.error("Parameter %s is still uninitialized after weight loading!", name)
                raise RuntimeError(f"Failed to materialize parameter {name}")

            # Make sure parameter is on the current device ID
            assert param.device == self.vllm_config.load_config.device

            # Get the underlying tensor
            tensor = param.data

            # Ensure tensor is on CUDA
            if tensor.device.type != "cuda":
                logger.warning("Parameter %s is not on CUDA, skipping", name)
                continue

            # Get rebuild info for IPC sharing
            _, rebuild_args = reduce_tensor(tensor)
            self._model_parameters_ipc_info[name] = (
                CUDATensorRebuildInfo.from_rebuild_args(rebuild_args)
            )

        logger.info(
            "[MODEL-LOAD] Model initialized with %d parameters ready for IPC sharing",
            len(self._model_parameters_ipc_info),
        )

    def _initialize_fake_distributed_environment(self):
        """Initialize distributed environment in fake mode for correct weight loading."""
        logger.info(
            "[DIST-INIT] Starting companion shadow process initialization:\n"
            "  local_rank=%d, rank=%d (TP/PP), world_size=%d (TP/PP)\n"
            "  TP=%d, PP=%d, DP=%d, Expert Parallel enabled=%s\n"
            "  companion_master_port=%d",
            self.local_rank, self.global_rank, self.world_size,
            self.vllm_config.parallel_config.tensor_parallel_size,
            self.vllm_config.parallel_config.pipeline_parallel_size,
            self.vllm_config.parallel_config.data_parallel_size,
            self.vllm_config.parallel_config.enable_expert_parallel,
            self.companion_master_port,
        )
        
        init_method = get_distributed_init_method(
            self.vllm_config.parallel_config.data_parallel_master_ip,
            self.companion_master_port
        )
        
        logger.info(
            "[DIST-INIT] About to call parallel_state.init_distributed_environment:\n"
            "  init_method=%s\n"
            "  backend=%s (will use gloo for fake run)\n"
            "  world_size=%d (TP*PP), rank=%d (TP/PP), local_rank=%d\n"
            "  DP adjustment will be handled by parallel_state",
            init_method,
            FAKE_DISTRIBUTED_BACKEND,
            self.world_size,
            self.global_rank,
            self.local_rank,
        )
        
        # Check if torch.distributed is already initialized
        import torch.distributed
        if torch.distributed.is_initialized():
            logger.warning(
                "[DIST-INIT] torch.distributed is already initialized! "
                "Current rank=%d, world_size=%d",
                torch.distributed.get_rank() if torch.distributed.is_initialized() else -1,
                torch.distributed.get_world_size() if torch.distributed.is_initialized() else -1,
            )
        
        # Set the process type in the config
        # This determines which DP port to use for rendezvous
        if self.vllm_config and self.vllm_config.parallel_config:
            # Determine if this is a warm spare companion based on context
            # (This would need to be passed from the coordinator)
            is_warm_spare = getattr(self, 'is_warm_spare', False)
            if is_warm_spare:
                self.vllm_config.parallel_config.process_type = "warm_spare_companion"
            else:
                self.vllm_config.parallel_config.process_type = "primary_companion"

        # Set the current vLLM config so init_distributed_environment can access it
        set_current_vllm_config(self.vllm_config)
        
        logger.info("[DIST-INIT] Calling init_distributed_environment NOW...")
        
        init_distributed_environment(
            world_size=self.world_size,
            rank=self.global_rank,
            distributed_init_method=init_method,
            local_rank=self.local_rank,
            backend=FAKE_DISTRIBUTED_BACKEND,
        )
        
        logger.info(
            "[DIST-INIT] ✓ parallel_state.init_distributed_environment completed successfully!\n"
            "  torch.distributed.is_initialized()=%s\n"
            "  torch.distributed rank=%d, world_size=%d",
            torch.distributed.is_initialized(),
            torch.distributed.get_rank() if torch.distributed.is_initialized() else -1,
            torch.distributed.get_world_size() if torch.distributed.is_initialized() else -1,
        )

        # Initialize model parallel groups
        tp_size = self.vllm_config.parallel_config.tensor_parallel_size
        pp_size = self.vllm_config.parallel_config.pipeline_parallel_size
        
        logger.info(
            "[DIST-INIT] About to call parallel_state.initialize_model_parallel:\n"
            "  tensor_model_parallel_size=%d\n"
            "  pipeline_model_parallel_size=%d",
            tp_size, pp_size
        )
        
        parallel_state.initialize_model_parallel(
            tensor_model_parallel_size=tp_size,
            pipeline_model_parallel_size=pp_size,
        )
        
        logger.info("[DIST-INIT] ✓ parallel_state.initialize_model_parallel completed successfully!")
        
        # Log information about created process groups
        pp_group = parallel_state.get_pp_group()
        world_group = parallel_state.get_world_group()
        
        # Get DP and EP groups if they exist
        try:
            dp_group = parallel_state.get_dp_group()
            dp_size = dp_group.world_size
            dp_rank = dp_group.rank_in_group
        except (AssertionError, AttributeError):
            # DP group may not be initialized
            dp_size = 1
            dp_rank = 0
            
        try:
            ep_group = parallel_state.get_ep_group()
            ep_size = ep_group.world_size
            ep_rank = ep_group.rank_in_group
        except (AssertionError, AttributeError):
            # EP group may not be initialized
            ep_size = 1
            ep_rank = 0
        
        logger.info(
            "[DIST-INIT] Process groups created:\n"
            "  TP group: size=%d, rank=%d\n"
            "  PP group: size=%d, rank=%d\n"
            "  DP group: size=%d, rank=%d\n"
            "  EP group: size=%d, rank=%d\n"
            "  World group: size=%d, rank=%d",
            parallel_state.get_tensor_model_parallel_world_size(),
            parallel_state.get_tensor_model_parallel_rank(),
            pp_group.world_size,
            pp_group.rank_in_group,
            dp_size,
            dp_rank,
            ep_size,
            ep_rank,
            world_group.world_size,
            world_group.rank,
        )

        logger.info(
            "[DIST-INIT] ✅ COMPLETE - fake distributed environment initialized successfully:\n"
            "  local_rank=%d, global_rank=%d, world_size=%d\n"
            "  TP=%d (rank %d), PP=%d (rank %d), DP=%d (rank %d), EP=%d (rank %d)",
            self.local_rank,
            self.global_rank,
            self.world_size,
            tp_size,
            parallel_state.get_tensor_model_parallel_rank(),
            pp_size,
            pp_group.rank_in_group,
            self.vllm_config.parallel_config.data_parallel_size,
            dp_rank,
            ep_size,
            ep_rank,
        )

    def get_model_parameters_ipc_info(self) -> dict[str, CUDATensorRebuildInfo]:
        """Get IPC rebuild info for all model parameters."""
        return self._model_parameters_ipc_info
