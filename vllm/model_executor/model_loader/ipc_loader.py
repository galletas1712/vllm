# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
from collections.abc import Generator
import contextlib
import copy

import torch
from torch import nn

from vllm.config import ModelConfig, VllmConfig
from vllm.companion.multiproc_companion_client import MultiProcCompanionClient
from vllm.distributed import get_world_group
from vllm.logger import init_logger
from vllm.model_executor.model_loader.base_loader import BaseModelLoader
from vllm.model_executor.model_loader.utils import (
    initialize_model,
    process_weights_after_loading,
    set_default_torch_dtype,
)
from vllm.model_executor.parameter import UninitializedParameterFromTensor

logger = init_logger(__name__)

class IPCModelLoader(BaseModelLoader):
    """Model loader that retrieves weights via IPC from a model server.
    
    This loader connects to a companion server that has
    pre-loaded the model weights and retrieves them via CUDA IPC. This allows
    multiple processes to share the same GPU memory for model weights.
    
    The loader automatically obtains rank information from vLLM's parallel_state
    module, which is initialized during init_device() before model loading.
    This ensures the loader connects to the correct companion server instance
    based on the worker's rank.
    """

    def __init__(self, vllm_config: VllmConfig):
        self.vllm_config = VllmConfig(
            model_config=copy.deepcopy(vllm_config.model_config),
            parallel_config=copy.deepcopy(vllm_config.parallel_config),
            cache_config=copy.deepcopy(vllm_config.cache_config),
            device_config=copy.deepcopy(vllm_config.device_config),
            load_config=copy.deepcopy(vllm_config.load_config),
            companion_config=copy.deepcopy(vllm_config.companion_config),
        )

        if not self.vllm_config.load_config.enable_companion_process:
            raise ValueError(
                "IPCModelLoader requires enable_companion_process=True in LoadConfig"
            )

        self.client = None  # Will be initialized when we know the model

        logger.info("IPC model loader initialized")
    
    def _validate_companion_server_availability(self) -> None:
        """Basic validation that a companion is addressable for current device."""
        if not self.client:
            return
        logical_device = torch.cuda.current_device()
        total_gpus = torch.cuda.device_count()
        if logical_device < 0 or logical_device >= total_gpus:
            raise RuntimeError(
                f"Current CUDA device {logical_device} is out of visible range (0..{total_gpus-1})."
            )
        logger.debug(
            "[IPC-LOADER] Using current logical CUDA device %d to select companion",
            logical_device,
        )

    def download_model(self, model_config: ModelConfig) -> None:
        """Connect to the model server and wait for model to be ready."""
        assert self.vllm_config is not None, "vllm_config not set"

        if self.client is None:
            # Initialize MultiProc client
            try:
                if not self.vllm_config or not self.vllm_config.companion_config:
                    raise ValueError("VllmConfig with CompanionConfig is required for IPC loading")

                # Select companion based on the worker's physical CUDA device id
                # that corresponds to the current logical device. If
                # CUDA_VISIBLE_DEVICES is set, map logical->physical via it.
                logical_device = torch.cuda.current_device()
                import os
                visible = os.environ.get("CUDA_VISIBLE_DEVICES")
                if visible:
                    try:
                        physical_device = int(visible.split(",")[logical_device].strip())
                    except Exception:
                        physical_device = logical_device
                else:
                    physical_device = logical_device

                logger.info(
                    "[IPC-LOADER] Creating companion client: logical %d -> physical %d",
                    logical_device, physical_device,
                )

                self.client = MultiProcCompanionClient(
                    self.vllm_config.companion_config, physical_device)
                logger.info(
                    "MultiProc companion client initialized for physical device %d with coordinator at: tcp://127.0.0.1:%d",
                    physical_device, self.vllm_config.companion_config.coordinator_port,
                )

                # Validate that companion server exists for our device
                self._validate_companion_server_availability()
            except Exception as e:
                logger.error(
                    "Error creating MultiProc companion client: %s", e)
                raise

    def get_all_weights(
        self,
        model_config: ModelConfig,
        model: nn.Module,
    ) -> Generator[tuple[str, torch.Tensor], None, None]:
        """Get all weights from the model server via IPC."""
        # First ensure the model is loaded on the server
        self.download_model(model_config)

        # Both clients now have the same API - they return rebuild info
        logger.info("Retrieving model parameters rebuild info from companion...")
        
        try:
            # MultiProc client returns rebuild info directly
            logger.info("[IPC-LOADER] Requesting model parameters via MultiProc client")
            model_parameters_rebuild_info = self.client.get_model_parameters(
                vllm_config=self.vllm_config
            )
        except Exception as e:
            logger.error("[IPC-LOADER] Error getting tensor rebuild info: %s", e)
            raise RuntimeError(
                f"Error getting tensor rebuild info: {e}") from e

        logger.info(
            "Retrieved rebuild info for %d parameters",
            len(model_parameters_rebuild_info)
        )

        # Reconstruct and yield each tensor - same for both backends
        for name, rebuild_info in model_parameters_rebuild_info.items():
            try:
                parameter = self.client.reconstruct_parameter(rebuild_info)

                # Verify we got a tensor on a valid device
                if not parameter.is_cuda:
                    raise RuntimeError(
                        f"Reconstructed tensor is not on CUDA: "
                        f"{parameter.device}"
                    )

                logger.debug(
                    "[IPC-LOADER] Reconstructed parameter %s shape=%s device=%s",
                    name, parameter.shape, parameter.device)

                yield name, parameter
            except Exception as e:
                logger.error(
                    "[IPC-LOADER] Failed to reconstruct parameter %s: %s", name, e)
                raise RuntimeError(
                    f"Failed to reconstruct parameter {name}: {e}") from e

    def load_weights(self, model: nn.Module,
                     model_config: ModelConfig) -> None:
        """Load weights into the model using IPC."""
        weights_to_load = {name for name, _ in model.named_parameters()}

        # NOTE: we manually assign weights here, since our model is already
        # remotely initialized
        weight_dict = dict(self.get_all_weights(model_config, model))

        for name, param in model.named_parameters():
            if name in weight_dict:
                if isinstance(param, UninitializedParameterFromTensor):
                    param.materialize(weight_dict[name])
                else:
                    param.data = weight_dict[name]
            else:
                logger.warning("Weight %s not found in IPC weights", name)

        # Check if all expected weights were loaded
        loaded_weights = set(weight_dict.keys())
        weights_not_loaded = weights_to_load - loaded_weights
        if weights_not_loaded:
            raise ValueError(
                f"Following weights were not initialized from IPC: "
                f"{weights_not_loaded}"
            )

        logger.info("Finished loading parameters via IPC: %d mapped, %d missing",
                    len(loaded_weights), len(weights_not_loaded))

    def load_model(self, vllm_config: VllmConfig,
                   model_config: ModelConfig) -> nn.Module:
        """Load a model with the given configurations."""
        device_config = vllm_config.device_config
        load_config = vllm_config.load_config
        load_device = device_config.device if load_config.device is None else \
                      load_config.device
        target_device = torch.device(load_device)
        with set_default_torch_dtype(model_config.dtype):
            with target_device:
                model = initialize_model(vllm_config=vllm_config,
                                         model_config=model_config)

            logger.debug("Loading weights on %s ...", load_device)
            
            # Log rank information for debugging
            # By this point, the distributed environment is already initialized
            world_group = get_world_group()
            logger.info(
                "IPC loader using rank info from parallel_state: "
                "local_rank=%d, global_rank=%d, world_size=%d",
                world_group.local_rank, world_group.rank, world_group.world_size
            )

            self.load_weights(model, model_config)
            # Quantization does not happen in `load_weights` but after it
            process_weights_after_loading(model, model_config, target_device)
        return model.eval()

    def __del__(self):
        """Clean up the client connection when the loader is destroyed."""
        # Clean up client
        if hasattr(self, "client") and self.client is not None:
            with contextlib.suppress(Exception):
                self.client.close()
