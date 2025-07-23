# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
from collections.abc import Generator
import copy

import torch
from torch import nn

from vllm.config import LoadConfig, ModelConfig, VllmConfig
from vllm.logger import init_logger
from vllm.model_executor.model_loader.base_loader import BaseModelLoader
from vllm.model_executor.model_loader.utils import initialize_model, process_weights_after_loading, set_default_torch_dtype
from vllm.model_executor.parameter import UninitializedParameterFromTensor

# Import IPC client components
# These will be available when stage_3 is in the path
try:
    from stage_3.model_client import ModelClient
except ImportError:
    # For development/testing, assume stage_3 is in the Python path
    import sys

    sys.path.append("/home/schwinns/cuda-ipc-poc")
    from stage_3.model_client import ModelClient

logger = init_logger(__name__)


class IPCModelLoader(BaseModelLoader):
    """Model loader that retrieves weights via IPC from a model server."""

    def __init__(self, load_config: LoadConfig):
        super().__init__(load_config)

        if not load_config.enable_ipc_loading:
            raise ValueError(
                "IPCModelLoader requires enable_ipc_loading=True in LoadConfig"
            )

        self.server_address = load_config.ipc_server_address
        self.sub_port = load_config.ipc_sub_port
        self.req_port = load_config.ipc_req_port
        self.client = None  # Will be initialized when we know the model
        self.vllm_config = None

        logger.info(
            "IPC model loader initialized. Will connect to server at %s",
            self.server_address,
        )

    def download_model(self, model_config: ModelConfig) -> None:
        """Connect to the model server and wait for model to be ready."""
        assert self.vllm_config is not None, "vllm_config not set"

        # Initialize client with the model name
        try:
            self.client = ModelClient(
                vllm_config=self.vllm_config,
                server_address=self.server_address,
                sub_port=self.sub_port,
                req_port=self.req_port,
            )
        except Exception as e:
            logger.error("Error connecting to model server: %s", e)
            raise

        # Wait for model to be ready with two-phase timeout
        logger.info("Waiting for model to be ready on server...")
        success, server_info = self.client.wait_for_model_ready(
            initial_timeout=15.0,  # 15 seconds to verify server is alive. TODO: Make this configurable
            loading_timeout=300.0,  # 5 minutes for model to load. TODO: Make this configurable
        )

        if not success:
            raise RuntimeError(
                "Failed to connect to model server or model not ready"
            )

        logger.info(
            "Model %s is ready on server (device: cuda:%s)",
            model_config.model,
            server_info.get("device_id", "unknown"),
        )

    def get_all_weights(
        self,
        model_config: ModelConfig,
        model: nn.Module,
    ) -> Generator[tuple[str, torch.Tensor], None, None]:
        """Get all weights from the model server via IPC."""
        # First ensure the model is loaded on the server
        self.download_model(model_config)

        # Get tensor rebuild info from server
        logger.info("Retrieving model parameters rebuild info from server...")
        try:
            model_parameters_rebuild_info = self.client.get_model_parameters()
        except Exception as e:
            raise RuntimeError(f"Error getting tensor rebuild info: {e}")

        logger.info(
            "Retrieved rebuild info for %d parameters", len(model_parameters_rebuild_info)
        )

        # Reconstruct and yield each tensor
        for name, rebuild_info in model_parameters_rebuild_info.items():
            try:
                parameter = self.client.reconstruct_parameter(rebuild_info)

                # Verify we got a tensor on a valid device
                if not parameter.is_cuda:
                    raise RuntimeError(
                        f"Reconstructed tensor is not on CUDA: {parameter.device}"
                    )

                logger.debug(
                    "Reconstructed parameter %s with shape %s on device %s",
                    name,
                    parameter.shape,
                    parameter.device,
                )

                yield name, parameter
            except Exception as e:
                logger.error("Failed to reconstruct parameter %s: %s", name, e)
                raise

    def load_weights(self, model: nn.Module, model_config: ModelConfig) -> None:
        """Load weights into the model using IPC."""
        weights_to_load = {name for name, _ in model.named_parameters()}

        # NOTE: we manually assign weights here, since our model is already remotely initialized
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

        logger.info("Successfully loaded all weights via IPC")

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

            # NOTE: we need to set this here for the client to work
            self.vllm_config = copy.deepcopy(vllm_config)
            self.vllm_config.parallel_config.dry_local_rank = torch.cuda.current_device()
            self.vllm_config.parallel_config.dry_global_rank = torch.distributed.get_rank()
            self.vllm_config.parallel_config.dry_world_size = torch.distributed.get_world_size()

            self.load_weights(model, model_config)
            # Quantization does not happen in `load_weights` but after it
            process_weights_after_loading(model, model_config, target_device)
        return model.eval()

    def __del__(self):
        """Clean up the client connection when the loader is destroyed."""
        if hasattr(self, "client") and self.client is not None:
            self.client.close()
