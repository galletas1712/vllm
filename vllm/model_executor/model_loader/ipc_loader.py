# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
from collections.abc import Generator

import torch
from torch import nn

from vllm.config import LoadConfig, ModelConfig
from vllm.logger import init_logger
from vllm.model_executor.model_loader.base_loader import BaseModelLoader

# Import IPC client components
# These will be available when stage_3 is in the path
try:
    from stage_3.model_client import ModelClient
    from stage_3.model_instance_manager import CUDATensorRebuildInfo
except ImportError:
    # For development/testing, assume stage_3 is in the Python path
    import sys

    sys.path.append("/home/schwinns/cuda-ipc-poc")
    from stage_3.model_client import ModelClient
    from stage_3.model_instance_manager import CUDATensorRebuildInfo

logger = init_logger(__name__)


class IPCModelLoader(BaseModelLoader):
    """Model loader that retrieves weights via IPC from a model server."""

    def __init__(self, load_config: LoadConfig):
        super().__init__(load_config)

        if not load_config.enable_ipc_loading:
            raise ValueError(
                "IPCModelLoader requires enable_ipc_loading=True in LoadConfig"
            )

        # Initialize the model client
        try:
            self.client = ModelClient(
                server_address=load_config.ipc_server_address,
                req_port=load_config.ipc_req_port,
                sub_port=load_config.ipc_sub_port,
            )
        except Exception as e:
            logger.error("Error initializing IPC model loader: %s", e)
            raise

        logger.info(
            "Initialized IPC model loader connected to %s:%s",
            load_config.ipc_server_address,
            load_config.ipc_req_port,
        )

    def download_model(self, model_config: ModelConfig) -> None:
        """Request the model to be loaded on the server if not already loaded."""
        logger.info(
            "Requesting model %s to be loaded via IPC",
            model_config.model,
        )

        # Request model loading - client will handle device mapping
        response = self.client.load_model(model_config)

        # Check if this is an error response
        if response.get("type") == "error":
            raise RuntimeError(
                f"Server error: {response.get('message', 'Unknown error')}"
            )

        # Check the status
        status = response.get("status")
        if status is None:
            raise RuntimeError(f"Invalid response from server: {response}")

        if status.value == "loading":
            logger.info(
                "Model is being loaded on server, waiting for completion..."
            )
            success = self.client.wait_for_model_load(model_config, timeout=300)
            if not success:
                raise RuntimeError("Failed to load model on server")
        elif status.value == "already_exists":
            logger.info("Model already loaded on server")
        else:
            raise RuntimeError(f"Unexpected load status: {status}")



    def get_all_weights(
        self,
        model_config: ModelConfig,
        model: nn.Module,
    ) -> Generator[tuple[str, torch.Tensor], None, None]:
        """Get all weights from the model server via IPC."""
        # First ensure the model is loaded on the server
        self.download_model(model_config)

        # Get tensor rebuild info from server
        logger.info("Retrieving tensor rebuild info from server...")
        response = self.client.get_tensor_rebuild_info(model_config)

        if response["error"]:
            raise RuntimeError(
                f"Error getting tensor rebuild info: {response['error']}"
            )

        tensor_rebuild_info = response["tensor_rebuild_info"]
        if tensor_rebuild_info is None:
            raise RuntimeError("No tensor rebuild info received from server")

        logger.info(
            "Retrieved rebuild info for %d tensors", len(tensor_rebuild_info)
        )

        # Reconstruct and yield each tensor
        for name, rebuild_info in tensor_rebuild_info.items():
            try:
                tensor = self.client.reconstruct_tensor(rebuild_info)
                
                # Verify we got a tensor on a valid device
                if not tensor.is_cuda:
                    raise RuntimeError(
                        f"Reconstructed tensor is not on CUDA: {tensor.device}"
                    )
                
                logger.debug(
                    "Reconstructed tensor %s with shape %s on device %s",
                    name,
                    tensor.shape,
                    tensor.device,
                )
                
                yield name, tensor
            except Exception as e:
                logger.error("Failed to reconstruct tensor %s: %s", name, e)
                raise

    def load_weights(self, model: nn.Module, model_config: ModelConfig) -> None:
        """Load weights into the model using IPC."""
        weights_to_load = {name for name, _ in model.named_parameters()}

        # Use the model's load_weights method if available
        if hasattr(model, "load_weights") and callable(model.load_weights):
            loaded_weights = model.load_weights(
                self.get_all_weights(model_config, model)
            )

            # Verify all weights were loaded (for non-quantized models)
            if model_config.quantization is None and loaded_weights is not None:
                weights_not_loaded = weights_to_load - loaded_weights
                if weights_not_loaded:
                    raise ValueError(
                        f"Following weights were not initialized from IPC: "
                        f"{weights_not_loaded}"
                    )
        else:
            # Fallback: manually assign weights
            weight_dict = dict(self.get_all_weights(model_config, model))

            for name, param in model.named_parameters():
                if name in weight_dict:
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

    def __del__(self):
        """Clean up the client connection when the loader is destroyed."""
        if hasattr(self, "client"):
            self.client.close()
