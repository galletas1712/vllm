# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Companion server handler for ZMQ-based communication."""

import base64
import json
from dataclasses import dataclass
from typing import Dict, Optional

import cloudpickle
import torch

from vllm.logger import init_logger

from .ipc_utils import count_tensors_in_tree, encode_response, extract_module_tree_ipc_info
from .model_loader import VllmModelLoader

logger = init_logger(__name__)


# Response dataclasses
@dataclass
class SuccessResponse:
    """Success response with optional message and data."""

    success: bool = True
    message: str = "Success"


@dataclass
class ErrorResponse:
    """Error response with error message."""

    success: bool = False
    error: str = "Unknown error"


@dataclass
class LoadModelResponse(SuccessResponse):
    """Response to model loading with module tree for CUDA IPC."""

    module_tree: Optional[Dict] = None  # Serialized ModuleTreeNode (as dict)


class CompanionHandler:
    """
    Companion server handler that loads model weights and serves them via CUDA IPC.
    """

    def __init__(self, device_id: int, companion_master_port: int = 29500):
        """
        Initialize the companion handler.

        Args:
            device_id: Physical GPU device ID
            companion_master_port: Master port for CPU group initialization
        """
        self.device_id = device_id
        self.companion_master_port = companion_master_port

        # Model state
        self.model_loader: Optional[VllmModelLoader] = None
        self.model_config_hash: Optional[str] = None
        self.module_tree: Optional[Dict] = None

        # Set CUDA device
        torch.cuda.set_device(self.device_id)

        logger.info(f"Companion handler initialized for GPU {device_id}")

    def handle_request(self, request_json: str) -> str:
        """
        Handle a single load_model request and return encoded response.

        Request format:
        {
            "config_pickled": str,  # Base64-encoded cloudpickled VllmConfig
            "local_rank": int,
            "global_rank": int,
            "world_size": int
        }

        Returns:
            Base64-encoded cloudpickled response dict
        """
        request = json.loads(request_json)

        try:
            # Unpickle the VllmConfig
            config_pickled = request["config_pickled"]
            config = cloudpickle.loads(base64.b64decode(config_pickled))

            local_rank = request["local_rank"]
            global_rank = request["global_rank"]
            world_size = request["world_size"]

            assert self.device_id == local_rank, (
                f"Device ID {self.device_id} != local_rank {local_rank}"
            )

            # Create model loader to compute hash
            if self.model_loader is None:
                self.model_loader = VllmModelLoader(
                    config=config,
                    device_id=self.device_id,
                    local_rank=local_rank,
                    global_rank=global_rank,
                    world_size=world_size,
                    companion_master_port=self.companion_master_port,
                )

            config_hash = self.model_loader.compute_config_hash(
                config, local_rank, global_rank, world_size
            )

            # Check if we already have this model loaded
            if self.model_config_hash == config_hash:
                logger.info(f"Model already loaded with hash {config_hash}")
                response = LoadModelResponse(
                    message="Model already loaded", module_tree=self.module_tree
                )
                return encode_response(response)

            # Check if we have a different model loaded
            if self.model_config_hash is not None:
                response = ErrorResponse(
                    error=f"Different model already loaded (hash: {self.model_config_hash})"
                )
                return encode_response(response)

            # Load the model
            logger.info(f"Loading model with config hash {config_hash}")

            # Initialize distributed environment and load model
            model = self.model_loader.load_model_weights()

            # Extract IPC info for entire module tree
            module_tree_node = extract_module_tree_ipc_info(model)
            self.module_tree = module_tree_node.to_dict()  # Store as dict
            self.model_config_hash = config_hash

            # Count tensors for logging
            param_count, buffer_count, tensor_attr_count = count_tensors_in_tree(
                module_tree_node
            )
            logger.info(
                f"Extracted module tree: {param_count} parameters, "
                f"{buffer_count} buffers, {tensor_attr_count} tensor attributes"
            )

            # Return module tree directly in the response
            response = LoadModelResponse(
                message=(
                    f"Model loaded successfully with {param_count} parameters, "
                    f"{buffer_count} buffers, {tensor_attr_count} tensor attributes"
                ),
                module_tree=self.module_tree,
            )
            return encode_response(response)

        except Exception as e:
            logger.exception(f"Failed to load model: {e}")
            response = ErrorResponse(error=str(e))
            return encode_response(response)
