# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION
# & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""
Model loader that retrieves weights from Dynamo companion server
via CUDA IPC.
"""

from torch import nn
import torch

from vllm.config import LoadConfig, VllmConfig, ModelConfig
from vllm.logger import init_logger
from vllm.model_executor.model_loader.weight_utils import (
    load_model_via_companion,
)
from vllm.model_executor.model_loader.utils import (
    initialize_model,
    set_default_torch_dtype,
)
from vllm.distributed import get_world_group

logger = init_logger(__name__)


class DynamoCompanionLoader:
    """
    Model loader that retrieves weights from a Dynamo companion server
    via CUDA IPC.

    This loader connects to a companion server that has already loaded
    the model weights, and retrieves them via CUDA IPC for fast,
    memory-efficient weight sharing.
    """

    def __init__(self, load_config: LoadConfig):
        self.load_config = load_config

    def load_weights(self, model: nn.Module, model_config: ModelConfig) -> None:
        """
        Load weights into a model from the Dynamo companion server via CUDA IPC.
        This will use the @dynamo_worker decorated function to load the model weights.
        """
        logger.info("Loading weights from Dynamo companion server via CUDA IPC")

        # Validate we're on a CUDA device (device index already set by worker)
        device_config = self.vllm_config.device_config
        assert device_config.device_type == "cuda", (
            "Companion loader only supports CUDA devices, "
            f"got device_type={device_config.device_type}"
        )

        # Use the current CUDA device (already set by the worker in init_device)
        device_id = torch.cuda.current_device()
        logger.info(f"Loading model weights on CUDA device {device_id}")

        load_model_via_companion(
            target_model=model,
            vllm_config=self.vllm_config,
            device_id=device_id,
            local_rank=get_world_group().local_rank,
            global_rank=get_world_group().rank,
            world_size=get_world_group().world_size,
        )
        logger.info(
            f"Successfully loaded model from companion server via CUDA IPC on device {device_id}"
        )

    def load_model(
        self,
        vllm_config: VllmConfig,
        model_config,
    ) -> nn.Module:
        """
        Load model from companion server via CUDA IPC.

        This follows the pattern from meta_load.py:
        1. Create a meta model (uninitialized)
        2. Load weights from companion server via CUDA IPC
        3. Import weights into the meta model

        Args:
            vllm_config: vLLM configuration
            model_config: Model configuration
        Returns:
            Model with weights loaded via CUDA IPC
        """

        logger.info("Loading model from Dynamo companion server via CUDA IPC")

        # Validate device type
        assert vllm_config.device_config.device_type == "cuda", (
            "Companion loader only supports CUDA devices, "
            f"got device_type={vllm_config.device_config.device_type}"
        )

        # Step 1: Create meta model (uninitialized)
        # Same pattern as meta_load.py
        logger.info("Creating meta model...")
        logger.debug(f"Device config type: {vllm_config.device_config.device_type}")
        logger.debug(f"Current CUDA device: {torch.cuda.current_device()}")

        with torch.device("meta"), set_default_torch_dtype(model_config.dtype):
            meta_model = initialize_model(
                vllm_config, model_config=model_config
            )
        self.vllm_config = vllm_config
        self.load_weights(meta_model, model_config)

        # Debug: Check if any tensors are still on meta device
        # Try to import from installed package
        try:
            from dynamo.companion.utils import check_for_meta_tensors

            meta_tensors = check_for_meta_tensors(meta_model)
        except ImportError:
            # Fall back to basic check
            meta_tensors = [
                f"param:{name}"
                for name, p in meta_model.named_parameters()
                if p.device.type == "meta"
            ]
            meta_tensors.extend(
                [
                    f"buffer:{name}"
                    for name, b in meta_model.named_buffers()
                    if b.device.type == "meta"
                ]
            )

        if meta_tensors:
            logger.warning(
                "Found %d tensors still on meta device after loading: %s",
                len(meta_tensors),
                meta_tensors[:10],
            )
        else:
            logger.info(
                "All tensors successfully materialized from meta to CUDA"
            )

        return meta_model
