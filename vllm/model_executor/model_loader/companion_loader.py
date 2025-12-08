# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""
Model loader that retrieves weights from companion server via CUDA IPC.
"""

import torch
from torch import nn

from vllm.config import LoadConfig, ModelConfig, VllmConfig
from vllm.distributed import get_world_group
from vllm.logger import init_logger
from vllm.model_executor.model_loader.utils import initialize_model
from vllm.utils.torch_utils import set_default_torch_dtype


logger = init_logger(__name__)


class CompanionLoader:
    """
    Model loader that retrieves weights from a companion server via CUDA IPC.

    This loader connects to a companion server that has already loaded
    the model weights, and retrieves them via CUDA IPC for fast,
    memory-efficient weight sharing.
    """

    def __init__(self, load_config: LoadConfig):
        self.load_config = load_config

    def load_weights(self, model: nn.Module, model_config: ModelConfig) -> None:
        """
        Load weights into a model from the companion server via CUDA IPC.
        """
        logger.info("Loading weights from companion server via CUDA IPC")

        # Validate we're on a CUDA device (device index already set by worker)
        device_config = self.vllm_config.device_config
        assert device_config.device_type == "cuda", (
            "Companion loader only supports CUDA devices, "
            f"got device_type={device_config.device_type}"
        )

        # Use the current CUDA device (already set by the worker in init_device)
        device_id = torch.cuda.current_device()
        logger.info(f"Loading model weights on CUDA device {device_id}")
        
        from vllm.companion import CompanionClient

        # Compute port from base port + local_rank
        # TODO: check if local_rank actually works for multi-node
        base_port = self.vllm_config.load_config.companion_port
        port = base_port + get_world_group().local_rank

        logger.info(
            f"Connecting to companion server on port {port} "
            f"(base_port={base_port}, local_rank={get_world_group().local_rank})"
        )

        with CompanionClient(host="localhost", port=port) as client:
            client.load_model(
                target_model=model,
                vllm_config=self.vllm_config,
                local_rank=get_world_group().local_rank,
                global_rank=get_world_group().rank,
                world_size=get_world_group().world_size,
            )

        logger.info(
            f"Successfully loaded model from companion server "
            f"via CUDA IPC on device {device_id}"
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
        logger.info("Loading model from companion server via CUDA IPC")

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
            meta_model = initialize_model(vllm_config, model_config=model_config)
        self.vllm_config = vllm_config
        self.load_weights(meta_model, model_config)

        # Debug: Check if any tensors are still on meta device
        from vllm.companion.ipc_utils import check_for_meta_tensors

        meta_tensors = check_for_meta_tensors(meta_model)

        if meta_tensors:
            logger.warning(
                "Found %d tensors still on meta device after loading: %s",
                len(meta_tensors),
                meta_tensors[:10],
            )
        else:
            logger.info("All tensors successfully materialized from meta to CUDA")

        return meta_model
