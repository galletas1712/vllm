# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""Simple companion client for GPU workers to retrieve model parameters."""

import pickle
from typing import Optional

import torch
import zmq

from vllm.config import VllmConfig
from vllm.config.companion import CompanionConfig
from vllm.distributed import get_world_group
from vllm.logger import init_logger
from vllm.companion.messages import (
    GetModelParametersRequest,
    ModelParametersResponse,
    CUDATensorRebuildInfo,
)

logger = init_logger(__name__)


class MultiProcCompanionClient:
    """
    Simple client for GPU workers to get model parameters from
    companion servers.
    """
    
    def __init__(self, companion_config: CompanionConfig):
        """
        Initialize the client.
        
        Args:
            companion_config: CompanionConfig object containing coordinator address
                and other companion settings.
        """
        if not companion_config or not companion_config.coordinator_address:
            raise ValueError("CompanionConfig with coordinator_address is required")
        
        self.companion_config = companion_config
        self.coordinator_address = companion_config.coordinator_address
        
        # ZMQ context and socket
        self.context = zmq.Context()
        self.socket = self.context.socket(zmq.REQ)
        self.socket.connect(self.coordinator_address)
        
        logger.info("Companion client connected to %s",
                   self.coordinator_address)
    
    def get_model_parameters(
            self, vllm_config: VllmConfig,
            device_id: Optional[int] = None) -> dict[str, CUDATensorRebuildInfo]:
        """
        Get model parameters from companion server.
        
        Args:
            vllm_config: VllmConfig for the model
            device_id: GPU device ID (default: current device)
            
        Returns:
            Dict of parameter name to CUDATensorRebuildInfo
        """
        if device_id is None:
            device_id = torch.cuda.current_device()
        
        # Get rank information from the distributed environment
        # Send local TP/PP rank and let companion server handle DP adjustment
        # This makes it consistent with how workers are initialized
        world_group = get_world_group()
        wg_rank = world_group.rank
        wg_world_size = world_group.world_size
        
        tp_pp_world_size = vllm_config.parallel_config.world_size
        
        # For companions, we pass the local TP/PP rank and world size
        # The companion server will handle DP adjustment in parallel_state
        if wg_world_size == tp_pp_world_size:
            # World group is TP*PP only (no DP adjustment yet)
            local_tp_pp_rank = wg_rank
            local_tp_pp_world = wg_world_size
        else:
            # World group already includes DP
            # Extract the local TP/PP rank by undoing DP adjustment
            dp_rank = vllm_config.parallel_config.data_parallel_rank
            local_tp_pp_rank = wg_rank - (dp_rank * tp_pp_world_size)
            local_tp_pp_world = tp_pp_world_size

        # Use physical device id for local_rank (for logging/device selection)
        local_rank = device_id if device_id is not None else local_tp_pp_rank

        logger.info(
            "[COMPANION-CLIENT] Sending local TP/PP rank=%d (TP*PP world=%d) "
            "DP rank=%d to companion for DP adjustment",
            local_tp_pp_rank, local_tp_pp_world,
            vllm_config.parallel_config.data_parallel_rank)
        
        # Create request with local TP/PP rank info
        # Companion server will handle DP adjustment like workers do
        request = GetModelParametersRequest(
            vllm_config=vllm_config,
            device_id=device_id,
            local_rank=local_rank,
            global_rank=local_tp_pp_rank,  # Pass local TP/PP rank
            world_size=local_tp_pp_world    # Pass TP*PP world size
        )
        
        # Send request and get response
        self.socket.send(pickle.dumps(request))
        response: ModelParametersResponse = pickle.loads(self.socket.recv())
        
        if not response.success:
            # Check if it's a hash mismatch error
            if response.error and response.error.startswith("hash_mismatch:"):
                parts = response.error.split(":")
                if len(parts) >= 3:
                    server_hash = parts[1]
                    request_hash = parts[2]
                    raise RuntimeError(
                        f"Model configuration mismatch: Companion server has "
                        f"already loaded a model with hash {server_hash}, but "
                        f"received request for different model with hash "
                        f"{request_hash}. This typically happens when a warm "
                        f"spare worker requests a different model "
                        f"configuration than the primary worker. Each "
                        f"companion server can "
                        f"only serve one model configuration."
                    )
            raise RuntimeError(
                f"Failed to get model parameters: {response.error}")
        
        # Return the CUDATensorRebuildInfo objects directly (same as Dynamo)
        model_parameters = response.model_parameters or {}
        
        logger.info("Received %d model parameters from companion", len(model_parameters))
        return model_parameters
    
    def reconstruct_parameter(self, rebuild_info: CUDATensorRebuildInfo) -> torch.Tensor:
        """Reconstruct a model parameter tensor from rebuild info using CUDA IPC.
        
        Args:
            rebuild_info: CUDATensorRebuildInfo object containing CUDA IPC rebuild information
            
        Returns:
            Reconstructed tensor on the current device
        """
        from torch.multiprocessing.reductions import rebuild_cuda_tensor
        
        # Ensure we have a CUDATensorRebuildInfo object
        if not isinstance(rebuild_info, CUDATensorRebuildInfo):
            raise ValueError(f"Expected CUDATensorRebuildInfo, got {type(rebuild_info)}")
        
        # Get the logical device (what this process sees after CUDA_VISIBLE_DEVICES mapping)
        logical_device = torch.cuda.current_device()
        
        # The server device in rebuild_info is the physical device
        # We need to adjust it to our logical device for reconstruction
        server_device = rebuild_info.device
        
        logger.debug(
            "Reconstructing tensor: server_device=%d → client_logical_device=%d",
            server_device,
            logical_device,
        )
        
        # Modify the device in the rebuild args
        rebuild_info.device = logical_device
        
        # Get rebuild args from the CUDATensorRebuildInfo object
        rebuild_args = rebuild_info.to_rebuild_args()
        
        # Reconstruct the tensor
        tensor = rebuild_cuda_tensor(*rebuild_args)
        
        return tensor
    
    def close(self):
        """Close the client connection."""
        self.socket.close()
        self.context.term()


