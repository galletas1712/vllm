# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""Message types for MultiProc companion server/client communication."""

from typing import Optional, Any
from dataclasses import dataclass
import pickle
import hashlib
import torch


@dataclass
class CUDATensorRebuildInfo:
    """Information needed to rebuild a CUDA tensor via IPC."""
    tensor_type: type[torch.Tensor]
    tensor_size: torch.Size
    tensor_stride: tuple[int, ...]
    tensor_offset: int
    storage_type: type
    tensor_dtype: torch.dtype
    device: int  # This is the CUDA Device ID
    ipc_handle: bytes
    storage_size_bytes: int
    storage_offset_bytes: int
    tensor_requires_grad: bool
    ref_counter_handle: bytes
    ref_counter_offset: int
    event_handle: bytes
    event_sync_required: bool

    @classmethod
    def from_rebuild_args(cls, rebuild_args: tuple) -> "CUDATensorRebuildInfo":
        return cls(*rebuild_args)

    def to_rebuild_args(self) -> tuple:
        return (
            self.tensor_type,
            self.tensor_size,
            self.tensor_stride,
            self.tensor_offset,
            self.storage_type,
            self.tensor_dtype,
            self.device,
            self.ipc_handle,
            self.storage_size_bytes,
            self.storage_offset_bytes,
            self.tensor_requires_grad,
            self.ref_counter_handle,
            self.ref_counter_offset,
            self.event_handle,
            self.event_sync_required,
        )


@dataclass
class GetModelParametersRequest:
    """Request to get model parameters from the companion server."""
    vllm_config: Any  # VllmConfig object (pickled directly)
    device_id: int
    local_rank: int
    global_rank: int
    world_size: int
    
    def compute_hash(self) -> str:
        """Compute a hash for this request configuration."""
        config_data = {
            'vllm_config': self.vllm_config.compute_hash(),
            'local_rank': self.local_rank,
            'global_rank': self.global_rank,
            'world_size': self.world_size
        }
        return hashlib.md5(pickle.dumps(config_data)).hexdigest()


@dataclass
class ModelParametersResponse:
    """Response containing model parameters for IPC sharing."""
    success: bool
    model_parameters: Optional[dict[str, CUDATensorRebuildInfo]] = None  # Dict of name -> CUDATensorRebuildInfo
    error: Optional[str] = None