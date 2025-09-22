# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""Message types for MultiProc companion server/client communication."""

from typing import Optional, Any
from dataclasses import dataclass
from enum import Enum
import pickle
import hashlib
import torch


# Enums
class RequestType(Enum):
    """Types of requests that can be sent to coordinator/companion."""
    GET_COMPANION_STATUS = "get_companion_status"
    HANDSHAKE = "handshake"
    LOAD_MODEL = "load_model"
    GET_MODEL_PARAMETERS_REBUILD_INFO = "get_model_parameters_rebuild_info"


class ResponseType(Enum):
    """Types of responses that can be received from coordinator/companion."""
    COMPANION_STATUS = "companion_status"
    HANDSHAKE = "handshake"
    LOAD_MODEL = "load_model"
    MODEL_PARAMETERS_REBUILD_INFO = "model_parameters_rebuild_info"


class CompanionState(Enum):
    """State of an individual companion server."""
    INITIALIZING = "initializing"
    LOADING = "loading"
    READY = "ready"
    ERROR = "error"


# Status request/response messages
@dataclass
class GetCompanionStatusRequest:
    """Request to get the status of a specific companion."""
    device_id: int
    request_type: RequestType = RequestType.GET_COMPANION_STATUS


@dataclass
class GetCompanionStatusResponse:
    """Response with companion status information."""
    device_id: int
    state: CompanionState
    error_message: Optional[str] = None
    response_type: ResponseType = ResponseType.COMPANION_STATUS


# Handshake messages
@dataclass
class HandshakeRequest:
    """Request to perform handshake with coordinator or companion."""
    device_id: Optional[int] = None  # None for coordinator handshake
    request_type: RequestType = RequestType.HANDSHAKE


@dataclass
class HandshakeResponse:
    """Response to handshake request."""
    success: bool
    message: str = "OK"
    response_type: ResponseType = ResponseType.HANDSHAKE


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


# Model loading messages (split from GetModelParameters)
@dataclass
class LoadModelRequest:
    """Request to load model on companion server."""
    vllm_config: Any  # VllmConfig object
    device_id: int
    local_rank: int
    global_rank: int
    world_size: int
    request_type: RequestType = RequestType.LOAD_MODEL
    
    def compute_hash(self) -> str:
        """Compute a hash for this request configuration."""
        config_data = {
            'model_config_hash': self.vllm_config.model_config.compute_hash(),
            'parallel_config_hash': (
                self.vllm_config.parallel_config.compute_hash()),
            'cache_config_hash': self.vllm_config.cache_config.compute_hash(),
            'device_config_hash': self.vllm_config.device_config.compute_hash(),
            'load_config_hash': self.vllm_config.load_config.compute_hash(),
            'lora_config_hash': (
                self.vllm_config.lora_config.compute_hash() 
                if self.vllm_config.lora_config else "None"),
            'local_rank': self.local_rank,
            'global_rank': self.global_rank,
            'world_size': self.world_size
        }
        return hashlib.md5(pickle.dumps(config_data)).hexdigest()


@dataclass
class LoadModelResponse:
    """Response to model loading request."""
    success: bool
    error: Optional[str] = None
    response_type: ResponseType = ResponseType.LOAD_MODEL


@dataclass
class GetModelParametersRebuildInfoRequest:
    """Request to get model parameter rebuild info after model is loaded."""
    device_id: int
    request_type: RequestType = RequestType.GET_MODEL_PARAMETERS_REBUILD_INFO


@dataclass
class ModelParametersRebuildInfoResponse:
    """Response containing model parameter rebuild info."""
    success: bool
    model_parameters: Optional[dict[str, CUDATensorRebuildInfo]] = None
    error: Optional[str] = None
    response_type: ResponseType = ResponseType.MODEL_PARAMETERS_REBUILD_INFO