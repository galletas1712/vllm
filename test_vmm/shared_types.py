"""Shared data types for VMM-based IPC."""

from dataclasses import dataclass
from typing import Optional, Tuple
import torch


@dataclass
class TensorIPCInfo:
    """Information needed to import a tensor via VMM IPC.

    This is what gets sent from server to client for each tensor.

    For quantized types (INT4, FP4, etc.):
    - dtype is the STORAGE dtype (e.g., torch.uint8 for packed INT4)
    - scalar_type_id contains vLLM's ScalarType.id describing the actual type
    - pack_factor indicates how many values are packed per storage element
    """
    # Tensor shape and type
    shape: Tuple[int, ...]
    strides: Tuple[int, ...]
    dtype: torch.dtype  # Storage dtype (may be container for packed types)

    # Storage info
    storage_size_bytes: int
    storage_offset: int  # Offset in elements

    # VMM allocation info
    allocation_size: int  # Aligned allocation size

    # For verification - just the sum
    checksum: float

    # Quantization metadata (optional)
    # If set, this is a vLLM ScalarType.id describing the actual quantized dtype
    # The tensor's torch.dtype is just the storage container
    scalar_type_id: Optional[int] = None

    # Pack factor for sub-byte types (e.g., 2 for INT4 packed in uint8)
    pack_factor: int = 1


def compute_checksum(tensor: torch.Tensor) -> float:
    """Compute checksum as the sum of all elements.

    Simple and deterministic across processes.
    Handles FP8 and other non-standard dtypes by converting to float32 first.
    """
    # FP8 types can't convert directly to float64, go through float32
    dtype_str = str(tensor.dtype)
    if 'float8' in dtype_str:
        # FP8 -> float32 -> float64 -> sum
        return float(tensor.to(torch.float32).to(torch.float64).sum().cpu().item())
    return float(tensor.to(torch.float64).sum().cpu().item())
