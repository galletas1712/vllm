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


def compute_sum_of_squares(tensor: torch.Tensor) -> float:
    """Compute sum of squares of all elements.

    This is a more rigorous check than simple sum since it's sensitive
    to the magnitude of each element, not just the aggregate.
    """
    dtype_str = str(tensor.dtype)
    if 'float8' in dtype_str:
        t = tensor.to(torch.float32).to(torch.float64)
    else:
        t = tensor.to(torch.float64)
    return float((t * t).sum().cpu().item())


def compute_aggregate_checksum(tensors: dict) -> float:
    """Compute aggregate sum-of-squares across all tensors.

    This verifies that ALL tensor data is correctly shared by computing
    a single aggregate value that depends on every element of every tensor.

    Args:
        tensors: Dict mapping names to tensors

    Returns:
        Sum of (sum of squares) for each tensor, plus a weighted component
        that encodes tensor ordering and sizes.
    """
    total = 0.0
    for i, (name, tensor) in enumerate(sorted(tensors.items())):
        # Sum of squares for this tensor
        sos = compute_sum_of_squares(tensor)
        # Weight by position to catch ordering issues
        weighted = sos * (i + 1)
        # Add tensor size component to catch shape mismatches
        size_component = tensor.numel() * (i + 1) * 0.001
        total += weighted + size_component
    return total
