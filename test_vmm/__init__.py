"""VMM-based IPC test module.

Demonstrates using cuMem VMM API for zero-copy tensor sharing between processes:

1. Server (companion):
   - Creates tensors with VMM allocator
   - Exports shareable handles (file descriptors)
   - Sends FDs + metadata to client

2. Client (vLLM worker):
   - Reserves VA (no physical memory)
   - Imports shareable handles from server
   - Maps reserved VAs to imported allocations
   - Creates tensors via DLPack (true zero-copy)

Run test:
    python -m test_vmm.run_test --device 0
"""

from .shared_types import TensorIPCInfo, compute_checksum
from .deferred_allocator import (
    DeferredAllocator,
    VAReservation,
    MappedAllocation,
    SleepState,
    tensor_from_vmm,
)

__all__ = [
    'TensorIPCInfo',
    'compute_checksum',
    'DeferredAllocator',
    'VAReservation',
    'MappedAllocation',
    'SleepState',
    'tensor_from_vmm',
]
