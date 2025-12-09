"""Shareable CuMem Allocator for VMM-based tensor sharing.

This module provides an allocator that supports:
1. Deferred allocation: reserve VA without physical memory
2. External import: map imported FDs to reserved VAs
3. Sleep/wake: backup to CPU, unmap, re-import, restore

Usage:
    allocator = get_allocator()

    # In deferred mode, tensors reserve VA but have no physical memory
    with allocator.use_deferred_pool():
        tensor = torch.empty(1024, 1024, device='cuda')
        # tensor.data_ptr() is valid VA, but no physical memory yet

    # Import external FD to map physical memory
    allocator.import_and_map(tensor.data_ptr(), fd, size)
    # Now tensor has physical memory from server!

    # Sleep: backup to CPU, unmap
    state = allocator.sleep_tensor(tensor)

    # Wake: re-import, restore
    allocator.wake_tensor(state, new_fd)
"""

import os
import sys
import ctypes
from contextlib import contextmanager
from dataclasses import dataclass
from typing import Dict, Optional, Tuple

import torch

# Add csrc to path for the extension
_csrc_path = os.path.join(os.path.dirname(__file__), 'csrc')
if _csrc_path not in sys.path:
    sys.path.insert(0, _csrc_path)

try:
    import _shareable_cumem_ext as _ext
    _EXTENSION_AVAILABLE = True
except ImportError as e:
    print(f"Warning: _shareable_cumem_ext extension not available: {e}")
    _EXTENSION_AVAILABLE = False
    _ext = None


def _find_library_path(lib_name: str) -> Optional[str]:
    """Find path of a loaded shared library."""
    try:
        with open("/proc/self/maps") as f:
            for line in f:
                if lib_name in line:
                    start = line.index("/")
                    return line[start:].strip()
    except:
        pass
    return None


@dataclass
class AllocationInfo:
    """Information about a tracked allocation."""
    va: int
    size: int
    aligned_size: int
    handle: int
    is_deferred: bool
    tag: str


@dataclass
class SleepState:
    """State saved when a tensor is put to sleep."""
    va: int
    size: int
    aligned_size: int
    cpu_backup: torch.Tensor
    shape: Tuple[int, ...]
    dtype: torch.dtype
    strides: Tuple[int, ...]
    tag: str


class ShareableCuMemAllocator:
    """
    Singleton shareable CUDA memory allocator using VMM API.

    Supports deferred allocation mode where allocations only reserve VA
    without allocating physical memory. Later, import_and_map() maps
    external memory (from another process) to the reserved VA.
    """

    _instance: Optional["ShareableCuMemAllocator"] = None
    default_tag: str = "default"

    @classmethod
    def get_instance(cls) -> "ShareableCuMemAllocator":
        """Get the singleton instance."""
        if not _EXTENSION_AVAILABLE:
            raise RuntimeError("C extension not available")
        if cls._instance is None:
            cls._instance = cls()
        return cls._instance

    def __init__(self):
        if not _EXTENSION_AVAILABLE:
            raise RuntimeError("C extension not available")

        # Check for incompatible PyTorch settings
        conf = os.environ.get("PYTORCH_CUDA_ALLOC_CONF", "")
        if "expandable_segments:True" in conf:
            raise RuntimeError(
                "Expandable segments are not compatible with ShareableCuMemAllocator"
            )

        self.device = torch.cuda.current_device()
        self.granularity = _ext.get_granularity(self.device)

        # Track allocations
        self.allocations: Dict[int, AllocationInfo] = {}
        self.sleep_states: Dict[int, SleepState] = {}
        self.current_tag = self.default_tag

        # Initialize extension with callbacks
        self._malloc_callback = self._on_malloc
        self._free_callback = self._on_free
        _ext.init_module(self._malloc_callback, self._free_callback)

        # Find the library path for CUDAPluggableAllocator
        self._lib_path = _find_library_path("_shareable_cumem_ext")
        if not self._lib_path:
            import glob
            so_files = glob.glob(os.path.join(_csrc_path, "_shareable_cumem_ext*.so"))
            if so_files:
                self._lib_path = so_files[0]

        if not self._lib_path:
            raise RuntimeError("Could not find _shareable_cumem_ext shared library")

        print(f"[ShareableCuMemAllocator] Initialized, granularity={self.granularity}")

    def _on_malloc(self, va: int, size: int, aligned_size: int, handle: int, is_deferred: int):
        """Callback when memory is allocated."""
        info = AllocationInfo(
            va=va,
            size=size,
            aligned_size=aligned_size,
            handle=handle,
            is_deferred=bool(is_deferred),
            tag=self.current_tag,
        )
        self.allocations[va] = info

    def _on_free(self, va: int):
        """Callback when memory is freed."""
        self.allocations.pop(va, None)
        self.sleep_states.pop(va, None)

    @contextmanager
    def use_deferred_pool(self, tag: Optional[str] = None):
        """
        Context manager for deferred allocation mode.

        Inside this context, all CUDA allocations will only reserve VA
        without allocating physical memory. Use import_and_map() later
        to map external memory to these VAs.
        """
        if tag is None:
            tag = self.default_tag

        old_tag = self.current_tag
        self.current_tag = tag

        _ext.set_deferred_mode(True)

        allocator = torch.cuda.memory.CUDAPluggableAllocator(
            self._lib_path, "my_malloc", "my_free"
        )
        mem_pool = torch.cuda.memory.MemPool(allocator._allocator)

        try:
            with torch.cuda.memory.use_mem_pool(mem_pool):
                yield mem_pool
        finally:
            _ext.set_deferred_mode(False)
            self.current_tag = old_tag

    def import_and_map(self, va: int, fd: int, size: int) -> int:
        """
        Import external FD and map to reserved VA.

        Args:
            va: Virtual address (from tensor.data_ptr())
            fd: File descriptor from server (via SCM_RIGHTS)
            size: Size of the allocation

        Returns:
            The imported handle
        """
        handle = _ext.import_and_map(va, fd, size, self.device)

        # Update tracking to reflect actual mapped size (may differ from original alloc)
        granularity = self.granularity
        aligned_size = ((size + granularity - 1) // granularity) * granularity

        if va in self.allocations:
            self.allocations[va].handle = handle
            self.allocations[va].is_deferred = False
            self.allocations[va].size = size
            self.allocations[va].aligned_size = aligned_size

        return handle

    def unmap(self, va: int):
        """
        Unmap imported memory (for sleep).

        Args:
            va: Virtual address to unmap
        """
        _ext.unmap_imported(va)

        if va in self.allocations:
            self.allocations[va].handle = 0

    def sleep_tensor(self, tensor: torch.Tensor) -> SleepState:
        """
        Put a tensor to sleep - backup to CPU, unmap.

        Args:
            tensor: The tensor to sleep

        Returns:
            SleepState that can be used to wake the tensor
        """
        va = tensor.data_ptr()

        if va not in self.allocations:
            raise ValueError(f"Tensor VA 0x{va:x} not in tracked allocations")

        info = self.allocations[va]

        # Backup to CPU
        cpu_backup = tensor.detach().cpu().clone()

        # Unmap (but VA reservation remains)
        self.unmap(va)

        state = SleepState(
            va=va,
            size=info.size,
            aligned_size=info.aligned_size,
            cpu_backup=cpu_backup,
            shape=tuple(tensor.shape),
            dtype=tensor.dtype,
            strides=tuple(tensor.stride()),
            tag=info.tag,
        )

        self.sleep_states[va] = state
        return state

    def wake_tensor(self, state: SleepState, fd: int) -> None:
        """
        Wake a tensor from sleep - re-import FD, restore data.

        Args:
            state: SleepState from sleep_tensor()
            fd: Fresh FD from server
        """
        va = state.va

        # Re-import and map
        self.import_and_map(va, fd, state.aligned_size)

        # Copy backup data back to GPU using cudaMemcpy
        _cudart = ctypes.CDLL("libcudart.so")
        cpu_ptr = state.cpu_backup.data_ptr()
        size_bytes = state.cpu_backup.numel() * state.cpu_backup.element_size()
        _cudart.cudaMemcpy(
            ctypes.c_void_p(va),
            ctypes.c_void_p(cpu_ptr),
            ctypes.c_size_t(size_bytes),
            ctypes.c_int(1)  # cudaMemcpyHostToDevice
        )
        torch.cuda.synchronize()

        # Remove from sleep states
        self.sleep_states.pop(va, None)

    def get_allocation_info(self, va: int) -> Optional[AllocationInfo]:
        """Get allocation info for a VA."""
        return self.allocations.get(va)

    def is_sleeping(self, va: int) -> bool:
        """Check if a VA is in sleep state."""
        return va in self.sleep_states


def get_allocator() -> ShareableCuMemAllocator:
    """Get the ShareableCuMemAllocator singleton."""
    return ShareableCuMemAllocator.get_instance()
