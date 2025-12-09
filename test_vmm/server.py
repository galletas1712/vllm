"""Server process - creates tensors with VMM allocator and exports handles.

This simulates the companion process that owns the model weights.
"""

import ctypes
from ctypes import c_void_p, c_size_t, c_int, c_ulonglong, byref, Structure
from dataclasses import dataclass
from typing import Dict, List, Tuple
import os

import torch

from shared_types import TensorIPCInfo, compute_checksum


# CUDA constants and types
CU_MEM_ALLOCATION_TYPE_PINNED = 0x1
CU_MEM_LOCATION_TYPE_DEVICE = 0x1
CU_MEM_ACCESS_FLAGS_PROT_READWRITE = 0x3
CU_MEM_ALLOC_GRANULARITY_MINIMUM = 0x0
CU_MEM_HANDLE_TYPE_POSIX_FILE_DESCRIPTOR = 0x1

CUdeviceptr = c_ulonglong
CUmemGenericAllocationHandle = c_ulonglong

_cuda = ctypes.CDLL("libcuda.so.1")
_cudart = ctypes.CDLL("libcudart.so")


class CUmemLocation(Structure):
    _fields_ = [("type", c_int), ("id", c_int)]


class CUmemAllocationProp(Structure):
    _fields_ = [
        ("type", c_int),
        ("requestedHandleTypes", c_int),
        ("location", CUmemLocation),
        ("win32HandleMetaData", c_void_p),
        ("allocFlags_compressionType", ctypes.c_uint),
        ("allocFlags_gpuDirectRDMACapable", ctypes.c_uint),
        ("allocFlags_usage", ctypes.c_uint),
        ("allocFlags_reserved", ctypes.c_uint * 4),
    ]


class CUmemAccessDesc(Structure):
    _fields_ = [("location", CUmemLocation), ("flags", c_int)]


def _check(result: int, name: str):
    if result != 0:
        err = ctypes.c_char_p()
        _cuda.cuGetErrorString(result, byref(err))
        raise RuntimeError(f"{name}: {err.value.decode() if err.value else result}")


def _ensure_ctx(device: int):
    ctx = c_void_p()
    _cuda.cuCtxGetCurrent(byref(ctx))
    if not ctx.value:
        _cuda.cuDevicePrimaryCtxRetain(byref(ctx), device)
        _cuda.cuCtxSetCurrent(ctx)


def _get_granularity(device: int) -> int:
    prop = CUmemAllocationProp()
    prop.type = CU_MEM_ALLOCATION_TYPE_PINNED
    prop.location.type = CU_MEM_LOCATION_TYPE_DEVICE
    prop.location.id = device
    prop.requestedHandleTypes = CU_MEM_HANDLE_TYPE_POSIX_FILE_DESCRIPTOR
    gran = c_size_t()
    _check(_cuda.cuMemGetAllocationGranularity(byref(gran), byref(prop), 0), "granularity")
    return gran.value


def _align(size: int, gran: int) -> int:
    return ((size + gran - 1) // gran) * gran


@dataclass
class ServerAllocation:
    """A VMM allocation owned by the server."""
    va: int
    size: int  # Aligned
    handle: int  # CUmemGenericAllocationHandle
    fd: int = -1  # Exported shareable handle


class ServerAllocator:
    """VMM allocator for server - full allocation with shareable handle support."""

    def __init__(self, device: int = 0):
        self.device = device
        _ensure_ctx(device)
        self.granularity = _get_granularity(device)
        self.allocations: Dict[int, ServerAllocation] = {}

    def allocate(self, size: int) -> ServerAllocation:
        """Allocate VMM memory with shareable handle support."""
        _ensure_ctx(self.device)
        aligned = _align(size, self.granularity)

        # Set up allocation properties for shareable handle
        prop = CUmemAllocationProp()
        prop.type = CU_MEM_ALLOCATION_TYPE_PINNED
        prop.location.type = CU_MEM_LOCATION_TYPE_DEVICE
        prop.location.id = self.device
        prop.requestedHandleTypes = CU_MEM_HANDLE_TYPE_POSIX_FILE_DESCRIPTOR

        # Reserve VA
        va = CUdeviceptr()
        _check(_cuda.cuMemAddressReserve(byref(va), c_size_t(aligned), c_size_t(self.granularity), CUdeviceptr(0), c_ulonglong(0)), "reserve")

        # Create physical allocation
        handle = CUmemGenericAllocationHandle()
        r = _cuda.cuMemCreate(byref(handle), c_size_t(aligned), byref(prop), c_ulonglong(0))
        if r != 0:
            _cuda.cuMemAddressFree(va, c_size_t(aligned))
            _check(r, "create")

        # Map
        r = _cuda.cuMemMap(va, c_size_t(aligned), c_size_t(0), handle, c_ulonglong(0))
        if r != 0:
            _cuda.cuMemRelease(handle)
            _cuda.cuMemAddressFree(va, c_size_t(aligned))
            _check(r, "map")

        # Set access
        acc = CUmemAccessDesc()
        acc.location.type = CU_MEM_LOCATION_TYPE_DEVICE
        acc.location.id = self.device
        acc.flags = CU_MEM_ACCESS_FLAGS_PROT_READWRITE
        r = _cuda.cuMemSetAccess(va, c_size_t(aligned), byref(acc), c_size_t(1))
        if r != 0:
            _cuda.cuMemUnmap(va, c_size_t(aligned))
            _cuda.cuMemRelease(handle)
            _cuda.cuMemAddressFree(va, c_size_t(aligned))
            _check(r, "access")

        alloc = ServerAllocation(va=va.value, size=aligned, handle=handle.value)
        self.allocations[va.value] = alloc
        return alloc

    def export_handle(self, alloc: ServerAllocation) -> int:
        """Export allocation to shareable file descriptor."""
        if alloc.fd >= 0:
            return alloc.fd

        _ensure_ctx(self.device)
        fd = c_int()
        _check(_cuda.cuMemExportToShareableHandle(
            byref(fd),
            CUmemGenericAllocationHandle(alloc.handle),
            c_int(CU_MEM_HANDLE_TYPE_POSIX_FILE_DESCRIPTOR),
            c_ulonglong(0)
        ), "export")
        alloc.fd = fd.value
        return alloc.fd

    def free(self, alloc: ServerAllocation):
        """Free an allocation."""
        _ensure_ctx(self.device)
        _cuda.cuMemUnmap(CUdeviceptr(alloc.va), c_size_t(alloc.size))
        _cuda.cuMemRelease(CUmemGenericAllocationHandle(alloc.handle))
        _cuda.cuMemAddressFree(CUdeviceptr(alloc.va), c_size_t(alloc.size))
        if alloc.fd >= 0:
            os.close(alloc.fd)
        self.allocations.pop(alloc.va, None)


def copy_to_vmm(tensor: torch.Tensor, alloc: ServerAllocation):
    """Copy tensor data to VMM allocation."""
    src = tensor.data_ptr()
    dst = alloc.va
    size = tensor.numel() * tensor.element_size()
    _cudart.cudaMemcpy(c_void_p(dst), c_void_p(src), c_size_t(size), c_int(3))  # D2D
    torch.cuda.synchronize()


def create_test_tensors(device: int) -> Dict[str, torch.Tensor]:
    """Create test tensors of various types and sizes."""
    d = torch.device(f"cuda:{device}")
    tensors = {
        "small_f32": torch.randn(64, 64, dtype=torch.float32, device=d),
        "medium_f16": torch.randn(256, 256, dtype=torch.float16, device=d),
        "large_f32": torch.randn(1024, 1024, dtype=torch.float32, device=d),
        "vector_i64": torch.randint(0, 1000, (10000,), dtype=torch.int64, device=d),
    }
    if torch.cuda.is_bf16_supported():
        tensors["tiny_bf16"] = torch.randn(32, 32, dtype=torch.bfloat16, device=d)

    # FP8 types (PyTorch 2.1+, requires Hopper or later for native support)
    if hasattr(torch, 'float8_e4m3fn'):
        try:
            # Create FP8 tensor - cast from float16 to preserve reasonable values
            fp8_data = torch.randn(128, 128, dtype=torch.float16, device=d)
            tensors["fp8_e4m3"] = fp8_data.to(torch.float8_e4m3fn)
            print(f"[Server] FP8 E4M3 tensor created successfully")
        except Exception as e:
            print(f"[Server] FP8 E4M3 not supported: {e}")

    if hasattr(torch, 'float8_e5m2'):
        try:
            fp8_data = torch.randn(128, 128, dtype=torch.float16, device=d)
            tensors["fp8_e5m2"] = fp8_data.to(torch.float8_e5m2)
            print(f"[Server] FP8 E5M2 tensor created successfully")
        except Exception as e:
            print(f"[Server] FP8 E5M2 not supported: {e}")

    # Packed INT4 simulation (stored as uint8, 2 values per byte)
    # This demonstrates how quantized weights would be transferred
    packed_shape = (64, 32)  # Represents 64x64 INT4 values packed into 64x32 uint8
    tensors["packed_int4"] = torch.randint(0, 256, packed_shape, dtype=torch.uint8, device=d)

    return tensors


def server_main(send_fn, recv_fn, device: int = 0):
    """Server main loop.

    Args:
        send_fn: Function to send (name, info, fd) to client.
        recv_fn: Function to receive ack from client.
        device: CUDA device ID.
    """
    print(f"[Server] Starting on device {device}")
    torch.cuda.set_device(device)

    allocator = ServerAllocator(device)
    print(f"[Server] Granularity: {allocator.granularity} bytes")

    # Create test tensors
    tensors = create_test_tensors(device)
    allocations: Dict[str, ServerAllocation] = {}
    tensor_infos: Dict[str, TensorIPCInfo] = {}  # Store for wake phase

    try:
        # Phase 1: Initial Import
        print("\n" + "=" * 50)
        print("PHASE 1: Initial Export")
        print("=" * 50)

        for name, tensor in tensors.items():
            print(f"\n[Server] {name}: shape={tuple(tensor.shape)}, dtype={tensor.dtype}")

            # Allocate VMM memory
            size_bytes = tensor.numel() * tensor.element_size()
            alloc = allocator.allocate(size_bytes)
            allocations[name] = alloc
            print(f"[Server]   VA=0x{alloc.va:x}, size={alloc.size}")

            # Copy tensor to VMM
            copy_to_vmm(tensor, alloc)

            # Export shareable handle
            fd = allocator.export_handle(alloc)
            print(f"[Server]   FD={fd}")

            # Compute checksum
            checksum = compute_checksum(tensor)
            print(f"[Server]   Checksum={checksum}")

            # Create IPC info
            info = TensorIPCInfo(
                shape=tuple(tensor.shape),
                strides=tuple(tensor.stride()),
                dtype=tensor.dtype,
                storage_size_bytes=size_bytes,
                storage_offset=0,
                allocation_size=alloc.size,
                checksum=checksum,
            )
            tensor_infos[name] = info

            # Send to client
            send_fn(name, info, fd)

            # Wait for ack
            ack = recv_fn()
            print(f"[Server]   Client: {ack}")

        # Signal done with initial phase
        send_fn(None, None, -1)

        # Wait for client to signal sleep complete
        print("\n" + "=" * 50)
        print("PHASE 2: Waiting for Client Sleep")
        print("=" * 50)

        sleep_msg = recv_fn()
        print(f"[Server] Received: {sleep_msg}")

        if sleep_msg.get("status") == "sleep_complete":
            # Phase 3: Wake - re-export FDs
            print("\n" + "=" * 50)
            print("PHASE 3: Wake Export")
            print("=" * 50)

            wake_names = sleep_msg.get("names", [])
            for name in wake_names:
                if name not in allocations:
                    print(f"[Server] WARNING: Unknown tensor {name}")
                    continue

                alloc = allocations[name]
                info = tensor_infos[name]

                # Close old FD and export fresh one
                if alloc.fd >= 0:
                    os.close(alloc.fd)
                    alloc.fd = -1

                fd = allocator.export_handle(alloc)
                print(f"\n[Server] Wake {name}: new FD={fd}")

                # Send fresh FD to client
                send_fn(name, info, fd)

                # Wait for ack
                ack = recv_fn()
                print(f"[Server]   Client: {ack}")

            # Signal wake phase done
            send_fn(None, None, -1)

        # Wait for final result
        final = recv_fn()
        print(f"\n[Server] Final: {final}")

    finally:
        print("[Server] Cleanup...")
        for alloc in allocations.values():
            allocator.free(alloc)
        print("[Server] Done")
