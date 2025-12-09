// IPC-aware CUDAPluggableAllocator for VMM-based tensor sharing.
//
// This allocator supports two modes:
// 1. Deferred mode: my_malloc() only reserves VA, no physical allocation
// 2. Import mode: import external FD and map to reserved VA
//
// For IPC between companion (server) and vLLM worker (client):
// - Client uses deferred mode to reserve VAs
// - Client imports server's FDs and maps them to reserved VAs
// - Tensors naturally use the mapped memory
// - Sleep/wake preserves VAs while backing up data to CPU

#include <cuda.h>
#include <iostream>
#include <unordered_map>

#define PY_SSIZE_T_CLEAN
#include <Python.h>

// Error handling
static char error_msg[10240];
static CUresult error_code = CUDA_SUCCESS;

#define CUDA_CHECK(condition)                                           \
  do {                                                                  \
    CUresult error = condition;                                         \
    if (error != CUDA_SUCCESS) {                                        \
      error_code = error;                                               \
      const char* error_string;                                         \
      cuGetErrorString(error, &error_string);                           \
      snprintf(error_msg, sizeof(error_msg), "CUDA Error: %s at %s:%d", \
               error_string ? error_string : "unknown", __FILE__, __LINE__); \
      std::cerr << error_msg << std::endl;                              \
    }                                                                   \
  } while (0)

// Allocation tracking
struct AllocationInfo {
    CUdeviceptr va;
    size_t size;
    size_t aligned_size;
    int device;
    CUmemGenericAllocationHandle handle;  // 0 if not mapped (deferred mode)
    bool is_imported;  // true if memory was imported from external FD
};

static std::unordered_map<CUdeviceptr, AllocationInfo> g_allocations;

// Python callbacks for allocation tracking
static PyObject* g_python_malloc_callback = nullptr;
static PyObject* g_python_free_callback = nullptr;

// Deferred mode flag - when true, my_malloc only reserves VA
static bool g_deferred_mode = false;

// Helper to ensure CUDA context
static void ensure_context(int device) {
    CUcontext ctx;
    CUDA_CHECK(cuCtxGetCurrent(&ctx));
    if (!ctx) {
        CUDA_CHECK(cuDevicePrimaryCtxRetain(&ctx, device));
        CUDA_CHECK(cuCtxSetCurrent(ctx));
    }
}

// Get allocation granularity
static size_t get_granularity(int device) {
    CUmemAllocationProp prop = {};
    prop.type = CU_MEM_ALLOCATION_TYPE_PINNED;
    prop.location.type = CU_MEM_LOCATION_TYPE_DEVICE;
    prop.location.id = device;
    prop.requestedHandleTypes = CU_MEM_HANDLE_TYPE_POSIX_FILE_DESCRIPTOR;

    size_t granularity;
    CUDA_CHECK(cuMemGetAllocationGranularity(&granularity, &prop,
                                              CU_MEM_ALLOC_GRANULARITY_MINIMUM));
    return granularity;
}

// Align size to granularity
static size_t align_size(size_t size, size_t granularity) {
    return ((size + granularity - 1) / granularity) * granularity;
}

extern "C" {

// ---------------------------------------------------------------------------
// Pluggable allocator functions

void* my_malloc(ssize_t size, int device, CUstream stream) {
    ensure_context(device);

    size_t granularity = get_granularity(device);
    size_t aligned_size = align_size(size, granularity);

    // Reserve virtual address
    CUdeviceptr va;
    CUDA_CHECK(cuMemAddressReserve(&va, aligned_size, granularity, 0, 0));
    if (error_code != CUDA_SUCCESS) {
        return nullptr;
    }

    AllocationInfo info = {};
    info.va = va;
    info.size = size;
    info.aligned_size = aligned_size;
    info.device = device;
    info.handle = 0;
    info.is_imported = false;

    if (!g_deferred_mode) {
        // Normal mode: allocate and map physical memory
        CUmemAllocationProp prop = {};
        prop.type = CU_MEM_ALLOCATION_TYPE_PINNED;
        prop.location.type = CU_MEM_LOCATION_TYPE_DEVICE;
        prop.location.id = device;
        prop.requestedHandleTypes = CU_MEM_HANDLE_TYPE_POSIX_FILE_DESCRIPTOR;

        CUmemGenericAllocationHandle handle;
        CUDA_CHECK(cuMemCreate(&handle, aligned_size, &prop, 0));
        if (error_code != CUDA_SUCCESS) {
            cuMemAddressFree(va, aligned_size);
            return nullptr;
        }

        CUDA_CHECK(cuMemMap(va, aligned_size, 0, handle, 0));
        if (error_code != CUDA_SUCCESS) {
            cuMemRelease(handle);
            cuMemAddressFree(va, aligned_size);
            return nullptr;
        }

        CUmemAccessDesc access = {};
        access.location.type = CU_MEM_LOCATION_TYPE_DEVICE;
        access.location.id = device;
        access.flags = CU_MEM_ACCESS_FLAGS_PROT_READWRITE;

        CUDA_CHECK(cuMemSetAccess(va, aligned_size, &access, 1));
        if (error_code != CUDA_SUCCESS) {
            cuMemUnmap(va, aligned_size);
            cuMemRelease(handle);
            cuMemAddressFree(va, aligned_size);
            return nullptr;
        }

        info.handle = handle;
    }
    // else: deferred mode - VA reserved but no physical memory

    g_allocations[va] = info;

    // Call Python callback if set
    if (g_python_malloc_callback) {
        PyGILState_STATE gstate = PyGILState_Ensure();

        PyObject* args = Py_BuildValue("(KKKKi)",
            (unsigned long long)va,
            (unsigned long long)size,
            (unsigned long long)aligned_size,
            (unsigned long long)info.handle,
            g_deferred_mode ? 1 : 0);

        PyObject* result = PyObject_CallObject(g_python_malloc_callback, args);
        Py_DECREF(args);
        Py_XDECREF(result);

        if (PyErr_Occurred()) {
            PyErr_Print();
        }

        PyGILState_Release(gstate);
    }

    return (void*)va;
}

void my_free(void* ptr, ssize_t size, int device, CUstream stream) {
    CUdeviceptr va = (CUdeviceptr)ptr;

    auto it = g_allocations.find(va);
    if (it == g_allocations.end()) {
        std::cerr << "my_free: unknown pointer " << ptr << std::endl;
        return;
    }

    AllocationInfo& info = it->second;
    ensure_context(info.device);

    // Call Python callback if set
    if (g_python_free_callback) {
        PyGILState_STATE gstate = PyGILState_Ensure();

        PyObject* args = Py_BuildValue("(K)", (unsigned long long)va);
        PyObject* result = PyObject_CallObject(g_python_free_callback, args);
        Py_DECREF(args);
        Py_XDECREF(result);

        if (PyErr_Occurred()) {
            PyErr_Print();
        }

        PyGILState_Release(gstate);
    }

    // Unmap if mapped
    if (info.handle != 0) {
        CUDA_CHECK(cuMemUnmap(va, info.aligned_size));

        // Only release if we own the memory (not imported)
        if (!info.is_imported) {
            CUDA_CHECK(cuMemRelease(info.handle));
        }
    }

    // Free VA reservation
    CUDA_CHECK(cuMemAddressFree(va, info.aligned_size));

    g_allocations.erase(it);
}

// ---------------------------------------------------------------------------
// Python-exposed functions

// init_module(malloc_callback, free_callback)
static PyObject* py_init_module(PyObject* self, PyObject* args) {
    PyObject* malloc_cb = nullptr;
    PyObject* free_cb = nullptr;

    if (!PyArg_ParseTuple(args, "OO", &malloc_cb, &free_cb)) {
        return nullptr;
    }

    if (!PyCallable_Check(malloc_cb) || !PyCallable_Check(free_cb)) {
        PyErr_SetString(PyExc_TypeError, "Both arguments must be callables");
        return nullptr;
    }

    g_python_malloc_callback = malloc_cb;
    g_python_free_callback = free_cb;

    Py_RETURN_NONE;
}

// set_deferred_mode(enabled)
static PyObject* py_set_deferred_mode(PyObject* self, PyObject* args) {
    int enabled;
    if (!PyArg_ParseTuple(args, "p", &enabled)) {
        return nullptr;
    }
    g_deferred_mode = enabled;
    Py_RETURN_NONE;
}

// get_deferred_mode() -> bool
static PyObject* py_get_deferred_mode(PyObject* self, PyObject* args) {
    return PyBool_FromLong(g_deferred_mode);
}

// import_and_map(va, fd, size, device) -> handle
// Import external FD and map to reserved VA
static PyObject* py_import_and_map(PyObject* self, PyObject* args) {
    unsigned long long va;
    int fd;
    unsigned long long size;
    int device;

    if (!PyArg_ParseTuple(args, "KiKi", &va, &fd, &size, &device)) {
        return nullptr;
    }

    auto it = g_allocations.find((CUdeviceptr)va);
    if (it == g_allocations.end()) {
        PyErr_SetString(PyExc_ValueError, "VA not found in allocations");
        return nullptr;
    }

    AllocationInfo& info = it->second;

    if (info.handle != 0) {
        PyErr_SetString(PyExc_RuntimeError, "VA already mapped");
        return nullptr;
    }

    ensure_context(device);
    error_code = CUDA_SUCCESS;

    // Import the shareable handle
    CUmemGenericAllocationHandle handle;
    CUDA_CHECK(cuMemImportFromShareableHandle(
        &handle, (void*)(intptr_t)fd, CU_MEM_HANDLE_TYPE_POSIX_FILE_DESCRIPTOR));
    if (error_code != CUDA_SUCCESS) {
        PyErr_SetString(PyExc_RuntimeError, error_msg);
        return nullptr;
    }

    // Map to reserved VA
    size_t granularity = get_granularity(device);
    size_t aligned_size = align_size(size, granularity);

    CUDA_CHECK(cuMemMap((CUdeviceptr)va, aligned_size, 0, handle, 0));
    if (error_code != CUDA_SUCCESS) {
        cuMemRelease(handle);
        PyErr_SetString(PyExc_RuntimeError, error_msg);
        return nullptr;
    }

    // Set access permissions
    CUmemAccessDesc access = {};
    access.location.type = CU_MEM_LOCATION_TYPE_DEVICE;
    access.location.id = device;
    access.flags = CU_MEM_ACCESS_FLAGS_PROT_READWRITE;

    CUDA_CHECK(cuMemSetAccess((CUdeviceptr)va, aligned_size, &access, 1));
    if (error_code != CUDA_SUCCESS) {
        cuMemUnmap((CUdeviceptr)va, aligned_size);
        cuMemRelease(handle);
        PyErr_SetString(PyExc_RuntimeError, error_msg);
        return nullptr;
    }

    info.handle = handle;
    info.is_imported = true;
    // Update sizes to reflect what was actually mapped (may differ from original allocation)
    info.size = size;
    info.aligned_size = aligned_size;

    return PyLong_FromUnsignedLongLong((unsigned long long)handle);
}

// unmap_imported(va) - Unmap without releasing (for sleep)
static PyObject* py_unmap_imported(PyObject* self, PyObject* args) {
    unsigned long long va;

    if (!PyArg_ParseTuple(args, "K", &va)) {
        return nullptr;
    }

    auto it = g_allocations.find((CUdeviceptr)va);
    if (it == g_allocations.end()) {
        PyErr_SetString(PyExc_ValueError, "VA not found in allocations");
        return nullptr;
    }

    AllocationInfo& info = it->second;

    if (info.handle == 0) {
        PyErr_SetString(PyExc_RuntimeError, "VA not mapped");
        return nullptr;
    }

    ensure_context(info.device);
    error_code = CUDA_SUCCESS;

    CUDA_CHECK(cuMemUnmap((CUdeviceptr)va, info.aligned_size));
    if (error_code != CUDA_SUCCESS) {
        PyErr_SetString(PyExc_RuntimeError, error_msg);
        return nullptr;
    }

    // Release the imported handle (we need to re-import on wake)
    CUDA_CHECK(cuMemRelease(info.handle));

    info.handle = 0;

    Py_RETURN_NONE;
}

// get_allocation_info(va) -> (size, aligned_size, handle, is_imported, device)
static PyObject* py_get_allocation_info(PyObject* self, PyObject* args) {
    unsigned long long va;

    if (!PyArg_ParseTuple(args, "K", &va)) {
        return nullptr;
    }

    auto it = g_allocations.find((CUdeviceptr)va);
    if (it == g_allocations.end()) {
        PyErr_SetString(PyExc_ValueError, "VA not found in allocations");
        return nullptr;
    }

    AllocationInfo& info = it->second;

    return Py_BuildValue("(KKKpi)",
        (unsigned long long)info.size,
        (unsigned long long)info.aligned_size,
        (unsigned long long)info.handle,
        info.is_imported,
        info.device);
}

// get_granularity(device) -> int
static PyObject* py_get_granularity(PyObject* self, PyObject* args) {
    int device;
    if (!PyArg_ParseTuple(args, "i", &device)) {
        return nullptr;
    }

    ensure_context(device);
    size_t gran = get_granularity(device);

    return PyLong_FromSize_t(gran);
}

// reserve_va(size, device) -> va
// Directly reserve VA without going through PyTorch's allocation machinery
// This gives us exact control over the VA size (no MemPool rounding)
static PyObject* py_reserve_va(PyObject* self, PyObject* args) {
    unsigned long long size;
    int device;

    if (!PyArg_ParseTuple(args, "Ki", &size, &device)) {
        return nullptr;
    }

    ensure_context(device);
    error_code = CUDA_SUCCESS;

    size_t granularity = get_granularity(device);
    size_t aligned_size = align_size(size, granularity);

    // Reserve virtual address
    CUdeviceptr va;
    CUDA_CHECK(cuMemAddressReserve(&va, aligned_size, granularity, 0, 0));
    if (error_code != CUDA_SUCCESS) {
        PyErr_SetString(PyExc_RuntimeError, error_msg);
        return nullptr;
    }

    // Track this allocation (deferred - no physical memory yet)
    AllocationInfo info = {};
    info.va = va;
    info.size = size;
    info.aligned_size = aligned_size;
    info.device = device;
    info.handle = 0;
    info.is_imported = false;

    g_allocations[va] = info;

    // Call Python callback if set
    if (g_python_malloc_callback) {
        PyGILState_STATE gstate = PyGILState_Ensure();

        PyObject* args_cb = Py_BuildValue("(KKKKi)",
            (unsigned long long)va,
            (unsigned long long)size,
            (unsigned long long)aligned_size,
            (unsigned long long)0,  // no handle yet
            1);  // is_deferred = True

        PyObject* result = PyObject_CallObject(g_python_malloc_callback, args_cb);
        Py_DECREF(args_cb);
        Py_XDECREF(result);

        if (PyErr_Occurred()) {
            PyErr_Print();
        }

        PyGILState_Release(gstate);
    }

    return Py_BuildValue("(KK)", (unsigned long long)va, (unsigned long long)aligned_size);
}

// free_va(va) - Free a VA reservation (unmaps if mapped)
static PyObject* py_free_va(PyObject* self, PyObject* args) {
    unsigned long long va;

    if (!PyArg_ParseTuple(args, "K", &va)) {
        return nullptr;
    }

    auto it = g_allocations.find((CUdeviceptr)va);
    if (it == g_allocations.end()) {
        PyErr_SetString(PyExc_ValueError, "VA not found in allocations");
        return nullptr;
    }

    AllocationInfo& info = it->second;
    ensure_context(info.device);
    error_code = CUDA_SUCCESS;

    // Call Python callback if set
    if (g_python_free_callback) {
        PyGILState_STATE gstate = PyGILState_Ensure();

        PyObject* args_cb = Py_BuildValue("(K)", (unsigned long long)va);
        PyObject* result = PyObject_CallObject(g_python_free_callback, args_cb);
        Py_DECREF(args_cb);
        Py_XDECREF(result);

        if (PyErr_Occurred()) {
            PyErr_Print();
        }

        PyGILState_Release(gstate);
    }

    // Unmap if mapped
    if (info.handle != 0) {
        CUDA_CHECK(cuMemUnmap((CUdeviceptr)va, info.aligned_size));

        // Only release if we imported the memory
        if (info.is_imported) {
            CUDA_CHECK(cuMemRelease(info.handle));
        }
    }

    // Free VA reservation
    CUDA_CHECK(cuMemAddressFree((CUdeviceptr)va, info.aligned_size));

    g_allocations.erase(it);

    Py_RETURN_NONE;
}

// ---------------------------------------------------------------------------
// Module definition

static PyMethodDef module_methods[] = {
    {"init_module", py_init_module, METH_VARARGS,
     "Initialize module with malloc/free callbacks"},
    {"set_deferred_mode", py_set_deferred_mode, METH_VARARGS,
     "Enable/disable deferred allocation mode"},
    {"get_deferred_mode", py_get_deferred_mode, METH_NOARGS,
     "Get current deferred mode state"},
    {"import_and_map", py_import_and_map, METH_VARARGS,
     "Import external FD and map to reserved VA"},
    {"unmap_imported", py_unmap_imported, METH_VARARGS,
     "Unmap imported memory (for sleep)"},
    {"get_allocation_info", py_get_allocation_info, METH_VARARGS,
     "Get allocation info for a VA"},
    {"get_granularity", py_get_granularity, METH_VARARGS,
     "Get VMM allocation granularity for device"},
    {"reserve_va", py_reserve_va, METH_VARARGS,
     "Reserve VA directly (bypasses PyTorch allocation sizing)"},
    {"free_va", py_free_va, METH_VARARGS,
     "Free a VA reservation"},
    {nullptr, nullptr, 0, nullptr}
};

static struct PyModuleDef shareable_cumem_module = {
    PyModuleDef_HEAD_INIT,
    "_shareable_cumem_ext",
    "Shareable CUDA memory allocator using VMM for cross-process tensor sharing",
    -1,
    module_methods
};

PyMODINIT_FUNC PyInit__shareable_cumem_ext(void) {
    return PyModule_Create(&shareable_cumem_module);
}

}  // extern "C"
