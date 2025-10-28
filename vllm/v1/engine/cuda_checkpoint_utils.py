# SPDX-License-Identifier: Apache-2.0
# Utilities for CUDA checkpointing and CRIU integration.
from vllm.logger import init_logger

logger = init_logger(__name__)

# Try to import cuda-python
try:
    from cuda import cuda
    cuda_available = True
except Exception:  # pragma: no cover
    logger.warning("cuda-python package not found. CUDA checkpointing will not be available. "
                   "Install with: pip install cuda-python")
    cuda = None
    cuda_available = False


def checkpoint_cuda_process(pid: int) -> None:
    """Lock and checkpoint a CUDA process using the CUDA checkpoint API."""
    if not cuda_available:
        raise RuntimeError("cuda-python package not available")

    logger.info("Locking CUDA process (PID: %d)...", pid)

    # Lock the CUDA process
    err, = cuda.cuCheckpointProcessLock(pid, None)
    if err != cuda.CUresult.CUDA_SUCCESS:
        _, name_ptr = cuda.cuGetErrorName(err)
        error_name = name_ptr.decode() if isinstance(name_ptr, bytes) else str(name_ptr)
        raise RuntimeError(f"Failed to lock CUDA process: {error_name}")
    logger.info("CUDA process locked")

    # Checkpoint the CUDA process
    logger.info("Checkpointing CUDA process (PID: %d)...", pid)
    err, = cuda.cuCheckpointProcessCheckpoint(pid, None)
    if err != cuda.CUresult.CUDA_SUCCESS:
        _, name_ptr = cuda.cuGetErrorName(err)
        error_name = name_ptr.decode() if isinstance(name_ptr, bytes) else str(name_ptr)
        raise RuntimeError(f"Failed to checkpoint CUDA process: {error_name}")
    logger.info("CUDA process checkpointed")


def restore_cuda_process(pid: int) -> None:
    """Restore and unlock a CUDA process using the CUDA checkpoint API.

    NOTE: This function should not be used with CRIU restore; CRIU will
    recreate process state, then normal vLLM wake_up() repopulates memory.
    """
    if not cuda_available:
        raise RuntimeError("cuda-python package not available")

    logger.info("Restoring CUDA process (PID: %d)...", pid)

    err, = cuda.cuCheckpointProcessRestore(pid, None)
    if err != cuda.CUresult.CUDA_SUCCESS:
        _, name_ptr = cuda.cuGetErrorName(err)
        error_name = name_ptr.decode() if isinstance(name_ptr, bytes) else str(name_ptr)
        raise RuntimeError(f"Failed to restore CUDA process: {error_name}")
    logger.info("CUDA process restored")

    logger.info("Unlocking CUDA process (PID: %d)...", pid)
    err, = cuda.cuCheckpointProcessUnlock(pid, None)
    if err != cuda.CUresult.CUDA_SUCCESS:
        _, name_ptr = cuda.cuGetErrorName(err)
        error_name = name_ptr.decode() if isinstance(name_ptr, bytes) else str(name_ptr)
        raise RuntimeError(f"Failed to unlock CUDA process: {error_name}")
    logger.info("CUDA process unlocked")
