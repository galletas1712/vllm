"""Utilities for CUDA checkpointing and CRIU integration."""
from vllm.logger import init_logger

logger = init_logger(__name__)

# Try to import cuda-python
try:
    import cuda.bindings.driver as cuda
    cuda_available = True
except ImportError:
    logger.warning("cuda-python package not found. CUDA checkpointing will not be available. "
                   "Install with: pip install cuda-python")
    cuda = None
    cuda_available = False


def checkpoint_cuda_process(pid: int) -> None:
    """Lock and checkpoint a CUDA process using the CUDA checkpoint API."""
    if not cuda_available:
        raise RuntimeError("cuda-python package not available")
    
    logger.info(f"Locking CUDA process (PID: {pid})...")
    
    # Lock the CUDA process
    err, = cuda.cuCheckpointProcessLock(pid, None)
    if err != cuda.CUresult.CUDA_SUCCESS:
        error_name = cuda.cuGetErrorName(err)[1].decode()
        raise RuntimeError(f"Failed to lock CUDA process: {error_name}")
    logger.info("CUDA process locked")
    
    # Checkpoint the CUDA process
    logger.info(f"Checkpointing CUDA process (PID: {pid})...")
    err, = cuda.cuCheckpointProcessCheckpoint(pid, None)
    if err != cuda.CUresult.CUDA_SUCCESS:
        error_name = cuda.cuGetErrorName(err)[1].decode()
        raise RuntimeError(f"Failed to checkpoint CUDA process: {error_name}")
    logger.info("CUDA process checkpointed")


def restore_cuda_process(pid: int) -> None:
    """Restore and unlock a CUDA process using the CUDA checkpoint API.
    
    NOTE: This function should *not* be called when using CRIU, which means
    it's mostly just an artifact for debugging purposes.
    """
    if not cuda_available:
        raise RuntimeError("cuda-python package not available")
    
    logger.info(f"Restoring CUDA process (PID: {pid})...")
    
    # Restore the CUDA process from checkpoint
    err, = cuda.cuCheckpointProcessRestore(pid, None)
    if err != cuda.CUresult.CUDA_SUCCESS:
        error_name = cuda.cuGetErrorName(err)[1].decode()
        raise RuntimeError(f"Failed to restore CUDA process: {error_name}")
    logger.info("CUDA process restored")
    
    # Unlock the CUDA process
    logger.info(f"Unlocking CUDA process (PID: {pid})...")
    err, = cuda.cuCheckpointProcessUnlock(pid, None)
    if err != cuda.CUresult.CUDA_SUCCESS:
        error_name = cuda.cuGetErrorName(err)[1].decode()
        raise RuntimeError(f"Failed to unlock CUDA process: {error_name}")
    logger.info("CUDA process unlocked")


def get_cuda_process_state(pid: int) -> str:
    """Get the current state of a CUDA process."""
    if not cuda_available:
        return "CUDA not available"
    
    err, state = cuda.cuCheckpointProcessGetState(pid)
    if err != cuda.CUresult.CUDA_SUCCESS:
        error_name = cuda.cuGetErrorName(err)[1].decode()
        logger.warning(f"Failed to get CUDA process state: {error_name}")
        return "UNKNOWN"
    
    # Map state enum to string
    state_names = {
        0: "RUNNING",
        1: "LOCKED", 
        2: "CHECKPOINTED",
    }
    return state_names.get(state, f"UNKNOWN({state})")