"""Utilities for CUDA checkpointing and CRIU integration."""
import subprocess
import shutil
from vllm.logger import init_logger

logger = init_logger(__name__)

# Check if cuda-checkpoint utility is available
cuda_checkpoint_available = shutil.which("cuda-checkpoint") is not None
if not cuda_checkpoint_available:
    logger.warning("cuda-checkpoint utility not found in PATH. CUDA checkpointing will not be available.")


def _run_cuda_checkpoint_cmd(args: list[str]) -> subprocess.CompletedProcess:
    """Run cuda-checkpoint command and return the result."""
    if not cuda_checkpoint_available:
        raise RuntimeError("cuda-checkpoint utility not available")
    
    cmd = ["cuda-checkpoint"] + args
    result = subprocess.run(cmd, capture_output=True, text=True)
    
    if result.returncode != 0:
        error_msg = result.stderr.strip() if result.stderr else "Unknown error"
        raise RuntimeError(f"cuda-checkpoint command failed: {error_msg}")
    
    return result


def checkpoint_cuda_process(pid: int) -> None:
    """Lock and checkpoint a CUDA process using the cuda-checkpoint utility."""
    logger.info(f"Locking CUDA process (PID: {pid})...")
    
    # Lock the CUDA process
    _run_cuda_checkpoint_cmd(["--action", "lock", "--pid", str(pid)])
    logger.info("CUDA process locked")
    
    # Checkpoint the CUDA process
    logger.info(f"Checkpointing CUDA process (PID: {pid})...")
    _run_cuda_checkpoint_cmd(["--action", "checkpoint", "--pid", str(pid)])
    logger.info("CUDA process checkpointed")


def restore_cuda_process(pid: int) -> None:
    """Restore and unlock a CUDA process using the cuda-checkpoint utility.
    
    NOTE: This function should *not* be called when using CRIU, which means
    it's mostly just an artifact for debugging purposes.
    """
    logger.info(f"Restoring CUDA process (PID: {pid})...")
    
    # Restore the CUDA process from checkpoint
    _run_cuda_checkpoint_cmd(["--action", "restore", "--pid", str(pid)])
    logger.info("CUDA process restored")
    
    # Unlock the CUDA process
    logger.info(f"Unlocking CUDA process (PID: {pid})...")
    _run_cuda_checkpoint_cmd(["--action", "unlock", "--pid", str(pid)])
    logger.info("CUDA process unlocked")


def get_cuda_process_state(pid: int) -> str:
    """Get the current state of a CUDA process."""
    if not cuda_checkpoint_available:
        return "CUDA not available"
    
    try:
        result = _run_cuda_checkpoint_cmd(["--get-state", "--pid", str(pid)])
        output = result.stdout.strip()
        
        # Parse the output to extract the state
        # The output format is typically: "CUDA checkpoint state for PID <pid>: <state>"
        if ":" in output:
            state = output.split(":")[-1].strip()
            return state
        else:
            logger.warning(f"Unexpected output format: {output}")
            return "UNKNOWN"
    except RuntimeError as e:
        logger.warning(f"Failed to get CUDA process state: {e}")
        return "UNKNOWN"