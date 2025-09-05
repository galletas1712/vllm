"""Utilities for CUDA checkpointing and CRIU integration."""
import subprocess
import time
from pathlib import Path
from typing import Optional, Any

from vllm.logger import init_logger

logger = init_logger(__name__)

# Try to import cuda-python
try:
    from cuda import cuda
    cuda_available = True
except ImportError:
    logger.warning("cuda-python package not found. CUDA checkpointing will not be available. "
                   "Install with: pip install cuda-python")
    cuda = None
    cuda_available = False


def init_cuda_driver() -> None:
    """Initialize the CUDA Driver API."""
    if not cuda_available:
        raise RuntimeError("cuda-python package not available. Install with: pip install cuda-python")
    
    err, = cuda.cuInit(0)
    if err != cuda.CUresult.CUDA_SUCCESS:
        raise RuntimeError(f"CUDA initialization failed with error: {cuda.cuGetErrorName(err)[1].decode()}")
    logger.info("CUDA Driver API initialized for checkpointing")


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
    """Restore and unlock a CUDA process using the CUDA checkpoint API."""
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


def compute_checkpoint_hash(vllm_config) -> str:
    """Compute a hash that uniquely identifies a vLLM configuration for checkpointing."""
    # Use VllmConfig's built-in compute_hash method
    return vllm_config.compute_hash()


class CheckpointManager:
    """Manages checkpointing and restoration of vLLM workers."""
    
    def __init__(self, base_dir: str = "/tmp/vllm_checkpoints", config_hash: Optional[str] = None, dp_rank: int = 0):
        self.base_dir = Path(base_dir)
        self.base_dir.mkdir(parents=True, exist_ok=True)
        
        if config_hash:
            # Include dp_rank in checkpoint directory name
            self.checkpoint_dir = self.base_dir / f"ckpt_{config_hash}_dp{dp_rank}"
            self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        else:
            # Legacy mode without hash
            self.checkpoint_dir = self.base_dir
        
        # Initialize CUDA driver if available
        if cuda_available:
            try:
                init_cuda_driver()
            except Exception as e:
                logger.warning(f"Failed to initialize CUDA driver: {e}")
    
    def cuda_checkpoint_worker(self, worker_info: dict[str, Any], rank: int) -> None:
        """Checkpoint CUDA state for a single worker (can run concurrently)."""
        pid = worker_info["pid"]
        if not cuda_available:
            return
        try:
            checkpoint_cuda_process(pid)
            logger.info("CUDA state checkpointed for worker %s (PID %s)", rank, pid)
        except Exception as e:
            logger.warning("Failed to checkpoint CUDA state for worker %s: %s", rank, e)

    def criu_dump_all_workers(self, pids: list[int]) -> None:
        """CRIU-dump all workers."""
        dump_cmd = [
            "criu", "dump",
            "--shell-job",
            "--images-dir", str(self.checkpoint_dir),
        ]
        for pid in pids:
            dump_cmd.extend(["--tree", str(pid)])
        dump_cmd += [
            "-o", str(self.checkpoint_dir / "criu-dump.log"),
            "-v4",
            "--ext-unix-sk",
            "--tcp-established",  # Include TCP connections in dump to avoid errors
            "--skip-in-flight",  # But skip connections with in-flight data
            "--external", "mnt[/dev/shm]:shm",
            "--link-remap",
            "--force-irmap",  # Force recreation of link remaps even if they exist
            "--manage-cgroups=ignore",  # Ignore cgroups in containerized environment
        ]

        try:
            result = subprocess.run(dump_cmd,
                                    check=True,
                                    capture_output=True,
                                    text=True)
            logger.info("CRIU checkpoint successful for pids %s", pids)
            if result.stdout:
                logger.debug("CRIU stdout:\n%s", result.stdout)
                
            # # Clean up link_remap files after successful dump to avoid conflicts
            # # with subsequent worker dumps that share the same deleted files
            # try:
            #     shm_dir = Path("/dev/shm")
            #     if shm_dir.exists():
            #         for p in shm_dir.glob("link_remap.*"):
            #             p.unlink()
            #             logger.debug("Removed link_remap file: %s", p)
            # except Exception as e:
            #     logger.debug("Failed to clean link_remap files: %s", e)
                
        except subprocess.CalledProcessError as e:
            logger.error("CRIU dump failed for pids %s. See CRIU logs at %s",
                         pids, self.checkpoint_dir / 'criu-dump.log')
            if e.stdout:
                logger.error("CRIU stdout:\n%s", e.stdout)
            if e.stderr:
                logger.error("CRIU stderr:\n%s", e.stderr)
            raise
    
    def restore_worker(self, rank: int) -> int:
        """Restore a worker process from checkpoint. Returns PID."""
        worker_dir = self.checkpoint_dir / f"worker_{rank}"
        if not worker_dir.exists():
            raise FileNotFoundError(f"No checkpoint found for worker {rank}")
        
        # Restore with CRIU
        images_dir = worker_dir / "criu_images"
        work_dir = worker_dir / "criu_work"
        work_dir.mkdir(exist_ok=True)
        pidfile = images_dir / "restored.pid"
        if pidfile.exists():
            pidfile.unlink()
        
        restore_cmd = [
            "criu", "restore",
            "--shell-job",
            "--images-dir", str(images_dir),
            "--work-dir", str(work_dir),
            "-o", str(images_dir / "criu-restore.log"),
            "-v4",
            "--ext-unix-sk",  # Allow external unix sockets
            "--tcp-close",  # Close all TCP connections instead of restoring them
            "--pidfile", str(pidfile),
            # Match the dump-time external mount id for /dev/shm
            "--external", "mnt[shm]:/dev/shm",
            "--manage-cgroups=ignore",  # Ignore cgroups in containerized environment
        ]
        
        # Start CRIU but don't wait for it
        proc = subprocess.Popen(restore_cmd, stdout=subprocess.PIPE, 
                            stderr=subprocess.PIPE, text=True)
        
        # Wait for pidfile to be created (indicates successful restore)
        for _ in range(50):  # 5 second timeout
            if pidfile.exists():
                new_pid = int(pidfile.read_text().strip())
                logger.info(f"CRIU restore successful for worker {rank}, PID: {new_pid}")

                # Resume CUDA state if available
                if cuda_available:
                    try:
                        restore_cuda_process(new_pid)
                        logger.info(f"CUDA state restored for worker {rank} (PID {new_pid})")
                    except Exception as e:
                        logger.warning(f"Failed to restore CUDA state: {e}")
                
                return new_pid
            time.sleep(0.1)

        # If we get here, restore failed
        proc.terminate()
        raise RuntimeError(f"CRIU restore timeout for worker {rank}")

    def cleanup(self) -> None:
        """Clean up checkpoint directory."""
        import shutil
        if self.checkpoint_dir.exists():
            shutil.rmtree(self.checkpoint_dir)
            logger.info(f"Cleaned up checkpoint directory: {self.checkpoint_dir}")
    
    @classmethod
    def find_checkpoint(cls, base_dir: str, config_hash: str, dp_rank: int = 0) -> Optional["CheckpointManager"]:
        """Try to find an existing checkpoint matching the config hash and parallel config."""
        ckpt_dir = Path(base_dir) / f"ckpt_{config_hash}_dp{dp_rank}"
        if ckpt_dir.exists():
            return cls(base_dir=base_dir, config_hash=config_hash, dp_rank=dp_rank)
        return None
