# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""General utilities for checkpoint/restore operations."""

import os
import time
from typing import Optional

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


# General checkpoint utilities

def read_comm(pid: int) -> str:
    """Best-effort read of the process command name."""
    try:
        with open(f"/proc/{pid}/comm", encoding="utf-8") as f:
            return f.read().strip()
    except Exception:
        return "?"


def read_cmdline(pid: int) -> str:
    """Best-effort read of the full command line."""
    try:
        with open(f"/proc/{pid}/cmdline", "rb") as f:
            raw = f.read()
        if not raw:
            return ""
        parts = [p.decode("utf-8", "ignore") for p in raw.split(b"\x00") if p]
        return " ".join(parts)
    except Exception:
        return ""


def collect_process_tree_pids(root_pid: int) -> set[int]:
    """Recursively collect PIDs in the process tree rooted at root_pid.

    Uses /proc/<pid>/task/<pid>/children to discover descendants.
    Best-effort: missing /proc entries are ignored.
    """
    pending: list[int] = [root_pid]
    seen: set[int] = set[int]()
    while pending:
        pid = pending.pop()
        if pid in seen:
            continue
        seen.add(pid)
        children_path = f"/proc/{pid}/task/{pid}/children"
        try:
            with open(children_path, encoding="utf-8") as f:
                content = f.read().strip()
        except FileNotFoundError:
            continue
        except Exception:
            continue
        if not content:
            continue
        for token in content.split():
            try:
                child = int(token)
            except ValueError:
                continue
            pending.append(child)
    return seen


def process_is_leaf(pid: int) -> bool:
    """Check if a process is a leaf (has no children).

    Args:
        pid: Process ID to check

    Returns:
        True if the process has no children, False otherwise
    """
    children_path = f"/proc/{pid}/task/{pid}/children"
    try:
        with open(children_path, encoding="utf-8") as f:
            content = f.read().strip()
        return not content  # Empty means no children
    except Exception:
        # If we can't read, assume it's a leaf to be safe
        return True


def verify_processes_exited(pre_dump_pids: set[int], timeout_s: float = 5.0) -> list[int]:
    """Verify that all processes from pre-dump snapshot have exited.

    Args:
        pre_dump_pids: Set of PIDs that existed before CRIU dump
        timeout_s: Maximum time to wait for processes to exit

    Returns:
        List of PIDs that are still running after timeout
    """
    deadline = time.time() + timeout_s
    lingering: set[int] = set()

    while time.time() < deadline:
        lingering = {
            pid for pid in pre_dump_pids
            if os.path.exists(f"/proc/{pid}")
        }
        if not lingering:
            break
        time.sleep(0.05)

    if lingering:
        logger.error(
            "CRIU dump verification: %d lingering PIDs: %s",
            len(lingering),
            sorted(lingering)
        )
        # Collect info about lingering processes
        for pid in sorted(lingering):
            try:
                comm = read_comm(pid)
                cmdline = read_cmdline(pid)
                logger.error("  PID %d (%s): %s", pid, comm, cmdline)
            except Exception:
                logger.error("  PID %d (unable to read info)", pid)
    else:
        logger.info("CRIU dump verification: all pre-dump PIDs have exited")

    return list(lingering)


def get_tty_info(pid: int) -> tuple[str, str]:
    """Get TTY device info for a process.

    Args:
        pid: Process ID to get TTY info for

    Returns:
        Tuple of (rdev, dev) as hex strings for CRIU
    """
    try:
        # Get the TTY device from /proc/PID/fd/0
        tty_path = f"/proc/{pid}/fd/0"
        st = os.stat(tty_path)

        # Format as hex values for CRIU
        rdev = f"{st.st_rdev:x}"
        dev = f"{st.st_dev:x}"

        return rdev, dev
    except Exception as e:
        logger.warning("Could not get TTY info: %s", e)
        return "", ""


# CUDA checkpoint utilities

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
    """Restore a CUDA process using the CUDA checkpoint API."""
    if not cuda_available:
        raise RuntimeError("cuda-python package not available")

    logger.info("Restoring CUDA process (PID: %d)...", pid)

    err, = cuda.cuCheckpointProcessRestore(pid, None)
    if err != cuda.CUresult.CUDA_SUCCESS:
        _, name_ptr = cuda.cuGetErrorName(err)
        error_name = name_ptr.decode() if isinstance(name_ptr, bytes) else str(name_ptr)
        raise RuntimeError(f"Failed to restore CUDA process: {error_name}")
    logger.info("CUDA process restored")

    # Unlock the CUDA process
    logger.info("Unlocking CUDA process (PID: %d)...", pid)

    err, = cuda.cuCheckpointProcessUnlock(pid, None)
    if err != cuda.CUresult.CUDA_SUCCESS:
        _, name_ptr = cuda.cuGetErrorName(err)
        error_name = name_ptr.decode() if isinstance(name_ptr, bytes) else str(name_ptr)
        raise RuntimeError(f"Failed to unlock CUDA process: {error_name}")
    logger.info("CUDA process unlocked")


def process_has_nvidia_fd(pid: int) -> bool:
    """Check if a process has any NVIDIA device file descriptors open.

    Args:
        pid: Process ID to check

    Returns:
        True if the process has /dev/nvidia* FDs open, False otherwise
    """
    fd_dir = f"/proc/{pid}/fd"
    if not os.path.exists(fd_dir):
        return False
    try:
        for fd in os.listdir(fd_dir):
            try:
                link = os.readlink(os.path.join(fd_dir, fd))
                if link.startswith("/dev/nvidia"):
                    return True
            except Exception:
                continue
    except Exception:
        pass
    return False


def find_gpu_worker_pids(root_pid: int) -> list[int]:
    """Find all GPU worker processes in the process tree.

    Returns PIDs of leaf processes that use GPU (workers).
    """
    all_pids = collect_process_tree_pids(root_pid)

    # Find leaf processes (no children)
    leaf_pids = []
    for pid in all_pids:
        if process_is_leaf(pid):
            leaf_pids.append(pid)

    # Filter for GPU-using processes
    gpu_pids = []
    for pid in leaf_pids:
        if process_has_nvidia_fd(pid):
            gpu_pids.append(pid)

    return gpu_pids


def validate_cuda_process_tree(root_pid: int) -> tuple[list[int], list[int]]:
    """Validate that all processes with NVIDIA fds are leaf processes.

    Args:
        root_pid: Root process ID of the tree to validate

    Returns:
        Tuple of (valid_cuda_pids, invalid_cuda_pids) where:
        - valid_cuda_pids: List of PIDs that have nvidia fds and are leaves
        - invalid_cuda_pids: List of PIDs that have nvidia fds but are NOT leaves

    Raises:
        RuntimeError: If any non-leaf process has NVIDIA file descriptors
    """
    all_pids = collect_process_tree_pids(root_pid)

    valid_cuda_pids = []
    invalid_cuda_pids = []

    for pid in all_pids:
        if process_has_nvidia_fd(pid):
            if process_is_leaf(pid):
                valid_cuda_pids.append(pid)
            else:
                invalid_cuda_pids.append(pid)

    return valid_cuda_pids, invalid_cuda_pids


def get_processes_with_nvidia_fds(pids: list[int]) -> list[int]:
    """Get list of PIDs that still have NVIDIA device file descriptors.

    Args:
        pids: List of PIDs to check

    Returns:
        List of PIDs that have /dev/nvidia* FDs open
    """
    return [pid for pid in pids if process_has_nvidia_fd(pid)]


def assert_no_nvidia_fds_in_tree(root_pid: int) -> None:
    """Assert that no processes in the tree have NVIDIA FDs after checkpoint.

    Args:
        root_pid: Root process ID of the tree to check

    Raises:
        RuntimeError: If any process still has NVIDIA FDs
    """
    all_pids = collect_process_tree_pids(root_pid)
    remaining = get_processes_with_nvidia_fds(list(all_pids))

    if remaining:
        # Collect detailed info about processes that still have FDs
        detailed_info = []
        for pid in remaining:
            try:
                comm = read_comm(pid)
                cmdline = read_cmdline(pid)
                detailed_info.append(f"PID {pid} ({comm}): {cmdline}")
            except Exception:
                detailed_info.append(f"PID {pid}")

        raise RuntimeError(
            f"CUDA checkpoint failed: {len(remaining)} process(es) still have "
            f"/dev/nvidia* FDs open.\n"
            f"Processes with NVIDIA FDs:\n" + "\n".join(detailed_info)
        )

    logger.info("Verified: No processes in tree have /dev/nvidia* FDs")


def format_cuda_checkpoint_results(succeeded: list[int],
                                  failed: list[tuple[int, str]]) -> str:
    """Format CUDA checkpoint results for logging.

    Args:
        succeeded: List of PIDs that were successfully checkpointed
        failed: List of (pid, error_message) tuples for failed checkpoints

    Returns:
        Formatted string summarizing the results
    """
    msg_parts = []

    if succeeded:
        msg_parts.append(f"CUDA checkpoint succeeded for {len(succeeded)} PIDs: {succeeded}")

    if failed:
        msg_parts.append(f"CUDA checkpoint failed for {len(failed)} PIDs:")
        for pid, error in failed:
            msg_parts.append(f"  PID {pid}: {error}")

    return "\n".join(msg_parts)


def checkpoint_cuda_processes_from_pids(cuda_pids: list[int]) -> tuple[list[int], list[tuple[int, str]]]:
    """Checkpoint CUDA processes from a list of PIDs.

    This function attempts to checkpoint each CUDA process and returns
    success/failure results.

    Args:
        cuda_pids: List of PIDs to checkpoint

    Returns:
        Tuple of (succeeded, failed) where:
        - succeeded: List of PIDs that were successfully checkpointed
        - failed: List of (pid, error_message) tuples for failed checkpoints
    """
    if not cuda_pids:
        return [], []

    if not cuda_available:
        raise RuntimeError(
            "cuda-python is not installed; install 'cuda-python' to "
            "enable CUDA API checkpointing")

    # Two-phase approach for multi-process: lock all, then checkpoint all.
    # This approximates a cohort-style checkpoint across ranks and avoids
    # capturing inconsistent cross-process CUDA/NCCL/IPC state.

    locked: list[int] = []
    lock_failed: list[tuple[int, str]] = []

    # Phase 1: Lock all CUDA processes
    for pid in cuda_pids:
        try:
            err, = cuda.cuCheckpointProcessLock(pid, None)  # type: ignore[union-attr]
            if err != cuda.CUresult.CUDA_SUCCESS:  # type: ignore[union-attr]
                _, name_ptr = cuda.cuGetErrorName(err)  # type: ignore[union-attr]
                error_name = name_ptr.decode() if isinstance(name_ptr, bytes) else str(name_ptr)
                raise RuntimeError(f"Failed to lock CUDA process: {error_name}")
            locked.append(pid)
        except Exception as e:
            error_msg = str(e)
            lock_failed.append((pid, error_msg))
            logger.debug("cuCheckpoint lock failed for PID %d: %s", pid, error_msg)

    # Phase 2: Checkpoint all locked processes
    succeeded: list[int] = []
    failed: list[tuple[int, str]] = list(lock_failed)

    for pid in locked:
        try:
            err, = cuda.cuCheckpointProcessCheckpoint(pid, None)  # type: ignore[union-attr]
            if err != cuda.CUresult.CUDA_SUCCESS:  # type: ignore[union-attr]
                _, name_ptr = cuda.cuGetErrorName(err)  # type: ignore[union-attr]
                error_name = name_ptr.decode() if isinstance(name_ptr, bytes) else str(name_ptr)
                raise RuntimeError(f"Failed to checkpoint CUDA process: {error_name}")
            succeeded.append(pid)
        except Exception as e:
            error_msg = str(e)
            failed.append((pid, error_msg))
            logger.debug("cuCheckpoint checkpoint failed for PID %d: %s", pid, error_msg)

    return succeeded, failed


def format_cuda_restore_results(succeeded: list[int],
                                failed: list[tuple[int, str]]) -> str:
    """Format CUDA restore/unlock results for logging.

    Args:
        succeeded: List of PIDs that were successfully restored/unlocked
        failed: List of (pid, error_message) tuples for failures

    Returns:
        Formatted string summarizing the results
    """
    msg_parts = []

    if succeeded:
        msg_parts.append(f"CUDA restore/unlock succeeded for {len(succeeded)} PIDs: {succeeded}")

    if failed:
        msg_parts.append(f"CUDA restore/unlock failed for {len(failed)} PIDs:")
        for pid, error in failed:
            msg_parts.append(f"  PID {pid}: {error}")

    return "\n".join(msg_parts)


def restore_cuda_processes_from_pids(cuda_pids: list[int]) -> tuple[list[int], list[tuple[int, str]]]:
    """Restore and unlock CUDA processes from a list of PIDs.

    Args:
        cuda_pids: List of PIDs to restore and unlock

    Returns:
        Tuple of (succeeded, failed) where:
        - succeeded: List of PIDs that were successfully restored and unlocked
        - failed: List of (pid, error_message) tuples for failed operations
    """
    if not cuda_pids:
        return [], []

    if not cuda_available:
        raise RuntimeError(
            "cuda-python is not installed; install 'cuda-python' to "
            "enable CUDA API restore/unlock")

    # Two-phase approach for multi-process restore: restore all, then unlock all.
    # This mirrors how we checkpoint (lock all, then checkpoint all) and helps
    # avoid per-rank inconsistencies during restore/unlock.

    restored: list[int] = []
    restore_failed: list[tuple[int, str]] = []

    # Phase 1: Restore all CUDA processes
    for pid in cuda_pids:
        try:
            err, = cuda.cuCheckpointProcessRestore(pid, None)  # type: ignore[union-attr]
            if err != cuda.CUresult.CUDA_SUCCESS:  # type: ignore[union-attr]
                _, name_ptr = cuda.cuGetErrorName(err)  # type: ignore[union-attr]
                error_name = name_ptr.decode() if isinstance(name_ptr, bytes) else str(name_ptr)
                raise RuntimeError(f"Failed to restore CUDA process: {error_name}")
            restored.append(pid)
        except Exception as e:
            error_msg = str(e)
            restore_failed.append((pid, error_msg))
            logger.debug("cuCheckpoint restore failed for PID %d: %s", pid, error_msg)

    # Phase 2: Unlock all successfully restored processes
    succeeded: list[int] = []
    failed: list[tuple[int, str]] = list(restore_failed)

    for pid in restored:
        try:
            err, = cuda.cuCheckpointProcessUnlock(pid, None)  # type: ignore[union-attr]
            if err != cuda.CUresult.CUDA_SUCCESS:  # type: ignore[union-attr]
                _, name_ptr = cuda.cuGetErrorName(err)  # type: ignore[union-attr]
                error_name = name_ptr.decode() if isinstance(name_ptr, bytes) else str(name_ptr)
                raise RuntimeError(f"Failed to unlock CUDA process: {error_name}")
            succeeded.append(pid)
        except Exception as e:
            error_msg = str(e)
            failed.append((pid, error_msg))
            logger.debug("cuCheckpoint unlock failed for PID %d: %s", pid, error_msg)

    return succeeded, failed


# CRIU helper utilities

def ensure_dummy_criu_libdir(base_dir: str, dir_name: str = "noop-criu-libdir") -> str:
    """Ensure a dummy libdir exists to prevent CRIU from loading plugins.

    Returns the path to a directory that can be passed via --libdir to CRIU
    to avoid discovering system-wide plugins such as the CUDA plugin.
    """
    path = os.path.join(base_dir, dir_name)
    try:
        os.makedirs(path, exist_ok=True)
    except Exception as e:
        logger.warning("Failed to create dummy CRIU libdir %s: %s", path, e)
    return path


def snapshot_dev_shm_files_for_tree(root_pid: int) -> list[dict]:
    """Snapshot open /dev/shm files across a process tree.

    Returns a list of dict entries with keys: name, size, mode.
    For files opened multiple times with different sizes, the largest size
    is kept.
    """
    dev_shm_files: dict[str, dict] = {}
    try:
        tree_pids = collect_process_tree_pids(root_pid)
        for pid in tree_pids:
            fd_dir = f"/proc/{pid}/fd"
            if not os.path.isdir(fd_dir):
                continue
            for fd in os.listdir(fd_dir):
                fd_path = os.path.join(fd_dir, fd)
                try:
                    link = os.readlink(fd_path)
                except Exception:
                    continue
                # Normalize deleted marker appended by the kernel
                if link.endswith(" (deleted)"):
                    link = link[:-10]
                if not link.startswith("/dev/shm/"):
                    continue
                name = os.path.basename(link)
                # Stat via fd path to obtain mode/size of the opened file
                try:
                    st = os.stat(fd_path)
                    size = int(getattr(st, "st_size", 0))
                    mode = int(getattr(st, "st_mode", 0))
                except Exception:
                    size = 0
                    mode = 0o600
                entry = dev_shm_files.get(name)
                if entry is None or size > entry.get("size", 0):
                    dev_shm_files[name] = {"name": name, "size": size, "mode": mode}
        if dev_shm_files:
            logger.info("Captured %d /dev/shm files for restore: %s",
                        len(dev_shm_files), [f["name"] for f in dev_shm_files.values()])
    except Exception as e:
        logger.warning("Failed to snapshot /dev/shm files: %s", e)
    return list(dev_shm_files.values())


def precreate_dev_shm_files(files: list[dict]) -> None:
    """Pre-create /dev/shm files described by snapshot entries.

    Each entry should contain keys: name, size, mode.
    """
    try:
        if files:
            if not os.path.isdir("/dev/shm"):
                os.makedirs("/dev/shm", exist_ok=True)
            for shm in files:
                name = shm.get("name")
                if not name:
                    continue
                path = os.path.join("/dev/shm", name)
                # Skip if already exists
                if os.path.exists(path):
                    continue
                mode = int(shm.get("mode", 0o600)) & 0o777
                size = int(shm.get("size", 0))
                try:
                    fd = os.open(path, os.O_CREAT | os.O_RDWR, mode)
                    if size > 0:
                        try:
                            os.ftruncate(fd, size)
                        except Exception:
                            pass
                    os.close(fd)
                except Exception as e:
                    logger.warning("Failed to pre-create /dev/shm/%s: %s", name, e)
    except Exception as e:
        logger.warning("Error while preparing /dev/shm files: %s", e)

