# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
import asyncio
import contextlib
import errno
import fcntl
import multiprocessing
import os
import pty
import subprocess
import sys
import termios
import threading
import time
import traceback
import uuid
from collections.abc import AsyncGenerator, Mapping
from typing import Any, Optional, Union

import cloudpickle
import msgspec
import zmq
import zmq.asyncio

from vllm.config import ModelConfig, VllmConfig
from vllm.engine.arg_utils import AsyncEngineArgs
from vllm.engine.protocol import EngineClient
from vllm.inputs import PromptType
from vllm.inputs.preprocess import InputPreprocessor
from vllm.logger import init_logger
from vllm.lora.request import LoRARequest
from vllm.outputs import PoolingRequestOutput, RequestOutput
from vllm.pooling_params import PoolingParams
from vllm.sampling_params import SamplingParams
from vllm.tasks import SupportedTask
from vllm.transformers_utils.tokenizer import AnyTokenizer
from vllm.usage.usage_lib import UsageContext
from vllm.utils import Device, get_open_port
from vllm.v1.engine.async_llm import AsyncLLM
from vllm.v1.engine.output_processor import RequestOutputCollector
from vllm.v1.executor.abstract import Executor
from vllm.v1.metrics.loggers import StatLoggerFactory
from vllm.v1.engine.cuda_checkpoint_utils import (
    checkpoint_cuda_process, cuda_available,
)

logger = init_logger(__name__)

# RPC message types
class RPCMessageType(msgspec.Struct):
    request_id: str
    method: str
    args_pickle: bytes = b''  # Cloudpickle serialized args
    kwargs_pickle: bytes = b''  # Cloudpickle serialized kwargs
    is_generator: bool = False

class RPCResponse(msgspec.Struct):
    request_id: str
    result_pickle: bytes = b''  # Cloudpickle serialized result
    error: Optional[str] = None
    is_generator_item: bool = False
    generator_done: bool = False

class PropertyRequest(msgspec.Struct):
    request_id: str
    property_name: str

class PropertyResponse(msgspec.Struct):
    request_id: str
    value_pickle: bytes = b''  # Cloudpickle serialized value
    error: Optional[str] = None


def _get_tty_info(pid: int) -> tuple[str, str]:
    """Get TTY device info for a process."""
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


def _save_tty_id(checkpoint_dir: str, rdev: str, dev: str) -> None:
    """Persist the external TTY id used during dump for reuse on restore."""
    try:
        path = os.path.join(checkpoint_dir, "criu_external_tty_id.txt")
        with open(path, "w", encoding="utf-8") as f:
            f.write(f"tty[{rdev}:{dev}]\n")
    except Exception as e:
        logger.warning("Failed to save TTY id for CRIU restore: %s", e)


def _load_tty_id(checkpoint_dir: str) -> Optional[str]:
    """Load the external TTY id saved during dump, if present."""
    try:
        path = os.path.join(checkpoint_dir, "criu_external_tty_id.txt")
        if not os.path.exists(path):
            return None
        with open(path, encoding="utf-8") as f:
            value = f.read().strip()
            return value or None
    except Exception as e:
        logger.warning("Failed to load TTY id for CRIU restore: %s", e)
        return None


def _save_tree_pid(checkpoint_dir: str, pid: int) -> None:
    """Persist the original tree pid to wait for its exit on restore."""
    try:
        path = os.path.join(checkpoint_dir, "criu_tree_pid.txt")
        with open(path, "w", encoding="utf-8") as f:
            f.write(str(pid))
    except Exception as e:
        logger.warning("Failed to save tree pid for CRIU restore: %s", e)


def _load_tree_pid(checkpoint_dir: str) -> Optional[int]:
    """Load the original tree pid if it was persisted."""
    try:
        path = os.path.join(checkpoint_dir, "criu_tree_pid.txt")
        if not os.path.exists(path):
            return None
        with open(path, encoding="utf-8") as f:
            data = f.read().strip()
        return int(data) if data else None
    except Exception as e:
        logger.warning("Failed to load tree pid for CRIU restore: %s", e)
        return None


def _collect_process_tree_pids(root_pid: int) -> set[int]:
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


def _process_has_nvidia_fd(pid: int) -> bool:
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


def _wait_until_no_nvidia_fds(root_pid: int,
                              timeout_s: float = 10.0,
                              poll_interval_s: float = 0.05) -> list[int]:
    """Poll the process tree until no /dev/nvidia* FDs remain.

    Returns list of PIDs that still have NVIDIA FDs open on timeout.
    """
    deadline = time.time() + timeout_s
    remaining: list[int] = []
    while time.time() < deadline:
        pids = _collect_process_tree_pids(root_pid)
        remaining = [pid for pid in pids if _process_has_nvidia_fd(pid)]
        if not remaining:
            return []
        time.sleep(poll_interval_s)
    return remaining


def _find_gpu_worker_pids(root_pid: int) -> list[int]:
    """Find all GPU worker processes in the process tree.

    Returns PIDs of leaf processes that use GPU (workers).
    """
    all_pids = _collect_process_tree_pids(root_pid)

    # Find leaf processes (no children)
    leaf_pids = []
    for pid in all_pids:
        children_path = f"/proc/{pid}/task/{pid}/children"
        try:
            with open(children_path, encoding="utf-8") as f:
                content = f.read().strip()
            if not content:  # No children = leaf process
                leaf_pids.append(pid)
        except Exception:
            continue

    # Filter for GPU-using processes
    gpu_pids = []
    for pid in leaf_pids:
        if _process_has_nvidia_fd(pid):
            gpu_pids.append(pid)

    return gpu_pids


def _run_cuda_checkpoint(
        pids: list[int], action: str,
        cuda_checkpoint_path: str = "cuda-checkpoint") -> None:
    """Run cuda-checkpoint utility on specified PIDs.

    Args:
        pids: List of process PIDs to checkpoint
        action: Either "lock", "checkpoint", "restore", or "unlock"
        cuda_checkpoint_path: Path to cuda-checkpoint utility
    """
    if not pids:
        logger.warning("No PIDs provided for cuda-checkpoint")
        return

    # Check if cuda-checkpoint is available in PATH
    import shutil
    if not shutil.which(cuda_checkpoint_path):
        raise RuntimeError(
            f"cuda-checkpoint utility not found: '{cuda_checkpoint_path}'. "
            "Please ensure cuda-checkpoint is installed and in your PATH."
        )

    # Build command for each PID
    # The cuda-checkpoint utility uses --action <action> --pid <pid> syntax
    for pid in pids:
        cmd = [cuda_checkpoint_path, "--action", action, "--pid", str(pid)]

        logger.info("Running cuda-checkpoint: %s", ' '.join(cmd))

        try:
            result = subprocess.run(cmd, capture_output=True, text=True)
            if result.returncode != 0:
                logger.error("cuda-checkpoint %s failed for PID %d: %s",
                            action, pid, result.stderr)
                raise RuntimeError(
                    f"cuda-checkpoint {action} failed for PID {pid}: "
                    f"{result.stderr}")
            logger.info("cuda-checkpoint %s succeeded for PID %d", action, pid)
        except Exception as e:
            logger.error("Failed to run cuda-checkpoint for PID %d: %s", pid, e)
            raise


class AsyncLLMServer:
    """Server process that runs AsyncLLM and handles RPC calls."""

    def __init__(self, socket_url: str, vllm_config: VllmConfig,
                 executor_class: type[Executor], **kwargs):
        self.socket_url = socket_url
        self.vllm_config = vllm_config
        self.executor_class = executor_class
        self.kwargs = kwargs
        self.async_llm: Optional[AsyncLLM] = None
        self.running = True
        self.engine_ready = False

        # For handling async generators
        self.active_generators: dict[str, AsyncGenerator] = {}

    async def initialize(self):
        """Initialize the AsyncLLM instance and wait for engine to be ready."""
        logger.info("Initializing AsyncLLM...")
        self.async_llm = AsyncLLM(
            vllm_config=self.vllm_config,
            executor_class=self.executor_class,
            **self.kwargs
        )

        # The AsyncLLM constructor returns after starting the engine,
        # but the engine may still be initializing (loading model,
        # compiling, etc.)
        # Try to verify the engine is ready by calling a simple method
        max_wait_time = 900  # 15 minutes
        start_time = time.time()
        last_error = None

        while time.time() - start_time < max_wait_time:
            try:
                # Try to get model config - this will succeed when
                # engine is ready
                await self.async_llm.get_model_config()
                self.engine_ready = True
                logger.info("AsyncLLM engine is fully initialized and ready")
                return
            except Exception as e:
                last_error = e
                # Check if the engine is dead
                if (hasattr(self.async_llm, 'errored') and
                        self.async_llm.errored):
                    raise RuntimeError(
                        f"AsyncLLM engine failed during initialization: {e}"
                    ) from e
                # Engine not ready yet, wait a bit
                await asyncio.sleep(0.5)
                elapsed = time.time() - start_time
                if int(elapsed) % 10 == 0 and int(elapsed) > 0:
                    logger.info("Waiting for AsyncLLM engine to initialize... "
                               "(%d seconds elapsed)", int(elapsed))

        raise RuntimeError(
            f"AsyncLLM engine failed to initialize within {max_wait_time} "
            f"seconds. Last error: {last_error}")

    async def is_engine_ready(self) -> bool:
        """Check if the AsyncLLM engine is fully initialized and ready."""
        return self.engine_ready and self.async_llm is not None

    async def handle_rpc_request(self, msg: RPCMessageType) -> Any:
        """Handle an RPC request and return the result."""
        if self.async_llm is None:
            raise RuntimeError("AsyncLLM not initialized")

        # Handle special server-side methods
        if msg.method == "is_engine_ready":
            return await self.is_engine_ready()

        method = getattr(self.async_llm, msg.method)

        # Deserialize args and kwargs
        args = cloudpickle.loads(msg.args_pickle) if msg.args_pickle else ()
        kwargs = (cloudpickle.loads(msg.kwargs_pickle)
                  if msg.kwargs_pickle else {})

        if msg.is_generator:
            # For generator methods, we store the generator and
            # return items one by one
            generator = method(*args, **kwargs)
            self.active_generators[msg.request_id] = generator
            return None  # Initial response for generator
        else:
            # Regular method call
            result = method(*args, **kwargs)
            # Handle both sync and async methods
            if asyncio.iscoroutine(result):
                result = await result
            return result

    async def handle_generator_next(self,
                                     request_id: str) -> tuple[Any, bool]:
        """Get the next item from a generator."""
        if request_id not in self.active_generators:
            raise RuntimeError(
                f"No active generator for request {request_id}")

        generator = self.active_generators[request_id]
        try:
            item = await generator.__anext__()
            return item, False  # not done
        except StopAsyncIteration:
            del self.active_generators[request_id]
            return None, True  # done

    async def handle_property_request(self, msg: PropertyRequest) -> Any:
        """Handle a property access request."""
        if self.async_llm is None:
            raise RuntimeError("AsyncLLM not initialized")

        return getattr(self.async_llm, msg.property_name)

    async def run(self):
        """Main server loop."""
        ctx = zmq.asyncio.Context()
        socket = ctx.socket(zmq.DEALER)
        socket.bind(self.socket_url)

        logger.info("AsyncLLM server listening on %s", self.socket_url)

        # Wait for initial HELLO from client
        logger.debug("Waiting for HELLO message...")
        hello_msg = await socket.recv()
        logger.debug("Received message: %s", hello_msg)
        if hello_msg != b"HELLO":
            logger.warning("Expected HELLO, got %s", hello_msg)

        # Initialize AsyncLLM
        logger.debug("Initializing AsyncLLM...")
        await self.initialize()
        logger.debug("AsyncLLM initialized")

        # Send ready signal
        logger.debug("Sending READY signal...")
        await socket.send(b"READY")
        logger.debug("READY signal sent")

        try:
            while self.running:
                # Receive message
                frames = await socket.recv_multipart()
                msg_type = frames[0].decode()

                if msg_type == "RPC":
                    msg = msgspec.msgpack.decode(frames[1], type=RPCMessageType)
                    try:
                        if msg.method == "_generator_next":
                            # Special case for getting next item from generator
                            # Deserialize the request_id from args_pickle
                            request_id = cloudpickle.loads(msg.args_pickle)[0]
                            result, done = await self.handle_generator_next(
                                request_id)
                            response = RPCResponse(
                                request_id=msg.request_id,
                                result_pickle=(cloudpickle.dumps(result)
                                              if not done else b''),
                                is_generator_item=True,
                                generator_done=done
                            )
                        else:
                            result = await self.handle_rpc_request(msg)
                            response = RPCResponse(
                                request_id=msg.request_id,
                                result_pickle=cloudpickle.dumps(result)
                            )
                    except Exception as e:
                        response = RPCResponse(
                            request_id=msg.request_id,
                            error=(f"{type(e).__name__}: {str(e)}\n"
                                   f"{traceback.format_exc()}")
                        )
                    await socket.send_multipart(
                        [b"RPC_RESPONSE", msgspec.msgpack.encode(response)])

                elif msg_type == "PROPERTY":
                    msg = msgspec.msgpack.decode(
                        frames[1], type=PropertyRequest)
                    try:
                        value = await self.handle_property_request(msg)
                        response = PropertyResponse(
                            request_id=msg.request_id,
                            value_pickle=cloudpickle.dumps(value)
                        )
                    except Exception as e:
                        response = PropertyResponse(
                            request_id=msg.request_id,
                            error=f"{type(e).__name__}: {str(e)}"
                        )
                    await socket.send_multipart(
                        [b"PROPERTY_RESPONSE",
                         msgspec.msgpack.encode(response)])

                elif msg_type == "SHUTDOWN":
                    logger.info("Received shutdown signal")
                    break

        finally:
            logger.info("AsyncLLM server shutting down...")
            if self.async_llm:
                logger.info("Shutting down AsyncLLM instance...")
                self.async_llm.shutdown()
                logger.info("AsyncLLM instance shutdown complete")
            socket.close()
            ctx.term()
            logger.info("AsyncLLM server shutdown complete")


def run_async_llm_server(socket_url: str, vllm_config_pickle: bytes,
                        executor_class_pickle: bytes,
                        kwargs_pickle: bytes,
                        parent_pid_for_tty: Optional[int] = None,
                        parent_tty_slave_fd: Optional[int] = None):
    """Entry point for the subprocess running AsyncLLM."""
    # Set up a private controlling TTY if provided by the parent.
    if parent_pid_for_tty is not None and parent_tty_slave_fd is not None:
        try:
            fd_path = f"/proc/{parent_pid_for_tty}/fd/{parent_tty_slave_fd}"
            tty_fd = os.open(fd_path, os.O_RDWR | os.O_NOCTTY)
            with contextlib.suppress(Exception):
                os.setsid()
            tiocsctty = getattr(termios, 'TIOCSCTTY', 0x540E)
            with contextlib.suppress(Exception):
                fcntl.ioctl(tty_fd, tiocsctty, 0)
            with contextlib.suppress(Exception):
                os.dup2(tty_fd, 0)
                os.dup2(tty_fd, 1)
                os.dup2(tty_fd, 2)
        except Exception as e:
            # Fall back to default stdio if anything fails; logs still work.
            print(f"Failed to set controlling TTY: {e}", file=sys.stderr)
        finally:
            with contextlib.suppress(Exception):
                os.close(tty_fd)  # type: ignore[name-defined]

    # Ensure logger is initialized in subprocess AFTER stdio is set
    import logging
    logging.basicConfig(
        level=logging.DEBUG,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )

    try:
        print(f"AsyncLLM server starting on {socket_url}", file=sys.stderr)

        # Deserialize the configuration
        vllm_config = cloudpickle.loads(vllm_config_pickle)
        executor_class = cloudpickle.loads(executor_class_pickle)
        kwargs = cloudpickle.loads(kwargs_pickle)

        print(f"Model: {vllm_config.model_config.model}", file=sys.stderr)

        # Create and run the server
        server = AsyncLLMServer(
            socket_url,
            vllm_config,
            executor_class,
            **kwargs,
        )

        # Run the async event loop
        asyncio.run(server.run())
    except Exception as e:
        print(f"AsyncLLM server failed: {e}", file=sys.stderr)
        traceback.print_exc(file=sys.stderr)
        sys.exit(1)


class CheckpointableAsyncLLM(EngineClient):
    """
    A wrapper around AsyncLLM that runs it in a subprocess and
    communicates via ZMQ. Supports CRIU checkpoint/restore functionality.

    This class is designed for checkpointing workflows:
    1. Start the AsyncLLM subprocess
    2. Checkpoint the process using CRIU
    3. Restore from checkpoint later

    Example usage:
        # Start and checkpoint
        engine_args = AsyncEngineArgs(model="...")
        llm = CheckpointableAsyncLLM.from_engine_args(engine_args)
        await llm.criu_checkpoint("/path/to/checkpoint/dir")

        # Restore from existing checkpoint
        llm = CheckpointableAsyncLLM.from_engine_args(engine_args,
                                                       auto_start=False)
        llm.checkpoint_dir = "/path/to/existing/checkpoint"
        await llm.criu_resume()

        # Now use normally
        async for output in llm.generate(...):
            print(output)
    """

    def __init__(
        self,
        vllm_config: VllmConfig,
        executor_class: type[Executor],
        log_stats: bool,
        usage_context: UsageContext = UsageContext.ENGINE_CONTEXT,
        log_requests: bool = True,
        start_engine_loop: bool = True,
        stat_loggers: Optional[list[StatLoggerFactory]] = None,
        client_addresses: Optional[dict[str, str]] = None,
        client_count: int = 1,
        client_index: int = 0,
        auto_start: bool = True,
    ) -> None:
        self.vllm_config = vllm_config
        # Use TCP socket instead of IPC for better CRIU compatibility
        self.port = get_open_port()
        self.socket_url = f"tcp://127.0.0.1:{self.port}"
        self.process: Optional[multiprocessing.Process] = None
        self.ctx = zmq.asyncio.Context()
        self.socket: Optional[zmq.asyncio.Socket] = None
        self.checkpoint_dir: Optional[str] = None
        self._is_running = False
        self._subprocess_started = False
        # For CRIU TTY forwarding
        self._pty_master_fd: Optional[int] = None
        self._pty_forwarder_thread: Optional[threading.Thread] = None
        # Track restored process PIDs for cleanup
        self._restored_pids: set[int] = set()

        # Serialize configuration for subprocess
        self.vllm_config_pickle = cloudpickle.dumps(vllm_config)
        self.executor_class_pickle = cloudpickle.dumps(executor_class)
        kwargs = {
            'log_stats': log_stats,
            'usage_context': usage_context,
            'log_requests': log_requests,
            'start_engine_loop': start_engine_loop,
            'stat_loggers': stat_loggers,
            'client_addresses': client_addresses,
            'client_count': client_count,
            'client_index': client_index,
        }
        self.kwargs_pickle = cloudpickle.dumps(kwargs)

        # Only start the subprocess if auto_start is True
        # This allows creating the instance without starting a subprocess
        # when we plan to restore from CRIU
        if auto_start:
            self._start_subprocess()

    def _start_subprocess(self):
        """Start the AsyncLLM subprocess."""
        if self._subprocess_started:
            logger.warning("Subprocess already started")
            return

        ctx = multiprocessing.get_context('spawn')
        # Note: daemon=False is required because AsyncLLM may need to spawn
        # its own child processes (e.g., DPCoordinator for data parallel)
        # Create a private PTY for the child so it can become a session leader
        # with a controlling TTY. We pass the parent's PID and the slave fd
        # number so the child can open it via /proc and call TIOCSCTTY.
        pty_master_fd, pty_slave_fd = pty.openpty()
        os.set_inheritable(pty_slave_fd, True)
        self.process = ctx.Process(
            target=run_async_llm_server,
            args=(self.socket_url, self.vllm_config_pickle,
                  self.executor_class_pickle, self.kwargs_pickle,
                  os.getpid(), pty_slave_fd),
            daemon=False
        )
        self.process.start()
        self._is_running = True
        self._subprocess_started = True

        # Close the slave fd in parent process (child has it)
        os.close(pty_slave_fd)

        # Forward child's PTY master to our stdout so logs are visible
        # Do this BEFORE connecting so we can see any startup errors
        try:
            self._start_pty_forwarder(pty_master_fd)
        except Exception as e:
            logger.warning("Failed to start startup PTY forwarder: %s", e)

        # Connect to the subprocess
        self._connect()

    def _connect(self):
        """Connect to the AsyncLLM subprocess."""
        # Use a synchronous socket for initial connection
        sync_ctx = zmq.Context()
        sync_socket = sync_ctx.socket(zmq.DEALER)
        sync_socket.connect(self.socket_url)

        try:
            # Send initial hello to establish connection
            sync_socket.send(b"HELLO")
            logger.debug("Sent HELLO to subprocess")

            # Wait for ready signal
            start_time = time.time()
            timeout = 120  # 2 minutes for initial connection
            while time.time() - start_time < timeout:
                if sync_socket.poll(timeout=1000):  # 1 second timeout
                    msg = sync_socket.recv()
                    logger.debug("Received message: %s", msg)
                    if msg == b"READY":
                        logger.info("Connected to AsyncLLM subprocess")
                        # Now create the async socket for normal operations
                        self.socket = self.ctx.socket(zmq.DEALER)
                        self.socket.connect(self.socket_url)
                        return
                # Check if process is still alive
                if self.process and not self.process.is_alive():
                    # Try to get exit code for better error info
                    exit_code = self.process.exitcode
                    raise RuntimeError(
                        f"AsyncLLM subprocess died during startup "
                        f"(exit code: {exit_code})")
                # Log progress
                elapsed = time.time() - start_time
                if int(elapsed) % 10 == 0 and int(elapsed) > 0:
                    logger.info("Still waiting for AsyncLLM to initialize... "
                                "(%d seconds elapsed)", int(elapsed))
            raise TimeoutError(
                f"Timeout waiting for AsyncLLM subprocess to start "
                f"after {timeout} seconds")
        finally:
            sync_socket.close()
            sync_ctx.term()

    async def start(self) -> None:
        """Explicitly start the AsyncLLM subprocess.

        This is used when auto_start=False was passed to __init__.
        """
        if self._subprocess_started:
            logger.info("Subprocess already started")
            return

        self._start_subprocess()
        logger.info("AsyncLLM subprocess started successfully")

    async def wait_until_ready(self, timeout: float = 300.0) -> None:
        """Wait until the AsyncLLM engine is fully initialized and ready.

        This method ensures the engine has completed all initialization steps:
        - Model loaded
        - torch.compile completed (if applicable)
        - CUDA graphs captured
        - KV cache allocated
        - Engine ready to handle requests

        Args:
            timeout: Maximum time to wait in seconds
                    (default: 300.0 = 5 minutes). Set higher for large models
                    or when torch.compile is enabled.

        Raises:
            TimeoutError: If engine doesn't become ready within timeout
            RuntimeError: If engine fails during initialization
        """
        if not self._subprocess_started:
            raise RuntimeError(
                "AsyncLLM subprocess not started. Call start() first.")

        start_time = time.time()
        last_error = None

        logger.info("Waiting for AsyncLLM engine to be fully initialized...")

        while time.time() - start_time < timeout:
            try:
                # Check if the engine is fully ready
                is_ready = await self._rpc_call("is_engine_ready")
                if is_ready:
                    logger.info(
                        "AsyncLLM engine is fully initialized and ready")
                    return
            except Exception as e:
                last_error = e
                # Check if process is still alive
                if self.process and not self.process.is_alive():
                    raise RuntimeError(
                        f"AsyncLLM subprocess died during initialization: "
                        f"{last_error}"
                    ) from e

            # Engine not ready yet, wait a bit before retrying
            await asyncio.sleep(0.5)

            # Log progress
            elapsed = time.time() - start_time
            if int(elapsed) % 10 == 0 and int(elapsed) > 0:
                logger.info("Still waiting for engine initialization... "
                           "(%d seconds elapsed)", int(elapsed))

        raise TimeoutError(
            f"AsyncLLM engine did not become ready within {timeout} seconds. "
            f"Last error: {last_error}"
        )

    async def _rpc_call(self, method: str, *args, **kwargs) -> Any:
        """Make an RPC call to the AsyncLLM subprocess."""
        if not self._subprocess_started:
            raise RuntimeError(
                "AsyncLLM subprocess not started. "
                "Call start() first or use auto_start=True")
        if not self.socket:
            raise RuntimeError(
                "Not connected to AsyncLLM subprocess")

        request_id = str(uuid.uuid4())
        msg = RPCMessageType(
            request_id=request_id,
            method=method,
            args_pickle=cloudpickle.dumps(args),
            kwargs_pickle=cloudpickle.dumps(kwargs),
            is_generator=False
        )

        # Send request
        await self.socket.send_multipart(
            [b"RPC", msgspec.msgpack.encode(msg)])

        # Wait for response
        frames = await self.socket.recv_multipart()
        if frames[0] != b"RPC_RESPONSE":
            raise RuntimeError(f"Unexpected response type: {frames[0]}")

        response = msgspec.msgpack.decode(frames[1], type=RPCResponse)
        if response.error:
            raise RuntimeError(f"RPC error: {response.error}")

        return cloudpickle.loads(response.result_pickle)

    async def _rpc_generator(self, method: str, *args,
                             **kwargs) -> AsyncGenerator:
        """Make an RPC call that returns an async generator."""
        if not self._subprocess_started:
            raise RuntimeError(
                "AsyncLLM subprocess not started. "
                "Call start() first or use auto_start=True")
        if not self.socket:
            raise RuntimeError(
                "Not connected to AsyncLLM subprocess")

        request_id = str(uuid.uuid4())

        # Initial call to start the generator
        msg = RPCMessageType(
            request_id=request_id,
            method=method,
            args_pickle=cloudpickle.dumps(args),
            kwargs_pickle=cloudpickle.dumps(kwargs),
            is_generator=True
        )

        await self.socket.send_multipart(
            [b"RPC", msgspec.msgpack.encode(msg)])

        # Get initial response
        frames = await self.socket.recv_multipart()
        if frames[0] != b"RPC_RESPONSE":
            raise RuntimeError(f"Unexpected response type: {frames[0]}")

        response = msgspec.msgpack.decode(frames[1], type=RPCResponse)
        if response.error:
            raise RuntimeError(f"RPC error: {response.error}")

        # Now iterate through generator items
        while True:
            # Request next item
            next_msg = RPCMessageType(
                request_id=str(uuid.uuid4()),
                method="_generator_next",
                args_pickle=cloudpickle.dumps((request_id,))
            )

            await self.socket.send_multipart(
                [b"RPC", msgspec.msgpack.encode(next_msg)])

            frames = await self.socket.recv_multipart()
            if frames[0] != b"RPC_RESPONSE":
                raise RuntimeError(f"Unexpected response type: {frames[0]}")

            response = msgspec.msgpack.decode(frames[1], type=RPCResponse)
            if response.error:
                raise RuntimeError(f"RPC error: {response.error}")

            if response.generator_done:
                break

            yield cloudpickle.loads(response.result_pickle)

    async def _get_property(self, property_name: str) -> Any:
        """Get a property value from the AsyncLLM subprocess."""
        if not self._subprocess_started:
            raise RuntimeError(
                "AsyncLLM subprocess not started. "
                "Call start() first or use auto_start=True")
        if not self.socket:
            raise RuntimeError(
                "Not connected to AsyncLLM subprocess")

        request_id = str(uuid.uuid4())
        msg = PropertyRequest(request_id=request_id,
                              property_name=property_name)

        await self.socket.send_multipart(
            [b"PROPERTY", msgspec.msgpack.encode(msg)])

        frames = await self.socket.recv_multipart()
        if frames[0] != b"PROPERTY_RESPONSE":
            raise RuntimeError(f"Unexpected response type: {frames[0]}")

        response = msgspec.msgpack.decode(frames[1], type=PropertyResponse)
        if response.error:
            raise RuntimeError(f"Property error: {response.error}")

        return cloudpickle.loads(response.value_pickle)

    # Implement all AsyncLLM methods
    async def get_supported_tasks(self) -> tuple[SupportedTask, ...]:
        return await self._rpc_call("get_supported_tasks")

    async def add_request(
        self,
        request_id: str,
        prompt: PromptType,
        params: Union[SamplingParams, PoolingParams],
        arrival_time: Optional[float] = None,
        lora_request: Optional[LoRARequest] = None,
        tokenization_kwargs: Optional[dict[str, Any]] = None,
        trace_headers: Optional[Mapping[str, str]] = None,
        priority: int = 0,
        data_parallel_rank: Optional[int] = None,
    ) -> RequestOutputCollector:
        return await self._rpc_call(
            "add_request",
            request_id,
            prompt,
            params,
            arrival_time,
            lora_request,
            tokenization_kwargs,
            trace_headers,
            priority,
            data_parallel_rank
        )

    async def generate(
        self,
        prompt: PromptType,
        sampling_params: SamplingParams,
        request_id: str,
        lora_request: Optional[LoRARequest] = None,
        trace_headers: Optional[Mapping[str, str]] = None,
        priority: int = 0,
        data_parallel_rank: Optional[int] = None,
    ) -> AsyncGenerator[RequestOutput, None]:
        async for output in self._rpc_generator(
            "generate",
            prompt,
            sampling_params,
            request_id,
            lora_request,
            trace_headers,
            priority,
            data_parallel_rank
        ):
            yield output

    async def abort(self, request_id: Union[str, list[str]]) -> None:
        await self._rpc_call("abort", request_id)

    async def encode(
        self,
        prompt: PromptType,
        pooling_params: PoolingParams,
        request_id: str,
        lora_request: Optional[LoRARequest] = None,
        trace_headers: Optional[Mapping[str, str]] = None,
        priority: int = 0,
        truncate_prompt_tokens: Optional[int] = None,
        tokenization_kwargs: Optional[dict[str, Any]] = None,
    ) -> AsyncGenerator[PoolingRequestOutput, None]:
        async for output in self._rpc_generator(
            "encode",
            prompt,
            pooling_params,
            request_id,
            lora_request,
            trace_headers,
            priority,
            truncate_prompt_tokens,
            tokenization_kwargs
        ):
            yield output

    async def get_vllm_config(self) -> VllmConfig:
        return await self._rpc_call("get_vllm_config")

    async def get_model_config(self) -> ModelConfig:
        return await self._rpc_call("get_model_config")

    async def get_input_preprocessor(self) -> InputPreprocessor:
        return await self._rpc_call("get_input_preprocessor")

    async def get_tokenizer(self) -> AnyTokenizer:
        return await self._rpc_call("get_tokenizer")

    async def is_tracing_enabled(self) -> bool:
        return await self._rpc_call("is_tracing_enabled")

    async def do_log_stats(self) -> None:
        await self._rpc_call("do_log_stats")

    async def check_health(self) -> None:
        await self._rpc_call("check_health")

    async def start_profile(self) -> None:
        await self._rpc_call("start_profile")

    async def stop_profile(self) -> None:
        await self._rpc_call("stop_profile")

    async def reset_mm_cache(self) -> None:
        await self._rpc_call("reset_mm_cache")

    async def reset_prefix_cache(self,
                                 device: Optional[Device] = None) -> None:
        await self._rpc_call("reset_prefix_cache", device)

    async def sleep(self, level: int = 1) -> None:
        await self._rpc_call("sleep", level)

    async def wake_up(self, tags: Optional[list[str]] = None) -> None:
        await self._rpc_call("wake_up", tags)

    async def is_sleeping(self) -> bool:
        return await self._rpc_call("is_sleeping")

    async def add_lora(self, lora_request: LoRARequest) -> bool:
        return await self._rpc_call("add_lora", lora_request)

    async def remove_lora(self, lora_id: int) -> bool:
        return await self._rpc_call("remove_lora", lora_id)

    async def list_loras(self) -> set[int]:
        return await self._rpc_call("list_loras")

    async def pin_lora(self, lora_id: int) -> bool:
        return await self._rpc_call("pin_lora", lora_id)

    async def collective_rpc(
        self,
        method: str,
        timeout: Optional[float] = None,
        args: tuple = (),
        kwargs: Optional[dict] = None
    ):
        return await self._rpc_call(
            "collective_rpc", method, timeout, args, kwargs)

    async def wait_for_requests_to_drain(self, drain_timeout: int = 300):
        await self._rpc_call("wait_for_requests_to_drain", drain_timeout)

    async def scale_elastic_ep(self, new_data_parallel_size: int,
                               drain_timeout: int = 300):
        await self._rpc_call(
            "scale_elastic_ep", new_data_parallel_size, drain_timeout)

    # Properties
    @property
    def is_running(self) -> bool:
        return (self._is_running and
                (self.process is not None and self.process.is_alive()))

    @property
    def is_stopped(self) -> bool:
        return not self.is_running

    @property
    async def errored(self) -> bool:
        return await self._get_property("errored")

    @property
    async def dead_error(self):
        return await self._get_property("dead_error")

    # CRIU-specific methods
    async def criu_checkpoint(
            self, checkpoint_dir: str,
            cuda_checkpoint_path: str = "cuda-checkpoint") -> None:
        """Checkpoint the AsyncLLM subprocess using CRIU.

        Args:
            checkpoint_dir: Directory to save checkpoint files
            cuda_checkpoint_path: Unused; kept for backward compatibility
        """
        if not self._subprocess_started:
            raise RuntimeError(
                "AsyncLLM subprocess not started. "
                "Call start() first")
        if not self.process or not self.process.is_alive():
            raise RuntimeError("AsyncLLM subprocess is not running")

        self.checkpoint_dir = checkpoint_dir
        try:
            os.makedirs(checkpoint_dir, exist_ok=False)
        except FileExistsError as err:
            raise RuntimeError(
                "Checkpoint directory "
                f"{checkpoint_dir} already exists. "
                "Please delete it before checkpointing."
            ) from err

        # Sleep the model (level 1) to free GPU memory before checkpointing
        logger.info("Putting model to sleep (level 1) before checkpoint...")
        await self.sleep(level=1)
        logger.info("Model sleep completed")

        # Root of the subprocess tree
        root_pid = self.process.pid

        # Attempt CUDA checkpoint on all PIDs in the process tree. This will
        # cover workers and any parent process that might hold CUDA contexts.
        all_pids = sorted(_collect_process_tree_pids(root_pid))
        logger.info("CUDA checkpoint: attempting cuCheckpoint on PIDs: %s",
                    all_pids)
        if not cuda_available:
            raise RuntimeError(
                "cuda-python is not installed; install 'cuda-python' to "
                "enable CUDA API checkpointing")
        succeeded: list[int] = []
        failed: list[tuple[int, str]] = []
        for pid in all_pids:
            try:
                checkpoint_cuda_process(pid)
                succeeded.append(pid)
            except Exception as e:
                failed.append((pid, str(e)))
                logger.debug("cuCheckpoint failed for PID %d: %s", pid, e)

        logger.info("CUDA checkpoint succeeded for PIDs: %s", succeeded)
        if failed:
            logger.warning(
                "CUDA checkpoint skipped/failed for PIDs (likely no CUDA context): %s",
                failed)

        # Verify device FDs for diagnostics (they may remain open and be
        # handled by CRIU's CUDA plugin; this is informational only)
        remaining_nvidia_pids = [pid for pid in succeeded
                                 if _process_has_nvidia_fd(pid)]
        if remaining_nvidia_pids:
            logger.warning(
                "After CUDA API checkpoint, these PIDs still have /dev/nvidia* "
                "FDs open: %s. This can be normal; CRIU's CUDA plugin handles "
                "device FDs.",
                remaining_nvidia_pids)

        # Wait until all NVIDIA FDs are closed in the entire process tree.
        logger.info("Waiting for all /dev/nvidia* FDs in process tree to close...")
        still_open = _wait_until_no_nvidia_fds(root_pid)
        if still_open:
            raise RuntimeError(
                f"Timeout waiting for NVIDIA FDs to close. PIDs: {still_open}")
        logger.info("All /dev/nvidia* FDs closed; proceeding to CRIU dump")

        # Get TTY info from the subprocess and persist it for restore
        rdev, dev = _get_tty_info(root_pid)
        tty_external = f"tty[{rdev}:{dev}]" if rdev and dev else ""
        if tty_external:
            _save_tty_id(checkpoint_dir, rdev, dev)

        # Take a snapshot of the process tree (for post-dump verification)
        pre_dump_tree = _collect_process_tree_pids(root_pid)

        # Build CRIU dump command
        cmd = [
            "criu", "dump",
            "--shell-job",
            "--images-dir", checkpoint_dir,
            "-o", "criu-dump.log",
            "-v4",
            "--ext-unix-sk",
            "--tcp-established",
            "--external", "mnt[shm]:/dev/shm",
            "--link-remap",
            "--manage-cgroups=ignore",
            "--tree", str(root_pid)
        ]

        if tty_external:
            cmd.extend(["--external", tty_external])

        logger.info("Running CRIU dump: %s", ' '.join(cmd))

        # Run CRIU dump
        result = subprocess.run(cmd)
        if result.returncode != 0:
            raise RuntimeError(f"CRIU dump failed: {result.stderr}")

        logger.info(
            "Successfully checkpointed AsyncLLM to %s", checkpoint_dir)
        # Persist the tree pid so we can wait for its full exit on restore
        _save_tree_pid(checkpoint_dir, root_pid)

        # Reap the child process to avoid a zombie holding the PID.
        # This ensures /proc/<pid> disappears if the process is already dead.
        if self.process is not None:
            try:
                self.process.join(timeout=5)
            except Exception as err:
                raise RuntimeError("Failed to reap child process") from err

        # Verify that all processes in the pre-dump tree are gone.
        # This helps catch stray children that might linger due to plugins.
        deadline = time.time() + 5.0
        lingering: set[int] = set()
        while time.time() < deadline:
            lingering = {
                pid for pid in pre_dump_tree
                if os.path.exists(f"/proc/{pid}")
            }
            if not lingering:
                break
            time.sleep(0.05)
        if lingering:
            logger.error(
                "CRIU dump verification: lingering PIDs: %s",
                sorted(lingering),
            )
        else:
            logger.info("CRIU dump verification: all pre-dump PIDs exited")

        # The process is now frozen, mark it as not running
        self._is_running = False
        self.process = None

        # Close the socket connection
        if self.socket:
            self.socket.close()
            self.socket = None

    async def criu_resume(
            self, cuda_checkpoint_path: str = "cuda-checkpoint") -> None:
        """Restore from CRIU checkpoint.

        This method:
        1. Restores the AsyncLLM subprocess from CRIU checkpoint
        2. Re-establishes the ZMQ connection
        3. Wakes up the model to restore GPU memory

        Args:
            cuda_checkpoint_path: Path to cuda-checkpoint utility (not used)
        """
        if not self.checkpoint_dir:
            raise RuntimeError(
                "No checkpoint directory set. Call criu_checkpoint first.")

        # Load the TTY id used during dump (if any)
        tty_external = _load_tty_id(self.checkpoint_dir)

        # Ensure the original tree PID from dump is fully gone to avoid
        # PID collisions when restoring into the same PID namespace.
        original_pid = _load_tree_pid(self.checkpoint_dir)
        if original_pid:
            start = time.time()
            # Wait up to a short grace period since dump should have killed it
            while time.time() - start < 5.0:
                if os.system(f"kill -0 {original_pid} >/dev/null 2>&1") != 0:
                    break
                await asyncio.sleep(0.05)

            # Verify root PID is not taken now (PID could be reused by others)
            pid_path = f"/proc/{original_pid}"
            if os.path.exists(pid_path):
                # Try to read the cmdline of the holder for diagnostics
                holder = ""
                try:
                    cmd_path = os.path.join(pid_path, "cmdline")
                    with open(cmd_path, "rb") as f:
                        raw = f.read().replace(b"\x00", b" ")
                        holder = raw.decode("utf-8", "ignore").strip()
                except Exception:
                    holder = ""
                msg = (
                    "CRIU restore pre-check failed: root PID "
                    f"{original_pid} is in use in current PID namespace. "
                    "Restore will fail with EEXIST. Consider restoring in a "
                    "new PID namespace or wait until the PID is free."
                )
                if holder:
                    logger.error(
                        "%s Holder cmdline: %s",
                        msg,
                        holder,
                    )
                else:
                    logger.error("%s", msg)
                raise RuntimeError(msg)

        # Build CRIU restore command
        cmd = [
            "criu", "restore",
            "--shell-job",
            "--restore-detached",
            "--images-dir", self.checkpoint_dir,
            "-o", "criu-restore.log",
            "-v4",
            "--ext-unix-sk",
            "--tcp-established",
            "--external", "mnt[shm]:/dev/shm",
            "--link-remap",
            "--manage-cgroups=ignore",
        ]

        # Provide a valid TTY to CRIU using a Python-created pty, so we can
        # mirror logs back to this terminal and support non-TTY parents.
        # We pass the pty slave fd to CRIU and map it to the saved TTY id.
        pass_fds = ()
        pty_master_fd = None
        pty_slave_fd = None
        if tty_external:
            try:
                pty_master_fd, pty_slave_fd = pty.openpty()
                os.set_inheritable(pty_slave_fd, True)
                # Map the saved tty id to the pty slave fd in CRIU
                cmd.extend([
                    "--inherit-fd", f"fd[{pty_slave_fd}]:{tty_external}",
                ])
                pass_fds = (pty_slave_fd,)
            except Exception as e:
                logger.warning("Failed to create PTY for CRIU restore: %s", e)

        logger.info("Running CRIU restore: %s", ' '.join(cmd))

        # Run CRIU restore
        if pass_fds:
            result = subprocess.run(cmd, capture_output=True, text=True,
                                    pass_fds=pass_fds)
        else:
            result = subprocess.run(cmd, capture_output=True, text=True)
        if result.returncode != 0:
            raise RuntimeError(f"CRIU restore failed: {result.stderr}")

        logger.info("Successfully restored AsyncLLM from checkpoint")

        # Wait a bit for the process to fully restore
        await asyncio.sleep(1)

        # Find all processes using our ZMQ port
        self._restored_pids = self._find_processes_using_port(self.port)
        if self._restored_pids:
            logger.info("Found restored processes: %s",
                        sorted(self._restored_pids))
        else:
            logger.warning("Could not find restored processes using port %d",
                           self.port)

        # Re-establish connection to the restored process
        self._is_running = True
        # Mark subprocess as started after restore
        self._subprocess_started = True
        # Just create socket, don't do handshake (server is already running)
        self.socket = self.ctx.socket(zmq.DEALER)
        self.socket.connect(self.socket_url)

        # Wake up the model to restore GPU memory
        logger.info("Waking up model after restore...")
        await self.wake_up()
        logger.info("Model wake up completed")

        # Start forwarding the restored process output to our stdout
        if pty_master_fd is not None:
            try:
                self._start_pty_forwarder(pty_master_fd)
            except Exception as e:
                logger.warning("Failed to start PTY forwarder: %s", e)
        # Close the slave fd in this process; CRIU/restored proc holds it now
        if pty_slave_fd is not None:
            with contextlib.suppress(Exception):
                os.close(pty_slave_fd)

    def shutdown(self):
        """Shutdown the subprocess and clean up resources."""
        # Close PTY master if present
        if getattr(self, "_pty_master_fd", None) is not None:
            with contextlib.suppress(Exception):
                os.close(self._pty_master_fd)  # type: ignore[arg-type]
            self._pty_master_fd = None

        # Try to send shutdown signal if we have a socket connection
        # This works both for normal operation and after CRIU restore
        if self.socket and self._is_running:
            try:
                # Create a synchronous context for shutdown
                sync_ctx = zmq.Context()
                sync_socket = sync_ctx.socket(zmq.DEALER)
                sync_socket.connect(self.socket_url)

                # Send shutdown signal
                sync_socket.send_multipart([b"SHUTDOWN"])
                logger.info("Sent SHUTDOWN signal to AsyncLLM subprocess")

                # If we have a process reference (normal operation), wait for it
                if self.process and self.process.is_alive():
                    # Wait for the process to exit gracefully
                    # (give it more time)
                    shutdown_timeout = 30  # 30 seconds for graceful shutdown
                    self.process.join(timeout=shutdown_timeout)

                    if self.process.is_alive():
                        logger.warning(
                            "AsyncLLM subprocess did not exit gracefully "
                            "after %d seconds, terminating...",
                            shutdown_timeout)
                        self.process.terminate()
                        self.process.join(timeout=5)

                        if self.process.is_alive():
                            logger.error(
                                "AsyncLLM subprocess did not terminate, "
                                "killing...")
                            self.process.kill()
                            self.process.join()
                else:
                    # After CRIU restore, we don't have process reference
                    # Give processes time to shutdown gracefully
                    time.sleep(2)

                sync_socket.close()
                sync_ctx.term()
            except Exception as e:
                logger.error("Error during shutdown: %s", e)
                # Force kill if graceful shutdown fails and we have process ref
                if self.process and self.process.is_alive():
                    self.process.kill()

        # Close async socket
        if self.socket:
            self.socket.close()

        if self.ctx:
            self.ctx.term()

        self._is_running = False

    def __del__(self):
        self.shutdown()

    def _find_processes_using_port(self, port: int) -> set[int]:
        """Find all processes using the given TCP port.

        Returns a set of PIDs that have connections to the port.
        """
        pids = set()
        try:
            # Use lsof to find processes using the port
            result = subprocess.run(
                ['lsof', '-ti', f'tcp:{port}'],
                capture_output=True,
                text=True
            )
            if result.returncode == 0 and result.stdout:
                for line in result.stdout.strip().split('\n'):
                    try:
                        pid = int(line.strip())
                        pids.add(pid)
                    except ValueError:
                        continue
        except Exception as e:
            logger.warning("Could not find processes using port %d: %s",
                           port, e)

        # Alternative: check /proc/*/net/tcp for our port
        if not pids:
            try:
                port_hex = f"{port:04X}"
                # Check all processes
                for pid_dir in os.listdir("/proc"):
                    if not pid_dir.isdigit():
                        continue
                    try:
                        tcp_path = f"/proc/{pid_dir}/net/tcp"
                        if os.path.exists(tcp_path):
                            with open(tcp_path) as f:
                                content = f.read()
                                if port_hex in content:
                                    pids.add(int(pid_dir))
                    except Exception:
                        continue
            except Exception as e:
                logger.debug("Alternative port check failed: %s", e)

        return pids

    # ----- Internal helpers for CRIU PTY forwarding -----
    def _start_pty_forwarder(self, master_fd: int) -> None:
        """Forward data from PTY master fd to this process' stdout.

        This mirrors the restored process' stdout/stderr back
        into the controlling terminal running CheckpointableAsyncLLM.
        """
        self._pty_master_fd = master_fd
        logger.debug("Starting PTY forwarder for fd %d", master_fd)

        def _forward_loop(fd: int):
            try:
                os.set_blocking(fd, False)  # Non-blocking reads
                logger.debug("PTY forwarder started for fd %d", fd)
                while True:
                    try:
                        data = os.read(fd, 4096)
                        if not data:
                            break
                        # Write raw bytes to stdout to preserve ANSI
                        # sequences
                        sys.stdout.buffer.write(data)
                        sys.stdout.buffer.flush()
                    except OSError as e:
                        if e.errno == errno.EAGAIN:
                            time.sleep(0.01)
                            continue
                        raise
                    except InterruptedError:
                        continue
            except Exception as e:
                logger.debug("PTY forwarder stopped: %s", e)
            finally:
                with contextlib.suppress(Exception):
                    os.close(fd)
                if getattr(self, "_pty_master_fd", None) == fd:
                    self._pty_master_fd = None

        t = threading.Thread(target=_forward_loop,
                             args=(master_fd,),
                             name="criu-pty-forwarder",
                             daemon=True)
        t.start()
        self._pty_forwarder_thread = t

    @classmethod
    def from_vllm_config(
        cls,
        vllm_config: VllmConfig,
        start_engine_loop: bool = True,
        usage_context: UsageContext = UsageContext.ENGINE_CONTEXT,
        stat_loggers: Optional[list[StatLoggerFactory]] = None,
        enable_log_requests: bool = False,
        disable_log_stats: bool = False,
        client_addresses: Optional[dict[str, str]] = None,
        client_count: int = 1,
        client_index: int = 0,
        auto_start: bool = True,
    ) -> "CheckpointableAsyncLLM":
        """Create CheckpointableAsyncLLM from VllmConfig."""
        return cls(
            vllm_config=vllm_config,
            executor_class=Executor.get_class(vllm_config),
            start_engine_loop=start_engine_loop,
            stat_loggers=stat_loggers,
            log_requests=enable_log_requests,
            log_stats=not disable_log_stats,
            usage_context=usage_context,
            client_addresses=client_addresses,
            client_count=client_count,
            client_index=client_index,
            auto_start=auto_start,
        )

    @classmethod
    def from_engine_args(
        cls,
        engine_args: AsyncEngineArgs,
        start_engine_loop: bool = True,
        usage_context: UsageContext = UsageContext.ENGINE_CONTEXT,
        stat_loggers: Optional[list[StatLoggerFactory]] = None,
        auto_start: bool = True,
    ) -> "CheckpointableAsyncLLM":
        """Create CheckpointableAsyncLLM from EngineArgs."""
        vllm_config = engine_args.create_engine_config(usage_context)
        return cls(
            vllm_config=vllm_config,
            executor_class=Executor.get_class(vllm_config),
            log_requests=engine_args.enable_log_requests,
            log_stats=not engine_args.disable_log_stats,
            start_engine_loop=start_engine_loop,
            usage_context=usage_context,
            stat_loggers=stat_loggers,
            auto_start=auto_start,
        )
