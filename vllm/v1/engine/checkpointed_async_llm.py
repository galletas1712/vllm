# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
import asyncio
import multiprocessing
import os
import subprocess
import time
import traceback
import uuid
from collections.abc import AsyncGenerator, Mapping
from typing import Any, Optional, Union

import cloudpickle
import msgspec
import zmq
import zmq.asyncio
import pty
import sys
import threading
import contextlib
import fcntl
import termios

from vllm.config import VllmConfig
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
from vllm.utils import get_open_port
from vllm.v1.engine.async_llm import AsyncLLM
from vllm.v1.engine.output_processor import RequestOutputCollector
from vllm.v1.executor.abstract import Executor

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


def _save_process_tree_pids(checkpoint_dir: str, pids: set[int]) -> None:
    """Save the PIDs of all processes in the checkpointed tree."""
    try:
        path = os.path.join(checkpoint_dir, "criu_process_tree_pids.txt")
        with open(path, "w", encoding="utf-8") as f:
            # Sort for consistent ordering
            for pid in sorted(pids):
                f.write(f"{pid}\n")
    except Exception as e:
        logger.warning("Failed to save process tree PIDs: %s", e)


def _load_process_tree_pids(checkpoint_dir: str) -> set[int]:
    """Load the PIDs of all processes that were checkpointed."""
    try:
        path = os.path.join(checkpoint_dir, "criu_process_tree_pids.txt")
        if not os.path.exists(path):
            return set()
        
        pids = set()
        with open(path, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    try:
                        pids.add(int(line))
                    except ValueError:
                        continue
        return pids
    except Exception as e:
        logger.warning("Failed to load process tree PIDs: %s", e)
        return set()


def _collect_process_tree_pids(root_pid: int) -> set[int]:
    """Recursively collect PIDs in the process tree rooted at root_pid.

    Uses /proc/<pid>/task/<pid>/children to discover descendants.
    Best-effort: missing /proc entries are ignored.
    """
    pending: list[int] = [root_pid]
    seen: set[int] = set()
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
        
        # For handling async generators
        self.active_generators: dict[str, AsyncGenerator] = {}
        
    async def initialize(self):
        """Initialize the AsyncLLM instance."""
        self.async_llm = AsyncLLM(
            vllm_config=self.vllm_config,
            executor_class=self.executor_class,
            **self.kwargs
        )
        # Note: If in checkpoint mode, AsyncLLM will complete phase 1
        # initialization and then wait for resume_init() to be called
        
    async def handle_rpc_request(self, msg: RPCMessageType) -> Any:
        """Handle an RPC request and return the result."""
        if self.async_llm is None:
            raise RuntimeError("AsyncLLM not initialized")
            
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
            result = await method(*args, **kwargs)
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
            logger.warning("Failed to set controlling TTY: %s", e)
        finally:
            with contextlib.suppress(Exception):
                os.close(tty_fd)  # type: ignore[name-defined]

    # Ensure logger is initialized in subprocess AFTER stdio is set
    import logging
    logging.basicConfig(level=logging.DEBUG)
    
    try:
        # Deserialize the configuration
        vllm_config = cloudpickle.loads(vllm_config_pickle)
        executor_class = cloudpickle.loads(executor_class_pickle)
        kwargs = cloudpickle.loads(kwargs_pickle)
        
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
        logger.exception("AsyncLLM server failed: %s", e)
        raise


class CheckpointedAsyncLLM(EngineClient):
    """
    A wrapper around AsyncLLM that runs it in a subprocess and
    communicates via ZMQ. Supports CRIU checkpoint/restore functionality.
    
    This class is designed specifically for checkpoint mode where you want to:
    1. Initialize the model partially (phase 1)
    2. Checkpoint the process using CRIU
    3. Restore from checkpoint later
    4. Complete initialization (phases 2 and 3)
    
    Example usage:
        # Option 1: Fresh start with checkpoint/restore
        engine_args = AsyncEngineArgs(model="...", init_mode="checkpoint")
        llm = CheckpointedAsyncLLM.from_engine_args(engine_args)
        await llm.wait_until_checkpoint_ready()
        await llm.criu_checkpoint("/path/to/checkpoint/dir")
        await llm.criu_resume()
        
        # Option 2: Restore from existing checkpoint
        engine_args = AsyncEngineArgs(model="...", init_mode="checkpoint")
        llm = CheckpointedAsyncLLM.from_engine_args(engine_args, 
                                                     auto_start=False)
        llm.checkpoint_dir = "/path/to/existing/checkpoint"
        await llm.criu_resume()
        
        # Option 3: Explicit start control
        llm = CheckpointedAsyncLLM.from_engine_args(engine_args, 
                                                     auto_start=False)
        await llm.start()  # Start subprocess manually
        await llm.wait_until_checkpoint_ready()
        await llm.resume_init()  # Normal init without CRIU
        
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
        use_cached_outputs: bool = False,
        log_requests: bool = True,
        start_engine_loop: bool = True,
        stat_loggers: Optional[list] = None,
        client_addresses: Optional[dict[str, str]] = None,
        client_count: int = 1,
        client_index: int = 0,
        auto_start: bool = True,
    ) -> None:
        # Validate that we're in checkpoint mode
        if vllm_config.launch_config.init_mode != 'checkpoint':
            raise ValueError(
                "CheckpointedAsyncLLM only supports init_mode='checkpoint'")
            
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
            'use_cached_outputs': use_cached_outputs,
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
        
        # Connect to the subprocess
        self._connect()

        # Forward child's PTY master to our stdout so logs are visible
        try:
            self._start_pty_forwarder(pty_master_fd)
        except Exception as e:
            logger.warning("Failed to start startup PTY forwarder: %s", e)
        
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
            while time.time() - start_time < 30:  # 30 second timeout
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
                    raise RuntimeError(
                        "AsyncLLM subprocess died during startup")
            raise TimeoutError(
                "Timeout waiting for AsyncLLM subprocess to start")
        finally:
            sync_socket.close()
            sync_ctx.term()
        
    async def start(self) -> None:
        """Explicitly start the AsyncLLM subprocess.
        
        This is used when auto_start=False was passed to __init__.
        Call this before wait_until_checkpoint_ready() for fresh starts.
        """
        if self._subprocess_started:
            logger.info("Subprocess already started")
            return
        
        self._start_subprocess()
        logger.info("AsyncLLM subprocess started successfully")
        
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
            
    async def abort(self, request_id: str) -> None:
        await self._rpc_call("abort", request_id)
        
    async def encode(
        self,
        prompt: PromptType,
        pooling_params: PoolingParams,
        request_id: str,
        lora_request: Optional[LoRARequest] = None,
        trace_headers: Optional[Mapping[str, str]] = None,
        priority: int = 0,
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
            tokenization_kwargs
        ):
            yield output
            
    async def get_vllm_config(self) -> VllmConfig:
        return await self._rpc_call("get_vllm_config")
        
    async def get_model_config(self):
        return await self._rpc_call("get_model_config")
        
    async def get_decoding_config(self):
        return await self._rpc_call("get_decoding_config")
        
    async def get_input_preprocessor(self) -> InputPreprocessor:
        return await self._rpc_call("get_input_preprocessor")
        
    async def get_tokenizer(self,
                            lora_request: Optional[LoRARequest] = None
                            ) -> AnyTokenizer:
        return await self._rpc_call("get_tokenizer", lora_request)
        
    async def is_tracing_enabled(self) -> bool:
        return await self._rpc_call("is_tracing_enabled")
        
    async def do_log_stats(self, scheduler_outputs=None,
                           model_output=None) -> None:
        await self._rpc_call("do_log_stats", scheduler_outputs, model_output)
        
    async def check_health(self) -> None:
        await self._rpc_call("check_health")
        
    async def start_profile(self) -> None:
        await self._rpc_call("start_profile")
        
    async def stop_profile(self) -> None:
        await self._rpc_call("stop_profile")
        
    async def reset_mm_cache(self) -> None:
        await self._rpc_call("reset_mm_cache")
        
    async def reset_prefix_cache(self) -> None:
        await self._rpc_call("reset_prefix_cache")
        
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
        
    async def wait_until_checkpoint_ready(self) -> None:
        await self._rpc_call("wait_until_checkpoint_ready")
        
    async def wait_for_requests_to_drain(self,
                                         drain_timeout: int = 300):
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
    async def criu_checkpoint(self, checkpoint_dir: str) -> None:
        """Checkpoint the AsyncLLM subprocess using CRIU."""
        if not self._subprocess_started:
            raise RuntimeError(
                "AsyncLLM subprocess not started. "
                "Call start() or wait_until_checkpoint_ready() first")
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
        
        # Get TTY info from the subprocess and persist it for restore
        rdev, dev = _get_tty_info(self.process.pid)
        tty_external = f"tty[{rdev}:{dev}]" if rdev and dev else ""
        if tty_external:
            _save_tty_id(checkpoint_dir, rdev, dev)
        
        # Take a snapshot of the process tree (for post-dump verification)
        root_pid = self.process.pid
        pre_dump_tree = _collect_process_tree_pids(root_pid)
        
        # Save the process tree PIDs for use after restore
        _save_process_tree_pids(checkpoint_dir, pre_dump_tree)

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
            "--tree", str(self.process.pid)
        ]
        
        if tty_external:
            cmd.extend(["--external", tty_external])
            
        logger.info("Running CRIU dump: %s", ' '.join(cmd))
        
        # Run CRIU dump
        result = subprocess.run(cmd, capture_output=True, text=True)
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
            
    async def resume_init(self, after_criu_restore: bool = False) -> None:
        """Resume initialization by calling AsyncLLM's resume_init via RPC.
        
        This completes phases 2 and 3 of initialization for an AsyncLLM that
        was created with init_mode='checkpoint'.
        
        Args:
            after_criu_restore: If True, we're resuming after CRIU restore,
                                so CUDA has already been restored by CRIU.
        """
        await self._rpc_call("resume_init", after_criu_restore)
        logger.info("AsyncLLM resume_init completed")
        
    async def criu_resume(self) -> None:
        """Restore from CRIU checkpoint and then resume initialization.
        
        This method:
        1. Restores the AsyncLLM subprocess from CRIU checkpoint
        2. Re-establishes the ZMQ connection
        3. Calls resume_init() to complete initialization
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
        
        # Load the PIDs that were saved during checkpoint
        self._restored_pids = _load_process_tree_pids(self.checkpoint_dir)
        if self._restored_pids:
            logger.info("Loaded %d restored process PIDs from checkpoint: %s",
                        len(self._restored_pids), 
                        sorted(self._restored_pids))
        else:
            logger.warning("No process PIDs found in checkpoint directory")
        
        # Re-establish connection to the restored process
        self._is_running = True
        # Mark subprocess as started after restore
        self._subprocess_started = True
        # Just create socket, don't do handshake (server is already running)
        self.socket = self.ctx.socket(zmq.DEALER)
        self.socket.connect(self.socket_url)
        
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

        # Now complete initialization
        # Pass after_criu_restore=True since CRIU's CUDA plugin has 
        # already restored CUDA
        await self.resume_init(after_criu_restore=True)
        
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
                    # but we might have tracked PIDs
                    # if self._restored_pids:
                    #     logger.info("Terminating %d restored processes", 
                    #                 len(self._restored_pids))
                    #     # Give them a chance to exit gracefully
                    #     time.sleep(2)
                        
                    #     # Now terminate any that are still running
                    #     still_running = []
                    #     for pid in self._restored_pids:
                    #         try:
                    #             # Check if process still exists
                    #             os.kill(pid, 0)
                    #             still_running.append(pid)
                    #         except ProcessLookupError:
                    #             # Process already dead
                    #             pass
                        
                    #     if still_running:
                    #         logger.warning("Force killing %d processes that "
                    #                        "didn't exit gracefully: %s", 
                    #                        len(still_running), still_running)
                    #         for pid in still_running:
                    #             try:
                    #                 os.kill(pid, signal.SIGKILL)
                    #                 logger.info("Killed process %d", pid)
                    #             except ProcessLookupError:
                    #                 pass
                    #             except Exception as e:
                    #                 logger.error("Failed to kill process "
                    #                              "%d: %s", pid, e)
                    # else:
                    #     logger.warning("No process reference and no tracked "
                    #                    "PIDs - unable to ensure clean "
                    #                    "shutdown")
                    pass
                
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
        
    
    # ----- Internal helpers for CRIU PTY forwarding -----
    def _start_pty_forwarder(self, master_fd: int) -> None:
        """Forward data from PTY master fd to this process' stdout.

        This mirrors the restored process' stdout/stderr back 
        into the controlling terminal running CheckpointedAsyncLLM.
        """
        self._pty_master_fd = master_fd

        def _forward_loop(fd: int):
            try:
                while True:
                    try:
                        data = os.read(fd, 4096)
                    except InterruptedError:
                        continue
                    if not data:
                        break
                    try:
                        # Write raw bytes to stdout to preserve ANSI 
                        # sequences
                        sys.stdout.buffer.write(data)
                        sys.stdout.buffer.flush()
                    except Exception:
                        # If stdout is not available, drop output
                        pass
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
        stat_loggers: Optional[list] = None,
        enable_log_requests: bool = False,
        disable_log_stats: bool = False,
        client_addresses: Optional[dict[str, str]] = None,
        client_count: int = 1,
        client_index: int = 0,
        auto_start: bool = True,
    ) -> "CheckpointedAsyncLLM":
        """Create CheckpointedAsyncLLM from VllmConfig."""
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
        stat_loggers: Optional[list] = None,
        auto_start: bool = True,
    ) -> "CheckpointedAsyncLLM":
        """Create CheckpointedAsyncLLM from EngineArgs."""
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
