# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
import os
import pickle
import pty
import signal
import subprocess
import sys
import time
import threading
import weakref
import zmq
import psutil
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Any, Optional
from multiprocessing.process import BaseProcess
from dataclasses import dataclass

import vllm.envs as envs
from vllm.config import VllmConfig
from vllm.logger import init_logger
from vllm.utils import (get_distributed_init_method, get_loopback_ip, 
                        get_open_port, get_mp_context, decorate_logs,
                        set_process_title)
from vllm.worker.worker_base import WorkerWrapperBase
from vllm.v1.executor.checkpoint_utils import checkpoint_cuda_process
from vllm.v1.executor.abstract import FailureCallback
from vllm.v1.executor.multiproc_executor import (
    MultiprocExecutor, WorkerProc, WorkerProcHandle
)
from vllm.distributed.device_communicators.shm_broadcast import (
    Handle, MessageQueue)
from vllm.distributed.kv_transfer.kv_connector.utils import (
    KVOutputAggregator)
from vllm.executor.multiproc_worker_utils import (
    set_multiprocessing_worker_envs)

# Import TTY utilities from checkpointed_async_llm
from vllm.v1.engine.checkpointed_async_llm import (
    _get_tty_info, _save_tty_id, _save_tree_pid,
    _load_tty_id, _load_tree_pid
)
import contextlib

logger = init_logger(__name__)


@dataclass
class CheckpointedWorkerProcHandle(WorkerProcHandle):
    """Extended handle for checkpointed workers."""
    worker_listen_port: Optional[int] = None  # Port worker binds to receive messages
    checkpoint_dir: Optional[str] = None
    pty_master_fd: Optional[int] = None
    restored_pid: Optional[int] = None


class CheckpointedWorkerProc(WorkerProc):
    """Worker process wrapper that supports CRIU checkpoint/restore.
    
    This class operates in two modes:
    1. Pre-checkpoint: Uses ZMQ TCP for communication (no MessageQueues)
    2. Post-restore: Full WorkerProc functionality with MessageQueues
    
    After CRIU restore, it transitions to full WorkerProc behavior with
    MessageQueues and enters the worker_busy_loop.
    """
    
    def __init__(
        self,
        vllm_config: VllmConfig,
        local_rank: int,
        rank: int,
        distributed_init_method: str,
        fake_distributed_init_method: str,
        input_shm_handle: Optional[Handle] = None,
        parent_handshake_port: Optional[int] = None,
        checkpoint_mode: bool = False,
    ):
        self.rank = rank
        self.checkpoint_mode = checkpoint_mode
        # Port to send initial READY to parent
        self.parent_handshake_port = parent_handshake_port
        self.zmq_context: Optional[zmq.Context] = None
        # Socket worker binds to receive messages  
        self.worker_listen_socket: Optional[zmq.Socket] = None
        # Port worker listens on
        self.worker_listen_port: Optional[int] = None
        
        # In checkpoint mode, we don't initialize MessageQueues yet
        if checkpoint_mode:
            # Create the worker via WorkerWrapperBase so RPC method
            # signatures remain consistent with MultiprocExecutor.
            # Critically, this ensures methods like
            # initialize_from_config can accept list[...] and select the
            # per-rank config, preventing type errors after resume.
            self.init_worker_proc(
                vllm_config,
                local_rank,
                rank,
                distributed_init_method,
                fake_distributed_init_method,
            )
            
            # Set up ZMQ for communication
            self._setup_zmq_communication()
        else:
            # Normal mode - full initialization with MessageQueues
            super().__init__(
                vllm_config, local_rank, rank,
                distributed_init_method, fake_distributed_init_method,
                input_shm_handle
            )
    
    def _setup_zmq_communication(self):
        """Set up ZMQ socket for checkpoint-mode communication."""
        self.zmq_context = zmq.Context()
        # Worker binds a PULL socket to receive messages from parent
        self.worker_listen_socket = self.zmq_context.socket(zmq.PULL)
        # Bind to any available port
        self.worker_listen_port = get_open_port()
        self.worker_listen_socket.bind(f"tcp://127.0.0.1:{self.worker_listen_port}")
        logger.info("Worker %d: Bound listening socket on port %d", 
                    self.rank, self.worker_listen_port)
    
    def send_ready_checkpoint(self):
        """Send ready signal in checkpoint mode with worker's bound port."""
        # Send ready signal to parent via the initial connection
        # (parent_handshake_port is the parent's port we connect to for initial handshake)
        if self.parent_handshake_port is not None:
            # Create a temporary socket to send ready signal to parent
            handshake_socket = self.zmq_context.socket(zmq.PUSH)
            handshake_socket.connect(f"tcp://127.0.0.1:{self.parent_handshake_port}")
            handshake_socket.send_json({
                "status": "READY",
                "pid": os.getpid(),
                "rank": self.rank,
                "worker_listen_port": self.worker_listen_port,  # Tell parent our listening port
            })
            handshake_socket.close()
    
    def wait_for_checkpoint_and_resume(self):
        """Wait for checkpoint to complete, then wait for resume signal.
        
        This method:
        1. Blocks waiting for checkpoint completion (via ZMQ)
        2. After CRIU restore, continues blocking for resume signal
        3. Receives resume signal with input_shm_handle
        4. Initializes MessageQueues and transitions to full WorkerProc
        
        Returns True if successful, False otherwise.
        """
        if not self.worker_listen_socket:
            return False
            
        try:
            # Wait for checkpoint completion or resume signal
            # This blocks the process in a CRIU-friendly way
            logger.info("Worker %d: Waiting for resume signal on port %d...", 
                        self.rank, self.worker_listen_port)
            
            # Check if socket and context are still valid after restore
            try:
                # Try to get socket option to verify it's still valid
                hwm = self.worker_listen_socket.getsockopt(zmq.HWM)
                logger.info("Worker %d: Socket still valid, HWM=%d", self.rank, hwm)
            except Exception as e:
                logger.error("Worker %d: Socket invalid after restore: %s", self.rank, e)
                # Try to recreate the socket and context
                logger.info("Worker %d: Recreating ZMQ context and socket on port %d", 
                            self.rank, self.worker_listen_port)
                with contextlib.suppress(Exception):
                    self.worker_listen_socket.close()
                with contextlib.suppress(Exception):
                    self.zmq_context.term()
                
                # Create new context and socket
                self.zmq_context = zmq.Context()
                self.worker_listen_socket = self.zmq_context.socket(zmq.PULL)
                self.worker_listen_socket.bind(f"tcp://127.0.0.1:{self.worker_listen_port}")
                logger.info("Worker %d: Successfully recreated socket", self.rank)
            
            # Add a timeout to help debug
            self.worker_listen_socket.setsockopt(zmq.RCVTIMEO, 60000)  # 60 second timeout
            try:
                msg = self.worker_listen_socket.recv_json()
                logger.info("Worker %d: Received message: %s", self.rank, msg)
            except zmq.Again:
                logger.error("Worker %d: Timeout waiting for resume signal on port %d", 
                             self.rank, self.worker_listen_port)
                return False
            
            # After CRIU restore, we'll receive the resume signal here
            if msg.get("command") != "RESUME":
                logger.error("Worker %d: Expected RESUME, got %s", self.rank, msg)
                return False
            
            # Extract the input_shm_handle (sent as serialized bytes)
            input_shm_handle_data = msg.get("input_shm_handle")
            if not input_shm_handle_data:
                logger.error("Worker %d: No input_shm_handle in resume message", self.rank)
                return False
            
            # Extract the handle_recv_port for sending MQ handle back
            handle_recv_port = msg.get("handle_recv_port")
            if handle_recv_port is None:
                logger.error("Worker %d: No handle_recv_port in resume message", self.rank)
                return False
            
            # Save handle_recv_port for later use
            self._handle_recv_port = handle_recv_port
            
            # Deserialize the handle
            import pickle
            input_shm_handle = pickle.loads(
                bytes.fromhex(input_shm_handle_data))
            
            # Initialize MessageQueues and transition to full WorkerProc
            logger.info("Worker %d: Initializing MessageQueues...", self.rank)
            self.initialize_post_restore(input_shm_handle)
            logger.info("Worker %d: MessageQueues initialized", self.rank)
            
            # Clean up worker's listening socket (no longer needed)
            if self.worker_listen_socket:
                self.worker_listen_socket.close()
                self.worker_listen_socket = None
            # Keep zmq_context for now - we'll need it to send MQ handle
            
            logger.info("Worker %d: Successfully resumed from checkpoint", self.rank)
            return True
            
        except Exception as e:
            logger.exception("Worker %d: Failed during checkpoint/resume: %s", self.rank, e)
            return False
    
    def initialize_post_restore(self, input_shm_handle: Handle):
        """Initialize MessageQueues and other components after CRIU restore."""
        # Initialize MessageQueue for receiving SchedulerOutput
        self.rpc_broadcast_mq = MessageQueue.create_from_handle(
            input_shm_handle, self.worker.rank)
        
        # Initialize a message queue for sending the model output
        self.worker_response_mq = MessageQueue(1, 1)
    
    @staticmethod
    def make_checkpointed_worker_process(
        vllm_config: VllmConfig,
        local_rank: int,
        rank: int,
        distributed_init_method: str,
        fake_distributed_init_method: str,
        parent_handshake_port: int,
    ) -> BaseProcess:
        """Create a worker process for checkpoint mode."""
        context = get_mp_context()
        
        process_kwargs = {
            "vllm_config": vllm_config,
            "local_rank": local_rank,
            "rank": rank,
            "distributed_init_method": distributed_init_method,
            "fake_distributed_init_method": fake_distributed_init_method,
            "parent_handshake_port": parent_handshake_port,
        }
        
        proc = context.Process(
            target=CheckpointedWorkerProc.checkpointed_worker_main,
            kwargs=process_kwargs,
            name=f"VllmWorker-{rank}",
            daemon=False
        )
        
        proc.start()
        return proc
    
    @staticmethod
    def checkpointed_worker_main(
        vllm_config: VllmConfig,
        local_rank: int,
        rank: int,
        distributed_init_method: str,
        fake_distributed_init_method: str,
        parent_handshake_port: int,
    ):
        """Main function for checkpointed worker process."""
        # Become session leader and set controlling TTY
        os.setsid()

        try:
            # Create worker in checkpoint mode
            worker_proc = CheckpointedWorkerProc(
                vllm_config=vllm_config,
                local_rank=local_rank,
                rank=rank,
                distributed_init_method=distributed_init_method,
                fake_distributed_init_method=fake_distributed_init_method,
                parent_handshake_port=parent_handshake_port,
                checkpoint_mode=True,
            )
            
            # Initialize phase 1
            logger.info(f"Worker {rank}: Starting phase 1 initialization...")
            start_time = time.time()
            worker_proc.worker.phase_1_init()
            phase1_time = time.time() - start_time
            logger.info(f"Worker {rank}: Phase 1 initialization completed in {phase1_time:.2f} seconds")
            
            # Prepare for checkpoint
            logger.info(f"Worker {rank}: Preparing for checkpoint...")
            worker_proc.worker.prepare_for_checkpoint()
            
            # Send ready signal
            worker_proc.send_ready_checkpoint()
            
            # Wait for checkpoint to complete and then for resume signal
            # This blocks on ZMQ socket recv, which is CRIU-friendly
            logger.info(f"Worker {rank} ready for checkpointing. Waiting...")
            
            if not worker_proc.wait_for_checkpoint_and_resume():
                raise RuntimeError(f"Worker {rank}: Failed during checkpoint/resume")
            
            logger.info(f"Worker {rank}: Resumed after CRIU restore with MessageQueues initialized")
            
            # Extract handle_recv_port from the resume message (we should have saved it)
            handle_recv_port = getattr(worker_proc, '_handle_recv_port', None)
            if handle_recv_port is None:
                raise RuntimeError(f"Worker {rank}: No handle_recv_port available")
            
            # Now we have MessageQueues set up, we need to send the handle back to parent
            # We'll use ZMQ to send the worker_response_mq handle
            logger.info(f"Worker {rank}: Sending worker_response_mq handle to parent on port {handle_recv_port}...")
            
            # Create a new socket to send the handle
            handle_socket = worker_proc.zmq_context.socket(zmq.PUSH)
            handle_socket.connect(f"tcp://127.0.0.1:{handle_recv_port}")
            
            # Send the MessageQueue handle
            handle_msg = {
                "status": "MQ_READY",
                "rank": rank,
                "pid": os.getpid(),
                "worker_response_mq_handle": pickle.dumps(
                    worker_proc.worker_response_mq.export_handle()).hex()
            }
            handle_socket.send_json(handle_msg)
            handle_socket.close()
            logger.info(f"Worker {rank}: Sent worker_response_mq handle to parent")
            
            # Now we can clean up the ZMQ context
            if worker_proc.zmq_context:
                worker_proc.zmq_context.term()
                worker_proc.zmq_context = None
            
            # Ensure message queues are ready
            logger.info(f"Worker {rank}: Waiting for message queues to be ready...")
            worker_proc.rpc_broadcast_mq.wait_until_ready()
            worker_proc.worker_response_mq.wait_until_ready()
            logger.info(f"Worker {rank}: Message queues ready")
            
            # Send READY signal to parent through the MessageQueue
            # The parent is expecting (ResponseStatus, result) tuple
            logger.info(f"Worker {rank}: Sending READY response to parent via MessageQueue...")
            worker_proc.worker_response_mq.enqueue(
                (WorkerProc.ResponseStatus.SUCCESS, 
                 worker_proc.worker_response_mq.export_handle()))
            logger.info(f"Worker {rank}: READY response sent")
            
            # Enter the worker busy loop - this is the same as WorkerProc
            logger.info(f"Worker {rank}: Entering worker busy loop...")
            worker_proc.worker_busy_loop()
            
        except Exception as e:
            logger.exception(f"Worker {rank} process failed: %s", e)
            raise


class CheckpointedMultiprocExecutor(MultiprocExecutor):
    """Multiprocessing executor with CRIU checkpoint/restore support."""
    
    def __init__(self, vllm_config: VllmConfig) -> None:
        self.checkpoint_mode = vllm_config.launch_config.init_mode in ["save_checkpoint", "resume_checkpoint"]
        self.checkpoint_dir = os.path.join(
            vllm_config.launch_config.checkpoint_dir_root,
            (
                vllm_config.compute_hash() + 
                "_dp_" + str(vllm_config.parallel_config.data_parallel_rank)
            )
        )
        logger.info(f"Checkpoint directory: {self.checkpoint_dir}")

        # Store these before calling super().__init__
        self.distributed_init_method = None
        self.fake_distributed_init_method = None
        # Ensure attribute exists regardless of init mode
        self.io_thread_pool: Optional[ThreadPoolExecutor] = None
        super().__init__(vllm_config)
    
    def _init_executor(self) -> None:
        # Call parent's initialization logic first to set up basic executor state
        # but override worker creation for checkpoint modes
        
        # Set up distributed init methods (from parent)
        self.fake_distributed_init_method = get_distributed_init_method(
            get_loopback_ip(), get_open_port())
        self.distributed_init_method = get_distributed_init_method(
            get_loopback_ip(), get_open_port())
        
        if self.vllm_config.launch_config.init_mode == "resume_checkpoint":
            # Restore from checkpoint
            self._restore_from_checkpoint()
            # Set up remaining executor components
            self._setup_post_restore()
        elif self.vllm_config.launch_config.init_mode == "save_checkpoint":
            # Initialize for checkpoint
            self._init_for_checkpoint()
            # Perform checkpoint after phase 1 initialization
            self._save_checkpoint()
        else:
            # Normal initialization
            super()._init_executor()
    
    def _init_for_checkpoint(self) -> None:
        """Initialize executor for checkpoint mode."""
        # Set multiprocessing envs
        set_multiprocessing_worker_envs(self.parallel_config)
        
        # Initialize basic executor state
        self.world_size = self.parallel_config.world_size
        self.is_failed = False
        self.shutdown_event = threading.Event()
        self.failure_callback: Optional[FailureCallback] = None
        self.io_thread_pool: Optional[ThreadPoolExecutor] = None
        
        # Don't create MessageQueues here - they use semaphores!
        # We'll create them during restore instead
        self.rpc_broadcast_mq = None
        self.scheduler_output_handle = None
        
        # Create workers in checkpoint mode (using ZMQ only)
        self.workers, self.worker_pids = self._create_checkpointed_workers()
    
    def _setup_post_restore(self) -> None:
        """Set up executor components after restore."""
        # Initialize components that depend on workers
        self.output_rank = self._get_output_rank()
        self.has_connector = self.vllm_config.kv_transfer_config is not None
        self.kv_output_aggregator = KVOutputAggregator(
            self.parallel_config.world_size)
        
        # For pipeline parallel
        if self.max_concurrent_batches > 1:
            self.io_thread_pool = ThreadPoolExecutor(
                max_workers=1, thread_name_prefix="mp_exec_io")
        
        self._start_restored_worker_monitor()
    
    def restore_workers(self, after_criu_restore: bool = False):
        """
        This method is called after checkpoint restoration.
        For CRIU, the workers are already restored, so we don't need to do
        anything here. The parent's implementation starts a worker monitor
        that is not compatible with restored processes.
        """
        # Do nothing. The worker monitor for restored processes is started
        # in _complete_post_restore_init.
        pass

    def _create_checkpointed_workers(
            self) -> tuple[list[CheckpointedWorkerProcHandle], list[Optional[int]]]:
        """Create workers in checkpoint mode."""
        # Create ZMQ context and socket for parent process to receive ready signals
        parent_handshake_port = get_open_port()
        ctx = zmq.Context()
        handshake_socket = ctx.socket(zmq.PULL)
        handshake_socket.bind(f"tcp://127.0.0.1:{parent_handshake_port}")
        
        # Create PTY for the worker processes
        pty_master_fd, pty_slave_fd = pty.openpty()
        os.set_inheritable(pty_slave_fd, True)
        
        processes = []
        
        for rank in range(self.world_size):
            process = CheckpointedWorkerProc.make_checkpointed_worker_process(
                vllm_config=self.vllm_config,
                local_rank=rank,
                rank=rank,
                distributed_init_method=self.distributed_init_method,
                fake_distributed_init_method=self.fake_distributed_init_method,
                parent_handshake_port=parent_handshake_port,
            )
            processes.append(process)
        
        # Wait for all workers to be ready
        worker_handles = []
        pids = []
        
        try:
            handshake_socket.setsockopt(zmq.RCVTIMEO, 120000)  # 120 second timeout
            
            # Store the context for later use
            self._checkpoint_zmq_ctx = ctx
            
            for i in range(self.world_size):
                # PULL socket receives just the message
                msg_data = handshake_socket.recv()
                response = zmq.utils.jsonapi.loads(msg_data)
                if response["status"] != "READY":
                    raise RuntimeError(f"Worker {response.get('rank', '?')} failed to initialize")
                
                # Create a minimal handle for checkpoint mode
                handle = CheckpointedWorkerProcHandle(
                    proc=processes[response["rank"]],
                    rank=response["rank"],
                    worker_response_mq=None,  # No MQ in checkpoint mode
                    death_writer=None,
                    worker_listen_port=response["worker_listen_port"],  # Worker's listening port
                    pty_master_fd=pty_master_fd,
                    checkpoint_dir=None,  # Will be set during checkpoint
                )
                worker_handles.append(handle)
                pids.append(response["pid"])
                
                logger.info("✓ Worker %d initialized (PID: %d, port: %d)", 
                            response['rank'], response['pid'], response['worker_listen_port'])
            
            # Start a PTY forwarder for shared worker TTY so logs show up
            try:
                # Lazily create container for forwarders
                if not hasattr(self, "_pty_forwarders"):
                    self._pty_forwarders = []
                self._start_pty_forwarder(pty_master_fd)
            except Exception as e:
                logger.warning("Failed to start worker PTY forwarder: %s", e)
            
        finally:
            # Close the initial handshake socket - we don't need it anymore
            handshake_socket.close()
            os.close(pty_slave_fd)
        
        return worker_handles, pids
    
    def _save_checkpoint(self):
        """Save CRIU checkpoint of all workers."""
        logger.info("Saving checkpoint...")
        
        # Remove checkpoint directory if it exists
        import shutil
        if os.path.exists(self.checkpoint_dir):
            shutil.rmtree(self.checkpoint_dir)
        os.makedirs(self.checkpoint_dir, exist_ok=True)
        
        # Checkpoint all workers
        self._checkpoint_all_workers()
        
        logger.info(f"Checkpoint saved to {self.checkpoint_dir}")
        logger.info("Shutting down after checkpoint...")
        
        # Clean up ZMQ resources
        if hasattr(self, '_checkpoint_zmq_ctx') and self._checkpoint_zmq_ctx:
            self._checkpoint_zmq_ctx.term()
            self._checkpoint_zmq_ctx = None
        
        # Gracefully exit
        self.shutdown()
        sys.exit(0)
    
    def _checkpoint_all_workers(self):
        """Checkpoint all workers using CRIU."""
        # Checkpoint all workers serially
        logger.info("Starting serial checkpoint of %d workers...", len(self.workers))
        for i, worker in enumerate(self.workers):
            logger.info("Checkpointing worker %d/%d (rank %d)...", 
                        i + 1, len(self.workers), worker.rank)
            result = self._checkpoint_worker(worker)
            if not result["success"]:
                raise RuntimeError(f"Failed to checkpoint worker {result['rank']}: {result.get('error')}")
            logger.info("Successfully checkpointed worker %d (rank %d)", 
                        i + 1, worker.rank)
    
    def _checkpoint_worker(
            self, worker: CheckpointedWorkerProcHandle) -> dict[str, Any]:
        """Checkpoint a single worker with CRIU."""
        rank = worker.rank
        pid = worker.proc.pid
        worker_checkpoint_dir = os.path.join(self.checkpoint_dir, f"worker_{rank}")
        
        logger.info(f"Checkpointing worker {rank} (PID: {pid})...")
        
        try:
            # Create checkpoint directory
            os.makedirs(worker_checkpoint_dir, exist_ok=False)
            
            # Checkpoint CUDA state
            logger.info(f"Worker {rank}: Checkpointing CUDA state...")
            checkpoint_cuda_process(pid)
            
            # Remove semaphore files after CUDA checkpoint but before CRIU dump
            logger.info(f"Worker {rank}: Removing semaphore files before CRIU dump...")
            self._remove_semaphore_files(target_pid=pid)
            
            # Get TTY info and save it
            rdev, dev = _get_tty_info(pid)
            tty_external = f"tty[{rdev}:{dev}]" if rdev and dev else ""
            if tty_external:
                _save_tty_id(worker_checkpoint_dir, rdev, dev)
            
            # Save the worker PID for restore
            _save_tree_pid(worker_checkpoint_dir, pid)
            
            # Save worker info
            with open(os.path.join(worker_checkpoint_dir, "worker_info.pkl"), "wb") as f:
                pickle.dump({
                    "rank": rank,
                    "vllm_config": self.vllm_config,
                    "worker_listen_port": worker.worker_listen_port,  # Save worker's listening port
                }, f)
            
            # Build CRIU dump command
            cmd = [
                "criu", "dump",
                "--shell-job",
                "--images-dir", worker_checkpoint_dir,
                "-o", "criu-dump.log",
                "-v4",
                "--ext-unix-sk",
                "--skip-in-flight",
                "--tcp-close",
                "--external", "mnt[shm]:/dev/shm",
                "--link-remap",
                "--manage-cgroups=ignore",
                "--tree", str(pid)
            ]
            
            if tty_external:
                cmd.extend(["--external", tty_external])
            
            # Run CRIU dump
            result = subprocess.run(cmd, capture_output=True, text=True)
            if result.returncode != 0:
                raise RuntimeError(f"CRIU dump failed: {result.stderr}")
            
            return {
                "rank": rank,
                "success": True,
            }
            
        except Exception as e:
            logger.error(f"Failed to checkpoint worker {rank}: %s", e)
            return {
                "rank": rank,
                "success": False,
                "error": str(e),
            }
    
    def _remove_semaphore_files(self, target_pid: Optional[int] = None):
        """Remove semaphore files that could interfere with CRIU.
        
        Args:
            target_pid: If specified, only remove semaphores for this PID.
                       If None, remove semaphores for all processes.
        """
        if target_pid is not None:
            logger.info(f"Removing semaphore files for PID {target_pid}...")
            our_pids = [target_pid]
        else:
            logger.info("Removing semaphore files for all processes...")
            our_pids = [os.getpid()] + [w.proc.pid for w in self.workers]
        logger.info(f"Checking semaphore files for PIDs: {our_pids}")
        our_semaphores = set()
        
        # First, log all /dev/shm files that look like semaphores
        logger.info("All semaphore-like files in /dev/shm:")
        import glob
        all_sem_files = glob.glob("/dev/shm/sem.*")
        for sem_file in all_sem_files:
            logger.info(f"  - {sem_file}")
        
        # Check memory maps for our processes
        for pid in our_pids:
            try:
                proc = psutil.Process(pid)
                logger.debug(f"Checking memory maps for PID {pid}...")
                pid_semaphores = []
                for mmap in proc.memory_maps():
                    if '/dev/shm/sem.' in mmap.path:
                        our_semaphores.add(mmap.path)
                        pid_semaphores.append(mmap.path)
                if pid_semaphores:
                    logger.info(f"PID {pid} has semaphores: {pid_semaphores}")
            except Exception as e:
                logger.warning(f"Failed to get memory maps for PID {pid}: {e}")
        
        # Also check for semaphore files that match our PIDs
        for sem_file in all_sem_files:
            for pid in our_pids:
                if f"-{pid}-" in sem_file or f".{pid}." in sem_file:
                    logger.info(f"Found semaphore matching PID {pid}: {sem_file}")
                    our_semaphores.add(sem_file)
        
        logger.info(f"Total semaphores detected for our processes: {len(our_semaphores)}")
        for sem in sorted(our_semaphores):
            logger.info(f"  - {sem}")
        
        # Remove the semaphores
        removed_count = 0
        failed_count = 0
        for shm_file in our_semaphores:
            try:
                if os.path.exists(shm_file):
                    os.unlink(shm_file)
                    logger.info(f"Successfully removed {shm_file}")
                    removed_count += 1
                else:
                    logger.warning(f"Semaphore file doesn't exist on disk: {shm_file}")
            except Exception as e:
                logger.error(f"Failed to remove semaphore {shm_file}: {e}")
                failed_count += 1
        
        logger.info(f"Semaphore cleanup complete: removed={removed_count}, failed={failed_count}")
        
        # Log remaining semaphores after cleanup
        remaining_sem_files = glob.glob("/dev/shm/sem.*")
        if remaining_sem_files:
            logger.info(f"Remaining semaphore files after cleanup: {len(remaining_sem_files)}")
            for sem_file in remaining_sem_files:
                logger.info(f"  - {sem_file}")
    
    def _restore_from_checkpoint(self):
        """Restore workers from CRIU checkpoint."""
        logger.info(f"Restoring from checkpoint {self.checkpoint_dir}...")
        
        # Initialize basic executor state first
        self.world_size = self.parallel_config.world_size
        self.is_failed = False
        self.shutdown_event = threading.Event()
        self.failure_callback: Optional[FailureCallback] = None
        
        # MessageQueues will be created later in _complete_post_restore_init
        self.rpc_broadcast_mq = None
        self.scheduler_output_handle = None
        
        # Find worker checkpoint directories
        worker_dirs = []
        for entry in os.listdir(self.checkpoint_dir):
            if entry.startswith("worker_") and os.path.isdir(os.path.join(self.checkpoint_dir, entry)):
                worker_dirs.append(entry)
        
        if not worker_dirs:
            raise RuntimeError(f"No worker checkpoints found in {self.checkpoint_dir}")
        
        worker_dirs.sort()
        
        # Restore all workers in parallel
        restored_workers = []
        original_pids = []
        with ThreadPoolExecutor(max_workers=len(worker_dirs)) as executor:
            futures = []
            for worker_dir in worker_dirs:
                rank = int(worker_dir.split('_')[1])
                worker_checkpoint_dir = os.path.join(self.checkpoint_dir, worker_dir)
                future = executor.submit(self._restore_worker, worker_checkpoint_dir, rank)
                futures.append(future)
            
            for future in as_completed(futures):
                result = future.result()
                if result["success"]:
                    restored_workers.append(result["handle"])
                    if result.get("original_pid") is not None:
                        original_pids.append(result["original_pid"])
                else:
                    raise RuntimeError(f"Failed to restore worker {result['rank']}: {result.get('error')}")
        
        # Start PTY forwarders for restored workers (each has its own PTY)
        try:
            if not hasattr(self, "_pty_forwarders"):
                self._pty_forwarders = []
            for h in restored_workers:
                if getattr(h, "pty_master_fd", None):
                    self._start_pty_forwarder(h.pty_master_fd)  # type: ignore[arg-type]
        except Exception as e:
            logger.warning("Failed to start PTY forwarders for restored workers: %s", e)

        # Sort by rank
        restored_workers.sort(key=lambda h: h.rank)
        self.workers = restored_workers
        self.worker_pids = original_pids
        
        # Now complete initialization (phase 2 and 3)
        logger.info("Completing initialization after restore...")
        self._complete_post_restore_init()
    
    def _restore_worker(
            self, worker_checkpoint_dir: str, rank: int) -> dict[str, Any]:
        """Restore a single worker from CRIU checkpoint."""
        logger.info(f"Restoring worker {rank} from {worker_checkpoint_dir}...")
        
        try:
            # Load saved TTY and PID info
            tty_external = _load_tty_id(worker_checkpoint_dir)
            original_pid = _load_tree_pid(worker_checkpoint_dir)
            
            # Load worker info (has zmq_port)
            with open(os.path.join(worker_checkpoint_dir, "worker_info.pkl"), "rb") as f:
                worker_info = pickle.load(f)
            
            # Build CRIU restore command
            cmd = [
                "criu", "restore",
                "--shell-job",
                "--restore-detached",
                "--images-dir", worker_checkpoint_dir,
                "-o", "criu-restore.log",
                "-v4",
                "--ext-unix-sk",
                "--tcp-close",
                "--external", "mnt[shm]:/dev/shm",
                "--link-remap",
                "--manage-cgroups=ignore",
            ]
            # Ask CRIU to write the restored root PID to a file for monitoring
            pidfile_path = os.path.join(worker_checkpoint_dir, "restored_pid.txt")
            cmd.extend(["--pidfile", pidfile_path])
            
            # Set up PTY for restored process
            pty_master_fd = None
            if tty_external:
                pty_master_fd, pty_slave_fd = pty.openpty()
                os.set_inheritable(pty_slave_fd, True)
                cmd.extend([
                    "--inherit-fd", f"fd[{pty_slave_fd}]:{tty_external}",
                ])
                pass_fds = (pty_slave_fd,)
            else:
                pass_fds = ()
            
            # Run CRIU restore
            logger.info(f"Running CRIU restore command for worker {rank}...")
            result = subprocess.run(cmd, capture_output=True, text=True, pass_fds=pass_fds)
            if result.returncode != 0:
                # Check CRIU log for more details
                log_file = os.path.join(worker_checkpoint_dir, "criu-restore.log")
                if os.path.exists(log_file):
                    with open(log_file, 'r') as f:
                        logger.error(f"CRIU restore log for worker {rank}:\n{f.read()}")
                raise RuntimeError(f"CRIU restore failed: {result.stderr}")
            logger.info(f"CRIU restore completed for worker {rank}")
            
            # Read the restored PID if CRIU wrote it
            restored_pid: Optional[int] = None
            try:
                if os.path.exists(pidfile_path):
                    with open(pidfile_path, "r", encoding="utf-8") as f:
                        restored_pid = int(f.read().strip() or "0")
            except Exception as e:
                logger.warning("Failed to read restored PID for worker %d: %s", rank, e)
 
            # Close slave FD
            if tty_external:
                os.close(pty_slave_fd)
            
            # Check if worker process was restored
            if original_pid:
                if os.path.exists(f"/proc/{original_pid}"):
                    logger.info(f"Worker {rank} restored with original PID {original_pid}")
                else:
                    logger.warning(f"Worker {rank} not found with original PID {original_pid}")
                    # Try to find any VllmWorker process
                    try:
                        for proc in psutil.process_iter(['pid', 'name', 'cmdline']):
                            cmdline = proc.info.get('cmdline', [])
                            if cmdline and any(f'VllmWorker-{rank}' in str(arg) for arg in cmdline):
                                logger.info(f"Found worker {rank} with PID {proc.info['pid']}")
                                break
                    except Exception as e:
                        logger.warning(f"Failed to search for worker {rank}: {e}")
            
            # Check if the port is actually listening
            import socket as sock
            test_sock = sock.socket(sock.AF_INET, sock.SOCK_STREAM)
            result = test_sock.connect_ex(('127.0.0.1', worker_info.get('worker_listen_port')))
            test_sock.close()
            if result == 0:
                logger.info(f"Worker {rank} port {worker_info.get('worker_listen_port')} is listening")
            else:
                logger.warning(f"Worker {rank} port {worker_info.get('worker_listen_port')} is NOT listening")
            
            # Create handle for restored worker
            handle = CheckpointedWorkerProcHandle(
                proc=None,  # We don't have process object after restore
                rank=rank,
                worker_response_mq=None,  # Will be initialized after resume
                death_writer=None,
                pty_master_fd=pty_master_fd,
                worker_listen_port=worker_info.get('worker_listen_port'),  # Worker's listening port
                checkpoint_dir=worker_checkpoint_dir,
                restored_pid=restored_pid,
            )
            
            return {
                "rank": rank,
                "success": True,
                "handle": handle,
                "original_pid": original_pid,
            }
            
        except Exception as e:
            logger.error(f"Failed to restore worker {rank}: %s", e)
            return {
                "rank": rank,
                "success": False,
                "error": str(e),
            }
    
    def _complete_post_restore_init(self):
        """Complete initialization after CRIU restore."""
        logger.info("Sending resume signals to restored workers...")
        
        # Create message queues - these weren't created during checkpoint
        # to avoid semaphores that interfere with CRIU
        if self.rpc_broadcast_mq is None:
            max_chunk_bytes = envs.VLLM_MQ_MAX_CHUNK_BYTES_MB * 1024 * 1024
            self.rpc_broadcast_mq = MessageQueue(self.world_size,
                                                 self.world_size,
                                                 max_chunk_bytes=max_chunk_bytes)
            self.scheduler_output_handle = self.rpc_broadcast_mq.export_handle()
        
        # Create ZMQ context if needed
        if not hasattr(self, '_checkpoint_zmq_ctx') or self._checkpoint_zmq_ctx is None:
            self._checkpoint_zmq_ctx = zmq.Context()
        
        # Create a socket to receive worker MessageQueue handles
        handle_recv_port = get_open_port()
        handle_socket = self._checkpoint_zmq_ctx.socket(zmq.PULL)
        handle_socket.bind(f"tcp://127.0.0.1:{handle_recv_port}")
        
        # Store the port so workers know where to send their handles
        self._handle_recv_port = handle_recv_port
        
        # Send resume signals to all workers by connecting to their listening ports
        for worker in self.workers:
            try:
                logger.info("Connecting to worker %d on port %d...", 
                            worker.rank, worker.worker_listen_port)
                # Create a PUSH socket to send to this worker
                resume_socket = self._checkpoint_zmq_ctx.socket(zmq.PUSH)
                resume_socket.connect(f"tcp://127.0.0.1:{worker.worker_listen_port}")
                
                # Give socket time to connect
                time.sleep(0.1)
                
                # Serialize the input_shm_handle
                import pickle
                shm_handle_hex = pickle.dumps(self.scheduler_output_handle).hex()
                
                resume_msg = {
                    "command": "RESUME",
                    "input_shm_handle": shm_handle_hex,
                    "handle_recv_port": handle_recv_port,  # Where to send MQ handle
                }
                
                logger.info("Sending resume message to worker %d...", worker.rank)
                # Send resume message
                resume_socket.send_json(resume_msg)
                resume_socket.close()
                
                logger.info("Sent resume signal to worker %d on port %d", 
                            worker.rank, worker.worker_listen_port)
            except Exception as e:
                logger.error("Failed to send resume signal to worker %d: %s", 
                             worker.rank, e)
                raise
        
        # Now wait for workers to send their MessageQueue handles
        logger.info("Waiting for worker MessageQueue handles...")
        
        # Set timeout for receiving handles
        handle_socket.setsockopt(zmq.RCVTIMEO, 30000)  # 30 second timeout
        
        # Receive handles from all workers
        received_handles = {}
        received_pids: dict[int, int] = {}
        for _ in range(self.world_size):
            try:
                handle_msg = handle_socket.recv_json()
                if handle_msg.get("status") != "MQ_READY":
                    raise RuntimeError(f"Expected MQ_READY, got {handle_msg}")
                
                rank = handle_msg["rank"]
                mq_handle_data = handle_msg.get("worker_response_mq_handle")
                if not mq_handle_data:
                    raise RuntimeError(f"No worker_response_mq_handle from worker {rank}")
                
                # Deserialize the handle
                mq_handle = pickle.loads(bytes.fromhex(mq_handle_data))
                received_handles[rank] = mq_handle
                # Record PID for robust monitoring/shutdown
                pid_val = handle_msg.get("pid")
                if pid_val is not None:
                    try:
                        received_pids[rank] = int(pid_val)
                    except Exception:
                        pass
                logger.info(f"Received MessageQueue handle from worker {rank}")
                
            except zmq.Again as e:
                raise TimeoutError("Timeout waiting for worker MessageQueue handles") from e
        
        # Close the handle socket
        handle_socket.close()
        
        # Create MessageQueues from worker handles
        for worker in self.workers:
            if worker.rank not in received_handles:
                raise RuntimeError(f"No MessageQueue handle received from worker {worker.rank}")
            
            # Create MessageQueue from worker's handle
            worker.worker_response_mq = MessageQueue.create_from_handle(
                received_handles[worker.rank], 0)
            logger.info(f"Created MessageQueue for worker {worker.rank}")
        
        # Build PID map from pidfiles and verify against worker-reported PIDs
        restored_map_from_files: dict[int, int] = {}
        for worker in self.workers:
            if not worker.checkpoint_dir:
                raise RuntimeError(f"Worker {worker.rank} has no checkpoint directory")
            pidfile_path = os.path.join(worker.checkpoint_dir, "restored_pid.txt")
            if not os.path.exists(pidfile_path):
                raise RuntimeError(f"PID file {pidfile_path} does not exist")
            with open(pidfile_path, "r", encoding="utf-8") as f:
                pid_val = int(f.read().strip())
                restored_map_from_files[worker.rank] = pid_val
                logger.info("Loaded restored PID %d for worker %d from pidfile", pid_val, worker.rank)
            
            # Need to remove pidfile, otherwise subsequent restores will fail
            os.unlink(pidfile_path)
            logger.info("Removed pidfile %s", pidfile_path)
        
        # Combine and validate: prefer worker-reported PIDs when mismatch
        combined_pids: dict[int, int] = {}
        parent_pid = os.getpid()
        for worker in self.workers:
            file_pid = restored_map_from_files.get(worker.rank)
            msg_pid = received_pids.get(worker.rank) if 'received_pids' in locals() else None
            chosen: Optional[int] = None
            if file_pid is not None and msg_pid is not None:
                if file_pid != msg_pid:
                    logger.warning("PID mismatch for worker %d: pidfile=%d, reported=%d", worker.rank, file_pid, msg_pid)
                    chosen = msg_pid
                else:
                    chosen = file_pid
            elif file_pid is not None:
                chosen = file_pid
            elif msg_pid is not None:
                chosen = msg_pid
            if chosen is not None and chosen not in (0, 1, parent_pid):
                combined_pids[worker.rank] = chosen
        
        if combined_pids:
            self._restored_worker_pids = combined_pids
        
        # Wait for message queues to be ready
        logger.info("Waiting for rpc_broadcast_mq to be ready...")
        self.rpc_broadcast_mq.wait_until_ready()
        logger.info("rpc_broadcast_mq is ready")
        
        for worker in self.workers:
            logger.info("Waiting for worker %d response MQ to be ready...", worker.rank)
            worker.worker_response_mq.wait_until_ready()
            logger.info("Worker %d response MQ is ready", worker.rank)
            
            # Now receive the actual READY message from worker via MessageQueue
            logger.info("Waiting for READY message from worker %d...", worker.rank)
            status, response = worker.worker_response_mq.dequeue()
            if status != WorkerProc.ResponseStatus.SUCCESS:
                raise RuntimeError(f"Worker {worker.rank} failed to initialize: {response}")
            
            logger.info(f"Worker {worker.rank} is ready")
        
        # Continue with normal executor initialization
        self._setup_post_restore()
        
        logger.info("Post-restore initialization complete")
    
    def _start_restored_worker_monitor(self):
        """Start monitoring restored workers using PID-based approach."""
        logger.info("Starting restored worker monitor...")
        
        # Use PIDs obtained from pidfiles/worker reports
        if not self._restored_worker_pids:
            raise RuntimeError("No restored worker PIDs found")
        
        # Start monitoring thread
        self_ref = weakref.ref(self)
        
        def monitor_restored_workers():
            while True:
                _self = self_ref()
                if not _self or getattr(_self, 'shutting_down', False):
                    return
                
                # Check each worker PID
                for rank, pid in list(_self._restored_worker_pids.items()):
                    try:
                        # Check if process exists
                        os.kill(pid, 0)
                    except ProcessLookupError:
                        # Process died
                        logger.error(f"Restored worker {rank} (PID {pid}) died unexpectedly!")
                        _self.is_failed = True
                        
                        # Kill all other workers
                        for other_rank, other_pid in _self._restored_worker_pids.items():
                            if other_rank != rank:
                                try:
                                    os.kill(other_pid, signal.SIGKILL)
                                    logger.info(f"Killed worker {other_rank} (PID {other_pid})")
                                except Exception:
                                    pass
                        
                        # Invoke failure callback
                        _self.shutdown()
                        callback = _self.failure_callback
                        if callback is not None:
                            _self.failure_callback = None
                            callback()
                        return
                
                # Check every 0.1 seconds
                time.sleep(0.1)
        
        monitor_thread = threading.Thread(
            target=monitor_restored_workers,
            daemon=True,
            name="RestoredWorkerMonitor"
        )
        monitor_thread.start()
        logger.info("Restored worker monitor started")
    
    def shutdown(self):
        """Shutdown the executor, handling restored workers specially."""
        # Mark as shutting down first so monitors exit promptly and avoid race
        self.shutting_down = True
        self.shutdown_event.set()
        # Stop any PTY forwarders and close FDs
        if hasattr(self, '_pty_forwarders'):
            for t, fd in list(getattr(self, '_pty_forwarders', [])):
                with contextlib.suppress(Exception):
                    os.close(fd)
            self._pty_forwarders = []
        # Kill restored workers if we have their PIDs
        if hasattr(self, '_restored_worker_pids'):
            for rank, pid in self._restored_worker_pids.items():
                try:
                    os.kill(pid, signal.SIGTERM)
                    logger.info(f"Sent SIGTERM to restored worker {rank} (PID {pid})")
                except ProcessLookupError:
                    pass  # Already dead
                except Exception as e:
                    logger.warning(f"Failed to terminate restored worker {rank}: {e}")
            
            # Give them time to exit gracefully
            time.sleep(1)
            
            # Force kill any remaining
            for rank, pid in self._restored_worker_pids.items():
                try:
                    os.kill(pid, signal.SIGKILL)
                    logger.info(f"Force killed restored worker {rank} (PID {pid})")
                except ProcessLookupError:
                    pass  # Already dead
                except Exception as e:
                    logger.debug(f"Failed to kill restored worker {rank}: {e}")
        
        # For restored workers, additional cleanup
        if hasattr(self, '_restored_worker_pids'):
            
            # Clean up io_thread_pool if exists
            if self.io_thread_pool is not None:
                self.io_thread_pool.shutdown(wait=False, cancel_futures=True)
                self.io_thread_pool = None
            
            # Clean up message queues
            for worker in getattr(self, 'workers', []):
                worker.worker_response_mq = None
            self.rpc_broadcast_mq = None
        else:
            # Call parent's shutdown for normal cases
            super().shutdown()

    # ----- Internal helpers -----
    def _start_pty_forwarder(self, master_fd: int) -> None:
        """Forward data from PTY master fd to this process' stdout.

        This mirrors the restored workers' stdout/stderr back into
        the terminal running the parent (EngineCore/APIServer).
        """
        import threading
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
                        sys.stdout.buffer.write(data)
                        sys.stdout.buffer.flush()
                    except Exception:
                        pass
            except Exception as e:
                logger.debug("Worker PTY forwarder stopped: %s", e)
            finally:
                with contextlib.suppress(Exception):
                    os.close(fd)

        t = threading.Thread(target=_forward_loop,
                             args=(master_fd,),
                             name=f"WorkerPTYForwarder-{master_fd}",
                             daemon=True)
        t.start()
        # Keep track so we can close on shutdown
        if not hasattr(self, "_pty_forwarders"):
            self._pty_forwarders = []
        self._pty_forwarders.append((t, master_fd))
