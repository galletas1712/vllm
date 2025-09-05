# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""Checkpoint/Resume coordination utilities for vLLM engine."""

import zmq
from typing import Optional
from vllm.utils import make_zmq_socket, get_open_zmq_ipc_path
from vllm.logger import init_logger

logger = init_logger(__name__)


class CheckpointCoordinator:
    """Handles checkpoint/resume coordination via side-channel socket."""
    
    def __init__(self, role: str):
        """
        Args:
            role: Either 'client' (MPClient) or 'engine' (EngineCoreProc)
        """
        self.role = role
        self.ctx = zmq.Context()
        self.socket: Optional[zmq.Socket] = None
        self.address: Optional[str] = None
        
    def initialize(self, address: Optional[str] = None) -> str:
        """Initialize the side-channel socket.
        
        Args:
            address: Socket address. If None, creates a new IPC path (client only)
            
        Returns:
            The socket address being used
        """
        if self.role == "client" and address is None:
            # Client creates the socket
            self.address = get_open_zmq_ipc_path()
            self.socket = make_zmq_socket(
                self.ctx, self.address, zmq.ROUTER, bind=True
            )
        elif self.role == "engine":
            # Engine connects to the socket
            assert address is not None, "Engine must have address to connect to"
            self.address = address
            self.socket = make_zmq_socket(
                self.ctx, self.address, zmq.DEALER, bind=False
            )
        else:
            raise ValueError(f"Invalid role: {self.role}")
            
        logger.info("Checkpoint coordinator (%s) initialized on %s",
                    self.role, self.address)
        return self.address
        
    def send_checkpointed(self, engine_id: bytes, config_hash: str) -> None:
        """Send checkpointed signal from engine to client with config hash."""
        assert self.role == "engine" and self.socket is not None
        self.socket.send_multipart(
            [b"CHECKPOINTED", engine_id, config_hash.encode()])
        
    def wait_for_checkpointed(self, engine_id: bytes) -> str:
        """Wait for checkpointed signal from engine and return config hash."""
        assert self.role == "client" and self.socket is not None
        while True:
            identity, msg_type, eng_id, config_hash = self.socket.recv_multipart()
            if msg_type == b"CHECKPOINTED" and eng_id == engine_id:
                logger.info("Received checkpointed signal from engine %d",
                            int.from_bytes(engine_id, 'little'))
                return config_hash.decode()
                
    def send_resume(self, engine_id: bytes, config_hash: str) -> None:
        """Send resume signal from client to engine with config hash."""
        assert self.role == "client" and self.socket is not None
        self.socket.send_multipart([engine_id, b"RESUME", config_hash.encode()])
        
    def wait_for_resume(self) -> str:
        """Wait for resume signal from client and return config hash."""
        assert self.role == "engine" and self.socket is not None
        while True:
            msg_type, config_hash = self.socket.recv_multipart()
            if msg_type == b"RESUME":
                logger.info("Received resume signal")
                return config_hash.decode()
                
    def close(self) -> None:
        """Close the socket and context."""
        if self.socket is not None:
            self.socket.close(linger=0)
            self.socket = None
        self.ctx.term()
        logger.info("Checkpoint coordinator (%s) closed", self.role)
