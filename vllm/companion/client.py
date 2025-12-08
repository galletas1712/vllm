# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""ZMQ client for loading model weights from companion server."""

import base64
import json
from typing import Optional

import cloudpickle
import torch
import zmq

from vllm.config import VllmConfig
from vllm.logger import init_logger

from .ipc_utils import ModuleTreeNode, import_weights_from_tree

logger = init_logger(__name__)


class CompanionClient:
    """
    ZMQ client for loading model weights from companion server via CUDA IPC.

    This client connects to a companion server and retrieves CUDA IPC handles
    that allow zero-copy weight sharing.
    """

    def __init__(
        self,
        host: str = "localhost",
        port: int = 5555,
        timeout_ms: int = 600000,  # 10 minutes default
    ):
        """
        Initialize the companion client.

        Args:
            host: Companion server hostname
            port: Companion server port
            timeout_ms: Receive timeout in milliseconds
        """
        self.host = host
        self.port = port
        self.timeout_ms = timeout_ms

        self._context: Optional[zmq.Context] = None
        self._socket: Optional[zmq.Socket] = None

    def connect(self):
        """Establish connection to the companion server."""
        if self._socket is not None:
            return  # Already connected

        self._context = zmq.Context()
        self._socket = self._context.socket(zmq.REQ)

        # Set socket options
        self._socket.setsockopt(zmq.RCVTIMEO, self.timeout_ms)
        self._socket.setsockopt(zmq.SNDTIMEO, self.timeout_ms)
        self._socket.setsockopt(zmq.LINGER, 0)

        connect_address = f"tcp://{self.host}:{self.port}"
        self._socket.connect(connect_address)

        logger.info(f"Connected to companion server at {connect_address}")

    def close(self):
        """Close the connection."""
        if self._socket:
            self._socket.close()
            self._socket = None
        if self._context:
            self._context.term()
            self._context = None

    def load_model(
        self,
        target_model: torch.nn.Module,
        vllm_config: VllmConfig,
        local_rank: int,
        global_rank: int,
        world_size: int,
    ) -> None:
        """
        Load model weights from companion server into target model.

        Args:
            target_model: The model to load weights into (typically a meta model)
            vllm_config: vLLM configuration
            local_rank: Local rank of this worker
            global_rank: Global rank of this worker
            world_size: Total number of workers

        Raises:
            RuntimeError: If the companion server returns an error
            zmq.ZMQError: If there's a communication error
        """
        if self._socket is None:
            self.connect()

        # Prepare request
        request = {
            "config_pickled": base64.b64encode(
                cloudpickle.dumps(vllm_config)
            ).decode("utf-8"),
            "local_rank": local_rank,
            "global_rank": global_rank,
            "world_size": world_size,
        }

        logger.info(
            f"Sending load_model request to companion server "
            f"(local_rank={local_rank}, global_rank={global_rank})"
        )

        # Send request
        self._socket.send_string(json.dumps(request))

        # Receive response
        response_encoded = self._socket.recv_string()

        # Decode response
        response = cloudpickle.loads(base64.b64decode(response_encoded))

        if not response.get("success", False):
            error_msg = response.get("error", "Unknown error from companion server")
            raise RuntimeError(f"Companion server error: {error_msg}")

        logger.info(f"Companion server response: {response.get('message', 'OK')}")

        # Import weights from module tree
        module_tree_dict = response.get("module_tree")
        if module_tree_dict is None:
            raise RuntimeError("Companion server did not return module tree")

        module_tree = ModuleTreeNode.from_dict(module_tree_dict)
        import_weights_from_tree(target_model, module_tree)

        logger.info("Successfully imported weights from companion server")

    def __enter__(self):
        self.connect()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()
        return False
