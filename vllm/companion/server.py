# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""ZMQ REQ/REP server for companion process."""

import signal
import sys
from typing import Optional

import zmq

from vllm.logger import init_logger

from .handler import CompanionHandler

logger = init_logger(__name__)


class CompanionServer:
    """
    ZMQ REQ/REP server for serving model weights via CUDA IPC.

    This server listens for load_model requests from vLLM workers and
    returns CUDA IPC handles that allow zero-copy weight sharing.
    """

    def __init__(
        self,
        device_id: int,
        port: int,
        companion_master_port: int = 29700,
    ):
        """
        Initialize the companion server.

        Args:
            device_id: GPU device ID this server manages
            port: ZMQ port to listen on
            companion_master_port: Master port for distributed initialization
        """
        self.device_id = device_id
        self.port = port
        self.companion_master_port = companion_master_port

        self.handler = CompanionHandler(
            device_id=device_id,
            companion_master_port=companion_master_port,
        )

        self._context: Optional[zmq.Context] = None
        self._socket: Optional[zmq.Socket] = None
        self._running = False

    def _setup_signal_handlers(self):
        """Set up graceful shutdown handlers."""

        def signal_handler(signum, frame):
            logger.info(f"Received signal {signum}, shutting down...")
            self._running = False

        signal.signal(signal.SIGTERM, signal_handler)
        signal.signal(signal.SIGINT, signal_handler)

    def run(self):
        """
        Start the server and process requests.

        This method blocks until the server is shut down via signal.
        """
        self._setup_signal_handlers()
        self._running = True

        self._context = zmq.Context()
        self._socket = self._context.socket(zmq.REP)

        # Set socket options for robustness
        self._socket.setsockopt(zmq.LINGER, 0)  # Don't wait on close

        bind_address = f"tcp://*:{self.port}"
        self._socket.bind(bind_address)

        logger.info(
            f"Companion server started on port {self.port} "
            f"for GPU {self.device_id}"
        )

        try:
            while self._running:
                # Use poll to allow checking _running flag
                if self._socket.poll(timeout=1000):  # 1 second timeout
                    request = self._socket.recv_string()
                    logger.debug(f"Received request: {request[:100]}...")

                    response = self.handler.handle_request(request)

                    self._socket.send_string(response)
                    logger.debug("Sent response")
        except zmq.ZMQError as e:
            if self._running:  # Only log if not shutting down
                logger.exception(f"ZMQ error: {e}")
        finally:
            self._cleanup()

    def _cleanup(self):
        """Clean up ZMQ resources."""
        logger.info("Cleaning up companion server...")
        if self._socket:
            self._socket.close()
            self._socket = None
        if self._context:
            self._context.term()
            self._context = None
        logger.info("Companion server stopped")


def main():
    """Entry point for running companion server directly."""
    import argparse

    parser = argparse.ArgumentParser(
        description="vLLM Companion Server for CUDA IPC weight sharing"
    )
    parser.add_argument(
        "--device-id",
        type=int,
        required=True,
        help="GPU device ID this server manages",
    )
    parser.add_argument(
        "--port",
        type=int,
        default=5555,
        help="ZMQ port to listen on (default: 5555)",
    )
    parser.add_argument(
        "--companion-master-port",
        type=int,
        default=29700,
        help="Master port for distributed initialization (default: 29700)",
    )

    args = parser.parse_args()

    server = CompanionServer(
        device_id=args.device_id,
        port=args.port,
        companion_master_port=args.companion_master_port,
    )

    try:
        server.run()
    except KeyboardInterrupt:
        logger.info("Interrupted by user")
        sys.exit(0)


if __name__ == "__main__":
    main()
