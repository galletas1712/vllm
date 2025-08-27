# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""Simple coordinator for companion server/client communication."""

import asyncio
import multiprocessing as mp
import signal
from dataclasses import dataclass
from typing import Optional

import torch
import zmq
import zmq.asyncio

from vllm.companion.utils import get_free_port
from vllm.logger import init_logger

logger = init_logger(__name__)


@dataclass  
class CompanionProcess:
    """Information about a companion server process."""
    device_id: int
    process: mp.Process
    port: int


class MultiProcCoordinator:
    """
    Simple coordinator that manages companion servers.
    
    - Starts companion processes for each GPU
    - Routes client requests to appropriate companion
    - Monitors process health and restarts if needed
    """
    
    def __init__(self, coordinator_port: int, companion_master_port: int):
        self.coordinator_port = coordinator_port
        self.companion_master_port = companion_master_port
        self.companions: dict[int, CompanionProcess] = {}
        
        # ZMQ context and router socket for client requests
        self.context: Optional[zmq.asyncio.Context] = None
        self.router_socket: Optional[zmq.asyncio.Socket] = None
        
        # For coordinating distributed initialization
        self.pending_init_requests: dict = {}  # device_id -> (client_id, message)
        self.expected_companions = 0
        self.companions_ready = asyncio.Event()  # Set when all companions are ready
        
        # Shutdown flags
        self.shutdown_event = asyncio.Event()
        self.shutdown_called = False
        
        logger.info("Coordinator initialized on port %d", coordinator_port)
    
    def start_companion(self, device_id: int) -> CompanionProcess:
        """Start a companion server for a specific GPU."""
        if device_id in self.companions:
            comp = self.companions[device_id]
            if comp.process.is_alive():
                return comp
            # Process died, restart it
            logger.info("Restarting dead companion for GPU %d", device_id)
        
        # Assign a port for this companion
        companion_port = get_free_port()
        
        # Start the companion process
        from vllm.companion.multiproc_companion_server import (
            run_companion_server)
        
        # Use spawn context for CUDA compatibility
        ctx = mp.get_context('spawn')
        process = ctx.Process(
            target=run_companion_server,
            args=(device_id, companion_port, self.companion_master_port),
            name=f"Companion-GPU{device_id}"
        )
        process.start()
        
        companion = CompanionProcess(
            device_id=device_id,
            process=process,
            port=companion_port
        )
        
        self.companions[device_id] = companion
        logger.info("Started companion for GPU %d on port %d (PID: %d)",
                   device_id, companion_port, process.pid)
        
        return companion
    
    async def handle_client_request(self, client_id: bytes, message: bytes):
        """Route client request to appropriate companion.

        Note: For DP/EP, requests may carry a global world_size across DP to
        enable correct rank mapping on the companion side. We no longer use
        world_size to gate/batch local forwarding because a single local
        coordinator should not wait for remote ranks it cannot observe. Gloo
        rendezvous in the companion processes provides the necessary global
        synchronization.
        """
        import pickle
        
        # Decode the request to get device_id
        request = pickle.loads(message)
        device_id = request.device_id
        
        # Always forward immediately. Global sync happens inside companions.
        await self._forward_single_request(client_id, message, device_id)
    
    async def _broadcast_init_requests(self):
        """Broadcast all pending init requests to companions simultaneously."""
        tasks = []
        for device_id, (client_id, message) in self.pending_init_requests.items():
            task = asyncio.create_task(
                self._forward_single_request(client_id, message, device_id)
            )
            tasks.append(task)
        
        # Wait for all to complete
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        # Check if all succeeded
        for result in results:
            if isinstance(result, Exception):
                logger.error("Companion initialization failed: %s", result)
        
        # Mark companions as ready after first successful broadcast
        if not self.companions_ready.is_set():
            self.companions_ready.set()
            logger.info("All companions initialized and ready")
        
        # Clear pending requests
        self.pending_init_requests.clear()
    
    async def _forward_single_request(self, client_id: bytes, message: bytes, device_id: int):
        """Forward a single request to a companion."""
        # Ensure companion is running
        if device_id not in self.companions:
            self.start_companion(device_id)
            await asyncio.sleep(0.5)  # Give it time to start
        
        companion = self.companions.get(device_id)
        if not companion or not companion.process.is_alive():
            # Send error response
            import pickle
            error_response = pickle.dumps({
                'success': False,
                'error': f"Companion for GPU {device_id} not available"
            })
            await self.router_socket.send_multipart([
                client_id, b"", error_response
            ])
            return
        
        # Forward to companion and get response
        try:
            # Create temporary connection to companion
            req_socket = self.context.socket(zmq.REQ)
            req_socket.connect(f"tcp://127.0.0.1:{companion.port}")
            
            await req_socket.send(message)
            response = await req_socket.recv()
            
            req_socket.close()
            
            # Send response back to client
            await self.router_socket.send_multipart([
                client_id, b"", response
            ])
            
        except Exception as e:
            logger.error("Failed to forward request: %s", e)
            import pickle
            error_response = pickle.dumps({
                'success': False,
                'error': str(e)
            })
            await self.router_socket.send_multipart([
                client_id, b"", error_response
            ])
    
    async def start(self):
        """Start the coordinator service."""
        logger.info("Starting coordinator...")
        
        # Initialize ZMQ
        self.context = zmq.asyncio.Context()
        self.router_socket = self.context.socket(zmq.ROUTER)
        self.router_socket.bind(f"tcp://*:{self.coordinator_port}")
        
        # Start companions for available GPUs
        if torch.cuda.is_available():
            for device_id in range(torch.cuda.device_count()):
                self.start_companion(device_id)
        
        logger.info("Coordinator ready on port %d", self.coordinator_port)
        
        # Main message loop
        while not self.shutdown_event.is_set():
            try:
                if await self.router_socket.poll(timeout=1000):
                    frames = await self.router_socket.recv_multipart()
                    if len(frames) >= 3:
                        # Handle request (will coordinate distributed init if needed)
                        await self.handle_client_request(frames[0], frames[2])
                
                # Check companion health periodically
                for device_id, comp in list(self.companions.items()):
                    if not comp.process.is_alive():
                        logger.warning("Companion for GPU %d died", device_id)
                        del self.companions[device_id]
                        self.start_companion(device_id)
                        
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error("Error in coordinator loop: %s", e)
    
    async def shutdown(self):
        """Shutdown the coordinator and all companions."""
        if self.shutdown_called:
            return  # Already shutting down
        self.shutdown_called = True
        
        logger.info("Shutting down coordinator...")
        self.shutdown_event.set()
        
        # Terminate all companion processes
        for comp in self.companions.values():
            if comp.process.is_alive():
                comp.process.terminate()
                comp.process.join(timeout=5)
                if comp.process.is_alive():
                    comp.process.kill()
                    comp.process.join()
        
        # Clean up ZMQ
        if self.router_socket:
            self.router_socket.close()
            self.router_socket = None
        if self.context:
            self.context.term()
            self.context = None
        
        logger.info("Coordinator shutdown complete")


def run_coordinator(port: int, companion_master_port: int):
    """Run the coordinator in a process.
    
    Args:
        port: Port for the coordinator to listen on
        companion_master_port: Master port for CPU group initialization
    """
    asyncio.set_event_loop(asyncio.new_event_loop())
    loop = asyncio.get_event_loop()
    
    coordinator = MultiProcCoordinator(port, companion_master_port)
    
    def signal_handler(sig, frame):
        loop.create_task(coordinator.shutdown())
    
    signal.signal(signal.SIGTERM, signal_handler)
    signal.signal(signal.SIGINT, signal_handler)
    
    try:
        loop.run_until_complete(coordinator.start())
    except KeyboardInterrupt:
        pass
    finally:
        loop.run_until_complete(coordinator.shutdown())
        loop.close()