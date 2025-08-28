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
from vllm.companion.messages import ModelParametersResponse
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
        self.companions_ready = asyncio.Event()  # Set when all companions are ready
        self.pending_init_requests: dict = {}  # (device_id, client_id) -> (client_id, message, needs_empty_delim)
        self.distributed_initialized = False
        
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
    
    async def _wait_for_companions_ready(self) -> None:
        """Wait for all companion servers to be ready."""
        if not self.companions:
            return
        
        max_wait = 30  # Maximum 30 seconds
        check_interval = 0.1  # Check every 100ms
        start_time = asyncio.get_event_loop().time()
        
        while asyncio.get_event_loop().time() - start_time < max_wait:
            all_ready = True
            for device_id, companion in self.companions.items():
                if not companion.process.is_alive():
                    logger.warning("Companion for GPU %d died during startup", device_id)
                    # Restart it
                    self.start_companion(device_id)
                    all_ready = False
                    continue
                    
                # Check if companion is responding to pings
                try:
                    req_socket = self.context.socket(zmq.REQ)
                    req_socket.setsockopt(zmq.RCVTIMEO, 100)  # 100ms timeout
                    req_socket.connect(f"tcp://127.0.0.1:{companion.port}")
                    
                    # Send a minimal ping request
                    import pickle
                    from vllm.companion.messages import GetModelParametersRequest
                    ping_request = GetModelParametersRequest(
                        vllm_config=None,  # Will be handled as ping
                        device_id=device_id,
                        local_rank=0,
                        global_rank=0,
                        world_size=1,
                        ping_only=True
                    )
                    await req_socket.send(pickle.dumps(ping_request))
                    await req_socket.recv()  # Just check if we get a response
                    req_socket.close()
                except Exception:
                    # Not ready yet
                    all_ready = False
                    req_socket.close()
            
            if all_ready:
                logger.info("All companion servers are ready")
                return
                
            await asyncio.sleep(check_interval)
        
        raise RuntimeError(f"Companion servers failed to start within {max_wait} seconds")
    
    async def handle_client_request(self, client_id: bytes, message: bytes,
                                   needs_empty_delim: bool):
        """Route client request to appropriate companion.

        Note: For DP/EP, requests may carry a global world_size across DP to
        enable correct rank mapping on the companion side. We no longer use
        world_size to gate/batch local forwarding because a single local
        coordinator should not wait for remote ranks it cannot observe. Gloo
        rendezvous in the companion processes provides the necessary global
        synchronization.
        """
        import pickle
        
        try:
            # Decode the request to get device_id
            request = pickle.loads(message)
            
            # Handle ping/health check messages
            if not hasattr(request, 'device_id'):
                # This is a health check from an empty dict, just respond with success
                response = pickle.dumps(ModelParametersResponse(
                    success=True,
                    model_parameters=None,
                    error='ping: pong'
                ))
                if needs_empty_delim:
                    await self.router_socket.send_multipart(
                        [client_id, b"", response])
                else:
                    await self.router_socket.send_multipart(
                        [client_id, response])
                return
            
            # Check if this is a ping request (vllm_config is None and ping_only is True)
            if hasattr(request, 'ping_only') and request.ping_only and request.vllm_config is None:
                # This is a proper ping request, respond immediately
                response = pickle.dumps(ModelParametersResponse(
                    success=True,
                    model_parameters=None,
                    error='ping: pong'
                ))
                if needs_empty_delim:
                    await self.router_socket.send_multipart(
                        [client_id, b"", response])
                else:
                    await self.router_socket.send_multipart(
                        [client_id, response])
                return
            
            device_id = request.device_id
        except Exception as e:
            # Failed to decode request
            logger.error("Failed to decode request: %s", e)
            error_response = pickle.dumps(ModelParametersResponse(
                success=False,
                model_parameters=None,
                error=f"Invalid request format: {e}"
            ))
            if needs_empty_delim:
                await self.router_socket.send_multipart(
                    [client_id, b"", error_response])
            else:
                await self.router_socket.send_multipart(
                    [client_id, error_response])
            return
        
        # Check if this is a model load request (has vllm_config and not a ping)
        if hasattr(request, 'vllm_config') and request.vllm_config is not None:
            # If distributed not initialized yet, this is an init request
            if not self.distributed_initialized:
                # Collect init request for broadcast
                request_key = (device_id, client_id)
                self.pending_init_requests[request_key] = (client_id, message, needs_empty_delim)
                
                # Get unique devices we've collected so far
                unique_devices = set(key[0] for key in self.pending_init_requests)
                expected_count = torch.cuda.device_count() if torch.cuda.is_available() else 1
                
                # When we have at least one request per device, broadcast init
                if len(unique_devices) >= expected_count:
                    logger.info("Broadcasting %d init requests to %d devices for GLOO rendezvous",
                               len(self.pending_init_requests), len(unique_devices))
                    await self._broadcast_init_requests()
                    self.distributed_initialized = True
                else:
                    logger.debug("Collected init request for device %d (%d/%d devices covered)",
                                device_id, len(unique_devices), expected_count)
            else:
                # Distributed already initialized, forward without blocking
                asyncio.create_task(
                    self._forward_single_request(client_id, message, device_id,
                                                 needs_empty_delim)
                )
        else:
            # Not a model load - forward without blocking (e.g., ping requests)
            asyncio.create_task(
                self._forward_single_request(client_id, message, device_id,
                                             needs_empty_delim)
            )
    

    async def _broadcast_init_requests(self):
        """Broadcast all pending init requests simultaneously.
        
        This ensures all companion servers enter GLOO rendezvous together
        for distributed initialization.
        """
        if not self.pending_init_requests:
            return
        
        # Create all sockets and send tasks first, then await them together
        # This ensures true simultaneous sending for GLOO rendezvous
        sockets = []
        client_infos = []
        send_tasks = []
        
        for (device_id, client_id), (cid, msg, needs_empty) in self.pending_init_requests.items():
            companion = self.companions.get(device_id)
            if not companion or not companion.process.is_alive():
                # Send error response to client
                import pickle
                error_response = pickle.dumps(ModelParametersResponse(
                    success=False,
                    model_parameters=None,
                    error=f"Companion for GPU {device_id} not available"
                ))
                if needs_empty:
                    await self.router_socket.send_multipart(
                        [cid, b"", error_response])
                else:
                    await self.router_socket.send_multipart(
                        [cid, error_response])
                continue
            
            # Use DEALER socket for async communication
            dealer_socket = self.context.socket(zmq.DEALER)
            dealer_socket.connect(f"tcp://127.0.0.1:{companion.port}")
            
            # Create send task but don't await yet
            # DEALER sockets need empty delimiter frame for REP socket compatibility
            send_task = dealer_socket.send_multipart([b"", msg])
            
            sockets.append(dealer_socket)
            client_infos.append((cid, needs_empty, device_id))
            send_tasks.append(send_task)
        
        # Now await all sends together - this ensures they happen simultaneously
        await asyncio.gather(*send_tasks)
        
        # Now collect all responses using async polling
        # This allows companions to respond in any order after GLOO completes
        poller = zmq.asyncio.Poller()
        for socket in sockets:
            poller.register(socket, zmq.POLLIN)
        
        responses_received = 0
        socket_to_info = dict(zip(sockets, client_infos))
        
        while responses_received < len(sockets):
            # Poll with timeout to detect stuck companions
            events = dict(await poller.poll(60000))  # 60 second timeout
            
            if not events:
                logger.error("Timeout waiting for companion responses during init broadcast")
                # Send error to remaining clients
                for socket in sockets:
                    if socket in socket_to_info:
                        client_id, needs_empty_delim, device_id = socket_to_info[socket]
                        import pickle
                        error_response = pickle.dumps(ModelParametersResponse(
                            success=False,
                            model_parameters=None,
                            error=f"Companion for GPU {device_id} timed out during init"
                        ))
                        if needs_empty_delim:
                            await self.router_socket.send_multipart(
                                [client_id, b"", error_response])
                        else:
                            await self.router_socket.send_multipart(
                                [client_id, error_response])
                break
            
            for socket in events:
                if events[socket] == zmq.POLLIN:
                    client_id, needs_empty_delim, device_id = socket_to_info.pop(socket)
                    try:
                        # DEALER receives [empty, response]
                        frames = await socket.recv_multipart()
                        response = frames[-1]  # Last frame is the actual response
                        
                        # Send response back to client
                        if needs_empty_delim:
                            await self.router_socket.send_multipart(
                                [client_id, b"", response])
                        else:
                            await self.router_socket.send_multipart(
                                [client_id, response])
                    except Exception as e:
                        logger.error("Failed to get response from companion %d: %s", device_id, e)
                        # Send error to client
                        import pickle
                        error_response = pickle.dumps(ModelParametersResponse(
                            success=False,
                            model_parameters=None,
                            error=str(e)
                        ))
                        if needs_empty_delim:
                            await self.router_socket.send_multipart(
                                [client_id, b"", error_response])
                        else:
                            await self.router_socket.send_multipart(
                                [client_id, error_response])
                    
                    poller.unregister(socket)
                    socket.close()
                    responses_received += 1
        
        # Close any remaining sockets
        for socket in sockets:
            socket.close()
        
        # Clear pending requests
        self.pending_init_requests.clear()
        logger.info("Distributed init broadcast complete")
    
    async def _forward_single_request(self, client_id: bytes, message: bytes,
                                      device_id: int,
                                      needs_empty_delim: bool):
        """Forward a single request to a companion."""
        # Ensure companion is running
        if device_id not in self.companions:
            self.start_companion(device_id)
        
        companion = self.companions.get(device_id)
        if not companion or not companion.process.is_alive():
            # Send error response
            import pickle
            error_response = pickle.dumps(ModelParametersResponse(
                success=False,
                model_parameters=None,
                error=f"Companion for GPU {device_id} not available"
            ))
            if needs_empty_delim:
                await self.router_socket.send_multipart(
                    [client_id, b"", error_response])
            else:
                await self.router_socket.send_multipart(
                    [client_id, error_response])
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
            if needs_empty_delim:
                await self.router_socket.send_multipart(
                    [client_id, b"", response])
            else:
                await self.router_socket.send_multipart(
                    [client_id, response])
            
        except Exception as e:
            logger.error("Failed to forward request: %s", e)
            import pickle
            error_response = pickle.dumps(ModelParametersResponse(
                success=False,
                model_parameters=None,
                error=str(e)
            ))
            if needs_empty_delim:
                await self.router_socket.send_multipart(
                    [client_id, b"", error_response])
            else:
                await self.router_socket.send_multipart(
                    [client_id, error_response])
    
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
        
        logger.info("Coordinator ready on port %d with %d companion servers", 
                   self.coordinator_port, len(self.companions))
        
        # Main message loop
        while not self.shutdown_event.is_set():
            try:
                if await self.router_socket.poll(timeout=1000):
                    frames = await self.router_socket.recv_multipart()
                    # ROUTER can receive either:
                    # - [identity, empty, payload] from DEALER-style clients
                    # - [identity, payload] from REQ-style clients (our validation ping)
                    if len(frames) >= 3:
                        await self.handle_client_request(frames[0], frames[2],
                                                         True)
                    elif len(frames) == 2:
                        await self.handle_client_request(frames[0], frames[1],
                                                         False)
                
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