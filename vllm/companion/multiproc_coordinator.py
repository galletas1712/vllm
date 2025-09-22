# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""Simple coordinator for companion server/client communication."""

import asyncio
import time
import multiprocessing as mp
import pickle
import signal
from dataclasses import dataclass
from typing import Optional

import zmq
import zmq.asyncio

from vllm.companion.utils import get_free_port
import torch
from vllm.companion.messages import (
    RequestType,
    ResponseType,
    CompanionState,
    GetCompanionStatusResponse,
    HandshakeRequest,
    HandshakeResponse,
    LoadModelResponse,
    ModelParametersRebuildInfoResponse,
)
from vllm.logger import init_logger
from vllm.utils import make_zmq_socket

logger = init_logger(__name__)


@dataclass  
class CompanionProcess:
    """Information about a companion server process."""
    device_id: int
    process: mp.Process
    data_port: int      # Port for model parameter requests
    status_port: int    # Port for status queries


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
        
        # Assign ports for this companion
        data_port = get_free_port()
        status_port = get_free_port()
        
        # Start the companion process
        from vllm.companion.multiproc_companion_server import (
            run_companion_server)
        
        # Use spawn context for CUDA compatibility
        ctx = mp.get_context('spawn')
        process = ctx.Process(
            target=run_companion_server,
            args=(device_id, data_port, status_port, self.companion_master_port),
            name=f"Companion-GPU{device_id}"
        )
        process.start()
        
        companion = CompanionProcess(
            device_id=device_id,
            process=process,
            data_port=data_port,
            status_port=status_port
        )
        
        self.companions[device_id] = companion
        logger.info("Started companion for GPU %d on ports %d/%d (PID: %d)",
                   device_id, data_port, status_port, process.pid)
        
        return companion
    
    async def _handshake_with_companion(self, companion: CompanionProcess) -> bool:
        """Perform handshake with a companion server to ensure it's ready.
        
        Returns:
            True if handshake successful, False otherwise
        """
        max_retries = 30  # 30 seconds total
        retry_delay = 1.0  # 1 second between retries
        
        for attempt in range(max_retries):
            try:
                # Create temporary connection to companion's status port
                req_socket = make_zmq_socket(
                    self.context,
                    f"tcp://127.0.0.1:{companion.status_port}",
                    zmq.REQ,
                    bind=False
                )
                req_socket.setsockopt(zmq.RCVTIMEO, 1000)  # 1 second timeout
                req_socket.setsockopt(zmq.SNDTIMEO, 1000)  # 1 second timeout
                
                # Send handshake request
                handshake_req = HandshakeRequest(device_id=companion.device_id)
                await req_socket.send(pickle.dumps(handshake_req))
                
                # Wait for response
                response_data = await req_socket.recv()
                response = pickle.loads(response_data)
                
                req_socket.close()
                
                # Check if it's a valid handshake response
                if (hasattr(response, 'response_type') and 
                    response.response_type == ResponseType.HANDSHAKE and
                    response.success):
                    logger.info("Handshake successful with companion for GPU %d", 
                               companion.device_id)
                    return True
                else:
                    logger.warning("Invalid handshake response from companion GPU %d: %s",
                                 companion.device_id, response)
                    
            except zmq.error.Again:
                logger.error("Handshake timeout for companion GPU %d (attempt %d/%d)",
                           companion.device_id, attempt + 1, max_retries)
            except Exception as e:
                logger.error("Handshake error for companion GPU %d: %s",
                           companion.device_id, e)
            
            # Check if process is still alive
            if not companion.process.is_alive():
                logger.error("Companion process for GPU %d died during handshake",
                           companion.device_id)
                return False
            
            await asyncio.sleep(retry_delay)
        
        logger.error("Failed to handshake with companion for GPU %d after %d attempts",
                   companion.device_id, max_retries)
        return False
    
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
        
        try:
            logger.debug(
                "[COORDINATOR] Received client request (bytes=%d)",
                len(message))
            # Decode the request
            request = pickle.loads(message)
            
            # Get request type
            request_type = getattr(request, 'request_type', None)
            if request_type is None:
                logger.error("Request missing request_type field: %s", type(request))
                error_response = pickle.dumps(ModelParametersRebuildInfoResponse(
                    success=False,
                    model_parameters=None,
                    error="Request missing request_type field",
                    response_type=ResponseType.MODEL_PARAMETERS_REBUILD_INFO
                ))
                await self.router_socket.send_multipart(
                    [client_id, b'', error_response])
                return
            
            # Handle requests based on type
            if request_type == RequestType.HANDSHAKE:
                # Check if this is a coordinator handshake or companion handshake
                device_id = getattr(request, 'device_id', None)
                if device_id is None:
                    # Coordinator handshake (no device_id)
                    logger.info("[COORDINATOR] Processing coordinator HANDSHAKE request")
                    response = HandshakeResponse(success=True, message="Coordinator ready")
                    response_data = pickle.dumps(response)
                    logger.info("[COORDINATOR] Sending HANDSHAKE response (bytes=%d)", len(response_data))
                    # For REQ clients, ROUTER must send: [identity, empty, data]
                    await self.router_socket.send_multipart(
                        [client_id, b'', response_data])
                    logger.info("[COORDINATOR] HANDSHAKE response sent")
                else:
                    # Companion handshake - forward to companion's status port
                    logger.info("[COORDINATOR] Forwarding HANDSHAKE request to companion device %d", device_id)
                    await self._forward_request(client_id, message, device_id, use_status_port=True)
                
            elif request_type == RequestType.GET_COMPANION_STATUS:
                # Forward to companion's status port synchronously
                await self._forward_request(client_id, message, request.device_id, use_status_port=True)
                
            elif request_type in (RequestType.LOAD_MODEL, RequestType.GET_MODEL_PARAMETERS_REBUILD_INFO):
                # Forward to companion's data port synchronously
                await self._forward_request(client_id, message, request.device_id, use_status_port=False)
                
            else:
                # Unknown request type
                logger.error("Unknown request type: %s", request_type)
                error_response = pickle.dumps(ModelParametersRebuildInfoResponse(
                    success=False,
                    model_parameters=None,
                    error=f"Unknown request type: {request_type}",
                    response_type=ResponseType.MODEL_PARAMETERS_REBUILD_INFO
                ))
                await self.router_socket.send_multipart(
                    [client_id, b'', error_response])
            
        except Exception as e:
            # Failed to decode request
            logger.error("Failed to decode request: %s", e)
            error_response = pickle.dumps(ModelParametersRebuildInfoResponse(
                success=False,
                model_parameters=None,
                error=f"Invalid request format: {e}",
                response_type=ResponseType.MODEL_PARAMETERS_REBUILD_INFO
            ))
            await self.router_socket.send_multipart(
                [client_id, error_response])
    

    
    async def _forward_request(self, client_id: bytes, message: bytes,
                               device_id: int, use_status_port: bool = False):
        """Forward a request to a companion synchronously.
        
        Args:
            client_id: ZMQ client identity
            message: Serialized request message
            device_id: GPU device ID
            use_status_port: If True, use status port; otherwise use data port
        """
        t0 = time.perf_counter()
        
        companion = self.companions.get(device_id)
        if not companion or not companion.process.is_alive():
            # Send appropriate error response based on request type
            import pickle
            request = pickle.loads(message)
            request_type = getattr(request, 'request_type', None)
            
            if request_type == RequestType.HANDSHAKE:
                error_response = pickle.dumps(HandshakeResponse(
                    success=False,
                    message=f"Companion for GPU {device_id} not available",
                    response_type=ResponseType.HANDSHAKE
                ))
            elif request_type == RequestType.GET_COMPANION_STATUS:
                error_response = pickle.dumps(GetCompanionStatusResponse(
                    device_id=device_id,
                    state=CompanionState.ERROR,
                    error_message=f"Companion for GPU {device_id} not available",
                    response_type=ResponseType.COMPANION_STATUS
                ))
            elif request_type == RequestType.LOAD_MODEL:
                error_response = pickle.dumps(LoadModelResponse(
                    success=False,
                    error=f"Companion for GPU {device_id} not available",
                    response_type=ResponseType.LOAD_MODEL
                ))
            elif request_type == RequestType.GET_MODEL_PARAMETERS_REBUILD_INFO:
                error_response = pickle.dumps(ModelParametersRebuildInfoResponse(
                    success=False,
                    model_parameters=None,
                    error=f"Companion for GPU {device_id} not available",
                    response_type=ResponseType.MODEL_PARAMETERS_REBUILD_INFO
                ))
            else:
                # Fallback for unknown request types
                error_response = pickle.dumps(ModelParametersRebuildInfoResponse(
                    success=False,
                    model_parameters=None,
                    error=f"Companion for GPU {device_id} not available",
                    response_type=ResponseType.MODEL_PARAMETERS_REBUILD_INFO
                ))
            
            await self.router_socket.send_multipart(
                [client_id, error_response])
            return
        
        # Forward to companion and get response
        try:
            # Create temporary connection to companion
            port = companion.status_port if use_status_port else companion.data_port
            req_socket = make_zmq_socket(
                self.context,
                f"tcp://127.0.0.1:{port}",
                zmq.REQ,
                bind=False
            )
            # Add timeouts - model loading already happened before requests arrive
            req_socket.setsockopt(zmq.RCVTIMEO, 10000)   # 10 seconds
            req_socket.setsockopt(zmq.SNDTIMEO, 5000)    # 5 seconds
            
            await req_socket.send(message)
            logger.debug("[COORDINATOR] Forwarded request to companion gpu=%d port=%d (bytes=%d)",
                        device_id, port, len(message))
            response = await req_socket.recv()
            t1 = time.perf_counter()
            logger.debug("[COORDINATOR] Received response from companion gpu=%d in %.3fs (bytes=%d)",
                        device_id, t1 - t0, len(response))
            
            req_socket.close()
            
            # Send response back to client with empty delimiter for REQ clients
            await self.router_socket.send_multipart(
                [client_id, b'', response])
            
        except Exception as e:
            logger.error("Failed to forward request: %s", e)
            import pickle
            error_response = pickle.dumps(ModelParametersRebuildInfoResponse(
                success=False,
                model_parameters=None,
                error=str(e),
                response_type=ResponseType.MODEL_PARAMETERS_REBUILD_INFO
            ))
            await self.router_socket.send_multipart(
                [client_id, error_response])
    
    async def start(self):
        """Start the coordinator service."""
        logger.info("Starting coordinator...")
        
        # Initialize ZMQ
        self.context = zmq.asyncio.Context()
        self.router_socket = make_zmq_socket(
            self.context,
            f"tcp://*:{self.coordinator_port}",
            zmq.ROUTER,
            bind=True
        )
        
        # Start companions for available GPUs eagerly
        if torch.cuda.is_available():
            for device_id in range(torch.cuda.device_count()):
                self.start_companion(device_id)
        
        # Perform handshakes with all companions
        logger.info("Performing handshakes with %d companion servers...", 
                   len(self.companions))
        handshake_tasks = []
        for companion in self.companions.values():
            handshake_tasks.append(self._handshake_with_companion(companion))
        
        handshake_results = await asyncio.gather(*handshake_tasks)
        
        # Check if all handshakes succeeded
        failed_companions = []
        for i, (device_id, success) in enumerate(zip(self.companions.keys(), handshake_results)):
            if not success:
                failed_companions.append(device_id)
        
        if failed_companions:
            logger.error("Failed to handshake with companions for GPUs: %s", failed_companions)
            # Clean up failed companions
            for device_id in failed_companions:
                comp = self.companions.pop(device_id, None)
                if comp and comp.process.is_alive():
                    comp.process.terminate()
                    comp.process.join(timeout=5)
        
        logger.info("Coordinator ready on port %d with %d companion servers", 
                   self.coordinator_port, len(self.companions))
        
        # Main message loop
        while not self.shutdown_event.is_set():
            try:
                if await self.router_socket.poll(timeout=1000):
                    frames = await self.router_socket.recv_multipart()
                    # ROUTER can receive different frame counts depending on client type:
                    # - REQ clients: [identity, payload] (2 frames)
                    # - DEALER clients: [identity, empty, payload] (3 frames)
                    # - Some ZMQ versions may add extra frames
                    if len(frames) == 2:
                        # Standard REQ client
                        await self.handle_client_request(frames[0], frames[1])
                    elif len(frames) == 3:
                        # DEALER client or REQ with delimiter - use last frame as payload
                        await self.handle_client_request(frames[0], frames[2])
                    else:
                        logger.warning("Unexpected frame count: %d, frames: %s", 
                                     len(frames), [len(f) for f in frames])
                        
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