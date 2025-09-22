# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""Simple companion server for CUDA IPC weight sharing."""

import asyncio
import contextlib
import pickle
import signal
from concurrent.futures import ThreadPoolExecutor
from typing import Optional

import torch
import zmq
import zmq.asyncio

from vllm.logger import init_logger
from vllm.companion.model_instance_manager import ModelInstanceManager
from vllm.companion.messages import (
    CompanionState,
    GetCompanionStatusResponse,
    RequestType,
    ResponseType,
    HandshakeResponse,
    LoadModelRequest,
    LoadModelResponse,
    ModelParametersRebuildInfoResponse,
)
from vllm.utils import make_zmq_socket

logger = init_logger(__name__)


class MultiProcCompanionServer:
    """
    Simple companion server that loads model weights and serves them via
    CUDA IPC. Uses the same ModelInstanceManager as Dynamo companion.
    """
    
    def __init__(self, device_id: int, data_port: int, status_port: int, 
                 companion_master_port: int):
        """
        Initialize the companion server.
        
        Args:
            device_id: Physical GPU device ID  
            data_port: Port for model parameter requests
            status_port: Port for status queries
            companion_master_port: Master port for CPU group initialization
        """
        self.device_id = device_id
        self.data_port = data_port
        self.status_port = status_port
        self.companion_master_port = companion_master_port
        
        # Model cache - only one model per companion server
        self.model_manager: Optional[ModelInstanceManager] = None
        self.model_hash: Optional[str] = None
        self.model_parameters: Optional[dict] = None
        
        # State tracking - only track model loading state
        self.state = CompanionState.INITIALIZING
        self.error_message: Optional[str] = None
        
        # ZMQ sockets
        self.context: Optional[zmq.asyncio.Context] = None
        self.data_socket: Optional[zmq.asyncio.Socket] = None
        self.status_socket: Optional[zmq.asyncio.Socket] = None
        
        # Shutdown flags
        self.shutdown_event = asyncio.Event()
        self.shutdown_called = False
        
        # Distributed initialization state
        self.distributed_initialized = False
        
        # Model loading state
        self.model_loading_lock = asyncio.Lock()
        self.model_loading_task: Optional[asyncio.Task] = None
        self.executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="CompanionLoader")
        
        # Set CUDA device
        torch.cuda.set_device(self.device_id)
        
        logger.info("Companion server for GPU %d on ports %d/%d",
                   device_id, data_port, status_port)
        
        # Server is ready to receive requests immediately
        # Note: Model loading happens asynchronously later
    

    
    def _load_model_sync(self, request: LoadModelRequest) -> Optional[dict]:
        """Load model weights for the given configuration (runs in thread pool).
        
        Returns:
            Dict of model parameters if successful, None if hash mismatch.
        """
        config_hash = request.compute_hash()
        
        # If model is already loaded
        if self.model_hash is not None:
            # Check if it's the same model
            if config_hash == self.model_hash:
                logger.debug("Returning cached model parameters for hash %s",
                            config_hash)
                return self.model_parameters
            else:
                # Different model requested - return None to signal mismatch
                logger.warning(
                    "Model hash mismatch: server has %s, request has %s",
                    self.model_hash, config_hash
                )
                return None  # Signal hash mismatch
        
        # First model load
        logger.info("Loading model %s for GPU %d (rank %d/%d)...",
                    request.vllm_config.model_config.model,
                    self.device_id,
                    request.global_rank, request.world_size)
        
        # Update state to LOADING
        self.state = CompanionState.LOADING
        
        try:
            # Get companion_master_port from vllm_config if available,
            # otherwise use server's default
            companion_master_port = getattr(
                request.vllm_config.load_config, 'companion_master_port',
                self.companion_master_port
            )
            
            # Create ModelInstanceManager
            self.model_manager = ModelInstanceManager(
                vllm_config=request.vllm_config,
                device_id=self.device_id,
                local_rank=request.local_rank,
                global_rank=request.global_rank,
                world_size=request.world_size,
                companion_master_port=companion_master_port,
            )
            
            # Initialize distributed environment only once
            if not self.distributed_initialized:
                logger.info("[COMPANION-SERVER] Starting distributed initialization for device %d", self.device_id)
                self.model_manager.initialize_distributed()
                self.distributed_initialized = True
                logger.info("[COMPANION-SERVER] Distributed initialization complete for device %d", self.device_id)
            
            # Load model weights
            logger.info("[COMPANION-SERVER] Starting model weight loading for device %d", self.device_id)
            self.model_manager.load_model_weights()
            logger.info("[COMPANION-SERVER] Model weight loading complete for device %d", self.device_id)
            
            # Update state to READY
            self.state = CompanionState.READY
            
        except Exception as e:
            logger.exception("Failed to load model on device %d", self.device_id)
            self.state = CompanionState.ERROR
            self.error_message = str(e)
            raise
        
        # Get IPC info for parameters
        model_params_ipc = self.model_manager.get_model_parameters_ipc_info()
        
        # Return CUDATensorRebuildInfo objects directly (same as Dynamo)
        # Cache the results
        self.model_hash = config_hash
        self.model_parameters = model_params_ipc
        
        logger.info("Model loaded: %d parameters", len(model_params_ipc))
        
        return model_params_ipc
    
    async def _load_model_async(self, request: LoadModelRequest) -> Optional[dict]:
        """Load model weights asynchronously using thread pool.
        
        Returns:
            Dict of model parameters if successful, None if hash mismatch.
        """
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(self.executor, self._load_model_sync, request)
    

    
    async def handle_request(self, message: bytes) -> bytes:
        """Handle different types of requests."""
        try:
            request = pickle.loads(message)
            
            # Get request type
            request_type = getattr(request, 'request_type', None)
            if request_type is None:
                logger.error("Request missing request_type field: %s", type(request))
                response = ModelParametersRebuildInfoResponse(
                    success=False,
                    model_parameters=None,
                    error="Request missing request_type field",
                    response_type=ResponseType.MODEL_PARAMETERS_REBUILD_INFO
                )
                return pickle.dumps(response)
            
            # Handle requests based on type
            if request_type == RequestType.HANDSHAKE:
                # Simple handshake response
                response = HandshakeResponse(
                    success=True,
                    message=f"Companion server for GPU {self.device_id} ready"
                )
                return pickle.dumps(response)
            
            elif request_type == RequestType.GET_COMPANION_STATUS:
                # Return the current state
                response = GetCompanionStatusResponse(
                    device_id=self.device_id,
                    state=self.state,
                    error_message=self.error_message
                )
                return pickle.dumps(response)
            
            elif request_type == RequestType.LOAD_MODEL:
                config_hash = request.compute_hash()
                
                # Check for hash mismatch first
                if self.model_hash is not None and self.model_hash != config_hash:
                    logger.warning(
                        "Model hash mismatch: server has %s, request has %s",
                        self.model_hash, config_hash
                    )
                    response = LoadModelResponse(
                        success=False,
                        error=f"hash_mismatch:{self.model_hash}:{config_hash}",
                        response_type=ResponseType.LOAD_MODEL
                    )
                    return pickle.dumps(response)
                
                # If model is already loaded for this config, just return success
                if self.model_hash == config_hash and self.state == CompanionState.READY:
                    logger.debug("Model already loaded for hash %s", config_hash)
                    response = LoadModelResponse(
                        success=True,
                        error=None,
                        response_type=ResponseType.LOAD_MODEL
                    )
                    return pickle.dumps(response)
                
                # If not loading yet, start loading asynchronously
                if self.state == CompanionState.INITIALIZING:
                    # Start loading task if not already started
                    if self.model_loading_task is None or self.model_loading_task.done():
                        logger.info("Starting async model loading for hash %s", config_hash)
                        self.model_loading_task = asyncio.create_task(
                            self._load_model_async(request)
                        )
                    
                    # Return immediate ack
                    response = LoadModelResponse(
                        success=True,
                        error=None,
                        response_type=ResponseType.LOAD_MODEL
                    )
                    return pickle.dumps(response)
                
                # If already loading, just return ack
                if self.state == CompanionState.LOADING:
                    logger.debug("Model already loading for hash %s", config_hash)
                    response = LoadModelResponse(
                        success=True,
                        error=None,
                        response_type=ResponseType.LOAD_MODEL
                    )
                    return pickle.dumps(response)
                
                # If in error state, return error
                if self.state == CompanionState.ERROR:
                    response = LoadModelResponse(
                        success=False,
                        error=self.error_message,
                        response_type=ResponseType.LOAD_MODEL
                    )
                    return pickle.dumps(response)
            
            elif request_type == RequestType.GET_MODEL_PARAMETERS_REBUILD_INFO:
                # Request for model parameters - model should already be loaded
                if self.state != CompanionState.READY or self.model_parameters is None:
                    logger.error("Model not ready when parameters requested. State: %s", self.state)
                    response = ModelParametersRebuildInfoResponse(
                        success=False,
                        model_parameters=None,
                        error=f"Model not ready. Current state: {self.state}",
                        response_type=ResponseType.MODEL_PARAMETERS_REBUILD_INFO
                    )
                    return pickle.dumps(response)
                
                logger.debug("Returning model parameters for device %d", self.device_id)
                response = ModelParametersRebuildInfoResponse(
                    success=True,
                    model_parameters=self.model_parameters,
                    error=None,
                    response_type=ResponseType.MODEL_PARAMETERS_REBUILD_INFO
                )
                return pickle.dumps(response)
            
            else:
                # Unknown request type
                logger.error("Unknown request type: %s", request_type)
                response = ModelParametersRebuildInfoResponse(
                    success=False,
                    model_parameters=None,
                    error=f"Unknown request type: {request_type}",
                    response_type=ResponseType.MODEL_PARAMETERS_REBUILD_INFO
                )
                return pickle.dumps(response)
            
        except Exception as e:
            logger.exception("Failed to handle request: %s", e)
            response = ModelParametersRebuildInfoResponse(
                success=False,
                model_parameters=None,
                error=str(e),
                response_type=ResponseType.MODEL_PARAMETERS_REBUILD_INFO
            )
            return pickle.dumps(response)
    
    async def _handle_data_requests(self):
        """Handle model parameter requests on the data socket."""
        while not self.shutdown_event.is_set():
            try:
                if await self.data_socket.poll(timeout=1000):
                    request = await self.data_socket.recv()
                    response = await self.handle_request(request)
                    await self.data_socket.send(response)
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error("Error handling data request: %s", e)
    
    async def _handle_status_requests(self):
        """Handle status queries on the status socket."""
        while not self.shutdown_event.is_set():
            try:
                if await self.status_socket.poll(timeout=100):  # Shorter timeout for status
                    request = await self.status_socket.recv()
                    response = await self.handle_request(request)
                    await self.status_socket.send(response)
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error("Error handling status request: %s", e)
    
    async def start(self):
        """Start the companion server."""
        logger.info("Starting companion server for GPU %d...", self.device_id)
        
        # Initialize ZMQ
        self.context = zmq.asyncio.Context()
        
        # Create data socket for model parameter requests
        self.data_socket = make_zmq_socket(
            self.context,
            f"tcp://*:{self.data_port}",
            zmq.REP,
            bind=True
        )
        
        # Create status socket for quick status queries
        self.status_socket = make_zmq_socket(
            self.context,
            f"tcp://*:{self.status_port}",
            zmq.REP,
            bind=True
        )
        
        logger.info("Companion server for GPU %d listening on ports %d (data) and %d (status)",
                   self.device_id,
                   self.data_port, self.status_port)
        
        # Create tasks for handling both sockets
        data_task = asyncio.create_task(self._handle_data_requests())
        status_task = asyncio.create_task(self._handle_status_requests())
        
        try:
            # Wait for either task to complete or shutdown
            await asyncio.gather(data_task, status_task)
        except asyncio.CancelledError:
            pass
        finally:
            # Cancel both tasks
            data_task.cancel()
            status_task.cancel()
            with contextlib.suppress(BaseException):
                await asyncio.gather(data_task, status_task, return_exceptions=True)
    
    async def shutdown(self):
        """Shutdown the companion server."""
        if self.shutdown_called:
            return  # Already shutting down
        self.shutdown_called = True
        
        logger.info("Shutting down companion server for GPU %d...", self.device_id)
        self.shutdown_event.set()
        
        # Cancel any pending model loading
        if self.model_loading_task and not self.model_loading_task.done():
            self.model_loading_task.cancel()
        
        # Shutdown thread pool executor
        self.executor.shutdown(wait=False)
        
        # Clear cache
        self.model_manager = None
        self.model_hash = None
        self.model_parameters = None
        torch.cuda.empty_cache()
        
        # Clean up ZMQ
        if self.data_socket:
            self.data_socket.close()
            self.data_socket = None
        if self.status_socket:
            self.status_socket.close()
            self.status_socket = None
        if self.context:
            self.context.term()
            self.context = None


def run_companion_server(device_id: int, data_port: int, status_port: int,
                        companion_master_port: int):
    """Run the companion server in a process.
    
    Args:
        device_id: Physical GPU device ID
        data_port: Port for model parameter requests
        status_port: Port for status queries
        companion_master_port: Master port for CPU group initialization
    """
    # Don't set CUDA_VISIBLE_DEVICES - we need physical device IDs for IPC
    
    asyncio.set_event_loop(asyncio.new_event_loop())
    loop = asyncio.get_event_loop()
    
    server = MultiProcCompanionServer(device_id, data_port, status_port, 
                                     companion_master_port)
    
    def signal_handler(sig, frame):
        loop.create_task(server.shutdown())
    
    signal.signal(signal.SIGTERM, signal_handler)
    signal.signal(signal.SIGINT, signal_handler)
    
    try:
        loop.run_until_complete(server.start())
    except KeyboardInterrupt:
        pass
    finally:
        loop.run_until_complete(server.shutdown())
        loop.close()