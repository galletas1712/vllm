# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""Simple companion server for CUDA IPC weight sharing."""

import asyncio
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
    GetModelParametersRequest,
    ModelParametersResponse,
)

logger = init_logger(__name__)


class MultiProcCompanionServer:
    """
    Simple companion server that loads model weights and serves them via
    CUDA IPC. Uses the same ModelInstanceManager as Dynamo companion.
    """
    
    def __init__(self, device_id: int, port: int, companion_master_port: int):
        """
        Initialize the companion server.
        
        Args:
            device_id: Physical GPU device ID  
            port: Port for this companion server to listen on
            companion_master_port: Master port for CPU group initialization
        """
        self.device_id = device_id
        self.port = port
        self.companion_master_port = companion_master_port
        
        # Model cache - only one model per companion server
        self.model_manager: Optional[ModelInstanceManager] = None
        self.model_hash: Optional[str] = None
        self.model_parameters: Optional[dict] = None
        
        # ZMQ socket
        self.context: Optional[zmq.asyncio.Context] = None
        self.rep_socket: Optional[zmq.asyncio.Socket] = None
        
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
        
        logger.info("Companion server for GPU %d on port %d",
                   device_id, port)
    

    
    def _load_model_sync(self, request: GetModelParametersRequest) -> Optional[dict]:
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
        
        # Get IPC info for parameters
        model_params_ipc = self.model_manager.get_model_parameters_ipc_info()
        
        # Return CUDATensorRebuildInfo objects directly (same as Dynamo)
        # Cache the results
        self.model_hash = config_hash
        self.model_parameters = model_params_ipc
        
        logger.info("Model loaded: %d parameters", len(model_params_ipc))
        
        return model_params_ipc
    
    async def _load_model_async(self, request: GetModelParametersRequest) -> Optional[dict]:
        """Load model weights asynchronously using thread pool.
        
        Returns:
            Dict of model parameters if successful, None if hash mismatch.
        """
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(self.executor, self._load_model_sync, request)
    

    
    async def handle_request(self, message: bytes) -> bytes:
        """Handle a request for model parameters."""
        try:
            request: GetModelParametersRequest = pickle.loads(message)
            # Fast path for health checks: don't try to load the model
            if getattr(request, "ping_only", False):
                response = ModelParametersResponse(
                    success=True,
                    model_parameters=None,
                    error=None,
                )
                return pickle.dumps(response)
            
            # Check if model is already loaded (fast path)
            config_hash = request.compute_hash()
            if self.model_hash == config_hash and self.model_parameters is not None:
                logger.debug("Returning cached model parameters for hash %s", config_hash)
                response = ModelParametersResponse(
                    success=True,
                    model_parameters=self.model_parameters,
                    error=None
                )
                return pickle.dumps(response)
            
            # Check for hash mismatch (another fast path)
            if self.model_hash is not None and self.model_hash != config_hash:
                logger.warning(
                    "Model hash mismatch: server has %s, request has %s",
                    self.model_hash, config_hash
                )
                response = ModelParametersResponse(
                    success=False,
                    model_parameters=None,
                    error=f"hash_mismatch:{self.model_hash}:{config_hash}"
                )
                return pickle.dumps(response)
            
            # Need to load model - use async loading with lock to prevent duplicates
            async with self.model_loading_lock:
                # Check again in case another request loaded it while we waited
                if self.model_hash == config_hash and self.model_parameters is not None:
                    response = ModelParametersResponse(
                        success=True,
                        model_parameters=self.model_parameters,
                        error=None
                    )
                    return pickle.dumps(response)
                
                # Load model asynchronously
                model_params = await self._load_model_async(request)
                
                # Check if model load returned None (shouldn't happen now, but keep for safety)
                if model_params is None:
                    response = ModelParametersResponse(
                        success=False,
                        model_parameters=None,
                        error=f"hash_mismatch:{self.model_hash}:{config_hash}"
                    )
                    return pickle.dumps(response)
                
                response = ModelParametersResponse(
                    success=True,
                    model_parameters=model_params,
                    error=None
                )
                
                return pickle.dumps(response)
            
        except Exception as e:
            logger.exception("Failed to handle request: %s", e)
            response = ModelParametersResponse(
                success=False,
                model_parameters=None,
                error=str(e)
            )
            return pickle.dumps(response)
    
    async def start(self):
        """Start the companion server."""
        logger.info("Starting companion server...")
        
        # Initialize ZMQ
        self.context = zmq.asyncio.Context()
        self.rep_socket = self.context.socket(zmq.REP)
        self.rep_socket.bind(f"tcp://*:{self.port}")
        
        logger.info("Companion server listening on port %d", self.port)
        
        # Main loop
        while not self.shutdown_event.is_set():
            try:
                if await self.rep_socket.poll(timeout=1000):
                    request = await self.rep_socket.recv()
                    response = await self.handle_request(request)
                    await self.rep_socket.send(response)
                    
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error("Error in companion loop: %s", e)
    
    async def shutdown(self):
        """Shutdown the companion server."""
        if self.shutdown_called:
            return  # Already shutting down
        self.shutdown_called = True
        
        logger.info("Shutting down companion server...")
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
        if self.rep_socket:
            self.rep_socket.close()
            self.rep_socket = None
        if self.context:
            self.context.term()
            self.context = None


def run_companion_server(device_id: int, port: int, companion_master_port: int):
    """Run the companion server in a process.
    
    Args:
        device_id: Physical GPU device ID
        port: Port for this companion server
        companion_master_port: Master port for CPU group initialization
    """
    # Don't set CUDA_VISIBLE_DEVICES - we need physical device IDs for IPC
    
    asyncio.set_event_loop(asyncio.new_event_loop())
    loop = asyncio.get_event_loop()
    
    server = MultiProcCompanionServer(device_id, port, companion_master_port)
    
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