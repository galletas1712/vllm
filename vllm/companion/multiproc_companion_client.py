# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""Simple companion client for GPU workers to retrieve model parameters."""

import pickle

import torch
import zmq

from vllm.config import VllmConfig
from vllm.config.companion import CompanionConfig
from vllm.distributed import get_world_group
from vllm.logger import init_logger
from vllm.companion.messages import (
    CUDATensorRebuildInfo,
    GetCompanionStatusRequest,
    CompanionState,
    ResponseType,
    HandshakeRequest,
    LoadModelRequest,
    GetModelParametersRebuildInfoRequest,
    GetMemoryUsageRequest,
)
from vllm.utils import make_zmq_socket

logger = init_logger(__name__)


class MultiProcCompanionClient:
    """
    Simple client for GPU workers to get model parameters from
    companion servers.
    """
    
    def __init__(self, companion_config: CompanionConfig, device_id: int):
        """
        Initialize the client for a specific GPU device.
        
        Args:
            companion_config: CompanionConfig object containing coordinator port
                and other companion settings.
            device_id: Physical GPU device ID this client will manage
        """
        if not companion_config:
            raise ValueError("CompanionConfig is required")
        
        self.companion_config = companion_config
        # Construct coordinator address from localhost and port
        self.coordinator_address = f"tcp://127.0.0.1:{companion_config.coordinator_port}"
        self.device_id = device_id  # This client always manages this specific GPU
        
        # ZMQ context and socket
        self.context = zmq.Context()
        self.socket = make_zmq_socket(
            self.context,
            self.coordinator_address,
            zmq.REQ,
            bind=False
        )
        # Set reasonable timeouts - model loading happens during wait_for_coordinator_ready
        self.socket.setsockopt(zmq.RCVTIMEO, 10000)   # 10 seconds receive timeout
        self.socket.setsockopt(zmq.SNDTIMEO, 5000)    # 5 seconds send timeout
        
        logger.info("Companion client for device %d connecting to %s",
                   self.device_id, self.coordinator_address)
        
        # Perform handshake with coordinator first
        self._perform_coordinator_handshake()
        
        # Then perform handshake with companion server to ensure it's ready
        self._perform_server_handshake()
        
        logger.info("Companion client for device %d ready", self.device_id)
    
    def _perform_coordinator_handshake(self) -> None:
        """Perform handshake with coordinator to ensure it's alive."""
        logger.info("Performing handshake with coordinator")
        
        # Send handshake request to coordinator (no device_id needed)
        request = HandshakeRequest()
        request_data = pickle.dumps(request)
        logger.debug("Sending handshake request to coordinator (bytes=%d)", len(request_data))
        self.socket.send(request_data)
        logger.debug("Handshake request sent, waiting for response...")
        
        try:
            logger.debug("Calling socket.recv() with timeout %dms", self.socket.getsockopt(zmq.RCVTIMEO))
            response_data = self.socket.recv()
            logger.debug("Received response from coordinator (bytes=%d)", len(response_data))
            response = pickle.loads(response_data)
            logger.debug("Response unpickled: %s", response)
            
            # Verify we got the expected response type
            if getattr(response, 'response_type', None) != ResponseType.HANDSHAKE:
                raise RuntimeError(f"Unexpected response type during coordinator handshake: {response}")
            
            if not response.success:
                raise RuntimeError(f"Coordinator handshake failed: {getattr(response, 'message', 'Unknown error')}")
                
            logger.info("Handshake successful with coordinator")
                       
        except zmq.error.Again as e:
            logger.error("ZMQ timeout error: %s", e)
            raise RuntimeError("Timeout during handshake with coordinator") from None
        except Exception as e:
            logger.error("Unexpected error during handshake: %s", e)
            raise
    
    def _perform_server_handshake(self) -> None:
        """Perform handshake with companion server for our device.
        
        This ensures the companion server is started and ready to receive requests.
        """
        logger.info("Performing handshake with companion server for device %d", self.device_id)
        
        # Send handshake request with device_id to check companion server
        request = HandshakeRequest(device_id=self.device_id)
        self.socket.send(pickle.dumps(request))
        
        try:
            response_data = self.socket.recv()
            response = pickle.loads(response_data)
            
            # Verify we got the expected response type
            if getattr(response, 'response_type', None) != ResponseType.HANDSHAKE:
                raise RuntimeError(f"Unexpected response type during server handshake: {response}")
            
            # Check if handshake was successful
            if not response.success:
                raise RuntimeError(
                    f"Companion server handshake failed for device {self.device_id}: "
                    f"{getattr(response, 'message', 'Unknown error')}")
            
            logger.info("Handshake complete with companion server for device %d", 
                       self.device_id)
                       
        except zmq.error.Again:
            raise RuntimeError(
                f"Timeout during handshake with companion server for device {self.device_id}"
            ) from None
    
    def wait_for_server_ready(self, timeout: float = 300.0) -> None:
        """Wait for the companion server to finish loading the model.
        
        Args:
            timeout: Maximum time to wait in seconds (default: 5 minutes)
            
        Raises:
            RuntimeError: If server fails to load model or encounters an error
        """
        import time
        
        start_time = time.time()
        
        # Create a separate socket for status checks to avoid REQ/REP state issues
        status_socket = make_zmq_socket(
            self.context,
            self.coordinator_address,
            zmq.REQ,
            bind=False
        )
        status_socket.setsockopt(zmq.RCVTIMEO, 5000)   # 5 seconds timeout
        status_socket.setsockopt(zmq.SNDTIMEO, 5000)    # 5 seconds timeout
        
        try:
            while time.time() - start_time < timeout:
                try:
                    # Send status request for our companion
                    request = GetCompanionStatusRequest(device_id=self.device_id)
                    status_socket.send(pickle.dumps(request))
                    response_data = status_socket.recv()
                    response = pickle.loads(response_data)
                    
                    # Check response type
                    response_type = getattr(response, 'response_type', None)
                    if response_type != ResponseType.COMPANION_STATUS:
                        logger.warning("Unexpected response type: %s", response_type)
                        time.sleep(0.5)
                        continue
                    
                    # Check state
                    if response.state == CompanionState.READY:
                        logger.info("Companion server for GPU %d has finished loading model", self.device_id)
                        return
                    elif response.state == CompanionState.ERROR:
                        raise RuntimeError(
                            f"Companion server for GPU {self.device_id} encountered an error: {response.error_message}"
                        )
                    elif response.state == CompanionState.LOADING:
                        logger.debug("Companion server for GPU %d is loading model...", self.device_id)
                    else:
                        logger.debug("Companion server for GPU %d state: %s", self.device_id, response.state)
                    
                    time.sleep(0.5)  # Poll every 500ms
                    
                except zmq.error.Again:
                    logger.warning("Timeout waiting for companion server status")
                    time.sleep(1.0)
                except Exception as e:
                    logger.error("Error checking companion server status: %s", e)
                    time.sleep(1.0)
            
            raise RuntimeError(
                f"Companion server for GPU {self.device_id} failed to load model within {timeout} seconds"
            )
        finally:
            status_socket.close()
    
    def get_model_parameters(
            self, vllm_config: VllmConfig) -> dict[str, CUDATensorRebuildInfo]:
        """
        Get model parameters from companion server.
        
        This method will:
        1. Send LoadModelRequest to trigger model loading
        2. Poll until model is loaded
        3. Send GetModelParametersRebuildInfoRequest to get the actual parameters
        
        Args:
            vllm_config: VllmConfig for the model
            
        Returns:
            Dict of parameter name to CUDATensorRebuildInfo
        """
        # Use the device_id from initialization - this client always manages the same GPU
        device_id = self.device_id
        
        # Get rank information from the distributed environment
        # Send local TP/PP rank and let companion server handle DP adjustment
        # This makes it consistent with how workers are initialized
        world_group = get_world_group()
        wg_rank = world_group.rank
        wg_world_size = world_group.world_size
        
        tp_pp_world_size = vllm_config.parallel_config.world_size
        
        # For companions, we pass the local TP/PP rank and world size
        # The companion server will handle DP adjustment in parallel_state
        if wg_world_size == tp_pp_world_size:
            # World group is TP*PP only (no DP adjustment yet)
            local_tp_pp_rank = wg_rank
            local_tp_pp_world = wg_world_size
        else:
            # World group already includes DP
            # Extract the local TP/PP rank by undoing DP adjustment
            dp_rank = vllm_config.parallel_config.data_parallel_rank
            local_tp_pp_rank = wg_rank - (dp_rank * tp_pp_world_size)
            local_tp_pp_world = tp_pp_world_size

        # Use physical device id for local_rank (for logging/device selection)
        local_rank = device_id if device_id is not None else local_tp_pp_rank

        logger.info(
            "[COMPANION-CLIENT] Sending local TP/PP rank=%d (TP*PP world=%d) "
            "DP rank=%d to companion for DP adjustment",
            local_tp_pp_rank, local_tp_pp_world,
            vllm_config.parallel_config.data_parallel_rank)
        
        # Create request to trigger model loading
        load_request = LoadModelRequest(
            vllm_config=vllm_config,
            device_id=device_id,
            local_rank=local_rank,
            global_rank=local_tp_pp_rank,  # Pass local TP/PP rank
            world_size=local_tp_pp_world    # Pass TP*PP world size
        )
        
        # Send request to trigger model loading
        try:
            self.socket.send(pickle.dumps(load_request))
            response_data = self.socket.recv()
            load_response = pickle.loads(response_data)
            
            # Check if we got a load response
            response_type = getattr(load_response, 'response_type', None)
            if response_type != ResponseType.LOAD_MODEL:
                raise RuntimeError(f"Expected LOAD_MODEL response, got {response_type}")
            
            if not load_response.success:
                # Check for hash mismatch or other errors
                if load_response.error and load_response.error.startswith("hash_mismatch:"):
                    parts = load_response.error.split(":")
                    if len(parts) >= 3:
                        server_hash = parts[1]
                        request_hash = parts[2]
                        raise RuntimeError(
                            f"Model configuration mismatch: Companion server has "
                            f"already loaded a model with hash {server_hash}, but "
                            f"received request for different model with hash "
                            f"{request_hash}. This typically happens when a warm "
                            f"spare worker requests a different model "
                            f"configuration than the primary worker. Each "
                            f"companion server can "
                            f"only serve one model configuration."
                        )
                raise RuntimeError(f"Failed to trigger model loading: {load_response.error}")
                
            logger.info("Model loading triggered on companion server for GPU %d", device_id)
            
        except zmq.error.Again as e:
            raise RuntimeError(
                f"Timeout sending load request to companion server. "
                f"Device ID: {device_id}, Model: {vllm_config.model_config.model}"
            ) from e
        
        # Wait for the companion server to be ready
        logger.info("Waiting for companion server to load model on GPU %d...", device_id)
        self.wait_for_server_ready()
        
        # Now send request to get actual model parameters
        rebuild_request = GetModelParametersRebuildInfoRequest(device_id=device_id)
        
        try:
            self.socket.send(pickle.dumps(rebuild_request))
            response_data = self.socket.recv()
        except zmq.error.Again as e:
            raise RuntimeError(
                f"Timeout waiting for model parameters from companion server. "
                f"Device ID: {device_id}, Model: {vllm_config.model_config.model}"
            ) from e
        
        response = pickle.loads(response_data)
        
        # Check response type
        response_type = getattr(response, 'response_type', None)
        if response_type != ResponseType.MODEL_PARAMETERS_REBUILD_INFO:
            # Unexpected response type
            raise RuntimeError(
                f"Expected MODEL_PARAMETERS_REBUILD_INFO response, got {response_type}"
            )
        
        if not response.success:
            # Check if it's a hash mismatch error
            if response.error and response.error.startswith("hash_mismatch:"):
                parts = response.error.split(":")
                if len(parts) >= 3:
                    server_hash = parts[1]
                    request_hash = parts[2]
                    raise RuntimeError(
                        f"Model configuration mismatch: Companion server has "
                        f"already loaded a model with hash {server_hash}, but "
                        f"received request for different model with hash "
                        f"{request_hash}. This typically happens when a warm "
                        f"spare worker requests a different model "
                        f"configuration than the primary worker. Each "
                        f"companion server can "
                        f"only serve one model configuration."
                    )
            raise RuntimeError(
                f"Failed to get model parameters: {response.error}")
        
        # Return the CUDATensorRebuildInfo objects directly (same as Dynamo)
        model_parameters = response.model_parameters or {}
        
        logger.info("Received %d model parameters from companion", len(model_parameters))
        return model_parameters
    
    def reconstruct_parameter(self, rebuild_info: CUDATensorRebuildInfo) -> torch.Tensor:
        """Reconstruct a model parameter tensor from rebuild info using CUDA IPC.
        
        Args:
            rebuild_info: CUDATensorRebuildInfo object containing CUDA IPC rebuild information
            
        Returns:
            Reconstructed tensor on the current device
        """
        from torch.multiprocessing.reductions import rebuild_cuda_tensor
        
        # Ensure we have a CUDATensorRebuildInfo object
        if not isinstance(rebuild_info, CUDATensorRebuildInfo):
            raise ValueError(f"Expected CUDATensorRebuildInfo, got {type(rebuild_info)}")
        
        # Get the logical device (what this process sees after CUDA_VISIBLE_DEVICES mapping)
        logical_device = torch.cuda.current_device()
        
        # The server device in rebuild_info is the physical device
        # We need to adjust it to our logical device for reconstruction
        server_device = rebuild_info.device
        
        logger.debug(
            "Reconstructing tensor: server_device=%d → client_logical_device=%d",
            server_device,
            logical_device,
        )
        
        # Modify the device in the rebuild args
        rebuild_info.device = logical_device
        
        # Get rebuild args from the CUDATensorRebuildInfo object
        rebuild_args = rebuild_info.to_rebuild_args()
        
        # Reconstruct the tensor
        tensor = rebuild_cuda_tensor(*rebuild_args)
        
        return tensor
    
    def get_memory_usage(self) -> tuple[int, bool]:
        """Get memory usage information from the companion server.
        
        Returns:
            Tuple of (model_weights_bytes, is_model_loaded)
            - model_weights_bytes: Total bytes used by model weights (0 if not loaded)
            - is_model_loaded: Whether the model is currently loaded
            
        Raises:
            RuntimeError: If failed to get memory usage information
        """
        # Use the device_id from initialization
        device_id = self.device_id
        
        # Create request to get memory usage
        request = GetMemoryUsageRequest(device_id=device_id)
        
        try:
            self.socket.send(pickle.dumps(request))
            response_data = self.socket.recv()
            response = pickle.loads(response_data)
            
            # Check response type
            response_type = getattr(response, 'response_type', None)
            if response_type != ResponseType.MEMORY_USAGE:
                raise RuntimeError(
                    f"Expected MEMORY_USAGE response, got {response_type}"
                )
            
            if not response.success:
                raise RuntimeError(
                    f"Failed to get memory usage: {response.error}"
                )
            
            logger.debug(
                "Memory usage for GPU %d: %d bytes, loaded=%s",
                device_id, response.model_weights_bytes, response.is_model_loaded
            )
            
            return response.model_weights_bytes, response.is_model_loaded
            
        except zmq.error.Again as e:
            raise RuntimeError(
                f"Timeout getting memory usage from companion server. "
                f"Device ID: {device_id}"
            ) from e
    
    def close(self):
        """Close the client connection."""
        self.socket.close()
        self.context.term()


