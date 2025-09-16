# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
from collections.abc import Generator
import contextlib
import copy
import asyncio
import os
import uvloop

import torch
from torch import nn

from vllm.config import LoadConfig, ModelConfig, VllmConfig
from vllm.distributed import get_world_group
from vllm.logger import init_logger
from vllm.model_executor.model_loader.base_loader import BaseModelLoader
from vllm.model_executor.model_loader.utils import (
    initialize_model,
    process_weights_after_loading,
    set_default_torch_dtype,
)
from vllm.model_executor.parameter import UninitializedParameterFromTensor

logger = init_logger(__name__)

# Determine which companion backend to use
USE_MULTIPROC = os.environ.get("VLLM_IPC_USE_MULTIPROC", "1") == "1"

if USE_MULTIPROC:
    # Import MultiProc companion client (default)
    try:
        from vllm.companion.multiproc_companion_client import (
            MultiProcCompanionClient)
        logger.info("Using MultiProc companion backend for IPC loading")
    except ImportError as e:
        raise ImportError(
            "Failed to import MultiProc companion components. "
            "Make sure the vllm.companion module is available"
        ) from e
else:
    # Import Dynamo companion client
    try:
        from dynamo.runtime import DistributedRuntime
        from dynamo.companion.dynamo_companion_client import create_model_client
        logger.info("Using Dynamo companion backend for IPC loading")
    except ImportError as e:
        raise ImportError(
            "Failed to import Dynamo companion components. "
            "Make sure the companion module is installed"
        ) from e


class IPCModelLoader(BaseModelLoader):
    """Model loader that retrieves weights via IPC from a model server.
    
    This loader connects to a companion server (MultiProc or Dynamo) that has
    pre-loaded the model weights and retrieves them via CUDA IPC. This allows
    multiple processes to share the same GPU memory for model weights.
    
    The backend can be selected via the VLLM_IPC_USE_MULTIPROC environment
    variable (default: "1" for MultiProc, "0" for Dynamo).
    
    The loader automatically obtains rank information from vLLM's parallel_state
    module, which is initialized during init_device() before model loading.
    This ensures the loader connects to the correct companion server instance
    based on the worker's rank.
    """

    def __init__(self, vllm_config: VllmConfig):
        self.vllm_config = VllmConfig(
            model_config=copy.deepcopy(vllm_config.model_config),
            parallel_config=copy.deepcopy(vllm_config.parallel_config),
            cache_config=copy.deepcopy(vllm_config.cache_config),
            device_config=copy.deepcopy(vllm_config.device_config),
            load_config=copy.deepcopy(vllm_config.load_config),
            companion_config=copy.deepcopy(vllm_config.companion_config),
        )

        if not self.vllm_config.load_config.enable_companion_process:
            raise ValueError(
                "IPCModelLoader requires enable_companion_process=True in LoadConfig"
            )

        self.client = None  # Will be initialized when we know the model
        self.use_multiproc = USE_MULTIPROC
        self.runtime = None  # Only used for Dynamo backend

        logger.info(
            "IPC model loader initialized with %s backend",
            "MultiProc" if self.use_multiproc else "Dynamo"
        )
    
    def _validate_companion_server_availability(self) -> None:
        """Validate that a companion server exists for the computed physical device.
        
        Raises:
            RuntimeError: If no companion server is available for the target device.
        """
        if not self.use_multiproc or not self.client:
            return  # Skip validation for Dynamo backend
        
        # Get current logical device
        logical_device = torch.cuda.current_device()
        
        # Compute physical device ID based on DP rank (same logic as in client)
        physical_device = logical_device  # Default if no DP
        if hasattr(self.vllm_config.parallel_config, 'data_parallel_rank'):
            dp_rank = self.vllm_config.parallel_config.data_parallel_rank
            tp_size = self.vllm_config.parallel_config.tensor_parallel_size
            pp_size = self.vllm_config.parallel_config.pipeline_parallel_size
            tp_pp_size = tp_size * pp_size
            
            # Calculate physical device ID
            physical_device = dp_rank * tp_pp_size + logical_device
            
            logger.info(
                "[IPC-LOADER] Computed physical device mapping: "
                "logical device %d → physical device %d "
                "(DP rank=%d, TP=%d, PP=%d)",
                logical_device, physical_device, dp_rank, tp_size, pp_size
            )
        
        # Check if physical device is valid
        total_gpus = torch.cuda.device_count()
        
        # Note: torch.cuda.device_count() returns the number of *visible* GPUs
        # after CUDA_VISIBLE_DEVICES is applied. For proper validation, we need
        # to check against the total system GPUs. We'll use a heuristic here.
        import os
        cuda_visible_devices = os.environ.get('CUDA_VISIBLE_DEVICES', '')
        if cuda_visible_devices:
            # Worker has restricted GPU visibility, assume system has more GPUs
            # We can't validate precisely, but we can warn if it looks wrong
            visible_gpus = total_gpus
            # Estimate total system GPUs based on DP configuration
            estimated_total_gpus = self.vllm_config.parallel_config.data_parallel_size * tp_pp_size
            logger.info(
                "[IPC-LOADER] Worker has CUDA_VISIBLE_DEVICES=%s (sees %d GPU(s)). "
                "Estimated system has %d total GPUs based on DP=%d, TP=%d, PP=%d",
                cuda_visible_devices, visible_gpus, estimated_total_gpus,
                self.vllm_config.parallel_config.data_parallel_size, tp_size, pp_size
            )
            
            if physical_device >= estimated_total_gpus:
                raise RuntimeError(
                    f"Computed physical device {physical_device} exceeds estimated total GPUs ({estimated_total_gpus}). "
                    f"This worker with DP rank {dp_rank} and logical device {logical_device} "
                    f"cannot find a companion server. Check your DP/TP/PP configuration:\n"
                    f"  - DP size: {self.vllm_config.parallel_config.data_parallel_size}\n"
                    f"  - TP size: {tp_size}\n"
                    f"  - PP size: {pp_size}\n"
                    f"  - DP rank: {dp_rank}\n"
                    f"  - Expected physical device: {physical_device}\n"
                    f"  - CUDA_VISIBLE_DEVICES: {cuda_visible_devices}\n"
                    f"  - Visible GPUs to this worker: {visible_gpus}"
                )
        else:
            # No CUDA_VISIBLE_DEVICES restriction, we see all GPUs
            if physical_device >= total_gpus:
                raise RuntimeError(
                    f"Computed physical device {physical_device} exceeds available GPUs ({total_gpus}). "
                    f"This worker with DP rank {dp_rank} and logical device {logical_device} "
                    f"cannot find a companion server. Check your DP/TP/PP configuration:\n"
                    f"  - DP size: {self.vllm_config.parallel_config.data_parallel_size}\n"
                    f"  - TP size: {tp_size}\n"
                    f"  - PP size: {pp_size}\n"
                    f"  - DP rank: {dp_rank}\n"
                    f"  - Expected physical device: {physical_device}\n"
                    f"  - Available GPUs: {total_gpus}"
                )
        
        # Skip ping/healthcheck. Proceed directly; errors will surface on use.
        logger.debug(
            "[IPC-LOADER] Skipping companion ping; proceeding directly to model parameter requests"
        )

    def download_model(self, model_config: ModelConfig) -> None:
        """Connect to the model server and wait for model to be ready."""
        assert self.vllm_config is not None, "vllm_config not set"

        if self.client is None:
            if self.use_multiproc:
                # Initialize MultiProc client
                try:
                    if not self.vllm_config or not self.vllm_config.companion_config:
                        raise ValueError("VllmConfig with CompanionConfig is required for IPC loading")
                    
                    self.client = MultiProcCompanionClient(self.vllm_config.companion_config)
                    logger.info("MultiProc companion client initialized with coordinator at: %s",
                               self.vllm_config.companion_config.coordinator_address)
                    
                    # Validate that companion server exists for our device
                    self._validate_companion_server_availability()
                except Exception as e:
                    logger.error(
                        "Error creating MultiProc companion client: %s", e)
                    raise
            else:
                # Initialize Dynamo runtime and client
                try:
                    # Create a new event loop for this thread if needed
                    try:
                        loop = asyncio.get_event_loop()
                        if loop.is_closed():
                            raise RuntimeError("Event loop is closed")
                    except RuntimeError:
                        # No event loop in this thread, create one
                        uvloop.install()
                        loop = asyncio.new_event_loop()
                        asyncio.set_event_loop(loop)

                    # Create a Dynamo runtime instance
                    # Second parameter is whether it's static
                    # (False = dynamic, can discover services)
                    self.runtime = DistributedRuntime(loop, False)

                    async def _create():
                        # Get rank information from the distributed environment
                        # By the time this is called, init_device() has already
                        # initialized the distributed environment
                        world_group = get_world_group()
                        
                        return await create_model_client(
                            runtime=self.runtime,
                            vllm_config=self.vllm_config,
                            local_rank=world_group.local_rank,
                            global_rank=world_group.rank,
                            world_size=world_group.world_size,
                            namespace="companion",
                        )

                    self.client = loop.run_until_complete(_create())
                except Exception as e:
                    logger.error("Error creating Dynamo model client: %s", e)
                    raise

                # Wait for model to be ready with two-phase timeout
                # (blocking from sync context)
                logger.info("Waiting for model to be ready on server...")
                success, server_info = loop.run_until_complete(
                    self.client.wait_for_model_ready(
                        initial_timeout=15.0,  # TODO: make configurable
                        loading_timeout=300.0,
                    )
                )

                if not success:
                    raise RuntimeError(
                        "Failed to connect to model server or model not ready"
                    )

                logger.info(
                    "Model %s is ready on server (device: cuda:%s)",
                    model_config.model,
                    server_info.get("device_id", "unknown"),
                )

    def get_all_weights(
        self,
        model_config: ModelConfig,
        model: nn.Module,
    ) -> Generator[tuple[str, torch.Tensor], None, None]:
        """Get all weights from the model server via IPC."""
        # First ensure the model is loaded on the server
        self.download_model(model_config)

        # Both clients now have the same API - they return rebuild info
        logger.info(
            "Retrieving model parameters rebuild info from %s companion...",
            "MultiProc" if self.use_multiproc else "Dynamo"
        )
        
        try:
            if self.use_multiproc:
                # MultiProc client returns rebuild info directly
                logger.info("[IPC-LOADER] Requesting model parameters via MultiProc client")
                model_parameters_rebuild_info = self.client.get_model_parameters(
                    vllm_config=self.vllm_config,
                    device_id=torch.cuda.current_device()
                )
            else:
                # Dynamo client is async
                loop = asyncio.get_event_loop()
                logger.info("[IPC-LOADER] Requesting model parameters via Dynamo client (async)")
                model_parameters_rebuild_info = loop.run_until_complete(
                    self.client.get_model_parameters()
                )
        except Exception as e:
            logger.error("[IPC-LOADER] Error getting tensor rebuild info: %s", e)
            raise RuntimeError(
                f"Error getting tensor rebuild info: {e}") from e

        logger.info(
            "Retrieved rebuild info for %d parameters",
            len(model_parameters_rebuild_info)
        )

        # Reconstruct and yield each tensor - same for both backends
        for name, rebuild_info in model_parameters_rebuild_info.items():
            try:
                parameter = self.client.reconstruct_parameter(rebuild_info)

                # Verify we got a tensor on a valid device
                if not parameter.is_cuda:
                    raise RuntimeError(
                        f"Reconstructed tensor is not on CUDA: "
                        f"{parameter.device}"
                    )

                logger.debug(
                    "[IPC-LOADER] Reconstructed parameter %s shape=%s device=%s",
                    name, parameter.shape, parameter.device)

                yield name, parameter
            except Exception as e:
                logger.error(
                    "[IPC-LOADER] Failed to reconstruct parameter %s: %s", name, e)
                raise RuntimeError(
                    f"Failed to reconstruct parameter {name}: {e}") from e

    def load_weights(self, model: nn.Module,
                     model_config: ModelConfig) -> None:
        """Load weights into the model using IPC."""
        weights_to_load = {name for name, _ in model.named_parameters()}

        # NOTE: we manually assign weights here, since our model is already
        # remotely initialized
        weight_dict = dict(self.get_all_weights(model_config, model))

        for name, param in model.named_parameters():
            if name in weight_dict:
                if isinstance(param, UninitializedParameterFromTensor):
                    param.materialize(weight_dict[name])
                else:
                    param.data = weight_dict[name]
            else:
                logger.warning("Weight %s not found in IPC weights", name)

        # Check if all expected weights were loaded
        loaded_weights = set(weight_dict.keys())
        weights_not_loaded = weights_to_load - loaded_weights
        if weights_not_loaded:
            raise ValueError(
                f"Following weights were not initialized from IPC: "
                f"{weights_not_loaded}"
            )

        logger.info("Finished loading parameters via IPC: %d mapped, %d missing",
                    len(loaded_weights), len(weights_not_loaded))

    def load_model(self, vllm_config: VllmConfig,
                   model_config: ModelConfig) -> nn.Module:
        """Load a model with the given configurations."""
        device_config = vllm_config.device_config
        load_config = vllm_config.load_config
        load_device = device_config.device if load_config.device is None else \
                      load_config.device
        target_device = torch.device(load_device)
        with set_default_torch_dtype(model_config.dtype):
            with target_device:
                model = initialize_model(vllm_config=vllm_config,
                                         model_config=model_config)

            logger.debug("Loading weights on %s ...", load_device)
            
            # Log rank information for debugging
            # By this point, the distributed environment is already initialized
            world_group = get_world_group()
            logger.info(
                "IPC loader using rank info from parallel_state: "
                "local_rank=%d, global_rank=%d, world_size=%d",
                world_group.local_rank, world_group.rank, world_group.world_size
            )

            self.load_weights(model, model_config)
            # Quantization does not happen in `load_weights` but after it
            process_weights_after_loading(model, model_config, target_device)
        return model.eval()

    def __del__(self):
        """Clean up the client connection when the loader is destroyed."""
        if hasattr(self, "use_multiproc") and self.use_multiproc:
            # Clean up MultiProc client
            if hasattr(self, "client") and self.client is not None:
                with contextlib.suppress(Exception):
                    self.client.close()
        else:
            # Dynamo runtime cleanup happens automatically when it goes
            # out of scope
            if hasattr(self, "runtime") and self.runtime is not None:
                pass
