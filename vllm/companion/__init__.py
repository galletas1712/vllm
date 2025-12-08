# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""
vLLM Companion Server for CUDA IPC weight sharing.

The companion server loads model weights once and shares them with multiple
vLLM workers via CUDA IPC, enabling efficient multi-process inference without
duplicating weights in memory.

Server usage:
    python -m vllm.companion --device-id 0 --port 5555

Client usage (in vLLM worker):
    from vllm.companion import CompanionClient

    client = CompanionClient(host="localhost", port=5555)
    client.load_model(model, vllm_config, local_rank, global_rank, world_size)
"""

from vllm.companion.client import CompanionClient
from vllm.companion.handler import CompanionHandler
from vllm.companion.ipc_utils import (
    ModuleTreeNode,
    check_for_meta_tensors,
    import_weights_from_tree,
)
from vllm.companion.server import CompanionServer

__all__ = [
    "CompanionServer",
    "CompanionClient",
    "CompanionHandler",
    "ModuleTreeNode",
    "import_weights_from_tree",
    "check_for_meta_tensors",
]
