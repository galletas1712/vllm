# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""
MultiProc Companion System for CUDA IPC weight sharing.

Alternative to Dynamo-based companion servers, using only ZMQ.
"""

from vllm.companion.multiproc_companion_client import MultiProcCompanionClient
from vllm.companion.multiproc_companion_server import MultiProcCompanionServer
from vllm.companion.multiproc_coordinator import MultiProcCoordinator

__all__ = [
    "MultiProcCompanionClient",
    "MultiProcCompanionServer", 
    "MultiProcCoordinator",
]