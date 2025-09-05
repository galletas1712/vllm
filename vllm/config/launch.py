# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import hashlib
from dataclasses import field
from typing import Any, Literal, Optional

from pydantic.dataclasses import dataclass

from vllm.config.utils import config

InitMode = Literal["checkpoint", "resume", "normal"]


@config
@dataclass
class LaunchConfig:
    """Configuration for vLLM launch mode and initialization behavior."""

    init_mode: InitMode = "normal"
    """Initialization mode:
    - 'normal': Normal initialization (default)
    - 'checkpoint': Initialize workers (phase 1) and save checkpoint. Process will pause after checkpointing.
    - 'resume': Resume from a paused checkpoint process. Completes initialization phases 2 and 3.
    """

    def compute_hash(self) -> str:
        """
        Provide a hash that uniquely identifies all the configs
        that affect the structure of the computation graph.

        We don't include init_mode in the hash because it only affects launch behavior, not the computation graph structure.
        """
        # No factors to consider - init_mode doesn't affect computation graph
        factors: list[Any] = []
        hash_str = hashlib.md5(str(factors).encode(),
                               usedforsecurity=False).hexdigest()
        return hash_str
