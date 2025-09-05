# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import hashlib
from dataclasses import field
from typing import Any, Literal, Optional

from pydantic.dataclasses import dataclass

from vllm.config.utils import config

InitMode = Literal["save_checkpoint", "load_checkpoint", "resume_checkpoint"]


@config
@dataclass
class LaunchConfig:
    """Configuration for vLLM launch mode and initialization behavior."""

    init_mode: Optional[InitMode] = None
    """Initialization mode:
    - None: Normal initialization (default)
    - 'save_checkpoint': Initialize workers (phase 1) and save checkpoint, then exit
    - 'load_checkpoint': Load workers from checkpoint (phase 1 only), wait for resume_init API
    - 'resume_checkpoint': Load workers from checkpoint and automatically complete all initialization phases
    """

    def compute_hash(self) -> str:
        """
        Provide a hash that uniquely identifies all the configs
        that affect the structure of the computation graph.
        
        Note: init_mode is intentionally excluded from the hash because
        it only affects launch behavior, not the computation graph structure.
        This ensures that save_checkpoint and resume_checkpoint operations
        for the same model configuration produce identical hashes.
        """
        # No factors to consider - init_mode doesn't affect computation graph
        factors: list[Any] = []
        hash_str = hashlib.md5(str(factors).encode(),
                               usedforsecurity=False).hexdigest()
        return hash_str
