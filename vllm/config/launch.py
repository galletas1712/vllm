# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import hashlib
from typing import Any, Literal

from pydantic.dataclasses import dataclass

from vllm.config.utils import config

InitMode = Literal[
    "normal", "save_checkpoint", "resume_checkpoint", "checkpoint"
]


@config
@dataclass
class LaunchConfig:
    """Configuration for vLLM launch mode and initialization behavior."""

    init_mode: InitMode = "normal"
    """Initialization mode:
    - 'normal': Normal initialization (default)
    - 'save_checkpoint': Initialize workers (phase 1), save CRIU checkpoint,
      and exit gracefully
    - 'resume_checkpoint': Restore workers from CRIU checkpoint and complete
      initialization
    - 'checkpoint': Initialize workers (phase 1) and save checkpoint. Requires
      `resume_init` API to complete initialization (deprecated, use
      save_checkpoint instead).
    """

    checkpoint_dir_root: str = "/tmp"
    """Root directory for CRIU dump files."""

    def compute_hash(self) -> str:
        """
        Provide a hash that uniquely identifies all the configs
        that affect the structure of the computation graph.

        We don't include init_mode in the hash because it only affects
        launch behavior, not the computation graph structure.
        """
        # No factors to consider - init_mode doesn't affect computation graph
        factors: list[Any] = []
        hash_str = hashlib.md5(str(factors).encode(),
                               usedforsecurity=False).hexdigest()
        return hash_str
