# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import hashlib
from typing import Any, Literal, Optional

from pydantic.dataclasses import dataclass

from vllm.config.utils import config

InitMode = Literal["checkpoint", "normal"]


@config
@dataclass
class LaunchConfig:
    """Configuration for vLLM launch mode and initialization behavior."""

    init_mode: InitMode = "normal"
    """Initialization mode:
    - 'normal': Normal initialization (default)
    - 'checkpoint': Initialize workers (phase 1) and save checkpoint. 
      Requires `resume_init` API to complete initialization.
    """

    resume_port: Optional[int] = None
    """Port for the resume side channel socket when using checkpoint mode.
    If None, a random available port will be used.
    This is only used when init_mode='checkpoint'."""

    def compute_hash(self) -> str:
        """
        Provide a hash that uniquely identifies all the configs
        that affect the structure of the computation graph.

        We don't include init_mode or resume_port in the hash because they 
        only affect launch behavior, not the computation graph structure.
        """
        # No factors to consider - init_mode and resume_port don't affect
        # computation graph
        factors: list[Any] = []
        hash_str = hashlib.md5(str(factors).encode(),
                               usedforsecurity=False).hexdigest()
        return hash_str
