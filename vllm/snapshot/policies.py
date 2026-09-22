# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from vllm.v1.engine.core import EngineCore


class ReloadWeightsPolicy:
    """Compact local snapshots: discard GPU storage and reload model files."""

    def prepare(self, core: "EngineCore") -> None:
        core.model_executor.sleep(level=2)

    def restore(self, core: "EngineCore") -> None:
        core.model_executor.wake_up(tags=["weights"])
        core.collective_rpc("reload_weights")
        core.model_executor.wake_up(tags=["kv_cache"])
