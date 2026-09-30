# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from vllm.v1.engine.core import EngineCore


class OffloadWeightsPolicy:
    """Offload weights to host memory and discard the KV cache (sleep level 1).

    Requires --enable-sleep-mode.
    """

    def prepare(self, core: "EngineCore") -> None:
        # Level 1 discards KV blocks, so no cached prefix may point at them.
        core._reset_caches()
        core.collective_rpc("checkpoint_prepare")
        core.model_executor.sleep(level=1)

    def restore(self, core: "EngineCore") -> None:
        core.model_executor.wake_up()
        core.collective_rpc("checkpoint_restore")


class ReloadWeightsPolicy:
    """Compact local snapshots: discard GPU storage and reload model files."""

    def prepare(self, core: "EngineCore") -> None:
        core.model_executor.sleep(level=2)

    def restore(self, core: "EngineCore") -> None:
        core.model_executor.wake_up(tags=["weights"])
        core.collective_rpc("reload_weights")
        core.model_executor.wake_up(tags=["kv_cache"])
