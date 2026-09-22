# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from typing import TYPE_CHECKING, Protocol

if TYPE_CHECKING:
    from vllm.v1.engine.core import EngineCore


class CheckpointPolicy(Protocol):
    """Paired resource hooks on each EngineCore's execution thread.

    Scheduling is paused before prepare and remains paused through restore.
    Use core.collective_rpc() for worker-local hooks. Hooks must finish all
    work before returning and must not resume scheduling. Artifact validation
    and device checkpoint/restore belong to the external orchestrator.
    """

    def prepare(self, core: "EngineCore") -> None: ...

    def restore(self, core: "EngineCore") -> None: ...


class ResidentPolicy:
    """Keep GPU allocations and communications for the checkpoint backend."""

    def prepare(self, core: "EngineCore") -> None:
        pass

    def restore(self, core: "EngineCore") -> None:
        pass
