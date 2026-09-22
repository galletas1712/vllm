# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import asyncio
from pathlib import Path
from typing import Protocol


class CheckpointSignal(Protocol):
    """External readiness publication and resource-restored barrier."""

    async def ready(self) -> None: ...

    async def wait_for_restore(self) -> None: ...


class FileCheckpointSignal:
    """The ai-dynamo/snapshot workload contract. Marker contents are opaque.

    Each container needs its own directory. The orchestrator must remove an
    old restore-complete marker before restoring a captured process tree.
    """

    def __init__(self, directory: Path):
        self.directory = directory

    def start_capture(self) -> None:
        """Initialize a new source, never a restore destination."""
        self.directory.mkdir(parents=True, exist_ok=True)
        for name in ("ready-for-snapshot", "restore-complete"):
            (self.directory / name).unlink(missing_ok=True)

    async def ready(self) -> None:
        (self.directory / "ready-for-snapshot").touch()

    async def wait_for_restore(self) -> None:
        while not (self.directory / "restore-complete").exists():
            await asyncio.sleep(0.05)
