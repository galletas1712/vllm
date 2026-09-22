# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import asyncio
import socket
from functools import partial
from typing import Any

from vllm.snapshot.lifecycle.channel import ControlChannel


class CheckpointCoordinator:
    """Serialize HTTP lifecycle operations across all local frontends and engines."""

    def __init__(self, frontend_count: int):
        pairs = [socket.socketpair() for _ in range(frontend_count)]
        self.sockets = [pair[0] for pair in pairs]
        self.frontend_sockets = [pair[1] for pair in pairs]
        self.channels: list[ControlChannel] = []
        self.joined: set[int] = set()
        self.ready = asyncio.Event()
        self.lock = asyncio.Lock()
        self.state = "starting"

    def close_frontend_sockets(self) -> None:
        for sock in self.frontend_sockets:
            sock.close()

    def close(self) -> None:
        self.close_frontend_sockets()
        for sock in self.sockets:
            sock.close()

    async def run(
        self, *, policy: str, options: dict[str, Any], clear_cache: bool
    ) -> None:
        self.policy = policy
        self.options = options
        self.clear_cache = clear_cache
        try:
            for index, sock in enumerate(self.sockets):
                self.channels.append(
                    await ControlChannel.connect(sock, partial(self._handle, index))
                )
            await asyncio.gather(*(channel.task for channel in self.channels))
        finally:
            await asyncio.gather(*(channel.close() for channel in self.channels))

    async def _all(self, method: str, *args: Any) -> None:
        # Wait for every participant even when one fails; no preparation work
        # may still be running when the transition is reported as failed.
        results = await asyncio.gather(
            *(channel.call(method, *args) for channel in self.channels),
            return_exceptions=True,
        )
        for result in results:
            if isinstance(result, BaseException):
                raise result

    async def _handle(self, index: int, method: str) -> dict[str, str]:
        if method == "join":
            self.joined.add(index)
            if len(self.joined) == len(self.sockets):
                self.state = "running"
                self.ready.set()
            await self.ready.wait()
        elif method == "status":
            pass
        elif method in ("prepare", "resume"):
            async with self.lock:
                target = "prepared" if method == "prepare" else "running"
                if self.state == target:
                    return {"state": self.state}
                expected = "running" if method == "prepare" else "prepared"
                if self.state != expected:
                    raise RuntimeError(f"Cannot {method} checkpoint in {self.state}")
                self.state = "preparing" if method == "prepare" else "resuming"
                leader = self.channels[0]
                try:
                    if method == "prepare":
                        await self._all("quiesce")
                        await self._all("drain")
                        await leader.call(
                            "utility", "pause_scheduler", "wait", self.clear_cache
                        )
                        if self.clear_cache:
                            await self._all("clear_cache")
                        await leader.call(
                            "utility", "checkpoint_prepare", self.policy, self.options
                        )
                    else:
                        await leader.call("utility", "checkpoint_restore")
                        await leader.call("utility", "resume_scheduler")
                        await self._all("admit")
                    self.state = target
                except Exception:
                    self.state = "failed"
                    await self._all("quiesce")
                    raise
        else:
            raise ValueError(f"Unknown checkpoint operation: {method}")
        return {"state": self.state}
