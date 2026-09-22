# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import asyncio
import json
import socket
from collections.abc import Awaitable, Callable
from typing import Any

from vllm.logger import init_logger
from vllm.snapshot.lifecycle.signaling import CheckpointSignal

logger = init_logger(__name__)


class CheckpointCoordinator:
    """Own the startup barrier for an entire local serve process tree.

    All frontends connect to their engines before joining. Only one frontend
    broadcasts utility calls to all engines; the others remain parked. The
    internal sockets are captured with the process tree, so a reused image
    cannot observe leftover internal release files from an earlier restore.
    """

    def __init__(self, frontend_count: int):
        pairs = [socket.socketpair() for _ in range(frontend_count)]
        self.sockets = [pair[0] for pair in pairs]
        self.frontend_sockets = [pair[1] for pair in pairs]

    def close_frontend_sockets(self) -> None:
        for sock in self.frontend_sockets:
            sock.close()

    def close(self) -> None:
        self.close_frontend_sockets()
        for sock in self.sockets:
            sock.close()

    async def run(
        self,
        signal: CheckpointSignal,
        *,
        policy: str,
        options: dict[str, Any],
        clear_cache: bool,
    ) -> None:
        streams = []
        try:
            for sock in self.sockets:
                sock.setblocking(False)
                streams.append(await asyncio.open_connection(sock=sock))
            await asyncio.gather(*(_expect(reader, "ready") for reader, _ in streams))
            reader, writer = streams[0]

            async def call(method: str, *args: Any) -> None:
                writer.write(
                    json.dumps({"method": method, "args": args}).encode() + b"\n"
                )
                await writer.drain()
                await _expect(reader, "ok")

            await call("pause_scheduler", "wait", clear_cache)
            await call("checkpoint_prepare", policy, options)
            logger.info("Checkpoint group prepared; publishing readiness")
            await signal.ready()
            # No timeout and no rollback that could touch CUDA while the
            # external orchestrator is reconstructing GPU resources.
            await signal.wait_for_restore()
            logger.info("External restore complete; recovering checkpoint resources")
            await call("checkpoint_restore")
            await call("resume_scheduler")
            for _, writer in streams:
                writer.write(b'{"method":"serve","args":[]}\n')
                await writer.drain()
            logger.info("Checkpoint recovery complete; frontends released to serve")
        finally:
            for _, writer in streams:
                writer.close()
            await asyncio.gather(*(writer.wait_closed() for _, writer in streams))


async def _expect(reader: asyncio.StreamReader, message: str) -> None:
    line = await reader.readline()
    if not line or json.loads(line) != message:
        raise RuntimeError(
            f"Checkpoint participant did not acknowledge {message}: {line!r}"
        )


async def join_checkpoint(
    sock: socket.socket, call: Callable[..., Awaitable[Any]]
) -> None:
    """Join before starting a frontend listener; propagate utility failures."""
    sock.setblocking(False)
    reader, writer = await asyncio.open_connection(sock=sock)
    try:
        writer.write(b'"ready"\n')
        await writer.drain()
        while line := await reader.readline():
            command = json.loads(line)
            if command["method"] == "serve":
                return
            await call(command["method"], *command["args"])
            writer.write(b'"ok"\n')
            await writer.drain()
        raise RuntimeError("Checkpoint coordinator exited before releasing frontend")
    finally:
        writer.close()
        await writer.wait_closed()
