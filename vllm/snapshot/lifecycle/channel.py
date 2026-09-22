# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import asyncio
import json
import socket
from collections.abc import Awaitable, Callable
from typing import Any


class ControlChannel:
    """Duplex utility calls over a socket captured with the serving process tree."""

    def __init__(self, reader, writer, handle, disconnected):
        self.reader = reader
        self.writer = writer
        self.handle = handle
        self.disconnected = disconnected
        self.pending: dict[int, asyncio.Future] = {}
        self.handlers: set[asyncio.Task] = set()
        self.sequence = 0
        self.task = asyncio.create_task(self._read())

    @classmethod
    async def connect(
        cls,
        sock: socket.socket,
        handle: Callable[..., Awaitable[Any]],
        disconnected: Callable[[], None] = lambda: None,
    ) -> "ControlChannel":
        sock.setblocking(False)
        reader, writer = await asyncio.open_connection(sock=sock)
        return cls(reader, writer, handle, disconnected)

    async def _send(self, message: dict[str, Any]) -> None:
        self.writer.write(json.dumps(message).encode() + b"\n")
        await self.writer.drain()

    async def call(self, method: str, *args: Any) -> Any:
        if self.task.done():
            raise RuntimeError("Checkpoint control channel is closed")
        self.sequence += 1
        request_id = self.sequence
        future = asyncio.get_running_loop().create_future()
        self.pending[request_id] = future
        # Losing an HTTP caller must not cancel a service-wide transition.
        try:
            await self._send({"id": request_id, "method": method, "args": args})
            return await asyncio.shield(future)
        except asyncio.CancelledError:
            future.add_done_callback(lambda done: done.exception())
            raise
        except (ConnectionError, OSError):
            self.pending.pop(request_id, None)
            future.cancel()
            raise

    async def _dispatch(self, message: dict[str, Any]) -> None:
        response = {"id": message["id"]}
        try:
            response["result"] = await self.handle(message["method"], *message["args"])
        except Exception as error:
            response["error"] = str(error)
        await self._send(response)

    def _dispatched(self, task: asyncio.Task) -> None:
        self.handlers.discard(task)
        if not task.cancelled() and task.exception() is not None:
            self.writer.close()

    async def _read(self) -> None:
        try:
            while line := await self.reader.readline():
                message = json.loads(line)
                if "method" in message:
                    task = asyncio.create_task(self._dispatch(message))
                    self.handlers.add(task)
                    task.add_done_callback(self._dispatched)
                else:
                    future = self.pending.pop(message["id"])
                    if "error" in message:
                        future.set_exception(RuntimeError(message["error"]))
                    else:
                        future.set_result(message["result"])
            raise RuntimeError("Checkpoint control channel disconnected")
        finally:
            self.disconnected()
            for future in self.pending.values():
                future.set_exception(RuntimeError("Checkpoint control channel closed"))
            self.pending.clear()

    async def close(self) -> None:
        tasks = [self.task, *self.handlers]
        for task in tasks:
            task.cancel()
        await asyncio.gather(*tasks, return_exceptions=True)
        self.writer.close()
        await self.writer.wait_closed()
