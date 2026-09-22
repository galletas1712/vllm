# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import asyncio
from typing import Any

from fastapi import FastAPI, HTTPException
from starlette.responses import JSONResponse
from starlette.types import ASGIApp, Receive, Scope, Send

from vllm.snapshot.lifecycle.channel import ControlChannel


class CheckpointFrontend:
    """HTTP control and admission for one member of a serving process tree."""

    def __init__(self, call, clear_cache):
        self.channel: ControlChannel | None = None
        self.call = call
        self.clear_cache = clear_cache
        self.accepting = False
        self.active = 0
        self.app: FastAPI | None = None
        self.idle = asyncio.Event()
        self.idle.set()

    async def connect(self, sock) -> None:
        self.channel = await ControlChannel.connect(sock, self._handle, self.quiesce)
        await self.channel.call("join")
        self.accepting = True

    async def close(self) -> None:
        if self.channel is not None:
            await self.channel.close()

    def quiesce(self) -> None:
        self.accepting = False

    async def _handle(self, method: str, *args: Any) -> None:
        if method == "quiesce":
            self.quiesce()
        elif method == "drain":
            await self.idle.wait()
            # Responses background mode can outlive its HTTP request.
            assert self.app is not None
            responses = getattr(self.app.state, "openai_serving_responses", None)
            if responses is not None:
                await asyncio.gather(
                    *list(responses.background_tasks.values()), return_exceptions=True
                )
        elif method == "clear_cache":
            await self.clear_cache()
        elif method == "admit":
            self.accepting = True
        elif method == "utility":
            await self.call(*args)
        else:
            raise ValueError(f"Unknown checkpoint command: {method}")

    def attach(self, app: FastAPI) -> None:
        self.app = app

        async def operation(method: str) -> dict[str, str]:
            assert self.channel is not None
            try:
                return await self.channel.call(method)
            except RuntimeError as error:
                raise HTTPException(status_code=503, detail=str(error)) from error

        @app.post("/checkpoint/prepare")
        async def prepare():
            return await operation("prepare")

        @app.post("/checkpoint/resume")
        async def resume():
            return await operation("resume")

        @app.get("/checkpoint/status")
        async def status():
            return await operation("status")

        app.add_middleware(CheckpointAdmission, checkpoint=self)


class CheckpointAdmission:
    """Track the complete ASGI response, including streaming and background work."""

    def __init__(self, app: ASGIApp, checkpoint: CheckpointFrontend):
        self.app = app
        self.checkpoint = checkpoint

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        if scope["type"] not in ("http", "websocket") or scope["path"].removeprefix(
            scope.get("root_path", "")
        ).startswith("/checkpoint/"):
            await self.app(scope, receive, send)
            return
        checkpoint = self.checkpoint
        if not checkpoint.accepting:
            if scope["type"] == "websocket":
                await send({"type": "websocket.close", "code": 1013})
            else:
                await JSONResponse(
                    {"detail": "Service is quiesced for checkpointing"}, status_code=503
                )(scope, receive, send)
            return
        checkpoint.active += 1
        checkpoint.idle.clear()
        try:
            await self.app(scope, receive, send)
        finally:
            checkpoint.active -= 1
            if checkpoint.active == 0:
                checkpoint.idle.set()
