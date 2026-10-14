# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from typing import Any

from fastapi import APIRouter, FastAPI, Request
from fastapi.responses import JSONResponse
from pydantic import BaseModel

from vllm.engine.protocol import EngineClient
from vllm.v1.engine import CheckpointState


def engine_client(request: Request) -> EngineClient:
    return request.app.state.engine_client


router = APIRouter()


def _state_response(
    state: CheckpointState, error: str | None, status_code: int
) -> JSONResponse:
    content = {"state": state} if error is None else {"state": state, "error": error}
    return JSONResponse(content=content, status_code=status_code)


class CheckpointResumeRequest(BaseModel):
    # Environment of the pod restored into. A null value unsets a variable.
    env: dict[str, str | None] = {}
    # Engine arguments of the pod restored into.
    flags: dict[str, Any] = {}


@router.post("/checkpoint/resume")
async def checkpoint_resume(
    raw_request: Request, request: CheckpointResumeRequest | None = None
):
    """Bring back what preparation released and serve. Repeating is a no-op."""
    client = engine_client(raw_request)
    state, error = await client.checkpoint_state()
    if state in (
        CheckpointState.PREPARING,
        CheckpointState.RESUMING,
        CheckpointState.FAILED,
    ):
        return _state_response(state, error, 409)
    if state != CheckpointState.RUNNING:
        request = request or CheckpointResumeRequest()
        try:
            await client.checkpoint_restore(request.env, request.flags)
        except ValueError as e:
            return JSONResponse(content={"error": str(e)}, status_code=400)
        await client.wake_up(["kv_cache"])
    return _state_response(CheckpointState.RUNNING, None, 200)


@router.get("/checkpoint/status")
async def checkpoint_status(raw_request: Request):
    # Orchestrators poll this route to learn when the process is safe to capture.
    state, error = await engine_client(raw_request).checkpoint_state()
    status_code = 200 if state == CheckpointState.PREPARED else 503
    return _state_response(state, error, status_code)


def attach_router(app: FastAPI):
    app.include_router(router)
