# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from fastapi import APIRouter, FastAPI, Request
from fastapi.responses import JSONResponse, Response

from vllm.engine.protocol import EngineClient


def engine_client(request: Request) -> EngineClient:
    return request.app.state.engine_client


router = APIRouter()


@router.post("/checkpoint/prepare")
async def checkpoint_prepare(raw_request: Request):
    await engine_client(raw_request).checkpoint_prepare()
    return Response(status_code=200)


@router.post("/checkpoint/resume")
async def checkpoint_resume(raw_request: Request):
    await engine_client(raw_request).checkpoint_restore()
    return Response(status_code=200)


@router.get("/checkpoint/status")
async def checkpoint_status(raw_request: Request):
    # Orchestrators poll this route to learn when the process is safe to capture.
    prepared = await engine_client(raw_request).is_checkpoint_prepared()
    return JSONResponse(
        content={"prepared": prepared}, status_code=200 if prepared else 503
    )


def attach_router(app: FastAPI):
    app.include_router(router)
