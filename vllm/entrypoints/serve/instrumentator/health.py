# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project


from fastapi import APIRouter, Request
from fastapi.responses import Response

from vllm.engine.protocol import EngineClient
from vllm.logger import init_logger
from vllm.v1.engine import CheckpointState
from vllm.v1.engine.exceptions import EngineDeadError

logger = init_logger(__name__)


router = APIRouter()


def engine_client(request: Request) -> EngineClient:
    return request.app.state.engine_client


@router.get("/health", response_class=Response)
async def health(raw_request: Request) -> Response:
    """Health check."""
    client = engine_client(raw_request)
    if client is None:
        # Render-only servers have no engine; they are always healthy.
        return Response(status_code=200)
    try:
        await client.check_health()
    except EngineDeadError:
        return Response(status_code=503)
    # Engines started with --checkpoint-prepare-on-startup serve only after
    # /checkpoint/resume. Once serving, they stay serving, so the engine is
    # asked only until then.
    app_state = raw_request.app.state
    if not getattr(app_state, "engine_serving", False):
        if client.vllm_config.model_config.checkpoint_prepare_on_startup:
            state, _ = await client.checkpoint_state()
            if state != CheckpointState.RUNNING:
                return Response(status_code=503)
        app_state.engine_serving = True
    return Response(status_code=200)
