# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from unittest.mock import AsyncMock

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from vllm.entrypoints.serve.checkpoint.api_router import attach_router


@pytest.mark.cpu_test
@pytest.mark.parametrize("prepared", [False, True])
def test_checkpoint_routes(prepared):
    """Status returns 200 only once prepared, so orchestrators can poll it."""
    app = FastAPI()
    app.state.engine_client = engine = AsyncMock()
    engine.is_checkpoint_prepared.return_value = prepared
    attach_router(app)

    with TestClient(app) as client:
        assert client.post("/checkpoint/prepare").status_code == 200
        status = client.get("/checkpoint/status")
        assert client.post("/checkpoint/resume").status_code == 200

    assert status.status_code == (200 if prepared else 503)
    assert status.json() == {"prepared": prepared}
    engine.checkpoint_prepare.assert_awaited_once_with()
    engine.checkpoint_restore.assert_awaited_once_with()
