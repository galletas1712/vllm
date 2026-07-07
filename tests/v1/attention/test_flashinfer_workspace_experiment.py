# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

flashinfer = pytest.importorskip("vllm.v1.attention.backends.flashinfer")


def _builder():
    builder = flashinfer.FlashInferMetadataBuilder.__new__(
        flashinfer.FlashInferMetadataBuilder
    )
    builder._workspace_buffer = None
    builder._workspace_buffer_provider = None
    builder.device = "cuda"
    return builder


def test_native_workspace_provider_shares_lazy_allocation():
    owner = _builder()
    follower = _builder()
    workspace = SimpleNamespace(
        data_ptr=lambda: 0x123,
        numel=lambda: 16,
        element_size=lambda: 1,
    )
    owner._workspace_buffer = workspace
    follower.set_workspace_buffer_provider(owner._get_workspace_buffer)

    assert follower._get_workspace_buffer("prefill-wrapper") is workspace
    assert follower._workspace_buffer is owner._workspace_buffer


def test_native_workspace_provider_rejects_late_install():
    builder = _builder()
    builder._workspace_buffer = MagicMock()

    with pytest.raises(RuntimeError, match="after allocation"):
        builder.set_workspace_buffer_provider(MagicMock())


def test_direct_trt_workspace_stays_separate(monkeypatch):
    direct = SimpleNamespace(
        data_ptr=lambda: 0x222,
        numel=lambda: 16,
        element_size=lambda: 1,
    )
    monkeypatch.setattr(flashinfer, "trtllm_workspace_buffer", None)
    monkeypatch.setattr(flashinfer.torch, "zeros", lambda *args, **kwargs: direct)
    flashinfer._native_workspace_ptrs.clear()
    flashinfer._native_workspace_ptrs.add(0x111)

    assert flashinfer._get_trtllm_workspace_buffer() is direct
    assert direct.data_ptr() not in flashinfer._native_workspace_ptrs
