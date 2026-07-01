# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from contextlib import nullcontext
from types import SimpleNamespace
from unittest.mock import Mock

from vllm.model_executor.warmup.kernel_warmup import flashinfer_autotune


def test_flashinfer_one_sided_autotunes_collectively(monkeypatch):
    runner = SimpleNamespace(
        vllm_config=SimpleNamespace(
            parallel_config=SimpleNamespace(
                all2all_backend="flashinfer_nvlink_one_sided"
            )
        ),
        scheduler_config=SimpleNamespace(max_num_batched_tokens=1024),
        _dummy_run=Mock(),
    )
    world = SimpleNamespace(barrier=Mock())
    autotune = Mock(return_value=nullcontext())

    monkeypatch.setattr("vllm.utils.flashinfer.autotune", autotune)
    monkeypatch.setattr(
        "vllm.distributed.parallel_state.get_world_group", lambda: world
    )

    flashinfer_autotune(runner)

    autotune.assert_called_once_with()
    runner._dummy_run.assert_called_once_with(
        num_tokens=1024,
        skip_eplb=True,
        is_profile=True,
    )
    world.barrier.assert_called_once_with()
