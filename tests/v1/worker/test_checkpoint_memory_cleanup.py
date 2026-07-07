# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
from types import SimpleNamespace
from unittest.mock import MagicMock

from vllm.v1.engine import core
from vllm.v1.worker import gpu_worker


def test_engine_core_heap_cleanup_follows_worker_sleep(monkeypatch):
    calls = []
    engine = core.EngineCore.__new__(core.EngineCore)
    engine.model_executor = SimpleNamespace(
        sleep=lambda level: calls.append(("worker-sleep", level))
    )
    engine.vllm_config = SimpleNamespace(
        parallel_config=SimpleNamespace(data_parallel_index=2)
    )
    engine.pause_scheduler = MagicMock(return_value=None)
    monkeypatch.setattr(
        core.envs,
        "VLLM_EXPERIMENT_CHECKPOINT_HEAP_CLEANUP",
        True,
    )
    monkeypatch.setattr(
        core,
        "run_experimental_checkpoint_heap_cleanup",
        lambda role: calls.append(("heap-cleanup", role)),
    )

    engine.sleep(level=1)

    assert calls == [
        ("worker-sleep", 1),
        ("heap-cleanup", "vllm-engine-core-dp-rank-2"),
    ]


def test_gpu_worker_checkpoint_cleanup_order(monkeypatch):
    calls = []
    worker = gpu_worker.Worker.__new__(gpu_worker.Worker)
    worker.rank = 3
    worker._get_sleep_mode_backend = lambda: SimpleNamespace(
        suspend=lambda level: calls.append(("suspend", level))
    )
    memory_info = iter([(100, 1000), (200, 1000)])
    monkeypatch.setattr(
        gpu_worker.torch.accelerator,
        "synchronize",
        lambda: calls.append(("synchronize", None)),
    )
    monkeypatch.setattr(
        gpu_worker.torch.accelerator,
        "get_memory_info",
        lambda: next(memory_info),
    )
    monkeypatch.setattr(gpu_worker.current_platform, "is_rocm", lambda: False)
    monkeypatch.setattr(
        gpu_worker.envs,
        "VLLM_EXPERIMENT_CHECKPOINT_PINNED_HOST_CACHE_FLUSH",
        True,
    )
    monkeypatch.setattr(
        gpu_worker.envs,
        "VLLM_EXPERIMENT_CHECKPOINT_HEAP_CLEANUP",
        True,
    )
    monkeypatch.setattr(
        gpu_worker,
        "flush_experimental_pinned_host_cache",
        lambda role, rank: calls.append(("pinned-flush", role, rank)),
    )
    monkeypatch.setattr(
        gpu_worker,
        "run_experimental_checkpoint_heap_cleanup",
        lambda role: calls.append(("heap-cleanup", role)),
    )

    worker.sleep(level=1)

    assert calls == [
        ("synchronize", None),
        ("suspend", 1),
        ("synchronize", None),
        ("pinned-flush", "vllm-gpu-worker", 3),
        ("heap-cleanup", "vllm-gpu-worker-rank-3"),
    ]


def test_gpu_worker_cleanup_gates_default_off(monkeypatch):
    worker = gpu_worker.Worker.__new__(gpu_worker.Worker)
    worker.rank = 0
    worker._get_sleep_mode_backend = lambda: SimpleNamespace(suspend=lambda level: None)
    memory_info = iter([(100, 1000), (100, 1000)])
    monkeypatch.setattr(gpu_worker.torch.accelerator, "synchronize", lambda: None)
    monkeypatch.setattr(
        gpu_worker.torch.accelerator,
        "get_memory_info",
        lambda: next(memory_info),
    )
    monkeypatch.setattr(gpu_worker.current_platform, "is_rocm", lambda: False)
    monkeypatch.setattr(
        gpu_worker.envs,
        "VLLM_EXPERIMENT_CHECKPOINT_PINNED_HOST_CACHE_FLUSH",
        False,
    )
    monkeypatch.setattr(
        gpu_worker.envs,
        "VLLM_EXPERIMENT_CHECKPOINT_HEAP_CLEANUP",
        False,
    )
    pinned_flush = MagicMock()
    heap_cleanup = MagicMock()
    monkeypatch.setattr(
        gpu_worker, "flush_experimental_pinned_host_cache", pinned_flush
    )
    monkeypatch.setattr(
        gpu_worker, "run_experimental_checkpoint_heap_cleanup", heap_cleanup
    )

    worker.sleep(level=1)

    pinned_flush.assert_not_called()
    heap_cleanup.assert_not_called()
