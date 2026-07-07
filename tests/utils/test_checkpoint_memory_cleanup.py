# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

from vllm.utils import mem_utils


class _FakeFunction:
    def __init__(self, result):
        self.result = result
        self.calls = []

    def __call__(self, *args):
        self.calls.append(args)
        return self.result


class _FakeLibc:
    def __init__(self, trim_result=1):
        self.malloc_trim = _FakeFunction(trim_result)


def test_checkpoint_heap_cleanup_receipt(monkeypatch):
    libc = _FakeLibc()
    snapshots = iter(
        [
            {
                "rss_bytes": 100,
                "heap_used_bytes": 80,
                "heap_free_bytes": 20,
            },
            {
                "rss_bytes": 60,
                "heap_used_bytes": 50,
                "heap_free_bytes": 10,
            },
        ]
    )
    monkeypatch.setattr(mem_utils.ctypes, "CDLL", lambda _: libc)
    monkeypatch.setattr(
        mem_utils, "_process_host_memory_snapshot", lambda _: next(snapshots)
    )
    monkeypatch.setattr(mem_utils.gc, "collect", lambda: 7)
    monkeypatch.setattr(mem_utils.os, "getpid", lambda: 123)

    receipt = mem_utils.run_experimental_checkpoint_heap_cleanup("engine-core")

    assert receipt["pid"] == 123
    assert receipt["role"] == "engine-core"
    assert receipt["gc_collected"] == 7
    assert receipt["malloc_trim_result"] == 1
    assert receipt["before"]["rss_bytes"] == 100
    assert receipt["after"]["rss_bytes"] == 60
    assert libc.malloc_trim.calls == [(0,)]


def test_checkpoint_heap_cleanup_requires_malloc_trim(monkeypatch):
    monkeypatch.setattr(
        mem_utils.ctypes,
        "CDLL",
        lambda _: SimpleNamespace(),
    )

    with pytest.raises(RuntimeError, match="requires libc malloc_trim"):
        mem_utils.run_experimental_checkpoint_heap_cleanup("engine-core")


def test_pinned_host_cache_flush_receipt(monkeypatch):
    empty_cache = MagicMock()
    stats = MagicMock(side_effect=[{"reserved_bytes.all.current": 10}, {}])
    monkeypatch.setattr(mem_utils.torch._C, "_host_emptyCache", empty_cache)
    monkeypatch.setattr(mem_utils.torch.cuda.memory, "host_memory_stats", stats)
    monkeypatch.setattr(mem_utils.os, "getpid", lambda: 456)

    receipt = mem_utils.flush_experimental_pinned_host_cache("worker", 3)

    assert receipt["pid"] == 456
    assert receipt["rank"] == 3
    assert receipt["capability_path"] == "torch._C._host_emptyCache"
    assert receipt["host_memory_stats_before"] == {"reserved_bytes.all.current": 10}
    assert receipt["host_memory_stats_after"] == {}
    empty_cache.assert_called_once_with()


def test_pinned_host_cache_flush_requires_private_capability(monkeypatch):
    monkeypatch.delattr(mem_utils.torch._C, "_host_emptyCache", raising=False)

    with pytest.raises(RuntimeError, match=r"torch\._C\._host_emptyCache"):
        mem_utils.flush_experimental_pinned_host_cache("worker", 0)
