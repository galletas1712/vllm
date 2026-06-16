# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from types import SimpleNamespace

import torch

from vllm.v1.worker import gpu_worker
from vllm.v1.worker.gpu_worker import Worker


class _Allocator:
    def __init__(self):
        self.sleep_calls = []
        self.wake_calls = []

    def sleep(self, offload_tags=None):
        self.sleep_calls.append(offload_tags)

    def wake_up(self, tags=None):
        self.wake_calls.append(tags)


class _All2AllManager:
    def __init__(self):
        self.calls = []

    def pause(self):
        self.calls.append("pause")
        return True

    def resume(self):
        self.calls.append("resume")
        return True


class _FlashInferAllReduce:
    def __init__(self):
        self.calls = []

    def pause(self):
        self.calls.append("pause")
        return True

    def resume(self):
        self.calls.append("resume")
        return True


def test_sleep_wake_run_peer_resource_hooks(monkeypatch):
    allocator = _Allocator()
    all2all_manager = _All2AllManager()
    fi_ar_comm = _FlashInferAllReduce()
    worker = object.__new__(Worker)
    worker._sleep_saved_buffers = {}
    worker._paused_peer_resources = []
    worker.model_runner = SimpleNamespace(post_kv_cache_wake_up=lambda: None)

    monkeypatch.setattr(
        gpu_worker,
        "get_mem_allocator_instance",
        lambda: allocator,
    )
    monkeypatch.setattr(
        torch.cuda,
        "mem_get_info",
        lambda: (10, 20),
    )
    monkeypatch.setattr(
        gpu_worker,
        "get_ep_group",
        lambda: SimpleNamespace(
            device_communicator=SimpleNamespace(all2all_manager=all2all_manager),
        ),
    )
    monkeypatch.setattr(
        gpu_worker,
        "get_tp_group",
        lambda: SimpleNamespace(
            device_communicator=SimpleNamespace(fi_ar_comm=fi_ar_comm),
        ),
    )
    monkeypatch.setattr(
        torch.cuda,
        "is_available",
        lambda: False,
    )

    Worker.sleep(worker, level=1)
    Worker.wake_up(worker, tags=["weights"])
    Worker.wake_up(worker, tags=["kv_cache"])

    assert allocator.sleep_calls == [("weights",)]
    assert allocator.wake_calls == [["weights"], ["kv_cache"]]
    assert all2all_manager.calls == ["pause", "resume"]
    assert fi_ar_comm.calls == ["pause", "resume"]
