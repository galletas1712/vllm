# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from vllm.distributed.device_communicators.all2all import (
    NixlEPAll2AllManager,
    _NixlEPBufferState,
)


class _FakeNixlBuffer:
    def __init__(self):
        self.calls = []
        self.addresses = {"rdma_buffer": 1, "gpu_ctx": 2}

    def get_graph_visible_addresses(self):
        self.calls.append(("get_graph_visible_addresses",))
        return self.addresses

    def checkpoint_pause_preserve_va(self):
        self.calls.append(("checkpoint_pause_preserve_va",))
        return self.addresses

    def set_tcp_store_group(self, store):
        self.calls.append(("set_tcp_store_group", store))

    def checkpoint_resume_preserve_va(
        self, remote_ranks, activate=True, expected_addresses=None
    ):
        self.calls.append(
            (
                "checkpoint_resume_preserve_va",
                remote_ranks,
                activate,
                expected_addresses,
            )
        )

    def update_mask_buffer(self, rank, mask=False):
        self.calls.append(("update_mask_buffer", rank, mask))

    def validate_graph_visible_addresses(self, expected):
        self.calls.append(("validate_graph_visible_addresses", expected))
        return expected == self.addresses


class _FakeTcpStoreGroup:
    store = object()


def test_nixl_ep_checkpoint_hooks_restore_connected_and_active_ranks():
    buffer = _FakeNixlBuffer()
    manager = NixlEPAll2AllManager.__new__(NixlEPAll2AllManager)
    manager.tcp_store_group = _FakeTcpStoreGroup()
    previous_buffer = NixlEPAll2AllManager._buffer
    try:
        NixlEPAll2AllManager._buffer = _NixlEPBufferState(
            buffer=buffer,
            connected_ep_size=4,
            active_ep_size=2,
        )

        expected = manager.checkpoint_pause_preserve_va()
        manager.checkpoint_resume_preserve_va(expected)

        assert ("checkpoint_pause_preserve_va",) in buffer.calls
        assert (
            "checkpoint_resume_preserve_va",
            [0, 1, 2, 3],
            False,
            expected,
        ) in buffer.calls
        assert ("update_mask_buffer", 0, False) in buffer.calls
        assert ("update_mask_buffer", 1, False) in buffer.calls
        assert ("update_mask_buffer", 2, False) not in buffer.calls
        assert ("validate_graph_visible_addresses", expected) in buffer.calls
    finally:
        NixlEPAll2AllManager._buffer = previous_buffer


def test_nixl_ep_checkpoint_hooks_support_high_throughput_full_active_set():
    buffer = _FakeNixlBuffer()
    manager = NixlEPAll2AllManager.__new__(NixlEPAll2AllManager)
    manager.tcp_store_group = _FakeTcpStoreGroup()
    previous_buffer = NixlEPAll2AllManager._buffer
    try:
        NixlEPAll2AllManager._buffer = _NixlEPBufferState(
            buffer=buffer,
            connected_ep_size=4,
            active_ep_size=4,
        )

        expected = manager.checkpoint_pause_preserve_va()
        manager.checkpoint_resume_preserve_va(expected)

        assert (
            "checkpoint_resume_preserve_va",
            [0, 1, 2, 3],
            False,
            expected,
        ) in buffer.calls
        for rank in range(4):
            assert ("update_mask_buffer", rank, False) in buffer.calls
    finally:
        NixlEPAll2AllManager._buffer = previous_buffer
