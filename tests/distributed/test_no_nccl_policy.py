# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import os
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest
import torch

import vllm.envs as envs
from vllm.distributed.device_communicators.all2all import (
    FlashInferNVLinkOneSidedManager,
    FlashInferNVLinkTwoSidedManager,
)
from vllm.distributed.device_communicators.cuda_communicator import CudaCommunicator
from vllm.distributed.nccl_audit import (
    assert_no_nccl_communicators,
    get_nccl_audit_events,
    record_nccl_event,
    reset_nccl_audit_events,
)


def test_disable_nccl_env(monkeypatch):
    monkeypatch.setenv("VLLM_DISABLE_NCCL", "1")
    assert envs.VLLM_DISABLE_NCCL is True


def test_disable_nccl_removes_runtime_configuration(monkeypatch):
    monkeypatch.setenv("VLLM_DISABLE_NCCL", "1")
    legacy_env = {
        "NCCL_CHECKPOINT_SHIM": "/opt/nccl-checkpoint/lib/shim.so",
        "NCCL_DEBUG": "INFO",
        "TORCH_NCCL_ENABLE_MONITORING": "1",
        "DYN_SNAPSHOT_NCCL_KVS_ENDPOINT": "redis://legacy",
    }
    for name, value in legacy_env.items():
        monkeypatch.setenv(name, value)

    envs._remove_nccl_environment_for_no_nccl_policy()

    for name in legacy_env:
        assert name not in os.environ
    assert os.environ["VLLM_DISABLE_NCCL"] == "1"


def test_pynccl_creation_is_rejected_and_audited(monkeypatch):
    monkeypatch.setenv("VLLM_DISABLE_NCCL", "1")
    reset_nccl_audit_events()

    from vllm.distributed.device_communicators.pynccl import PyNcclCommunicator

    with pytest.raises(RuntimeError, match="forbidden"):
        PyNcclCommunicator(MagicMock(), "cuda:0")

    assert get_nccl_audit_events() == {"pynccl_communicator_create": 1}


def test_nccl_audit_rejects_recorded_creation_attempt():
    reset_nccl_audit_events()
    record_nccl_event("process_group_nccl_init")

    with pytest.raises(RuntimeError, match="creation was attempted"):
        assert_no_nccl_communicators()

    reset_nccl_audit_events()


def test_unsupported_collectives_fail_closed():
    communicator = CudaCommunicator.__new__(CudaCommunicator)
    communicator.disable_nccl = True
    communicator.world_size = 2
    communicator.rank_in_group = 0

    with pytest.raises(RuntimeError, match="reduce-scatter"):
        communicator.reduce_scatter(torch.empty(1))
    with pytest.raises(RuntimeError, match="GPU P2P"):
        communicator.send(torch.empty(1))
    with pytest.raises(RuntimeError, match="one size per rank"):
        communicator.all_gatherv(torch.empty(1), sizes=[1])
    with pytest.raises(RuntimeError, match="variable all-gather"):
        communicator.all_gatherv(torch.empty(1), sizes=[1, 2])


def test_all_gather_policy_overrides_nccl_symmetric_memory():
    communicator = CudaCommunicator.__new__(CudaCommunicator)
    communicator.disable_nccl = True
    input_ = torch.empty(1)
    expected = torch.empty(2)
    communicator._flashinfer_all_gather = MagicMock(return_value=expected)
    communicator._all_gather_symm_mem = MagicMock()

    with patch(
        "vllm.distributed.device_communicators.cuda_communicator."
        "should_nccl_symm_mem_ag_rs",
        return_value=True,
    ):
        assert communicator.all_gather(input_, dim=0) is expected

    communicator._flashinfer_all_gather.assert_called_once_with(input_, 0)
    communicator._all_gather_symm_mem.assert_not_called()


def test_deepep_v2_probe_is_skipped(monkeypatch):
    monkeypatch.setenv("VLLM_DISABLE_NCCL", "1")

    with (
        patch("vllm.utils.import_utils._has_module") as has_module,
        patch(
            "vllm.utils.import_utils._get_runtime_nccl_version"
        ) as get_runtime_nccl_version,
    ):
        from vllm.utils.import_utils import has_deep_ep_v2

        assert has_deep_ep_v2() is False

    has_module.assert_not_called()
    get_runtime_nccl_version.assert_not_called()


def test_stateless_process_group_nccl_is_rejected(monkeypatch):
    monkeypatch.setenv("VLLM_DISABLE_NCCL", "1")
    reset_nccl_audit_events()

    from vllm.platforms.cuda import CudaPlatformBase

    with pytest.raises(RuntimeError, match="ProcessGroupNCCL"):
        CudaPlatformBase.stateless_init_device_torch_dist_pg(
            "nccl", MagicMock(), 0, 1, MagicMock()
        )

    assert get_nccl_audit_events() == {"stateless_process_group_nccl_init": 1}
    reset_nccl_audit_events()


def test_two_sided_checkpoint_is_rejected():
    manager = FlashInferNVLinkTwoSidedManager.__new__(FlashInferNVLinkTwoSidedManager)
    manager.initialized = True

    with pytest.raises(NotImplementedError, match="one_sided"):
        manager.checkpoint_prepare()
    with pytest.raises(NotImplementedError, match="one_sided"):
        manager.checkpoint_restore()


def test_one_sided_restore_uses_fresh_control_backend():
    manager = FlashInferNVLinkOneSidedManager.__new__(FlashInferNVLinkOneSidedManager)
    manager.initialized = True
    manager.cpu_group = MagicMock()
    manager.moe_alltoall = MagicMock()
    fresh_backend = object()

    with patch(
        "vllm.distributed.device_communicators.mnnvl_compat.CustomCommunicator",
        return_value=fresh_backend,
    ):
        manager.checkpoint_prepare()
        manager.checkpoint_restore()

    manager.moe_alltoall.checkpoint_prepare.assert_called_once_with()
    manager.moe_alltoall.checkpoint_restore.assert_called_once_with(fresh_backend)


def test_no_nccl_rejects_elastic_ep(monkeypatch):
    monkeypatch.setenv("VLLM_DISABLE_NCCL", "1")
    config = SimpleNamespace(
        parallel_config=SimpleNamespace(
            enable_elastic_ep=True,
            pipeline_parallel_size=1,
        )
    )

    with (
        patch("vllm.config.get_current_vllm_config_or_none", return_value=config),
        pytest.raises(ValueError, match="elastic EP"),
    ):
        from vllm.distributed.parallel_state import init_distributed_environment

        init_distributed_environment()
