# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from collections import Counter
from typing import Any

_events: Counter[str] = Counter()


def record_nccl_event(event: str) -> None:
    _events[event] += 1


def get_nccl_audit_events() -> dict[str, int]:
    return dict(_events)


def reset_nccl_audit_events() -> None:
    _events.clear()


def assert_no_nccl_communicators() -> dict[str, str]:
    """Fail if the current worker created an NCCL group or communicator."""
    events = get_nccl_audit_events()
    if events:
        raise RuntimeError(f"NCCL creation was attempted: {events}")

    import torch.distributed as dist

    from vllm.distributed import parallel_state

    backends: dict[str, str] = {}
    if dist.is_initialized():
        backends["default"] = str(dist.get_backend())

    for name, group_ref in list(parallel_state._groups.items()):
        coordinator = group_ref()
        if coordinator is None:
            continue

        communicator: Any = coordinator.device_communicator
        if communicator is not None and getattr(
            communicator, "pynccl_comm", None
        ) is not None:
            raise RuntimeError(f"NCCL PyNccl communicator exists for group {name}")

        for kind in ("cpu_group", "device_group"):
            process_group = getattr(coordinator, kind, None)
            if process_group is not None:
                backends[f"{name}.{kind}"] = str(dist.get_backend(process_group))

    nccl_groups = {
        name: backend for name, backend in backends.items() if "nccl" in backend.lower()
    }
    if nccl_groups:
        raise RuntimeError(f"NCCL process groups exist: {nccl_groups}")
    return backends
