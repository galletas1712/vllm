# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from collections import Counter

_events: Counter[str] = Counter()


def record_nccl_event(event: str) -> None:
    _events[event] += 1


def get_nccl_audit_events() -> dict[str, int]:
    return dict(_events)


def reset_nccl_audit_events() -> None:
    _events.clear()
