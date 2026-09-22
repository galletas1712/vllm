# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Workload checkpoint lifecycle, independent of snapshot artifact management."""

from vllm.snapshot.lifecycle.policy import CheckpointPolicy, ResidentPolicy

__all__ = ["CheckpointPolicy", "ResidentPolicy"]
