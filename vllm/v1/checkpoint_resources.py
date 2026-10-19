# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Process state that a checkpoint must not contain, beyond device communicators.

A resource has the same hooks as a device communicator: `checkpoint_prepare()`
releases what reaches outside the process (network endpoints, NIC
registrations, memory shared with other processes) and `checkpoint_restore()`
brings it back, or creates it for the first time when the engine prepared at
startup.

A resource whose state cannot be fully released, such as one holding native
libraries with process-wide caches, sets `restartable` to False once it has
created that state. The engine then refuses to prepare before anything is
released, and keeps serving.

The engine and its workers each prepare and restore the resources of their
process. A worker that shares the engine's process finds them already handled.
"""

import weakref
from typing import Protocol


class CheckpointResource(Protocol):
    def checkpoint_prepare(self) -> None: ...

    def checkpoint_restore(self) -> None: ...


# An ordered weak set: a resource leaves once it is garbage collected.
_resources = weakref.WeakKeyDictionary[CheckpointResource, None]()
# Set while prepared. When the engine and a worker share a process, the second
# caller finds the resources already handled.
_prepared = False


def register_checkpoint_resource(resource: CheckpointResource) -> None:
    _resources[resource] = None


def checkpoint_blockers() -> list[str]:
    """Names of the resources of this process that cannot be released."""
    return [type(r).__name__ for r in _resources if not getattr(r, "restartable", True)]


def prepare_checkpoint_resources() -> None:
    global _prepared
    if _prepared:
        return
    for resource in reversed(list(_resources)):
        resource.checkpoint_prepare()
    _prepared = True


def restore_checkpoint_resources() -> None:
    global _prepared
    if _prepared:
        for resource in list(_resources):
            resource.checkpoint_restore()
        _prepared = False
