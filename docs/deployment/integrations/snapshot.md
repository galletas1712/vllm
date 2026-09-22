# Externally orchestrated startup checkpoints

`vllm serve` can prepare its complete local process tree for an external
checkpoint orchestrator, including [ai-dynamo/snapshot](https://github.com/ai-dynamo/snapshot).
The Python launcher owns the barrier for either the Python or Rust frontend.
The frontend connects to all engines before joining the barrier; HTTP and gRPC
serving start only after recovery finishes.

```bash
vllm serve Qwen/Qwen3-0.6B --snapshot-control-dir /snapshot-control
```

`VLLM_SNAPSHOT_CONTROL_DIR` or `SNAPSHOT_CONTROL_DIR` can supply the directory
instead. An explicit CLI directory takes precedence. Without a directory the
normal serving path is unchanged. The Rust frontend uses the same command with
`VLLM_USE_RUST_FRONTEND=1` and `VLLM_RUST_FRONTEND_PATH` pointing to a binary built
with checkpoint barrier support.

This is a startup barrier for a fixed local process tree, including multiple
Python API servers and local TP/PP/DP engines. It is not a live HTTP snapshot
API. Headless, standalone Python gRPC, multi-port DP supervisors, Ray executors,
elastic EP and remote engines are outside this launcher's barrier. A process
tree being locally supported does not certify its checkpoint compatibility.
The orchestrator owns that decision and must capture the entire tree, including
its IPC connections. Checkpointing only the EngineCore subtree is insufficient.

## Lifecycle and ownership

```mermaid
sequenceDiagram
    participant O as External orchestrator
    participant L as Python launcher
    participant F as All frontends (Python or Rust)
    participant E as All EngineCores and workers
    F->>E: Connect and initialize
    F->>L: Join startup barrier
    L->>F: Leader: pause_scheduler(wait, clear_cache)
    F->>E: Broadcast pause, drain, synchronize
    L->>F: Leader: checkpoint_prepare(policy, options)
    F->>E: Run policy on every core, synchronize workers
    E-->>L: All preparation completed (via leader)
    L-->>O: ready-for-snapshot
    Note over L,E: Frontends parked; scheduling paused
    O->>O: Prepare sharing, CUDA checkpoint, CRIU capture
    O->>O: CRIU restore, CUDA restore/unlock, sharing restore
    O-->>L: restore-complete
    L->>F: Leader: checkpoint_restore()
    F->>E: Recover every core's policy, synchronize
    L->>F: Leader: resume_scheduler()
    L->>F: Release all frontends to serve
```

The source removes stale readiness and release markers before initialization.
It creates `ready-for-snapshot` only after every frontend has joined and every
engine has prepared. It waits indefinitely for `restore-complete`; a timeout
does not grant permission to resume CUDA work. Marker contents are opaque.
Use a separate control directory for each independently captured container.

The orchestrator must remove an old `restore-complete` before restoring an image,
including each reuse of that image. Internal barriers use inherited Unix sockets
captured with the tree, rather than files that could retain an old release.
When snapshot mode is enabled, `SNAPSHOT_RESTORE_STANDBY=1` makes a newly launched
`vllm serve` placeholder idle before creating engines or touching the markers.
It never initializes a second model; the restored source tree crosses the barrier.

For ai-dynamo/snapshot, the agent owns the cuinterpose coordinator and native CUDA
checkpoint interfaces. Application preparation must finish before the agent
tears down shared GPU mappings. On restore, the agent must finish CUDA restore
and unlock **and** cuinterpose sharing reconstruction before writing
`restore-complete`. vLLM makes no cuinterpose control calls. Existing communicator
`checkpoint_prepare`/`checkpoint_restore` worker hooks are optional resource
operations, not a replacement for this group barrier.

`ready-for-snapshot` means workload preparation completed; it is not a manifest
or a compatibility certificate. `restore-complete` means external resources are
ready; vLLM recovery still follows. Normal serving health checks determine when
traffic can return. No extra success report to the orchestrator is required.

## Resource policies

The common `CheckpointPolicy` protocol lives in
`vllm/snapshot/lifecycle/policy.py`. Its synchronous `prepare(core)` and
`restore(core)` hooks execute on the EngineCore execution thread. The same
policy instance survives in the checkpoint. Constructor options are supplied
with `--snapshot-policy-options` as a JSON object.

The default `vllm.snapshot.lifecycle.ResidentPolicy` preserves weights, KV
allocations, caches and communications. It relies on the orchestrator's backend
to capture those resources. `--snapshot-clear-cache` independently clears prefix,
multimodal and encoder caches during pause. Policy hooks must complete their
operations before returning and must leave scheduling paused. Use
`core.collective_rpc` for actions that must execute inside GPU workers.

For example, an installed integration can opt into the existing communicator
hooks without forcing them on every checkpoint backend:

```python
# my_integration/checkpoint.py
class CommunicationPolicy:
    def __init__(self, release_communications=True):
        self.release_communications = release_communications

    def prepare(self, core):
        if self.release_communications:
            core.collective_rpc("checkpoint_prepare")

    def restore(self, core):
        if self.release_communications:
            core.collective_rpc("checkpoint_restore")
```

```bash
vllm serve Qwen/Qwen3-0.6B \
  --snapshot-control-dir /snapshot-control \
  --snapshot-policy my_integration.checkpoint.CommunicationPolicy \
  --snapshot-policy-options '{"release_communications": true}'
```

These hooks implement the communicators' existing behavior; they do not promise
to destroy and recreate every NCCL resource. A backend-specific policy can
instead release other resources, reload selected state, or read externally
updated configuration during recovery. Installed policy code is trusted workload
code, like a worker extension. It does not need a wrapper around `vllm serve`.

## Common APIs and the local snapshot implementation

| Seam | Common workload responsibility | Consumer-specific responsibility |
| --- | --- | --- |
| `vllm/snapshot/lifecycle/coordinator.py` | Aggregate frontend participation, prepare, park, recover, then serve | No artifact management |
| `vllm/snapshot/lifecycle/signaling.py` | `CheckpointSignal.ready()` / `wait_for_restore()`; file implementation for the workload contract | Orchestrator creates snapshots and supplies the release |
| `vllm/snapshot/lifecycle/policy.py` | Paired resource-policy interface and resident default | Select resource hooks and constructor options |
| `vllm/v1/engine/core.py` | `checkpoint_prepare(policy, options)` / `checkpoint_restore()` utility RPCs | No identity, model-output or manifest checks |
| `vllm/entrypoints/cli/serve.py` | Launcher owns the barrier and monitors its processes | Opt-in startup configuration |
| `vllm/entrypoints/launchers/api_server/entry.py` | Python participant before `build_and_serve` | Existing Python HTTP serving |
| `rust/src/server/src/checkpoint.rs` | Rust participant, using the same EngineCore utility RPCs | Existing Rust HTTP/gRPC serving |
| `vllm/snapshot/policies.py` | Implements the common resource protocol | Local `ReloadWeightsPolicy`: discard and reload |
| `vllm/snapshot/server.py` | Calls the common EngineCore resource API | Local canary, rehearsal, JSON release and listener configuration |
| `vllm/snapshot/controller.py`, `runtime.py`, `manifest.py` | None; common lifecycle modules do not import them | Local CRIU/CUDA tooling, storage, manifests, identity checks and oracle validation |

The existing `AsyncLLM.checkpoint_prepare()` and `checkpoint_restore()` methods
remain worker-communicator helpers. The new EngineCore utility methods have a
different scope: a selected resource policy, invoked after a completed group
pause. The coordinator broadcasts recovery to every engine before broadcasting
resume. Policies should use executor resource methods rather than
`EngineCore.wake_up()`, which can resume scheduling automatically.

`vllm snapshot create/restore` retains its local compatibility and canary checks.
Its child selects `ReloadWeightsPolicy` through the common EngineCore API, while
retaining its existing control-file schema and artifact controller. It does not
use the `vllm serve` frontend-group coordinator. ai-dynamo/snapshot does not
inherit this local controller, policy, manifest, rehearsal, or validation logic.

The startup path includes normal model initialization and engine warmup. It does
not force the local snapshot server's extra canary generation or rehearsal.
Live capture, changed network endpoint reconstruction, and cross-node recovery
require additional integration and backend validation.
