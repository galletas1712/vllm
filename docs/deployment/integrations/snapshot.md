# HTTP checkpoint lifecycle

`vllm serve` can prepare its complete local process tree for an external
checkpoint orchestrator, including [ai-dynamo/snapshot](https://github.com/ai-dynamo/snapshot).
Both Python and Rust frontends expose the same opt-in HTTP API:

```bash
vllm serve Qwen/Qwen3-0.6B --enable-checkpoint

curl --fail -X POST http://localhost:8000/checkpoint/prepare
# {"state":"prepared"}

# Capture the container. Later, restore the container and its external resources.
curl --fail -X POST http://localhost:8000/checkpoint/resume
# {"state":"running"}
```

The Rust frontend uses the same command with `VLLM_USE_RUST_FRONTEND=1` and
`VLLM_RUST_FRONTEND_PATH` pointing to a binary built with checkpoint control
support. The endpoints do not require `VLLM_SERVER_DEV_MODE`. If API keys are
configured, all three checkpoint routes require the usual bearer token.

| Endpoint | Contract |
| --- | --- |
| `POST /checkpoint/prepare` | Close admission across all frontends, drain requests and response streams, pause all engines, apply the configured policy and synchronize workers. Return `{"state":"prepared"}` only after preparation completes. |
| `POST /checkpoint/resume` | Recover resources on all engines, resume scheduling, then reopen frontend admission. Return `{"state":"running"}`. |
| `GET /checkpoint/status` | Report `starting`, `running`, `preparing`, `prepared`, `resuming`, or `failed` without invoking GPU operations. |

POST requests have no configuration payload. Policies are trusted workload code
selected at startup, not code selected by an HTTP caller. Repeating prepare
while prepared or resume while running is a no-op. Transitions are serialized
across all API processes. Losing an HTTP connection does not cancel a transition;
query status or retry the operation through a fresh connection. A policy failure
returns HTTP 503, leaves admission closed, and reports `failed`; restart the
workload rather than retrying partially executed resource hooks.

During preparation and until recovery finishes, ordinary routes (including
`/health` and development control endpoints) return HTTP 503. Checkpoint routes
remain available. Use `/checkpoint/status` for CPU-only liveness while parked,
and ordinary readiness/inference checks before sending traffic. Long-lived
streams and WebSockets must finish or disconnect before preparation completes.
Python Responses API background requests are also drained.

Address the specific pod or serving instance, not a load-balanced service that
might send prepare and resume to different instances. Wait for the complete
prepare response before capturing. After restore, connect anew to the restored
listener; the old external HTTP connection need not survive the checkpoint.

## Lifecycle and ownership

```mermaid
sequenceDiagram
    participant O as External orchestrator
    participant F as All frontends (Python or Rust)
    participant L as Python launcher coordinator
    participant E as All EngineCores and workers
    O->>F: POST /checkpoint/prepare
    F->>L: Prepare service
    L->>F: Close admission and drain responses
    L->>F: Leader: pause_scheduler(wait, clear_cache)
    F->>E: Broadcast pause and synchronize
    L->>F: Leader: checkpoint_prepare(policy, options)
    F->>E: Run policy on every core and synchronize
    L-->>O: 200 prepared, via frontend
    Note over F,E: HTTP control available; scheduling paused
    O->>O: Capture entire container
    O->>O: Restore container, CUDA and external sharing
    O->>F: POST /checkpoint/resume (new connection)
    F->>L: Resume service
    L->>F: Leader: checkpoint_restore()
    F->>E: Recover policy on every core and synchronize
    L->>F: Leader: resume_scheduler()
    L->>F: Reopen admission on all frontends
    L-->>O: 200 running, via frontend
```

The coordinator and its private Unix control sockets are captured with the
entire process tree. It owns sequencing, not artifact storage or compatibility.
Preparation does not create a snapshot or certify that a backend can capture
the selected resources. The orchestrator owns those decisions.

For ai-dynamo/snapshot, the agent owns cuinterpose coordination and native CUDA
checkpoint interfaces. Preparation must complete before the agent tears down
shared GPU mappings. The agent must finish CUDA restore/unlock **and** cuinterpose
sharing reconstruction before calling resume. vLLM makes no cuinterpose control
calls, reads no Snapshot-specific environment variables, and uses no readiness
or release files for this API. The orchestrator supplies a destination standby
process rather than launching a second inference engine before restore.

This API coordinates a fixed local process tree, including multiple Python API
servers and local TP/PP/DP engines. Headless serving, gRPC serving, multi-port DP
supervisors, Ray executors, elastic EP, and remote engines are outside its scope.
Changed network endpoints and cross-node restore require backend integration
and validation. Retaining a listener does not preserve the state of external
clients or peers. The orchestrator must preserve or reconstruct the internal
IPC connections and the service's configured listener address.

## Resource policies

The common `CheckpointPolicy` protocol lives in
`vllm/snapshot/lifecycle/policy.py`. Its synchronous `prepare(core)` and
`restore(core)` hooks execute on the EngineCore execution thread. The same
policy instance survives in the checkpoint. Constructor options are supplied
with `--snapshot-policy-options` as a JSON object.

The default `vllm.snapshot.lifecycle.ResidentPolicy` preserves weights, KV
allocations, caches and communications. It relies on the orchestrator's backend
to capture those resources. `--snapshot-clear-cache` independently clears prefix,
multimodal and encoder caches during preparation. Hooks must finish their
operations before returning and must leave scheduling paused. Use
`core.collective_rpc` for actions inside GPU workers.

For example, an installed integration can opt into existing communicator hooks:

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
  --enable-checkpoint \
  --snapshot-policy my_integration.checkpoint.CommunicationPolicy \
  --snapshot-policy-options '{"release_communications": true}'
```

These hooks implement the communicators' existing behavior; they do not promise
to destroy and recreate every NCCL resource. A backend-specific policy can
release other resources, reload selected state, or read externally updated
configuration during recovery. It requires no wrapper around `vllm serve`.

## Common APIs and the local snapshot implementation

| Module | Responsibility |
| --- | --- |
| `vllm/snapshot/lifecycle/coordinator.py` | Common service-wide preparation/recovery ordering and state; one leader broadcasts EngineCore calls. |
| `vllm/snapshot/lifecycle/channel.py` | Private duplex control transport between the launcher and Python frontends. Rust implements the same protocol. |
| `vllm/snapshot/lifecycle/frontend.py` | Python HTTP adapter, admission gate and response/background-request drain. |
| `rust/src/server/src/checkpoint.rs` | Rust HTTP adapter, response-body admission tracking and EngineCore utility calls. |
| `vllm/snapshot/lifecycle/policy.py` | Common paired resource-policy interface and resident default. |
| `vllm/v1/engine/core.py` | Common `checkpoint_prepare(policy, options)` / `checkpoint_restore()` utility RPCs. |
| `vllm/snapshot/policies.py` | Local `ReloadWeightsPolicy`: discard and reload. |
| `vllm/snapshot/server.py` | Local canary/rehearsal and JSON release protocol; calls the common EngineCore resource API. |
| `vllm/snapshot/controller.py`, `runtime.py`, `manifest.py` | Local artifact management, CRIU/CUDA tooling, identity checks and oracle validation. |

The existing `AsyncLLM.checkpoint_prepare()` and `checkpoint_restore()` methods
remain worker-communicator helpers. The EngineCore utility methods run a selected
resource policy after a completed pause. Rust calls those utilities directly;
it does not use AsyncLLM. Policies should use executor resource methods rather
than `EngineCore.wake_up()`, which can resume scheduling automatically.

`vllm snapshot create/restore` retains its local compatibility checks, canary,
control-file schema and controller. Its child selects `ReloadWeightsPolicy`
through the common EngineCore API. ai-dynamo/snapshot does not inherit that
policy, manifest, rehearsal, validation logic or file protocol.
