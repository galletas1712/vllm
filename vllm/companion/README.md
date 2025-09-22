# MultiProc Companion System for vLLM

A simple IPC weight sharing system for vLLM that doesn't require external dependencies like Dynamo runtime.

## Overview

The MultiProc Companion System allows multiple vLLM processes to share model weights via CUDA IPC, reducing GPU memory usage when running multiple instances of the same model.

## Components

### 1. Coordinator (`multiproc_coordinator.py`)
- Manages companion server processes (one per GPU)
- Routes client requests to the appropriate companion
- Automatically restarts failed companions

### 2. Companion Server (`multiproc_companion_server.py`)
- Loads model weights once per GPU
- Serves CUDA IPC handles to GPU workers
- Caches loaded models to avoid reloading

### 3. Client (`multiproc_companion_client.py`)
- Used by GPU workers to request model weights
- Connects to coordinator to get weights from companion
- Rebuilds tensors from CUDA IPC handles

## How It Works

1. **Startup**: When vLLM starts with `enable_companion_process=True`, the coordinator automatically starts
2. **Companion Creation**: Coordinator starts one companion server per GPU
3. **Weight Sharing**: GPU workers request weights from companions via CUDA IPC
4. **Memory Efficiency**: Multiple workers share the same GPU memory for model weights

Note: Even though companion processes use a fake distributed backend for device communication,
they still need real CPU groups (gloo backend) for node detection. The `companion_master_port`
configuration ensures these CPU groups don't conflict with vLLM's main distributed groups.

## Usage

The system starts automatically when IPC loading is enabled:

```python
from vllm import AsyncLLM
from vllm.engine.arg_utils import AsyncEngineArgs

engine_args = AsyncEngineArgs(
    model="meta-llama/Llama-2-7b-hf",
    enable_companion_process=True,  # Enables companion system
)

engine = AsyncLLM.from_engine_args(engine_args)
```

## Benefits

- **No External Dependencies** - Uses only ZMQ (already required by vLLM)
- **Automatic Management** - Companions start/stop with the engine
- **Resilient** - Companions persist if workers fail
- **Simple** - Minimal code, easy to understand and maintain