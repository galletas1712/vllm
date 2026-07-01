# Environment Variables

vLLM uses the following environment variables to configure the system:

!!! warning
    Please note that `VLLM_PORT` and `VLLM_HOST_IP` set the port and ip for vLLM's **internal usage**. It is not the port and ip for the API server. If you use `--host $VLLM_HOST_IP` and `--port $VLLM_PORT` to start the API server, it will not work.

    All environment variables used by vLLM are prefixed with `VLLM_`. **Special care should be taken for Kubernetes users**: please do not name the service as `vllm`, otherwise environment variables set by Kubernetes might conflict with vLLM's environment variables, because [Kubernetes sets environment variables for each service with the capitalized service name as the prefix](https://kubernetes.io/docs/concepts/services-networking/service/#environment-variables).

## Disabling NCCL

`VLLM_DISABLE_NCCL=1` is an experimental, fail-closed policy for deployments
that must not create NCCL process groups or PyNCCL communicators. It uses Gloo
for control groups, FlashInfer for supported CUDA all-reduce and equal-size
all-gather operations, and `flashinfer_nvlink_one_sided` for expert-parallel
dispatch and combine.

The policy requires pipeline parallel size 1. It rejects unsupported
reduce-scatter, variable all-gather, GPU point-to-point and broadcast,
split-group, elastic EP, EPLB, Ray, and weight-transfer paths instead of
falling back to NCCL. Normal behavior is unchanged when the variable is unset.

```python
--8<-- "vllm/envs.py:env-vars-definition"
```
