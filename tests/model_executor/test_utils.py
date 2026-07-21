# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import torch

from vllm.model_executor.utils import replace_parameter


def test_replace_parameter_prefer_copy_owns_storage():
    layer = torch.nn.Module()
    source = torch.arange(12, dtype=torch.float32).reshape(3, 4)
    view = source.transpose(0, 1)

    replace_parameter(layer, "weight", view, prefer_copy=True)

    assert torch.equal(layer.weight, view)
    assert layer.weight.shape == view.shape
    assert layer.weight.dtype == view.dtype
    assert layer.weight.device == view.device
    assert layer.weight.stride() == view.stride()
    assert layer.weight.untyped_storage().data_ptr() != (
        view.untyped_storage().data_ptr()
    )


def test_replace_parameter_prefer_copy_reuses_compatible_storage():
    layer = torch.nn.Module()
    replace_parameter(layer, "weight", torch.zeros(4, 3), prefer_copy=True)
    data_ptr = layer.weight.data_ptr()
    updated = torch.full_like(layer.weight, 7)

    replace_parameter(layer, "weight", updated, prefer_copy=True)

    assert layer.weight.data_ptr() == data_ptr
    assert torch.equal(layer.weight, updated)
