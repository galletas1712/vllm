# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import os

from vllm.triton_utils.importing import (
    HAS_TRITON,
    TritonLanguagePlaceholder,
    TritonPlaceholder,
)

if HAS_TRITON and os.environ.get("VLLM_IS_WORKER_PROCESS") == "1":
    import triton
    import triton.language as tl
    import triton.language.extra.libdevice as tldevice
else:
    triton = TritonPlaceholder()
    tl = TritonLanguagePlaceholder()
    tldevice = TritonLanguagePlaceholder()

__all__ = ["HAS_TRITON", "triton", "tl", "tldevice"]
