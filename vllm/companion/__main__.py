# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Entry point for running vLLM companion server.

Usage:
    python -m vllm.companion --device-id 0 --port 5555
"""

from vllm.companion.server import main

if __name__ == "__main__":
    main()
