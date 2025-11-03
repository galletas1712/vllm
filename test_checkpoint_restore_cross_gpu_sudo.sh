#!/bin/bash
# Helper script to run cross-GPU checkpoint/restore test with proper permissions

# Enable error handling
set -e

# Get the directory of this script
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"

echo "=== Cross-GPU Checkpoint/Restore Test ==="
echo "This test will:"
echo "1. Checkpoint on GPU 0"
echo "2. Restore on GPU 1"
echo ""
echo "Requirements:"
echo "- At least 2 GPUs"
echo "- NVIDIA persistence mode enabled"
echo "- CRIU installed"
echo ""

# Check GPU count
GPU_COUNT=$(nvidia-smi --query-gpu=count --format=csv,noheader | head -1)
if [ "$GPU_COUNT" -lt 2 ]; then
    echo "ERROR: This test requires at least 2 GPUs, but only $GPU_COUNT found"
    exit 1
fi
echo "Found $GPU_COUNT GPUs"

# Run the test with sudo and preserve environment
echo ""
echo "Running test with sudo..."
cd "$SCRIPT_DIR"
# Use the same pattern as the working test_checkpoint_restore_sudo.sh script
sudo -E env "PATH=$PATH" "PYTHONPATH=$PYTHONPATH" python3 "$SCRIPT_DIR/test_checkpoint_restore_cross_gpu.py" "$@"

