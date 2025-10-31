#!/bin/bash
# Test checkpoint/restore functionality with sudo
# This script runs the test with proper privileges for CRIU
#
# Usage:
#   ./test_checkpoint_restore_sudo.sh                    # Full checkpoint/restore test
#   ./test_checkpoint_restore_sudo.sh --restore-only --checkpoint-dir /path/to/checkpoint

echo "This test requires sudo privileges for CRIU operations."
echo "Running checkpoint/restore test..."

# Preserve the Python environment when using sudo
# Pass through all command-line arguments
sudo -E env "PATH=$PATH" "PYTHONPATH=$PYTHONPATH" "CUDA_VISIBLE_DEVICES=0,1" python3 /home/ubuntu/dynamo/vllm/test_checkpoint_restore.py "$@"
