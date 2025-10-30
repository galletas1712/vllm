#!/bin/bash
# Test checkpoint/restore functionality with sudo
# This script runs the test with proper privileges for CRIU

echo "This test requires sudo privileges for CRIU operations."
echo "Running checkpoint/restore test..."

# Preserve the Python environment when using sudo
sudo -E env "PATH=$PATH" "PYTHONPATH=$PYTHONPATH" python3 /home/ubuntu/dynamo/vllm/test_checkpoint_restore.py
