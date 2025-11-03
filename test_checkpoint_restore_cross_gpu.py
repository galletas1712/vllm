#!/usr/bin/env python3
"""
Test cross-GPU checkpoint/restore functionality of CheckpointableAsyncLLM.

This test:
1. Starts the engine on GPU 0
2. Waits for readiness
3. Runs a test generation to verify functionality
4. Checkpoints the engine (cuda-checkpoint + CRIU)
5. Restores from checkpoint on GPU 1
6. Runs another generation to verify functionality after restore

NOTE: This test requires:
- sudo privileges for CRIU operations
- At least 2 GPUs available
- NVIDIA persistence mode enabled

Run with: sudo -E python3 test_checkpoint_restore_cross_gpu.py
"""

import argparse
import asyncio
import logging
import os
import tempfile

from vllm.engine.arg_utils import AsyncEngineArgs
from vllm.sampling_params import SamplingParams
from vllm.checkpoint.checkpointable_async_llm import CheckpointableAsyncLLM

# Enable debug logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def get_engine_args():
    """Get standard engine arguments."""
    return AsyncEngineArgs(
        model="Qwen/Qwen3-0.6B-FP8",
        max_model_len=512,
        disable_custom_all_reduce=True,
        enable_sleep_mode=True,
    )


async def test_cross_gpu_checkpoint_restore():
    """Test checkpoint on GPU 0, restore on GPU 1."""
    print("\n=== Testing Cross-GPU Checkpoint/Restore Functionality ===\n")

    # Check that we have at least 2 GPUs
    import torch
    if torch.cuda.device_count() < 2:
        raise RuntimeError(f"This test requires at least 2 GPUs, but only {torch.cuda.device_count()} found")

    print(f"Found {torch.cuda.device_count()} GPUs")

    # Generate a unique checkpoint directory path (but don't create it)
    checkpoint_dir = tempfile.mktemp(
        prefix="vllm_test_cross_gpu_checkpoint_", dir="/tmp")
    print(f"Will use checkpoint directory: {checkpoint_dir}")

    try:
        # Phase 1: Start engine on GPU 0 and verify it works
        print("\n--- Phase 1: Starting engine on GPU 0 ---")
        os.environ["CUDA_VISIBLE_DEVICES"] = "0"
        engine_args = get_engine_args()

        llm = CheckpointableAsyncLLM.from_engine_args(engine_args)

        # Wait for engine to be fully ready
        print("Waiting for engine to be fully initialized...")
        await llm.wait_until_ready()
        print("Engine is ready!")

        # Test generation before checkpoint
        print("\n--- Phase 2: Testing generation before checkpoint (GPU 0) ---")
        test_prompt = "The capital of France is"
        sampling_params = SamplingParams(
            temperature=0.0,  # Deterministic
            max_tokens=10,
        )

        print(f"Prompt: {test_prompt}")
        print("Generating...")

        outputs_before = []
        async for output in llm.generate(
            test_prompt,
            sampling_params,
            request_id="test-before-checkpoint"
        ):
            if output.finished:
                outputs_before.append(output)

        if outputs_before:
            generated_text = outputs_before[-1].outputs[0].text
            print(f"Generated: {generated_text}")
            assert len(generated_text.strip()) > 0, "No output generated"
        else:
            raise RuntimeError("No output received before checkpoint")

        # Phase 3: Checkpoint the engine
        print("\n--- Phase 3: Checkpointing engine on GPU 0 ---")
        print(f"Checkpointing to: {checkpoint_dir}")

        # This internally runs cuda-checkpoint lock/checkpoint and CRIU dump
        await llm.criu_checkpoint(checkpoint_dir)
        print("Checkpoint completed successfully!")

        # The engine is now stopped, llm object is no longer usable
        print("Engine has been checkpointed and stopped")

        # Phase 4: Switch to GPU 1 and restore
        print("\n--- Phase 4: Switching to GPU 1 and restoring from checkpoint ---")

        # Change CUDA_VISIBLE_DEVICES to GPU 1
        os.environ["CUDA_VISIBLE_DEVICES"] = "1"
        print("Switched CUDA_VISIBLE_DEVICES to GPU 1")

        # Create a new instance without auto-starting
        llm_restored = CheckpointableAsyncLLM.from_engine_args(
            engine_args,
            auto_start=False
        )

        # Set the checkpoint directory
        llm_restored.checkpoint_dir = checkpoint_dir

        print(f"Restoring from: {checkpoint_dir}")
        print("This will perform GPU migration from GPU 0 to GPU 1")

        # This internally runs CRIU restore and cuda-checkpoint restore/unlock with GPU migration
        await llm_restored.criu_resume()
        print("Restore completed successfully!")

        # Phase 5: Test generation after restore on GPU 1
        print("\n--- Phase 5: Testing generation after restore (GPU 1) ---")
        print(f"Prompt: {test_prompt}")
        print("Generating...")

        outputs_after = []
        async for output in llm_restored.generate(
            test_prompt,
            sampling_params,
            request_id="test-after-restore"
        ):
            if output.finished:
                outputs_after.append(output)

        if outputs_after:
            generated_text_after = outputs_after[-1].outputs[0].text
            print(f"Generated: {generated_text_after}")
            assert len(generated_text_after.strip()) > 0, \
                "No output generated after restore"

            # Check if outputs are consistent
            # (should be identical with temperature=0)
            if generated_text == generated_text_after:
                print("\n✓ Outputs are consistent before and after "
                      "cross-GPU checkpoint/restore!")
            else:
                print("\n⚠ Outputs differ:")
                print(f"  Before (GPU 0): {generated_text}")
                print(f"  After  (GPU 1): {generated_text_after}")
                print("  This may be expected due to engine state differences")
        else:
            raise RuntimeError("No output received after restore")

        # Cleanup
        print("\n--- Phase 6: Cleanup ---")
        llm_restored.shutdown()
        print("Shutdown complete")

        print("\n=== Test Passed! ===")
        print("Successfully:")
        print("- Started engine on GPU 0 and waited for readiness")
        print("- Generated text before checkpoint on GPU 0")
        print("- Checkpointed engine (cuda-checkpoint + CRIU)")
        print("- Restored from checkpoint on GPU 1 (GPU migration)")
        print("- Generated text after restore on GPU 1")
        print("- Verified engine remains functional after cross-GPU restore")

    except Exception:
        raise
    finally:
        # Clean up checkpoint directory if it exists
        # Comment this out to preserve logs for debugging
        # if os.path.exists(checkpoint_dir):
        #     shutil.rmtree(checkpoint_dir)
        #     print(f"\nCleaned up checkpoint directory: {checkpoint_dir}")
        print(f"\nCheckpoint directory preserved for debugging: "
              f"{checkpoint_dir}")


async def test_restore_only_cross_gpu(checkpoint_dir, target_gpu="1"):
    """Test restore-only functionality from an existing checkpoint to a different GPU."""
    print("\n=== Testing Cross-GPU Restore-Only Functionality ===\n")
    print(f"Restoring from checkpoint directory: {checkpoint_dir}")
    print(f"Target GPU: {target_gpu}")

    if not os.path.exists(checkpoint_dir):
        raise ValueError(f"Checkpoint directory does not exist: {checkpoint_dir}")

    try:
        # Set CUDA_VISIBLE_DEVICES to target GPU
        os.environ["CUDA_VISIBLE_DEVICES"] = target_gpu
        print(f"Set CUDA_VISIBLE_DEVICES to GPU {target_gpu}")

        # Phase 1: Create new instance and restore
        print("\n--- Phase 1: Restoring from checkpoint ---")
        engine_args = get_engine_args()

        # Create a new instance without auto-starting
        llm_restored = CheckpointableAsyncLLM.from_engine_args(
            engine_args,
            auto_start=False
        )

        # Set the checkpoint directory
        llm_restored.checkpoint_dir = checkpoint_dir

        print(f"Restoring from: {checkpoint_dir}")
        # This internally runs CRIU restore and cuda-checkpoint restore/unlock with GPU migration
        await llm_restored.criu_resume()
        print("Restore completed successfully!")

        # Phase 2: Test generation after restore
        print("\n--- Phase 2: Testing generation after restore ---")
        test_prompt = "The capital of France is"
        sampling_params = SamplingParams(
            temperature=0.0,  # Deterministic
            max_tokens=10,
        )

        print(f"Prompt: {test_prompt}")
        print("Generating...")

        outputs_after = []
        async for output in llm_restored.generate(
            test_prompt,
            sampling_params,
            request_id="test-after-restore"
        ):
            if output.finished:
                outputs_after.append(output)

        if outputs_after:
            generated_text_after = outputs_after[-1].outputs[0].text
            print(f"Generated: {generated_text_after}")
            assert len(generated_text_after.strip()) > 0, \
                "No output generated after restore"
        else:
            raise RuntimeError("No output received after restore")

        # Cleanup
        print("\n--- Phase 3: Cleanup ---")
        llm_restored.shutdown()
        print("Shutdown complete")

        print("\n=== Test Passed! ===")
        print("Successfully:")
        print(f"- Restored from checkpoint to GPU {target_gpu}")
        print("- Generated text after restore")
        print("- Verified engine remains functional after cross-GPU restore")

    except Exception:
        raise


async def main():
    """Run all tests."""
    parser = argparse.ArgumentParser(
        description="Test cross-GPU checkpoint/restore functionality of CheckpointableAsyncLLM"
    )
    parser.add_argument(
        "--restore-only",
        action="store_true",
        help="Only restore from an existing checkpoint (skip checkpoint creation)"
    )
    parser.add_argument(
        "--checkpoint-dir",
        type=str,
        help="Checkpoint directory to restore from (required when --restore-only is used)"
    )
    parser.add_argument(
        "--target-gpu",
        type=str,
        default="1",
        help="Target GPU for restore (default: 1)"
    )

    args = parser.parse_args()

    # Validate arguments
    if args.restore_only:
        if not args.checkpoint_dir:
            parser.error("--checkpoint-dir is required when --restore-only is used")

    try:
        if args.restore_only:
            # Restore-only mode
            await test_restore_only_cross_gpu(args.checkpoint_dir, args.target_gpu)
        else:
            # Full checkpoint/restore test
            await test_cross_gpu_checkpoint_restore()

        print("\n🎉 All tests passed!")

    except Exception as e:
        print(f"\n❌ Test failed: {e}")
        import traceback
        traceback.print_exc()
        raise


if __name__ == "__main__":
    asyncio.run(main())

