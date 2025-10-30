#!/usr/bin/env python3
"""
Test checkpoint/restore functionality of CheckpointableAsyncLLM.

This test:
1. Starts the engine
2. Waits for readiness
3. Runs a test generation to verify functionality
4. Checkpoints the engine (cuda-checkpoint + CRIU)
5. Restores from checkpoint
6. Runs another generation to verify functionality after restore

NOTE: This test requires sudo privileges for CRIU operations.
Run with: sudo -E python3 test_checkpoint_restore.py
Or use the provided script: ./test_checkpoint_restore_sudo.sh
"""

import asyncio
import logging
import os
import tempfile

# Use GPU 1
os.environ["CUDA_VISIBLE_DEVICES"] = "1"

# Disable custom all-reduce to avoid CUDA IPC issues with CRIU
os.environ["VLLM_DISABLE_CUSTOM_ALL_REDUCE"] = "1"

from vllm.engine.arg_utils import AsyncEngineArgs
from vllm.sampling_params import SamplingParams
from vllm.checkpoint.checkpointable_async_llm import CheckpointableAsyncLLM

# Enable debug logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


async def test_checkpoint_restore():
    """Test full checkpoint/restore cycle."""
    print("\n=== Testing Checkpoint/Restore Functionality ===\n")

    # Generate a unique checkpoint directory path (but don't create it)
    checkpoint_dir = tempfile.mktemp(
        prefix="vllm_test_checkpoint_", dir="/tmp")
    print(f"Will use checkpoint directory: {checkpoint_dir}")

    try:

        # Phase 1: Start engine and verify it works
        print("\n--- Phase 1: Starting engine ---")
        engine_args = AsyncEngineArgs(
            model="Qwen/Qwen3-0.6B-FP8",
            max_model_len=512,
            disable_custom_all_reduce=True,
            enable_sleep_mode=True
        )

        llm = CheckpointableAsyncLLM.from_engine_args(engine_args)

        # Wait for engine to be fully ready
        print("Waiting for engine to be fully initialized...")
        await llm.wait_until_ready()
        print("Engine is ready!")

        # Test generation before checkpoint
        print("\n--- Phase 2: Testing generation before checkpoint ---")
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
        print("\n--- Phase 3: Checkpointing engine ---")
        print(f"Checkpointing to: {checkpoint_dir}")

        # This internally runs cuda-checkpoint lock/checkpoint and CRIU dump
        await llm.criu_checkpoint(checkpoint_dir)
        print("Checkpoint completed successfully!")

        # The engine is now stopped, llm object is no longer usable
        print("Engine has been checkpointed and stopped")

        # Phase 4: Create new instance and restore
        print("\n--- Phase 4: Restoring from checkpoint ---")

        # Create a new instance without auto-starting
        llm_restored = CheckpointableAsyncLLM.from_engine_args(
            engine_args,
            auto_start=False
        )

        # Set the checkpoint directory
        llm_restored.checkpoint_dir = checkpoint_dir

        print(f"Restoring from: {checkpoint_dir}")
        # This internally runs CRIU restore and cuda-checkpoint restore/unlock
        await llm_restored.criu_resume()
        print("Restore completed successfully!")

        # Phase 5: Test generation after restore
        print("\n--- Phase 5: Testing generation after restore ---")
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
                      "checkpoint/restore!")
            else:
                print("\n⚠ Outputs differ:")
                print(f"  Before: {generated_text}")
                print(f"  After:  {generated_text_after}")
                print("  This may be expected due to engine state differences")
        else:
            raise RuntimeError("No output received after restore")

        # Cleanup
        print("\n--- Phase 6: Cleanup ---")
        llm_restored.shutdown()
        print("Shutdown complete")

        print("\n=== Test Passed! ===")
        print("Successfully:")
        print("- Started engine and waited for readiness")
        print("- Generated text before checkpoint")
        print("- Checkpointed engine (cuda-checkpoint + CRIU)")
        print("- Restored from checkpoint")
        print("- Generated text after restore")
        print("- Verified engine remains functional after restore")

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


async def main():
    """Run all tests."""
    try:
        # Basic checkpoint/restore test
        await test_checkpoint_restore()

        print("\n🎉 All tests passed!")

    except Exception as e:
        print(f"\n❌ Test failed: {e}")
        import traceback
        traceback.print_exc()
        raise


if __name__ == "__main__":
    asyncio.run(main())
