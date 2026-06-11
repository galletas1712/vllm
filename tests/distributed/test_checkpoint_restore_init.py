# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from pathlib import Path

from vllm.distributed import parallel_state


def test_checkpoint_restore_init_method_uses_filestore(monkeypatch, tmp_path):
    filestore_path = tmp_path / "shared" / "main" / "torch_pg"
    rendezvous_file = tmp_path / "snapshot-control" / "rendezvous.json"

    monkeypatch.setattr(parallel_state.envs, "VLLM_ENABLE_CHECKPOINT_RESTORE", True)
    monkeypatch.setattr(
        parallel_state.envs,
        "VLLM_CHECKPOINT_RESTORE_FILESTORE_PATH",
        str(filestore_path),
    )
    monkeypatch.setenv("TORCH_C10D_RENDEZVOUS_FILE", str(rendezvous_file))

    init_method = parallel_state._checkpoint_restore_init_method(
        "tcp://old-master:29500"
    )

    assert init_method == f"file://{filestore_path}"
    assert Path(filestore_path).parent.is_dir()
