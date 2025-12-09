"""Client for VMM-based tensor sharing using ShareableCuMemAllocator.

Flow:
1. For each tensor from server:
   - Create deferred tensor (VA only, no physical memory)
   - Import server's FD and map to the tensor's VA
   - Tensor now has physical memory from server
2. Sleep/wake cycle for memory management
"""

import torch
from shared_types import TensorIPCInfo, compute_checksum
from shareable_cumem_allocator import get_allocator


def client_main(recv_fn, send_fn, device: int = 0):
    """Client main loop."""
    print(f"[Client] Starting on device {device}", flush=True)
    torch.cuda.set_device(device)

    allocator = get_allocator()
    print(f"[Client] Allocator ready, granularity={allocator.granularity}", flush=True)

    results = []
    tensors = {}
    tensor_infos = {}

    try:
        # Phase 1: Import tensors from server
        print("\n" + "=" * 50)
        print("PHASE 1: Deferred Allocation + Import")
        print("=" * 50)

        while True:
            name, info, fd = recv_fn()
            if name is None:
                print("[Client] Done signal received", flush=True)
                break

            print(f"\n[Client] {name}: shape={info.shape}, dtype={info.dtype}")
            print(f"[Client]   FD={fd}, alloc_size={info.allocation_size}")

            # Create deferred tensor (VA only) - one at a time to avoid suballocation
            with allocator.use_deferred_pool("weights"):
                tensor = torch.empty(
                    info.shape,
                    dtype=info.dtype,
                    device=f"cuda:{device}"
                )
            va = tensor.data_ptr()
            print(f"[Client]   Reserved VA=0x{va:x}")

            # Import and map the server's physical memory
            handle = allocator.import_and_map(va, fd, info.allocation_size)
            print(f"[Client]   Imported and mapped! handle={handle}")

            # Verify checksum (outside deferred pool to avoid suballocation)
            client_sum = compute_checksum(tensor)
            server_sum = info.checksum
            tol = abs(server_sum) * 1e-5 + 1e-5
            match = abs(client_sum - server_sum) < tol

            print(f"[Client]   Checksum: server={server_sum}, client={client_sum}")
            print(f"[Client]   {'MATCH!' if match else 'MISMATCH!'}")

            results.append({"name": name, "phase": "import", "match": match})
            tensors[name] = tensor
            tensor_infos[name] = info
            send_fn({"status": "ok", "match": match})

        # Phase 2: Sleep all tensors
        print("\n" + "=" * 50)
        print("PHASE 2: Sleep (unmap, backup to CPU)")
        print("=" * 50)

        sleep_states = {}
        for name, tensor in tensors.items():
            print(f"\n[Client] Sleeping {name}...")
            state = allocator.sleep_tensor(tensor)
            sleep_states[name] = state
            print(f"[Client]   VA=0x{state.va:x} unmapped, backed up to CPU")

        send_fn({"status": "sleep_complete", "names": list(sleep_states.keys())})

        # Phase 3: Wake all tensors
        print("\n" + "=" * 50)
        print("PHASE 3: Wake (re-import, remap, restore)")
        print("=" * 50)

        while True:
            name, info, fd = recv_fn()
            if name is None:
                print("[Client] Wake phase complete")
                break

            if name not in sleep_states:
                print(f"[Client] WARNING: Unknown tensor {name}")
                continue

            print(f"\n[Client] Waking {name}...")
            state = sleep_states[name]
            tensor = tensors[name]

            allocator.wake_tensor(state, fd)
            assert tensor.data_ptr() == state.va, "VA changed!"
            print(f"[Client]   VA unchanged at 0x{state.va:x}")

            client_sum = compute_checksum(tensor)
            server_sum = info.checksum
            tol = abs(server_sum) * 1e-5 + 1e-5
            match = abs(client_sum - server_sum) < tol

            print(f"[Client]   Checksum: server={server_sum}, client={client_sum}")
            print(f"[Client]   {'MATCH!' if match else 'MISMATCH!'}")

            results.append({"name": name, "phase": "wake", "match": match})
            send_fn({"status": "ok", "match": match})

        # Summary
        all_match = all(r["match"] for r in results)
        send_fn({"status": "complete", "all_match": all_match})

        print("\n" + "=" * 50)
        print("SUMMARY")
        print("=" * 50)
        for r in results:
            print(f"  {r['name']} ({r['phase']}): {'PASS' if r['match'] else 'FAIL'}")
        print("=" * 50)
        print(f"Overall: {'ALL PASSED' if all_match else 'SOME FAILED'}")

    except Exception as e:
        print(f"[Client] Error: {e}")
        import traceback
        traceback.print_exc()
        send_fn({"status": "error", "error": str(e)})

    print("[Client] Done")
