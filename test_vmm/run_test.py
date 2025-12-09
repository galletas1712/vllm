#!/usr/bin/env python3
"""Test VMM-based IPC with ShareableCuMemAllocator.

Spawns server and client processes, passes FDs via Unix socket.
Tests import, sleep/wake cycle with checksum verification.
"""

import os
import sys
import time
import pickle
import socket
import struct
import array
import multiprocessing as mp
from multiprocessing import Process

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))


def send_msg(sock, data: bytes, fd: int = -1):
    """Send message with optional FD via SCM_RIGHTS."""
    length = struct.pack('!I', len(data))
    full = length + data
    if fd >= 0:
        ancdata = [(socket.SOL_SOCKET, socket.SCM_RIGHTS, struct.pack('i', fd))]
        sock.sendmsg([full], ancdata)
    else:
        sock.sendall(full)


def recv_msg(sock):
    """Receive message with optional FD."""
    # Receive with ancillary data space
    msg, ancdata, _, _ = sock.recvmsg(65540, socket.CMSG_LEN(4))

    fd = -1
    for level, typ, data in ancdata:
        if level == socket.SOL_SOCKET and typ == socket.SCM_RIGHTS:
            fds = array.array('i')
            fds.frombytes(data)
            if fds:
                fd = fds[0]

    if len(msg) < 4:
        return None, -1

    length = struct.unpack('!I', msg[:4])[0]
    data = msg[4:]

    while len(data) < length:
        more = sock.recv(length - len(data))
        if not more:
            break
        data += more

    return pickle.loads(data), fd


def server_process(sock_path: str, device: int):
    """Server process entry point."""
    from server import server_main

    # Create server socket
    if os.path.exists(sock_path):
        os.unlink(sock_path)
    srv = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
    srv.bind(sock_path)
    srv.listen(1)

    print("[Server] Waiting for client...")
    conn, _ = srv.accept()
    print("[Server] Client connected")

    def send_fn(name, info, fd):
        send_msg(conn, pickle.dumps((name, info)), fd)

    def recv_fn():
        msg, _ = recv_msg(conn)
        return msg

    try:
        server_main(send_fn, recv_fn, device)
    finally:
        conn.close()
        srv.close()
        os.unlink(sock_path)


def client_process(sock_path: str, device: int):
    """Client process entry point."""
    from client import client_main

    # Wait for server
    time.sleep(0.5)

    conn = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
    conn.connect(sock_path)
    print("[Client] Connected to server")

    def recv_fn():
        msg, fd = recv_msg(conn)
        if msg is None:
            return None, None, -1
        name, info = msg
        return name, info, fd

    def send_fn(ack):
        send_msg(conn, pickle.dumps(ack))

    try:
        client_main(recv_fn, send_fn, device)
    finally:
        conn.close()


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", type=int, default=0)
    parser.add_argument("--sock", default="/tmp/vmm_test.sock")
    args = parser.parse_args()

    print("=" * 60)
    print("VMM IPC Test with Deferred Mapping")
    print("=" * 60)

    import torch
    if not torch.cuda.is_available():
        print("CUDA not available!")
        sys.exit(1)

    print(f"Device: {args.device} ({torch.cuda.get_device_name(args.device)})")
    print("=" * 60)

    # Start processes
    server = Process(target=server_process, args=(args.sock, args.device))
    client = Process(target=client_process, args=(args.sock, args.device))

    server.start()
    time.sleep(1)
    client.start()

    client.join(timeout=120)
    server.join(timeout=10)

    if client.exitcode == 0 and server.exitcode == 0:
        print("\nTest completed successfully!")
    else:
        print(f"\nTest failed! client={client.exitcode}, server={server.exitcode}")
        sys.exit(1)


if __name__ == "__main__":
    mp.set_start_method('spawn', force=True)
    main()
