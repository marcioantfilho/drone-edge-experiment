from __future__ import annotations

import argparse
import csv
import socket
import struct
import time
from pathlib import Path


HEADER = struct.Struct("!IQ")  # filename length, payload length


def recv_exact(sock: socket.socket, size: int) -> bytes:
    chunks = []
    remaining = size
    while remaining:
        chunk = sock.recv(min(1024 * 1024, remaining))
        if not chunk:
            raise ConnectionError("conexão encerrada antes do payload completo")
        chunks.append(chunk)
        remaining -= len(chunk)
    return b"".join(chunks)


def server(args: argparse.Namespace) -> int:
    output = args.output.resolve() if args.output else None
    if output:
        output.mkdir(parents=True, exist_ok=True)
    with socket.create_server((args.host, args.port), reuse_port=False) as listener:
        print(f"Servidor ouvindo em {args.host}:{args.port}")
        with listener.accept()[0] as conn:
            while True:
                raw = conn.recv(HEADER.size)
                if not raw:
                    break
                if len(raw) != HEADER.size:
                    raw += recv_exact(conn, HEADER.size - len(raw))
                name_size, payload_size = HEADER.unpack(raw)
                name = recv_exact(conn, name_size).decode("utf-8")
                payload = recv_exact(conn, payload_size)
                if output:
                    (output / Path(name).name).write_bytes(payload)
                conn.sendall(b"OK")
    return 0


def client(args: argparse.Namespace) -> int:
    images = sorted(p for p in args.input.iterdir() if p.is_file())
    if not images:
        raise FileNotFoundError(f"Nenhum arquivo em {args.input}")
    args.csv.parent.mkdir(parents=True, exist_ok=True)
    rows = []
    with socket.create_connection((args.host, args.port), timeout=args.timeout) as sock:
        for path in images:
            payload = path.read_bytes()
            header = HEADER.pack(len(path.name.encode("utf-8")), len(payload))
            started = time.perf_counter()
            sock.sendall(header + path.name.encode("utf-8") + payload)
            if recv_exact(sock, 2) != b"OK":
                raise ConnectionError("servidor não confirmou o payload")
            elapsed = time.perf_counter() - started
            rows.append({
                "file": path.name,
                "bytes": len(payload),
                "elapsed_ms": round(elapsed * 1000, 3),
                "throughput_mbps": round(len(payload) * 8 / elapsed / 1e6, 3),
            })
    with args.csv.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    print(f"{len(rows)} arquivos enviados; resultados em {args.csv}")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="role", required=True)
    srv = sub.add_parser("server")
    srv.add_argument("--host", default="0.0.0.0")
    srv.add_argument("--port", type=int, default=5000)
    srv.add_argument("--output", type=Path)
    cli = sub.add_parser("client")
    cli.add_argument("--host", required=True)
    cli.add_argument("--port", type=int, default=5000)
    cli.add_argument("--input", type=Path, required=True)
    cli.add_argument("--csv", type=Path, default=Path("results/transmission_real.csv"))
    cli.add_argument("--timeout", type=float, default=120.0)
    args = parser.parse_args()
    return server(args) if args.role == "server" else client(args)


if __name__ == "__main__":
    raise SystemExit(main())