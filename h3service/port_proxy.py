"""Forward the existing penetration target port to the Diffusers API."""
from __future__ import annotations

import asyncio
import os


LISTEN_HOST = os.getenv("H3_PROXY_LISTEN_HOST", "127.0.0.1")
LISTEN_PORT = int(os.getenv("H3_PROXY_LISTEN_PORT", "8188"))
UPSTREAM_HOST = os.getenv("H3_PROXY_UPSTREAM_HOST", "127.0.0.1")
UPSTREAM_PORT = int(os.getenv("H3_PROXY_UPSTREAM_PORT", "8190"))


async def pipe(source: asyncio.StreamReader, destination: asyncio.StreamWriter) -> None:
    try:
        while chunk := await source.read(64 * 1024):
            destination.write(chunk)
            await destination.drain()
    except (ConnectionError, asyncio.CancelledError):
        pass
    finally:
        if not destination.is_closing() and destination.can_write_eof():
            destination.write_eof()


async def handle(client_reader: asyncio.StreamReader, client_writer: asyncio.StreamWriter) -> None:
    upstream_writer: asyncio.StreamWriter | None = None
    try:
        upstream_reader, upstream_writer = await asyncio.open_connection(UPSTREAM_HOST, UPSTREAM_PORT)
        await asyncio.gather(
            pipe(client_reader, upstream_writer),
            pipe(upstream_reader, client_writer),
        )
    except (ConnectionError, OSError) as exc:
        print(f"upstream connection failed: {exc}", flush=True)
    finally:
        client_writer.close()
        if upstream_writer is not None:
            upstream_writer.close()
        await asyncio.gather(
            client_writer.wait_closed(),
            upstream_writer.wait_closed() if upstream_writer is not None else asyncio.sleep(0),
            return_exceptions=True,
        )


async def main() -> None:
    server = await asyncio.start_server(handle, LISTEN_HOST, LISTEN_PORT)
    print(f"TCP proxy {LISTEN_HOST}:{LISTEN_PORT} -> {UPSTREAM_HOST}:{UPSTREAM_PORT}", flush=True)
    async with server:
        await server.serve_forever()


if __name__ == "__main__":
    asyncio.run(main())
