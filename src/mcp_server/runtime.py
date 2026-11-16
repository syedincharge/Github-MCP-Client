"""Run async entry points without Windows' asyncio teardown noise.

On Windows the Proactor event loop closes sockets and subprocess pipes as it
shuts down. If the peer is already gone, `socket.shutdown()` raises
ConnectionResetError (WinError 10054) inside
`_ProactorBasePipeTransport._call_connection_lost`. It surfaces through the
loop's exception handler rather than the caller, so it appears *after* a run
has finished successfully and never affects the exit code or the results.
"""

import asyncio
from typing import Any

TEARDOWN_ERRORS = (ConnectionResetError, ConnectionAbortedError, BrokenPipeError)
DRAIN_SECONDS = 0.25


def _handler(loop: asyncio.AbstractEventLoop, context: dict) -> None:
    exc = context.get("exception")
    if isinstance(exc, TEARDOWN_ERRORS):
        return
    loop.default_exception_handler(context)


def run(coro) -> Any:
    """`asyncio.run`, minus the harmless teardown errors from closing transports."""

    async def wrapper() -> Any:
        asyncio.get_running_loop().set_exception_handler(_handler)
        try:
            return await coro
        finally:
            # Give transports a moment to close before the loop goes away.
            await asyncio.sleep(DRAIN_SECONDS)

    return asyncio.run(wrapper())
