"""A stand-in for anp-mcp in the toolbox tests: same name and version, a few tools that
succeed, fail, hang or kill the process. FAKE_VERSION overrides the reported version."""

import json
import os
import sys
import time
from pathlib import Path

from mcp.server import MCPServer

server = MCPServer("anp-mcp", instructions="Fake anp-mcp for tests.",
                   version=os.environ.get("FAKE_VERSION", "0.1.0"))


@server.tool()
def national_reconciliation(details: bool = False) -> dict:
    """Summary with the last month."""
    if os.environ.get("FAKE_WARMUP_FAILS"):
        raise ValueError("dataset unavailable")
    return {"summary": {"months_compared": 261, "last_month": "2026-09"}}


@server.tool()
def echo(text: str) -> dict:
    """Return the text."""
    return {"text": text, "pid": os.getpid()}


@server.tool()
def fail(text: str) -> dict:
    """Always raise."""
    raise ValueError(f"bad argument: {text}")


@server.tool()
def sleep(seconds: float) -> dict:
    """Sleep, then answer."""
    time.sleep(seconds)
    return {"slept": seconds}


@server.tool()
def die(times: int = 1) -> dict:
    """Exit the process, the first `times` calls across restarts (counted in a file)."""
    counter = Path(os.environ["FAKE_COUNTER"])
    n = int(counter.read_text()) if counter.exists() else 0
    counter.write_text(str(n + 1))
    if n < times:
        print("dying", file=sys.stderr, flush=True)
        os._exit(1)
    return {"survived_after": n}


if __name__ == "__main__":
    print(json.dumps({"started": os.getpid()}), file=sys.stderr, flush=True)
    server.run()
