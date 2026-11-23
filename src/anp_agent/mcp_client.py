"""AnpToolbox: a thin stdio client for anp-mcp (design, section 2).

One server process and session per agent process. A single owner task opens the session,
serves the calls in order from a queue, and closes it: the stdio transport's task groups must
be entered and left by the same task, while the graph may call from others.

    async with AnpToolbox.start(settings) as toolbox:
        toolbox.instructions, toolbox.tool_schemas, toolbox.latest_month
        record = await toolbox.call("field_group_production", {"field_group": "MERO"})

Failures at start (server does not start, initialize times out, wrong name or version, the
warm-up fails) raise ToolboxError before any LLM call. During a question, a tool error or a
timeout becomes the record's status; a broken transport restarts the server once and retries
that call once (the tools are read-only), then gives up: later calls fail fast.
"""

import asyncio
import contextlib
import json
import os
import sys
import time
from collections.abc import AsyncIterator
from dataclasses import dataclass, field
from pathlib import Path

from mcp import ClientSession
from mcp.client.stdio import StdioServerParameters, stdio_client
from mcp.shared.exceptions import MCPError
from mcp_types import CONNECTION_CLOSED, REQUEST_TIMEOUT

from anp_agent.config import Settings
from anp_agent.state import ToolRecord

SERVER_NAME, SERVER_VERSION = "anp-mcp", "0.1.0"
WARMUP_TOOL = "national_reconciliation"
# The warm-up downloads the dataset on first use, which can take far longer than a tool call.
WARMUP_TIMEOUT_S = 300.0
STDERR_LOG = Path("logs") / "anp-mcp.stderr.log"
TRANSPORT_ERRORS = (OSError, EOFError, ConnectionError)


class ToolboxError(RuntimeError):
    """The server cannot be used: fatal before any question."""


class _TransportError(Exception):
    pass


def default_server() -> StdioServerParameters:
    """anp-mcp from this environment, with the caller's environment (ANP_MCP_REVISION,
    ANP_MCP_DATA_DIR, the Hugging Face cache) passed through unchanged."""
    return StdioServerParameters(command=sys.executable, args=["-m", "anp_mcp.server"],
                                 env=dict(os.environ))


@dataclass
class _Request:
    name: str
    args: dict
    call_id: str
    future: asyncio.Future


@dataclass
class AnpToolbox:
    settings: Settings
    server: StdioServerParameters
    stderr_log: Path = STDERR_LOG
    instructions: str = ""
    tool_schemas: list[dict] = field(default_factory=list)
    latest_month: str | None = None
    restarts: int = 0
    _queue: asyncio.Queue = field(default_factory=asyncio.Queue)
    _unavailable: bool = False

    @classmethod
    @contextlib.asynccontextmanager
    async def start(cls, settings: Settings, server: StdioServerParameters | None = None,
                    stderr_log: Path = STDERR_LOG) -> AsyncIterator["AnpToolbox"]:
        toolbox = cls(settings, server or default_server(), stderr_log)
        ready = asyncio.get_running_loop().create_future()
        owner = asyncio.create_task(toolbox._serve(ready), name="anp-mcp session")
        try:
            await ready
            yield toolbox
        finally:
            await toolbox._queue.put(None)
            with contextlib.suppress(Exception):
                await asyncio.wait_for(owner, timeout=10)
            if not owner.done():
                owner.cancel()
                with contextlib.suppress(BaseException):
                    await owner

    async def call(self, name: str, args: dict, call_id: str = "") -> ToolRecord:
        """Run one tool call; never raises for tool, timeout or transport failures."""
        if self._unavailable:
            return _record(call_id, name, args, "server unavailable", "transport_error", 0.0)
        future = asyncio.get_running_loop().create_future()
        await self._queue.put(_Request(name, args, call_id, future))
        return await future

    # The owner task: everything below runs in it.

    async def _serve(self, ready: asyncio.Future) -> None:
        try:
            await self._serve_loop(ready)
        finally:  # whatever ended the loop, nobody waits forever
            self._unavailable = True
            self._drain("server unavailable")

    async def _serve_loop(self, ready: asyncio.Future) -> None:
        retry: _Request | None = None
        while True:
            restart = None
            try:
                async with self._session(first=not ready.done()) as session:
                    if not ready.done():
                        ready.set_result(None)
                    while True:
                        req, retrying = (retry, True) if retry else (await self._queue.get(), False)
                        retry = None
                        if req is None:
                            return
                        try:
                            req.future.set_result(await self._call(session, req))
                        except _TransportError as e:
                            if retrying:
                                self._give_up(req, e)
                                return
                            restart = req
                            break
            except Exception as e:  # noqa: BLE001 - start failure, or a dead child on exit
                if not ready.done():
                    ready.set_exception(e if isinstance(e, ToolboxError) else ToolboxError(
                        f"anp-mcp did not start: {_describe(e)}"))
                    return
                if restart is None:
                    if retry is not None:  # the restart itself failed
                        self._give_up(retry, e)
                    return
            if restart is None:
                return
            self.restarts += 1
            retry = restart

    @contextlib.asynccontextmanager
    async def _session(self, first: bool) -> AsyncIterator[ClientSession]:
        self.stderr_log.parent.mkdir(parents=True, exist_ok=True)
        with self.stderr_log.open("a", encoding="utf-8") as errlog:
            async with (stdio_client(self.server, errlog=errlog) as (read, write),
                        ClientSession(read, write) as session):
                await self._handshake(session, first)
                yield session

    async def _handshake(self, session: ClientSession, first: bool) -> None:
        try:
            async with asyncio.timeout(self.settings.server_init_timeout_s):
                init = await session.initialize()
                listed = await session.list_tools() if first else None
        except TimeoutError:
            raise ToolboxError(f"anp-mcp did not initialize within "
                               f"{self.settings.server_init_timeout_s:g} s") from None
        info = init.server_info
        if (info.name, info.version) != (SERVER_NAME, SERVER_VERSION):
            raise ToolboxError(f"expected {SERVER_NAME} {SERVER_VERSION}, "
                               f"got {info.name} {info.version}")
        if not first:
            return
        self.instructions = init.instructions or ""
        self.tool_schemas = [{"name": t.name, "description": t.description or "",
                              "input_schema": t.input_schema} for t in listed.tools]
        if WARMUP_TOOL not in {t["name"] for t in self.tool_schemas}:
            raise ToolboxError(f"anp-mcp does not list {WARMUP_TOOL}")
        # Outside any question budget; only latest_month is kept (never shown to the model).
        warm = await self._call(session, _Request(WARMUP_TOOL, {}, "warm-up", None),
                                timeout=WARMUP_TIMEOUT_S)
        last = ((warm["structured"] or {}).get("summary") or {}).get("last_month")
        if warm["status"] != "ok" or not isinstance(last, str):
            raise ToolboxError(f"warm-up {WARMUP_TOOL} failed: {warm['result_text'][:200]}")
        self.latest_month = last

    async def _call(self, session: ClientSession, req: _Request,
                    timeout: float | None = None) -> ToolRecord:
        timeout = timeout or self.settings.tool_timeout_s
        start = time.monotonic()

        def done(text, status, structured=None):
            return _record(req.call_id, req.name, req.args, text, status,
                           time.monotonic() - start, structured)

        try:
            result = await session.call_tool(req.name, req.args, read_timeout_seconds=timeout)
        except MCPError as e:
            if e.code == REQUEST_TIMEOUT:
                return done(f"timed out after {timeout:g} s", "timeout")
            if e.code == CONNECTION_CLOSED:
                raise _TransportError(e.message) from e
            return done(f"error: {e.message}", "tool_error")
        except TRANSPORT_ERRORS as e:
            raise _TransportError(_describe(e)) from e
        except Exception as e:  # noqa: BLE001 - anything else stays this call's error
            return done(f"error: {_describe(e)}", "tool_error")
        text = "\n".join(c.text for c in result.content if getattr(c, "type", "") == "text")
        if result.is_error:
            return done(text or "tool error", "tool_error")
        structured = result.structured_content
        if not isinstance(structured, dict):
            try:
                structured = json.loads(text)
            except json.JSONDecodeError:
                structured = None
        return done(text or json.dumps(structured or {}), "ok",
                    structured if isinstance(structured, dict) else None)

    def _give_up(self, req: _Request, error: BaseException) -> None:
        self._unavailable = True
        req.future.set_result(_record(req.call_id, req.name, req.args,
                                      f"server unavailable: {_describe(error)}",
                                      "transport_error", 0.0))
        self._drain("server unavailable")

    def _drain(self, text: str) -> None:
        """Fail the calls still queued."""
        while not self._queue.empty():
            other = self._queue.get_nowait()
            if other is not None and not other.future.done():
                other.future.set_result(_record(other.call_id, other.name, other.args, text,
                                                "transport_error", 0.0))


def _record(call_id, name, args, text, status, seconds, structured=None) -> ToolRecord:
    return ToolRecord(call_id=call_id, name=name, args=args, result_text=text,
                      structured=structured, status=status, seconds=round(seconds, 3))


def _describe(e: BaseException) -> str:
    if isinstance(e, BaseExceptionGroup) and e.exceptions:
        return _describe(e.exceptions[0])
    return f"{type(e).__name__}: {e}" if str(e) else type(e).__name__
