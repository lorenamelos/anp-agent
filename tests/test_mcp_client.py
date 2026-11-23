import asyncio
import os
import sys
from pathlib import Path

import pytest
from mcp.client.stdio import StdioServerParameters

from anp_agent.config import Settings
from anp_agent.mcp_client import AnpToolbox, ToolboxError

FAKE = Path(__file__).parent / "fake_server.py"
SETTINGS = Settings(tool_timeout_s=2, server_init_timeout_s=10)


def fake(tmp_path, **env):
    return StdioServerParameters(command=sys.executable, args=[str(FAKE)], env={
        **os.environ, "FAKE_COUNTER": str(tmp_path / "deaths"), **env})


def run(tmp_path, body, settings=SETTINGS, **env):
    async def main():
        async with AnpToolbox.start(settings, fake(tmp_path, **env),
                                    tmp_path / "stderr.log") as toolbox:
            return await body(toolbox)
    return asyncio.run(main())


def test_start_lists_tools_and_warms_up(tmp_path):
    async def body(tb):
        return tb.instructions, [t["name"] for t in tb.tool_schemas], tb.latest_month
    instructions, names, latest = run(tmp_path, body)
    assert instructions == "Fake anp-mcp for tests."
    assert {"national_reconciliation", "echo", "fail", "sleep", "die"} <= set(names)
    assert latest == "2026-09"
    assert "started" in (tmp_path / "stderr.log").read_text()  # stderr goes to the log


def test_call_ok_and_errors(tmp_path):
    async def body(tb):
        return [await tb.call("echo", {"text": "oi"}, "c1"), await tb.call("fail", {"text": "x"}),
                await tb.call("nope", {}), await tb.call("sleep", {"seconds": 5}),
                await tb.call("echo", {"text": "still here"})]
    ok, failed, unknown, slow, after = run(tmp_path, body)
    assert (ok["status"], ok["call_id"], ok["structured"]["text"]) == ("ok", "c1", "oi")
    assert '"text": "oi"' in ok["result_text"]
    assert (failed["status"], unknown["status"]) == ("tool_error", "tool_error")
    assert "nope" in unknown["result_text"]
    assert slow["status"] == "timeout" and slow["result_text"] == "timed out after 2 s"
    assert after["status"] == "ok"  # a timeout does not break the session


def test_transport_failure_restarts_once_and_retries(tmp_path):
    async def body(tb):
        first_pid = (await tb.call("echo", {"text": "a"}))["structured"]["pid"]
        died = await tb.call("die", {"times": 1})  # dies once, the retry survives
        pid = (await tb.call("echo", {"text": "b"}))["structured"]["pid"]
        return died, first_pid != pid, tb.restarts
    died, new_process, restarts = run(tmp_path, body)
    assert died["status"] == "ok" and died["structured"] == {"survived_after": 1}
    assert new_process and restarts == 1


def test_transport_failure_twice_gives_up(tmp_path):
    async def body(tb):
        return await tb.call("die", {"times": 5}), await tb.call("echo", {"text": "x"})
    died, later = run(tmp_path, body)
    assert died["status"] == "transport_error"
    assert later["status"] == "transport_error"  # fails fast, no new LLM-visible numbers


def test_calls_from_other_tasks(tmp_path):
    async def body(tb):
        return await asyncio.gather(*(asyncio.create_task(tb.call("echo", {"text": str(i)}))
                                      for i in range(5)))
    records = run(tmp_path, body)
    assert [r["structured"]["text"] for r in records] == ["0", "1", "2", "3", "4"]


@pytest.mark.parametrize("env, message", [
    ({"FAKE_VERSION": "0.2.0"}, "expected anp-mcp 0.1.0, got anp-mcp 0.2.0"),
    ({"FAKE_WARMUP_FAILS": "1"}, "warm-up national_reconciliation failed"),
])
def test_start_failures_are_fatal(tmp_path, env, message):
    async def body(tb):
        raise AssertionError("must not start")
    with pytest.raises(ToolboxError, match=message):
        run(tmp_path, body, **env)


def test_server_that_does_not_start(tmp_path):
    async def main():
        server = StdioServerParameters(command=sys.executable, args=["-c", "import sys; sys.exit(3)"])
        async with AnpToolbox.start(SETTINGS, server, tmp_path / "e.log"):
            pass
    with pytest.raises(ToolboxError, match="did not start"):
        asyncio.run(main())


@pytest.mark.integration
def test_real_server(tmp_path):
    async def main():
        async with AnpToolbox.start(Settings(), stderr_log=tmp_path / "e.log") as tb:
            rec = await tb.call("search_field_groups", {"text": "buzios"})
            return tb, rec
    tb, rec = asyncio.run(main())
    assert tb.latest_month == "2026-09" and len(tb.tool_schemas) == 5
    assert rec["status"] == "ok" and rec["structured"]["matches"][0]["field_group"] == "BÚZIOS"
