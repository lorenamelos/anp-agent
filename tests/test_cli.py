import contextlib
import json

import pytest

from anp_agent import cli
from anp_agent.mcp_client import ToolboxError
from tests.test_graph import FETCH, FakeToolbox, ScriptedLLM, say, use


@pytest.fixture
def fake_run(monkeypatch, tmp_path):
    """The CLI with a fake server and a scripted model, from a directory without .env."""
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("ANTHROPIC_API_KEY", "test-key")

    def setup(script, toolbox=None, start_error=None):
        @contextlib.asynccontextmanager
        async def start(settings):
            if start_error:
                raise start_error
            yield toolbox or FakeToolbox()
        monkeypatch.setattr(cli.AnpToolbox, "start", start)
        monkeypatch.setattr(cli, "make_llm", lambda s: ScriptedLLM(script=script, requests=[]))
    return setup


def test_prints_the_answer(fake_run, capsys):
    fake_run([use(FETCH), say("Em setembro de 2026, Búzios produziu 1.197.657 bbl/d.")])
    assert cli.main(["Quanto Búzios produziu em setembro de 2026?"]) == 0
    out, err = capsys.readouterr()
    assert out.startswith("Em setembro de 2026, Búzios produziu 1.197.657 bbl/d.\n\n")
    assert "Fontes: field_group_production(BÚZIOS, 2026-08..)" in out and err == ""


def test_verbose_summary_on_stderr(fake_run, capsys):
    fake_run([use(FETCH), say("Búzios: 1.197.657 bbl/d.")])
    assert cli.main(["-v", "Quanto Búzios produziu?"]) == 0
    err = capsys.readouterr().err
    assert "status=answered llm_calls=2 tool_calls=1" in err
    assert "calls: field_group_production=ok" in err


def test_json_record(fake_run, capsys):
    fake_run([use(FETCH), say("Búzios: 1.197.657 bbl/d.")])
    assert cli.main(["--json", "Quanto Búzios produziu?"]) == 0
    data = json.loads(capsys.readouterr().out)
    assert data["status"] == "answered" and data["model"] == "claude-haiku-5-5"
    assert data["tool_records"][0]["name"] == "field_group_production"
    assert "messages" not in data and len(data["llm_log"]) == 2


def test_failed_question_exits_1(fake_run, capsys):
    fake_run([use(FETCH)], toolbox=FakeToolbox(broken=True))
    assert cli.main(["Quanto Búzios produziu?"]) == 1
    assert "Não foi possível acessar os dados" in capsys.readouterr().out


def test_start_failures_exit_2(fake_run, capsys, monkeypatch):
    fake_run([], start_error=ToolboxError("expected anp-mcp 0.1.0, got anp-mcp 0.2.0"))
    assert cli.main(["q"]) == 2
    assert "expected anp-mcp 0.1.0" in capsys.readouterr().err
    monkeypatch.delenv("ANTHROPIC_API_KEY")
    assert cli.main(["q"]) == 2
    assert "ANTHROPIC_API_KEY is not set" in capsys.readouterr().err
    monkeypatch.setenv("ANTHROPIC_API_KEY", "x")
    monkeypatch.setenv("ANP_AGENT_MAX_TOOL_CALLS", "0")
    assert cli.main(["q"]) == 2
    assert "max_tool_calls must be positive" in capsys.readouterr().err
