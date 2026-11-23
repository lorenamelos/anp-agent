import asyncio
import json
from typing import Any

import pytest
from langchain_core.language_models import BaseChatModel
from langchain_core.messages import AIMessage, HumanMessage, ToolMessage
from langchain_core.outputs import ChatGeneration, ChatResult

from anp_agent.config import Settings
from anp_agent.graph import AgentContext, answer
from anp_agent.state import ToolRecord

QUESTION = "Quanto Búzios produziu em setembro de 2026?"
BUZIOS = {"field_group": "BÚZIOS", "unit": "bbl/d", "months": [
    {"month": "2026-08", "petroleum_bpd": 1179084.0, "active_wells": 36},
    {"month": "2026-09", "petroleum_bpd": 1197657.0, "active_wells": 37}]}
SCHEMAS = [
    {"name": "field_group_production", "description": "Monthly production of a field group.",
     "input_schema": {"type": "object", "properties": {
         "field_group": {"type": "string"},
         "start": {"type": "string", "default": "2024-01"},
         "end": {"type": "string", "default": "2026-12"}}, "required": ["field_group"]}},
    {"name": "search_field_groups", "description": "Find field groups.",
     "input_schema": {"type": "object", "properties": {"text": {"type": "string"}}}},
]


def usage(input=1000, output=100, cache_read=0, cache_write=0):
    return {"usage": {"input_tokens": input, "output_tokens": output,
                      "cache_read_input_tokens": cache_read,
                      "cache_creation_input_tokens": cache_write}}


def say(text, **u):
    return AIMessage(content=text, response_metadata=usage(**u))


def use(*calls, **u):
    """A turn asking for tools: use(("field_group_production", {...}), ...)."""
    return AIMessage(content="", response_metadata=usage(**u), tool_calls=[
        {"name": n, "args": a, "id": f"t{i}-{n}"} for i, (n, a) in enumerate(calls)])


class ScriptedLLM(BaseChatModel):
    """Answers from a script; records each request's messages and tool_choice."""
    script: list
    requests: list

    @property
    def _llm_type(self) -> str:
        return "scripted"

    def bind_tools(self, tools, tool_choice=None, **kwargs):
        return self.bind(tool_names=[t["name"] for t in tools], tool_choice=tool_choice)

    def _generate(self, messages, stop=None, run_manager=None, **kwargs) -> ChatResult:
        self.requests.append({"messages": list(messages), **kwargs})
        step = self.script.pop(0)
        if isinstance(step, Exception):
            raise step
        return ChatResult(generations=[ChatGeneration(message=step)])


class FakeToolbox:
    instructions = "Fake server instructions."
    tool_schemas = SCHEMAS
    latest_month = "2026-09"

    def __init__(self, results=None, broken=False):
        self.results = results or {"field_group_production": BUZIOS,
                                   "search_field_groups": {"matches": [{"field_group": "BÚZIOS"}]}}
        self.broken = broken
        self.calls = []

    async def call(self, name, args, call_id=""):
        self.calls.append((name, args))
        if self.broken:
            return ToolRecord(call_id=call_id, name=name, args=args, structured=None,
                              result_text="server unavailable", status="transport_error",
                              seconds=0.0)
        result = self.results[name]
        return ToolRecord(call_id=call_id, name=name, args=args, structured=result,
                          result_text=json.dumps(result), status="ok", seconds=0.01)


DEFAULT = Settings()


def run(script, settings=DEFAULT, toolbox=None, question=QUESTION) -> tuple[dict, Any, Any]:
    llm, toolbox = ScriptedLLM(script=script, requests=[]), toolbox or FakeToolbox()
    out = asyncio.run(answer(question, AgentContext(settings, toolbox, llm)))
    return out, llm, toolbox


FETCH = ("field_group_production", {"field_group": "BÚZIOS", "start": "2026-08"})


def test_grounded_answer():
    out, llm, tb = run([use(FETCH), say("Em setembro de 2026, Búzios produziu 1.197.657 bbl/d.")])
    assert out["status"] == "answered" and out["regen_count"] == 0
    assert out["answer"] == (
        "Em setembro de 2026, Búzios produziu 1.197.657 bbl/d.\n\n"
        "Dados de jul–set/2026 ainda podem ser revisados pela ANP.\n\n"
        "Fontes: field_group_production(BÚZIOS, 2026-08..)")
    assert tb.calls == [FETCH]
    assert out["tool_calls_used"] == 1 and out["llm_calls"] == 2
    assert out["tokens_used"] == 2 * 1100
    # every request offers the anp-mcp tools and the calculator, with tool_choice auto
    assert llm.requests[0]["tool_names"] == [
        "field_group_production", "search_field_groups", "calculate"]
    assert all(r["tool_choice"] is None for r in llm.requests)
    # the tool result went back to the model as a tool_result
    tool_msg = llm.requests[1]["messages"][-1]
    assert isinstance(tool_msg, ToolMessage) and "1197657" in tool_msg.content
    assert [e["mode"] for e in out["llm_log"]] == ["normal", "normal"]


def test_calculator_grounds_a_derived_number():
    mean = ("calculate", {"operation": "mean", "values": [1179084, 1197657], "unit": "bbl/d"})
    out, _, _ = run([use(FETCH), use(mean), say("A média foi de 1.188.370 bbl/d.")])
    assert out["status"] == "answered", out["answer"]
    assert out["calculator_calls_used"] == 1
    assert "calculate(mean, 2 valores)" in out["answer"]
    assert "field_group_production(BÚZIOS, 2026-08..)" in out["answer"]  # transitive


def test_calculator_rejects_an_ungrounded_input():
    bad = ("calculate", {"operation": "mean", "values": [1200000, 1197657]})
    out, llm, _ = run([use(FETCH), use(bad), say("Búzios: 1.197.657 bbl/d.")])
    error = llm.requests[2]["messages"][-1]
    assert isinstance(error, ToolMessage) and error.status == "error"
    assert "1200000 is not in this question's tool results" in error.content
    assert out["calculator_calls_used"] == 1 and out["status"] == "answered"


def test_one_regeneration_fixes_the_answer():
    out, llm, _ = run([use(FETCH), say("Búzios produziu 1.197.657 bbl/d, 35% do pré-sal."),
                       say("Búzios produziu 1.197.657 bbl/d.")])
    assert out["status"] == "answered" and out["regen_count"] == 1
    note = llm.requests[2]["messages"][-1]
    assert isinstance(note, HumanMessage) and '"35%"' in note.content
    assert [r["ok"] for r in out["reports"]] == [False, True]


def test_redaction_after_a_failed_regeneration():
    out, _, _ = run([use(FETCH), say("Búzios produziu 1.197.657 bbl/d, 35% do pré-sal."),
                     say("Búzios produziu 1.197.657 bbl/d, cerca de 40% do pré-sal.")])
    assert out["status"] == "redacted" and out["removed"] == ["40%"]
    assert "cerca de [não confirmado] do pré-sal" in out["answer"]
    assert out["answer"].endswith('Removido por não ter origem confirmada: "40%"')


def test_tool_cap_refuses_and_moves_to_final_mode():
    calls = [("field_group_production", {"field_group": g}) for g in ("BÚZIOS", "MERO", "TUPI")]
    out, llm, tb = run([use(*calls), say("Búzios: 1.197.657 bbl/d. Não consultei Tupi.")],
                       settings=Settings(max_tool_calls=2))
    assert [c[1]["field_group"] for c in tb.calls] == ["BÚZIOS", "MERO"]
    refused = llm.requests[1]["messages"]
    assert refused[-2].content == "not executed: tool-call budget reached"  # a tool_result
    assert "field_group_production(TUPI)" in refused[-1].content  # the final-mode note
    assert llm.requests[1]["tool_choice"] == {"type": "none"}
    assert out["status"] == "budget_reached" and out["notices"][0] == (
        "Resposta limitada pelo orçamento por pergunta: 1 consulta não foi feita "
        "(field_group_production para TUPI).")


def test_last_allowed_llm_call_is_final():
    out, llm, tb = run([use(FETCH), use(("search_field_groups", {"text": "x"})),
                        say("Búzios: 1.197.657 bbl/d.")], settings=Settings(max_llm_calls=2))
    assert llm.requests[1]["tool_choice"] == {"type": "none"}
    # the model asked for a tool anyway: it gets a refusal, and no LLM call is left
    assert tb.calls == [FETCH]
    assert out["llm_calls"] == 2 and out["status"] == "budget_reached"


def test_no_llm_call_fits_the_token_limit():
    big = {"unit": "bbl/d", "months": [{"month": f"20{y:02d}-{m:02d}", "petroleum_bpd": 1e6 + m}
                                       for y in range(5, 26) for m in range(1, 13)]}
    settings = Settings(max_tokens_per_question=6000, max_output_tokens=500)
    out, llm, _ = run([use(FETCH, input=1500)], settings=settings,
                      toolbox=FakeToolbox({"field_group_production": big}))
    assert len(llm.requests) == 1  # the second call would not fit
    assert out["status"] == "budget_reached" and out["token_limited"]
    assert out["answer"].startswith("Não foi possível concluir a resposta dentro do limite")
    assert "field_group_production(BÚZIOS, 2026-08..): {" in out["answer"]
    assert "o limite de tokens foi atingido" in out["answer"]
    assert out["tokens_used"] <= settings.max_tokens_per_question


def test_server_unavailable():
    out, llm, _ = run([use(FETCH)], toolbox=FakeToolbox(broken=True))
    assert out["status"] == "tool_unavailable" and len(llm.requests) == 1
    assert out["answer"] == "Não foi possível acessar os dados (o servidor anp-mcp não respondeu)."


def test_api_error_gives_status_error_with_partial_state():
    out, _, _ = run([use(FETCH), RuntimeError("overloaded")])
    assert out["status"] == "error" and "overloaded" in out["error"]
    assert out["tool_calls_used"] == 1


def test_english_question():
    out, _, _ = run([use(FETCH), say("In September 2026 Búzios produced 1,197,657 bbl/d.")],
                    question="How much did Búzios produce in September 2026?")
    assert out["language"] == "en"
    assert out["answer"].endswith("Sources: field_group_production(BÚZIOS, 2026-08..)")


def test_no_regeneration_without_room():
    settings = Settings(max_tokens_per_question=8000, max_output_tokens=500)
    out, llm, _ = run([use(FETCH, input=2500),
                       say("Búzios produziu 1.197.657 bbl/d, 35% do pré-sal.", input=2600)],
                      settings=settings)
    assert len(llm.requests) == 2 and out["regen_count"] == 0  # straight to redaction
    assert out["status"] == "redacted" and out["removed"] == ["35%"]


def test_final_note_is_sent_once():
    calls = [("field_group_production", {"field_group": g}) for g in ("BÚZIOS", "MERO")]
    out, llm, _ = run([use(*calls), say("Búzios: 35%."), say("Búzios: 1.197.657 bbl/d.")],
                      settings=Settings(max_tool_calls=1))
    notes = [m for m in llm.requests[2]["messages"]
             if isinstance(m, HumanMessage) and "Limite de consultas" in str(m.content)]
    assert len(notes) == 1 and llm.requests[2]["tool_choice"] == {"type": "none"}
    assert out["regen_count"] == 1 and out["status"] == "budget_reached"


@pytest.mark.integration
def test_real_toolbox_end_to_end(tmp_path):
    """The real anp-mcp server through AnpToolbox, a scripted model: no API spend."""
    from anp_agent.mcp_client import AnpToolbox

    async def main():
        async with AnpToolbox.start(DEFAULT, stderr_log=tmp_path / "e.log") as toolbox:
            llm = ScriptedLLM(script=[
                use(("field_group_production", {"field_group": "BÚZIOS", "start": "2026-09"})),
                say("Em setembro de 2026, Búzios produziu 1.197.657 bbl/d, com 37 poços.")],
                requests=[])
            return await answer(QUESTION, AgentContext(DEFAULT, toolbox, llm))

    out = asyncio.run(main())
    assert out["status"] == "answered", out["answer"]
    assert out["answer"].endswith("Fontes: field_group_production(BÚZIOS, 2026-09..)")
    assert out["tool_records"][0]["structured"]["months"][0]["petroleum_bpd"] == 1197657.0
