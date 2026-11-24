"""The agent graph (design, section 1.1).

    start -> agent <-> tools
             agent -> guardrail -> finalize                 (all numbers grounded)
                      guardrail -> feedback -> agent        (one regeneration, if it fits)
                      guardrail -> redact -> finalize       (otherwise)
             agent -> finalize                              (no LLM call fits: tool results)
             tools -> finalize                              (server unavailable)

The state holds only serialisable per-question data; the settings, the MCP toolbox, the LLM
and the prices come from the run context. Final mode keeps the tool definitions in the
request with `tool_choice: none`: the conversation already holds tool_use and tool_result
blocks, which a request without tools cannot carry, and the cached prefix stays the same.
"""

import asyncio
import json
from dataclasses import dataclass, field
from typing import Annotated, Any, Literal, Protocol, TypedDict

from langchain_core.language_models import BaseChatModel
from langchain_core.messages import AIMessage, AnyMessage, HumanMessage, ToolMessage
from langgraph.errors import GraphRecursionError
from langgraph.graph import END, START, StateGraph
from langgraph.graph.message import add_messages
from langgraph.runtime import Runtime

from anp_agent import calculator, emergency, footer, prompts
from anp_agent.budget import (
    CALCULATOR,
    MESSAGE_OVERHEAD,
    Mode,
    Usage,
    admit,
    estimate_tokens,
    next_prompt,
    plan_call,
)
from anp_agent.config import Settings
from anp_agent.guardrail.match import check
from anp_agent.guardrail.redact import redact
from anp_agent.guardrail.sources import Sources, build_sources
from anp_agent.llm import system_message, usage_of, usd
from anp_agent.state import ToolRecord

Status = Literal["answered", "redacted", "budget_reached", "tool_unavailable", "error"]
NOT_EXECUTED = "not executed: {kind} budget reached"


class Toolbox(Protocol):
    instructions: str
    tool_schemas: list[dict]
    latest_month: str | None

    async def call(self, name: str, args: dict, call_id: str = "") -> ToolRecord: ...


@dataclass
class AgentContext:
    settings: Settings
    toolbox: Toolbox
    llm: BaseChatModel
    prices: dict = field(default_factory=dict)

    @property
    def tools(self) -> list[dict]:
        return [*self.toolbox.tool_schemas, calculator.SCHEMA]

    @property
    def system(self) -> str:
        return prompts.system_prompt(self.toolbox.instructions)


class AgentState(TypedDict, total=False):
    question: str
    language: str
    messages: Annotated[list[AnyMessage], add_messages]
    tool_records: list[ToolRecord]
    # R4
    tool_calls_used: int
    calculator_calls_used: int
    tokens_used: int  # all billed tokens: input, output, cache read, cache write
    usd_used: float  # recorded only, never a limit
    llm_calls: int
    last_usage: dict | None  # of the last LLM call, for the next projection
    llm_log: list[dict]  # per call: mode, projected prompt, actual prompt
    final_mode: bool
    token_limited: bool
    not_executed: list[dict]  # calls refused by a cap: name + args
    unavailable: bool
    no_llm: bool  # no LLM call fitted: the draft lists the tool results
    # R1
    draft: str
    regen_count: int
    can_regenerate: bool  # set by the guardrail: one more LLM call fits
    reports: list[dict]  # one per draft checked: ok + ungrounded spans
    removed: list[str]
    # Output
    answer: str
    status: Status
    notices: list[str]


# Helpers


def text_of(message: AIMessage) -> str:
    if isinstance(message.content, str):
        return message.content
    return "".join(b.get("text", "") for b in message.content
                   if isinstance(b, dict) and b.get("type") == "text")


def last_ai(state: AgentState) -> AIMessage:
    return next(m for m in reversed(state["messages"]) if isinstance(m, AIMessage))


def as_calls(records: list[ToolRecord]) -> list[dict]:
    """Tool records in the shape the provenance sources and the footer read."""
    return [{"name": r["name"], "args": r["args"], "is_error": r["status"] != "ok",
             "result": r["structured"] if r["structured"] is not None else r["result_text"]}
            for r in records]


def provenance(ctx: AgentContext, question: str, records: list[ToolRecord]) -> Sources:
    """Sources 1-5 of the design (section 1.3), then the calculator results in call order."""
    schemas = {t["name"]: t["input_schema"] for t in ctx.toolbox.tool_schemas}
    sources = build_sources(question, as_calls(records), ctx.toolbox.instructions, schemas)
    for i, r in enumerate(records, start=1):
        if r["name"] == CALCULATOR and r["status"] == "ok":
            calculator.add_result(sources, i, r["args"], r["structured"])
    return sources


def projected_prompt(ctx: AgentContext, state: AgentState, extra: list[AnyMessage] = ()) -> int:
    """Tokens the next call will send (design, section 3)."""
    messages = [*state["messages"], *extra]
    last = state.get("last_usage")
    if last is None:
        initial = estimate_tokens(ctx.system) + estimate_tokens(ctx.tools) + sum(
            estimate_tokens(m.content) for m in messages[1:])
        return next_prompt(None, 0, initial)
    i = max(k for k, m in enumerate(messages) if isinstance(m, AIMessage))
    added = sum(estimate_tokens(m.content) + MESSAGE_OVERHEAD for m in messages[i + 1:])
    return next_prompt(Usage(**last), added, 0)


def room_for_tokens(ctx: AgentContext, state: AgentState, prompt: int) -> bool:
    s = ctx.settings
    return state["tokens_used"] + prompt + s.max_output_tokens <= s.max_tokens_per_question


# Nodes


def start(state: AgentState, runtime: Runtime[AgentContext]) -> dict:
    question = state["question"]
    return {
        "language": footer.detect_language(question),
        "messages": [system_message(runtime.context.system), HumanMessage(question)],
        "tool_records": [], "tool_calls_used": 0, "calculator_calls_used": 0,
        "tokens_used": 0, "usd_used": 0.0, "llm_calls": 0, "last_usage": None, "llm_log": [],
        "final_mode": False, "token_limited": False, "not_executed": [], "unavailable": False,
        "no_llm": False, "regen_count": 0, "can_regenerate": False, "reports": [], "removed": [], "notices": [],
    }


async def agent(state: AgentState, runtime: Runtime[AgentContext]) -> dict:
    ctx = runtime.context
    lang = state["language"]
    prompt = projected_prompt(ctx, state)
    mode = plan_call(ctx.settings, tokens_used=state["tokens_used"], llm_calls=state["llm_calls"],
                     prompt=prompt, tools_tokens=0, final_mode=state["final_mode"])
    token_limited = state["token_limited"] or not room_for_tokens(ctx, state, prompt)
    if mode is Mode.NONE:
        draft = emergency.render(state["tool_records"], state["not_executed"], token_limited, lang)
        return {"no_llm": True, "token_limited": token_limited, "draft": draft}
    extra: list[AnyMessage] = []
    if mode is Mode.FINAL and not any(e["mode"] == "final" for e in state["llm_log"]):
        refused = ", ".join(footer.describe_call(c, lang) for c in state["not_executed"])
        missing = prompts.MISSING[lang].format(calls=refused) if refused else ""
        extra.append(HumanMessage(prompts.FINAL_NOTE[lang].format(missing=missing)))
        prompt = projected_prompt(ctx, state, extra)
    choice = {"type": "none"} if mode is Mode.FINAL else None
    model = ctx.llm.bind_tools(ctx.tools, tool_choice=choice)
    response = await model.ainvoke([*state["messages"], *extra])
    usage = usage_of(response)
    cost = usd(usage, ctx.settings.model, ctx.prices) or 0.0
    log = {"call": state["llm_calls"] + 1, "mode": mode.value, "projected_prompt": prompt,
           "actual_prompt": usage.prompt, **vars(usage)}
    return {
        "messages": [*extra, response],
        "tokens_used": state["tokens_used"] + usage.total,
        "usd_used": state["usd_used"] + cost,
        "llm_calls": state["llm_calls"] + 1,
        "last_usage": vars(usage),
        "llm_log": [*state["llm_log"], log],
        "final_mode": state["final_mode"] or mode is Mode.FINAL,
        "token_limited": token_limited,
    }


async def tools(state: AgentState, runtime: Runtime[AgentContext]) -> dict:
    ctx = runtime.context
    calls = last_ai(state).tool_calls
    admission = admit(ctx.settings, [c["name"] for c in calls],
                      tool_calls_used=state["tool_calls_used"],
                      calculator_calls_used=state["calculator_calls_used"])
    records = list(state["tool_records"])
    not_executed = list(state["not_executed"])
    messages = []
    # in final mode the model should not ask for tools; if it does, nothing runs
    runs = [False] * len(calls) if state["final_mode"] else admission.run
    for call, run in zip(calls, runs, strict=True):
        name, args, call_id = call["name"], call["args"], call["id"]
        if not run:
            kind = "calculator" if name == CALCULATOR else "tool-call"
            record = ToolRecord(call_id=call_id, name=name, args=args, structured=None,
                                result_text=NOT_EXECUTED.format(kind=kind),
                                status="not_executed", seconds=0.0)
            not_executed.append({"name": name, "args": args})
        elif name == CALCULATOR:
            record = run_calculator(ctx, state["question"], records, call)
        else:
            record = await ctx.toolbox.call(name, args, call_id)
        if record["status"] != "not_executed":
            records.append(record)
        messages.append(ToolMessage(content=record["result_text"], tool_call_id=call_id,
                                    name=name, status="success" if record["status"] == "ok"
                                    else "error"))
    return {
        "messages": messages, "tool_records": records, "not_executed": not_executed,
        "tool_calls_used": (state["tool_calls_used"] if state["final_mode"]
                            else admission.tool_calls_used),
        "calculator_calls_used": (state["calculator_calls_used"] if state["final_mode"]
                                  else admission.calculator_calls_used),
        "final_mode": state["final_mode"] or admission.refused,
        "unavailable": any(r["status"] == "transport_error" for r in records),
    }


def run_calculator(ctx: AgentContext, question: str, records: list[ToolRecord],
                   call: dict) -> ToolRecord:
    """The local calculate tool, against the sources of the calls made so far."""
    record = {"call_id": call["id"], "name": CALCULATOR, "args": call["args"], "seconds": 0.0}
    try:
        result = calculator.calculate(call["args"], provenance(ctx, question, records))
    except calculator.CalcError as e:
        return ToolRecord(**record, result_text=f"error: {e}", structured=None,
                          status="tool_error")
    return ToolRecord(**record, result_text=json.dumps(result), structured=result, status="ok")


def guardrail(state: AgentState, runtime: Runtime[AgentContext]) -> dict:
    draft = text_of(last_ai(state))
    report = check(draft, provenance(runtime.context, state["question"], state["tool_records"]))
    summary = {"ok": report.ok, "ungrounded": [v.claim.text for v in report.ungrounded]}
    regen = (not report.ok and state["regen_count"] == 0
             and can_regenerate(runtime.context, state, summary["ungrounded"]))
    return {"draft": draft, "reports": [*state["reports"], summary], "can_regenerate": regen}


def feedback_message(language: str, ungrounded: list[str]) -> HumanMessage:
    numbers = "; ".join(f'"{n}"' for n in ungrounded)
    return HumanMessage(prompts.FEEDBACK[language].format(numbers=numbers))


def feedback(state: AgentState) -> dict:
    message = feedback_message(state["language"], state["reports"][-1]["ungrounded"])
    return {"messages": [message], "regen_count": state["regen_count"] + 1}


def redact_node(state: AgentState, runtime: Runtime[AgentContext]) -> dict:
    sources = provenance(runtime.context, state["question"], state["tool_records"])
    text, removed = redact(state["draft"], check(state["draft"], sources), state["language"])
    return {"draft": text, "removed": removed}


def finalize(state: AgentState, runtime: Runtime[AgentContext]) -> dict:
    ctx = runtime.context
    lang = state["language"]
    if state["unavailable"]:
        return {"answer": prompts.UNAVAILABLE[lang], "status": "tool_unavailable", "notices": []}
    records = state["tool_records"]
    draft = state.get("draft", "")
    report = check(draft, provenance(ctx, state["question"], records))
    # the emergency answer lists what was left out itself: no R4 notice on top of it
    not_executed = [] if state["no_llm"] else state["not_executed"]
    token_limited = state["token_limited"] and not state["no_llm"]
    budget = footer.budget_notice(not_executed, token_limited, lang)
    recent = (footer.recent_notice(ctx.toolbox.latest_month, lang)
              if ctx.toolbox.latest_month
              and footer.uses_recent_months(report, ctx.toolbox.latest_month) else None)
    answer = footer.finalize(draft, language=lang, calls=as_calls(records), report=report,
                             removed=state["removed"], latest_month=ctx.toolbox.latest_month,
                             not_executed=not_executed, token_limited=token_limited)
    if budget or state["no_llm"]:
        status = "budget_reached"
    elif state["removed"]:
        status = "redacted"
    else:
        status = "answered"
    return {"answer": answer, "status": status, "notices": [n for n in (budget, recent) if n]}


# Routing


def after_agent(state: AgentState) -> str:
    if state["no_llm"]:
        return "finalize"
    # every tool_use needs a tool_result, even one refused in final mode
    return "tools" if last_ai(state).tool_calls else "guardrail"


def after_tools(state: AgentState) -> str:
    return "finalize" if state["unavailable"] else "agent"


def after_guardrail(state: AgentState) -> str:
    if state["reports"][-1]["ok"]:
        return "finalize"
    return "feedback" if state["can_regenerate"] else "redact"


def can_regenerate(ctx: AgentContext, state: AgentState, ungrounded: list[str]) -> bool:
    """One more LLM call, with the feedback message, fits the budget."""
    prompt = projected_prompt(ctx, state, [feedback_message(state["language"], ungrounded)])
    return plan_call(ctx.settings, tokens_used=state["tokens_used"],
                     llm_calls=state["llm_calls"], prompt=prompt, tools_tokens=0,
                     final_mode=state["final_mode"]) is not Mode.NONE


def build_graph():
    g = StateGraph(AgentState, context_schema=AgentContext)
    g.add_node("start", start)
    g.add_node("agent", agent)
    g.add_node("tools", tools)
    g.add_node("guardrail", guardrail)
    g.add_node("feedback", feedback)
    g.add_node("redact", redact_node)
    g.add_node("finalize", finalize)
    g.add_edge(START, "start")
    g.add_edge("start", "agent")
    g.add_conditional_edges("agent", after_agent, ["tools", "guardrail", "finalize"])
    g.add_conditional_edges("tools", after_tools, ["agent", "finalize"])
    g.add_conditional_edges("guardrail", after_guardrail, ["finalize", "feedback", "redact"])
    g.add_edge("feedback", "agent")
    g.add_edge("redact", "finalize")
    g.add_edge("finalize", END)
    return g.compile()


GRAPH = build_graph()


async def answer(question: str, ctx: AgentContext) -> dict[str, Any]:
    """Run one question. Never raises for the per-question failures of design section 2:
    a wall-clock timeout, the recursion backstop or an API error give status "error" with the
    partial state."""
    state: dict[str, Any] = {"question": question}
    try:
        async with asyncio.timeout(ctx.settings.question_timeout_s):
            async for values in GRAPH.astream(
                    {"question": question}, context=ctx, stream_mode="values",
                    config={"recursion_limit": ctx.settings.recursion_limit}):
                state = values
    except TimeoutError:
        return {**state, "status": "error",
                "error": f"question timed out after {ctx.settings.question_timeout_s:g} s"}
    except GraphRecursionError:
        return {**state, "status": "error", "error": "recursion limit reached"}
    except Exception as e:  # noqa: BLE001 - e.g. the API failed after the SDK's retries
        return {**state, "status": "error", "error": f"{type(e).__name__}: {e}"}
    return state
