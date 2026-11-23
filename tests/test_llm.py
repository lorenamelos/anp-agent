import asyncio
import os
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest
from anthropic.types import Message
from langchain_core.messages import HumanMessage

from anp_agent.budget import Usage
from anp_agent.config import Settings
from anp_agent.llm import load_prices, make_llm, system_message, usage_of, usd

ROOT = Path(__file__).resolve().parents[1]
RAW = {"input_tokens": 12, "output_tokens": 34, "cache_read_input_tokens": 5000,
       "cache_creation_input_tokens": 700}


def fake_response(**content):
    return Message.model_validate({
        "id": "msg_1", "type": "message", "role": "assistant", "model": "claude-haiku-5-5",
        "content": [content or {"type": "text", "text": "ok"}], "stop_reason": "end_turn",
        "stop_sequence": None, "usage": RAW})


@pytest.fixture
def fake_api(monkeypatch):
    """Replace the HTTP call of ChatAnthropic; record the request payload."""
    from langchain_anthropic import ChatAnthropic

    sent = {}

    def use(response):
        async def acreate(self, payload):
            sent.update(payload)
            return SimpleNamespace(parse=lambda: response)
        monkeypatch.setattr(ChatAnthropic, "_acreate", acreate)
        return sent
    return use


def invoke(llm):
    return asyncio.run(llm.ainvoke([system_message("Be brief."), HumanMessage("hi")]))


def test_make_llm():
    llm = make_llm(Settings(model="claude-sonnet-5-5", max_output_tokens=1000))
    assert (llm.model, llm.max_tokens, llm.max_retries) == ("claude-sonnet-5-5", 1000, 2)


def test_usage_comes_from_the_raw_block_not_usage_metadata(fake_api):
    sent = fake_api(fake_response())
    msg = invoke(make_llm(Settings()))
    assert usage_of(msg) == Usage(input=12, output=34, cache_read=5000, cache_write=700)
    # LangChain folds the cache tokens into input_tokens: using it would count them twice
    assert msg.usage_metadata["input_tokens"] == 12 + 5000 + 700
    # the cache breakpoint reaches the request
    assert sent["system"][0]["cache_control"] == {"type": "ephemeral"}


def test_tool_use_turn_keeps_the_usage(fake_api):
    sent = fake_api(fake_response(type="tool_use", id="t1", name="echo", input={"x": 1}))
    llm = make_llm(Settings()).bind_tools([{"name": "echo", "description": "Echo.",
                                            "input_schema": {"type": "object"}}])
    msg = invoke(llm)
    assert msg.tool_calls[0]["name"] == "echo" and usage_of(msg).total == 12 + 34 + 5000 + 700
    assert sent["tools"][0]["name"] == "echo" and "tool_choice" not in sent


def test_usd_matches_the_harness():
    sys.path.insert(0, str(ROOT / "vendor" / "anp-mcp"))
    try:
        from evals.run_evals import Costs
    finally:
        sys.path.pop(0)
    prices = load_prices()
    u = Usage(input=1200, output=340, cache_read=50_000, cache_write=7_000)
    for model in prices:
        harness = Costs(prices).add(model, SimpleNamespace(
            input_tokens=u.input, output_tokens=u.output, cache_read_input_tokens=u.cache_read,
            cache_creation_input_tokens=u.cache_write))
        assert usd(u, model, prices) == pytest.approx(harness)
    assert usd(u, "unknown-model", prices) is None


@pytest.mark.api
def test_usage_matches_a_direct_sdk_call():
    """Same request through the SDK and through LangChain, after the cache is written.
    Prints both counts. Costs well under US$ 0.001 with Haiku."""
    import anthropic
    from anp_mcp.server import INSTRUCTIONS, mcp
    from dotenv import load_dotenv

    load_dotenv(ROOT / ".env")
    assert os.environ.get("ANTHROPIC_API_KEY"), "ANTHROPIC_API_KEY missing in .env"
    settings = Settings(max_output_tokens=256)
    # a realistic cacheable prefix: the anp-mcp tools as listed, and the server instructions
    tools = [{"name": t.name, "description": t.description or "", "input_schema": t.input_schema}
             for t in asyncio.run(mcp.list_tools())]
    system = f"Answer in one word.\n\n{INSTRUCTIONS}"
    question = "Reply with the single word: ok"
    client = anthropic.Anthropic()

    def sdk():
        r = client.messages.create(
            model=settings.model, max_tokens=settings.max_output_tokens, tools=tools,
            system=[{"type": "text", "text": system, "cache_control": {"type": "ephemeral"}}],
            messages=[{"role": "user", "content": question}])
        return Usage.from_raw(r.usage.model_dump())

    warm = sdk()  # writes the cache
    msg = asyncio.run(make_llm(settings).bind_tools(tools).ainvoke(
        [system_message(system), HumanMessage(question)]))
    ours, direct = usage_of(msg), sdk()
    prices = load_prices()
    print(f"\nwarm-up SDK call:  {warm}")
    print(f"LangChain (ours):  {ours}  usd={usd(ours, settings.model, prices):.8f}")
    print(f"direct SDK call:   {direct}  usd={usd(direct, settings.model, prices):.8f}")
    print(f"LangChain usage_metadata: {dict(msg.usage_metadata)}")
    assert warm.cache_write > 0, "the prefix was not cached; the comparison needs it"
    assert (ours.input, ours.cache_read, ours.cache_write) == \
        (direct.input, direct.cache_read, direct.cache_write)
    assert ours.cache_read > 0
