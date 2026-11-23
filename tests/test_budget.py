import pytest

from anp_agent.budget import Mode, Usage, admit, estimate_tokens, next_prompt, plan_call
from anp_agent.config import Settings

S = Settings()  # 40,000 tokens, 2,048 output, 10 LLM calls, 6 + 4 calls


def test_usage_counts_every_kind_alike():
    raw = {"input_tokens": 300, "output_tokens": 200, "cache_read_input_tokens": 5000,
           "cache_creation_input_tokens": 1000}
    u = Usage.from_raw(raw)
    assert (u.prompt, u.total) == (6300, 6500)
    assert Usage.from_raw({"input_tokens": 10, "cache_read_input_tokens": None}).total == 10
    assert Usage.from_raw(None).total == 0


def test_estimate_and_next_prompt():
    assert estimate_tokens("x" * 101) == 51
    assert estimate_tokens({"a": 1}) == estimate_tokens('{"a": 1}')
    assert next_prompt(None, 0, initial=4000) == 4000
    last = Usage(input=300, output=200, cache_read=5000, cache_write=1000)
    assert next_prompt(last, 900, initial=4000) == 6300 + 200 + 900


def plan(tokens_used, prompt, *, llm_calls=1, tools_tokens=3000, final_mode=False):
    return plan_call(S, tokens_used=tokens_used, llm_calls=llm_calls, prompt=prompt,
                     tools_tokens=tools_tokens, final_mode=final_mode)


def test_plan_call_edges():
    room = 40_000 - 2_048
    assert plan(0, room) is Mode.NORMAL  # exactly at the limit
    assert plan(0, room + 1) is Mode.FINAL  # fits only without the tool definitions
    assert plan(10_000, room - 10_000 + 3_000) is Mode.FINAL
    assert plan(10_000, room - 10_000 + 3_001) is Mode.NONE
    assert plan(0, 5_000, final_mode=True) is Mode.FINAL  # a cap was reached: no tools


def test_plan_call_llm_call_backstop():
    assert plan(0, 5_000, llm_calls=8) is Mode.NORMAL
    assert plan(0, 5_000, llm_calls=9) is Mode.FINAL  # the last call has no tools
    assert plan(0, 5_000, llm_calls=10) is Mode.NONE


def test_projection_never_exceeds_the_limit():
    # whatever the model does, an allowed call costs at most prompt + max_output_tokens
    for used in range(0, 40_000, 997):
        for prompt in range(0, 40_000, 1_009):
            mode = plan(used, prompt)
            cost = {Mode.NORMAL: prompt, Mode.FINAL: prompt - 3000, Mode.NONE: None}[mode]
            if cost is not None:
                assert used + cost + S.max_output_tokens <= S.max_tokens_per_question


@pytest.mark.parametrize("names, used, run, after", [
    # 3 anp-mcp calls with 2 left: the first 2 run, the third is refused
    (["a", "b", "c"], (4, 0), [True, True, False], (6, 0)),
    # each kind against its own cap
    (["calculate", "a", "calculate"], (6, 3), [True, False, False], (6, 4)),
    (["a", "calculate"], (0, 0), [True, True], (1, 1)),
])
def test_admit(names, used, run, after):
    a = admit(S, names, tool_calls_used=used[0], calculator_calls_used=used[1])
    assert a.run == run and (a.tool_calls_used, a.calculator_calls_used) == after
    assert a.refused == (not all(run))
