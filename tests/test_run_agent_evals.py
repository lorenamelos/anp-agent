from agent_evals.run_agent_evals import Costs, agent_fields, as_run
from anp_agent.budget import Usage
from anp_agent.llm import load_prices, usd

RESULT = {
    "answer": "Búzios: 1.197.657 bbl/d.\n\nFontes: field_group_production(BÚZIOS)",
    "status": "answered", "usd_used": 0.0012, "regen_count": 1, "removed": [],
    "reports": [{"ok": False, "ungrounded": ["35%"]}, {"ok": True, "ungrounded": []}],
    "tool_records": [
        {"name": "field_group_production", "args": {"field_group": "BÚZIOS"},
         "result_text": "x" * 7000, "status": "ok"},
        {"name": "calculate", "args": {"operation": "sum", "values": [1, 2]},
         "result_text": "error: 1 is not in", "status": "tool_error"}],
    "llm_log": [{"input": 100, "output": 10, "cache_read": 2000, "cache_write": 0},
                {"input": 50, "output": 20, "cache_read": 2100, "cache_write": 300}],
    "tool_calls_used": 1, "calculator_calls_used": 1, "llm_calls": 2, "tokens_used": 4580,
}


def test_as_run_matches_the_harness_shape():
    run = as_run(RESULT, 3.21)
    assert set(run) == {"answer", "calls", "solver_usd", "input_tokens", "output_tokens",
                        "cache_read_tokens", "cache_write_tokens", "seconds"}
    assert (run["input_tokens"], run["output_tokens"], run["cache_read_tokens"],
            run["cache_write_tokens"]) == (150, 30, 4100, 300)
    assert len(run["calls"][0]["result"]) == 6000  # the harness's RESULT_LIMIT
    assert run["calls"][1] == {"name": "calculate", "args": {"operation": "sum", "values": [1, 2]},
                               "result": "error: 1 is not in", "is_error": True}
    assert run["seconds"] == 3.2


def test_agent_fields():
    f = agent_fields(RESULT, "abc1234")
    assert f["regenerated"] and f["ungrounded_before_regen"] == ["35%"]
    assert f["ungrounded_final"] == [] and f["agent_commit"] == "abc1234"


def test_solver_cost_in_the_harness_costs_matches_ours():
    from agent_evals.run_agent_evals import _add_solver_cost

    prices = load_prices()
    costs = Costs(prices)
    run = as_run(RESULT, 1)
    _add_solver_cost(costs, "claude-haiku-5-5", run)
    ours = usd(Usage(150, 30, 4100, 300), "claude-haiku-5-5", prices)
    assert costs.total == ours
