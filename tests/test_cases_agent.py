import string

import pytest
import yaml

from agent_evals.run_agent_evals import AGENT_INSTRUMENT, INSTRUMENTS, ROOT

CASES = yaml.safe_load((ROOT / "agent_evals" / "cases_agent.yaml").read_text())
V3 = yaml.safe_load((ROOT / "vendor" / "anp-mcp" / "evals" / "cases.yaml").read_text())
TOOLS = {"search_field_groups", "field_group_production", "top_changes",
         "national_reconciliation", "national_production"}


def test_schema():
    ids = [c["id"] for c in CASES]
    assert len(ids) == len(set(ids)) and not set(ids) & {c["id"] for c in V3}
    for c in CASES:
        assert {"id", "category", "question", "expect_tools", "rubric"} <= set(c)
        assert set(c["expect_tools"]) <= TOOLS, c["id"]  # anp-mcp tools only
        assert not c.get("expect_refusal") or c["expect_tools"] == [], c["id"]


def test_q17_is_kept_as_a_regression_case():
    v3_q17 = next(c for c in V3 if c["id"] == "q17")
    regression = [c for c in CASES if c.get("regression")]
    assert [(c["question"], c["rubric"]) for c in regression] == [
        (v3_q17["question"], v3_q17["rubric"])]


def test_instrument_names():
    assert set(INSTRUMENTS) == {"v3", AGENT_INSTRUMENT}


@pytest.mark.integration
def test_every_rubric_fills_from_the_ground_truth():
    truth = INSTRUMENTS[AGENT_INSTRUMENT][1]()
    for c in CASES:
        keys = {f for _, f, _, _ in string.Formatter().parse(c["rubric"]) if f}
        assert keys <= set(truth), (c["id"], keys - set(truth))
        assert "{" not in c["rubric"].format(**truth)
