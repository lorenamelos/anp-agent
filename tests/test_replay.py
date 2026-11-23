import json

from agent_evals.replay_guardrail import Calculator, cited, group, load, run
from anp_agent.guardrail.extract import Reading
from anp_agent.guardrail.sources import build_sources

SERIES = {"unit": "bbl/d", "months": [
    {"month": "2026-01", "petroleum_bpd": 100.0, "active_wells": 10},
    {"month": "2026-02", "petroleum_bpd": 200.0, "active_wells": 11},
    {"month": "2026-03", "petroleum_bpd": 300.0, "active_wells": 12},
]}


def calc():
    calls = [{"name": "t", "args": {}, "result": json.dumps(SERIES), "is_error": False}]
    return Calculator(build_sources("q", calls))


def test_calculator_operations():
    c = calc()
    assert c(Reading(200, 1), "bbl/d", False).startswith("mean(")  # mean of the 3 months
    assert c(Reading(600, 1), "bbl/d", False).startswith("sum(")
    assert c(Reading(150, 1), "bbl/d", False).startswith("mean(")  # months 1-2
    # 50% is 300 vs 200, and also a mean over a sum of two months: any one of them will do
    assert c(Reading(50, 1), "%", False).startswith(("percent_change(", "ratio("))
    assert c(Reading(1, 1), "bbl/d", False) is None


def test_calculator_does_not_mix_units():
    c = calc()
    # 2 is a difference of wells (12 - 10), never a percentage or a bbl/d value
    assert c(Reading(2, 1), "bbl/d", False) is None
    assert c(Reading(2, 1), "wells", False).startswith("difference(")


def test_cited_and_group():
    assert cited("35%", "says about 35% but the data gives 51%")
    assert cited("100 mil bbl/d", "drop of 'cerca de 100 mil' is wrong")
    assert not cited("989 mil bbl/d", "the smallest gap was in March")
    assert group({"grader_passed": True, "error_type": "none"}) == "passed"
    assert group({"grader_passed": False, "error_type": "wrong_number"}).endswith("ungrounded")
    assert group({"grader_passed": False, "error_type": "other"}) == "failed: other"


def test_replay_runs_on_the_vendored_records():
    records = load()
    assert len(records) == 126
    text, flagged = run()
    assert "## Without calculator" in text and "## With calculator (upper bound)" in text
    assert {f["mode"] for f in flagged} == {"base", "calc"}
