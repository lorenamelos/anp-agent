import json

import pytest

from anp_agent.guardrail.sources import BBL_TO_M3, build_sources

# Shapes as returned by anp-mcp v0.1.0 (values shortened).
FIELD = {"field_group": "MERO", "unit": "bbl/d", "months": [
    {"month": "2026-01", "petroleum_bpd": 669378.0, "active_wells": 20},
    {"month": "2026-02", "petroleum_bpd": 663823.0, "active_wells": 19},
]}
SEARCH = {"matches": [{
    "field_group": "BÚZIOS", "latest_month": "2026-09", "latest_petroleum_bpd": 1197657.0,
    "matched_fields": ["BÚZIOS_ECO"],
    "grouping_reason": "BÚZIOS_ECO (from 2021-09): 38 shared wells 2019-10..2026-09; 100% of volume",
}], "unit": "bbl/d (oil + condensate)"}
TOP = {"latest_month": "2026-09", "window_months": 6, "method": "mean of last window",
       "groups": [{"field_group": "RONCADOR", "before_bpd": 120000.0, "recent_bpd": 90000.0,
                   "change_bpd": -30000.0, "change_pct": -25.0}]}
RECON = {"summary": {"months_compared": 261, "months_outside_1pct": 0, "worst_month": "2008-03",
                     "worst_gap_pct": 0.517, "first_month": "2005-01", "last_month": "2026-09"},
         "definition": "gap = wells / national - 1 (petroleum, bbl/d)"}
NATIONAL_YEAR = {"frequency": "year", "unit": "bbl/d", "years": [
    {"year": 2024, "months_available": 12, "complete_year": True, "petroleum_bpd": 3360000.0,
     "onshore_bpd": 100000.0, "offshore_bpd": 3260000.0},
    {"year": 2026, "months_available": 9, "complete_year": False, "petroleum_bpd": 3900000.0,
     "onshore_bpd": 90000.0, "offshore_bpd": 3810000.0},
]}


def call(name, result, **args):
    return {"name": name, "args": args, "result": json.dumps(result), "is_error": False}


def by_label(s, fragment):
    found = [n for n in s.numbers if fragment in n.label]
    assert found, fragment
    return found


def test_tool_numbers_carry_unit_month_and_series():
    s = build_sources("q", [call("field_group_production", FIELD, field_group="MERO")])
    n = by_label(s, "months[1].petroleum_bpd")[0]
    assert (n.value, n.unit, n.month, n.days, n.pos) == (663823.0, "bbl/d", "2026-02", 28, 1)
    assert n.series == "call 1: months[].petroleum_bpd"
    assert by_label(s, "months[0].active_wells")[0].unit == "wells"
    assert (2026, 1) in s.months and 2026 in s.years


def test_numbers_and_dates_inside_strings():
    s = build_sources("q", [call("search_field_groups", SEARCH, text="buzios")])
    values = {n.value for n in s.numbers if n.origin == "tool"}
    assert {1197657.0, 38, 100}.issubset(values)
    assert {(2021, 9), (2019, 10), (2026, 9)}.issubset(s.months)


def test_signed_values_and_percent_keys():
    s = build_sources("q", [call("top_changes", TOP, months=6, direction="down")])
    assert by_label(s, "change_pct")[0].value == -25.0
    assert by_label(s, "change_pct")[0].unit == "%"
    assert by_label(s, "args.months")[0].value == 6


def test_threshold_in_a_key_name():
    s = build_sources("q", [call("national_reconciliation", RECON)])
    key = by_label(s, "key months_outside_1pct")[0]
    assert (key.value, key.unit) == (1.0, "%")
    assert (2008, 3) in s.months


def test_yearly_rows_have_days_only_when_complete():
    s = build_sources("q", [call("national_production", NATIONAL_YEAR, frequency="year")])
    rows = by_label(s, "petroleum_bpd")
    assert [r.days for r in rows] == [366, None]  # 2024 is a leap year; 2026 incomplete
    assert {2024, 2026}.issubset(s.years)


def test_question_instructions_and_constants():
    s = build_sources("Quando Búzios passou de 1 milhão de barris por dia?", [],
                      instructions="Monthly from 2005. ANP publishes about two months later.")
    assert any(n.origin == "question" and n.value == 1_000_000 for n in s.numbers)
    assert 2005 in s.years
    assert any(n.origin == "constant" and n.value == BBL_TO_M3 for n in s.numbers)


def test_failed_calls_give_no_numbers():
    bad = {"name": "field_group_production", "args": {}, "result": "Error: 12345",
           "is_error": True}
    s = build_sources("q", [bad])
    assert not any(n.origin == "tool" for n in s.numbers)


def test_truncated_result_is_scanned_and_reported():
    text = json.dumps(FIELD, indent=2)[:-60]  # cut mid-JSON, as stored results are
    s = build_sources("q", [{"name": "field_group_production", "args": {}, "result": text,
                             "is_error": False}])
    assert s.truncated == [1]
    first = next(n for n in s.numbers if n.label.startswith("call 1: petroleum_bpd#0"))
    assert (first.value, first.unit, first.month) == (669378.0, "bbl/d", "2026-01")


@pytest.mark.parametrize("bad", ["", "not json", "{"])
def test_garbage_results_do_not_crash(bad):
    s = build_sources("q", [{"name": "x", "args": {}, "result": bad, "is_error": False}])
    assert s.truncated == [1]
