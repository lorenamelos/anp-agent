import json

import pytest

from anp_agent.calculator import SCHEMA, CalcError, add_result, calculate
from anp_agent.guardrail.match import check
from anp_agent.guardrail.sources import build_sources

BUZIOS = {"field_group": "BÚZIOS", "unit": "bbl/d", "months": [
    {"month": "2026-08", "petroleum_bpd": 1179084.0, "active_wells": 36},
    {"month": "2026-09", "petroleum_bpd": 1197657.0, "active_wells": 37},
]}
NATIONAL = {"frequency": "month", "unit": "bbl/d", "months": [
    {"month": "2008-03", "petroleum_bpd": 1749888.0, "onshore_bpd": 186603.0,
     "offshore_bpd": 1563285.0}]}
RECON = {"summary": {"worst_month": "2008-03", "worst_gap_pct": 0.517}}


def sources(*results):
    calls = [{"name": "t", "args": {}, "result": json.dumps(r), "is_error": False}
             for r in results]
    return build_sources("q", calls)


def calc(s, **args):
    return calculate(args, s)


def test_operations():
    s = sources(BUZIOS)
    assert calc(s, operation="sum", values=[1179084, 1197657])["result"] == 2376741
    assert calc(s, operation="mean", values=[1179084, 1197657], unit="bbl/d") == {
        "operation": "mean", "inputs": [1179084, 1197657], "result": 1188370.5, "unit": "bbl/d"}
    assert calc(s, operation="difference", values=[1179084, 1197657])["result"] == 18573
    assert calc(s, operation="percent_change", values=[1179084, 1197657])["result"] == \
        pytest.approx(1.5752, abs=1e-4)
    share = calc(s, operation="ratio", values=[36, 37])
    assert share["unit"] == "%" and share["result"] == pytest.approx(97.2973, abs=1e-4)


def test_multiply_a_percentage():
    # q17: 0.517 % of the national total in March 2008
    s = sources(NATIONAL, RECON)
    out = calc(s, operation="multiply", values=[0.517, 1749888], percent=True, unit="bbl/d")
    assert out["result"] == pytest.approx(9046.9, abs=0.1) and out["unit"] == "bbl/d"


def test_convert():
    s = sources(BUZIOS)
    assert calc(s, operation="convert", values=[1197657], to_unit="m3/d")["result"] == \
        pytest.approx(190411.9, abs=0.1)
    assert calc(s, operation="convert", values=[1179084], to_unit="bbl",
                month="2026-08")["result"] == 1179084 * 31
    assert calc(s, operation="convert", values=[1197657], to_unit="bbl",
                year=2024)["result"] == 1197657 * 366


def test_inputs_must_be_grounded():
    s = sources(BUZIOS)
    with pytest.raises(CalcError, match="1200000 is not in this question's tool results"):
        calc(s, operation="mean", values=[1200000, 1197657])  # rounded: not as returned
    with pytest.raises(CalcError, match="not in"):
        calc(s, operation="convert", values=[3400000], to_unit="m3/d")  # from memory


def test_signed_inputs_follow_the_source_sign():
    s = sources({"groups": [{"change_bpd": -30000.0, "before_bpd": 120000.0}]})
    assert calc(s, operation="ratio", values=[-30000, 120000])["result"] == -25
    with pytest.raises(CalcError):
        calc(s, operation="ratio", values=[-120000, 120000])


@pytest.mark.parametrize("args, message", [
    ({"operation": "pow", "values": [1]}, "unknown operation"),
    ({"operation": "sum", "values": []}, "non-empty list"),
    ({"operation": "sum", "values": [True]}, "non-empty list"),
    ({"operation": "difference", "values": [36]}, "takes 2"),
    ({"operation": "convert", "values": [1197657], "to_unit": "bbl"}, "needs month"),
    ({"operation": "convert", "values": [1197657], "to_unit": "bbl", "month": "2026-13"},
     "YYYY-MM"),
    ({"operation": "ratio", "values": [36, 0]}, "not in|division"),
])
def test_bad_arguments(args, message):
    with pytest.raises(CalcError, match=message):
        calculate(args, sources(BUZIOS))


def test_results_ground_answers_and_later_calculations():
    s = sources(BUZIOS)
    args = {"operation": "mean", "values": [1179084, 1197657], "unit": "bbl/d"}
    mean = calculate(args, s)
    assert not check("média de 1.188.370 bbl/d", s).ok  # derived: not grounded yet
    add_result(s, 2, args, mean)
    report = check("média de 1.188.370 bbl/d", s)
    assert report.ok  # truncated 1,188,370.5
    used = report.verdicts[0].used[0]
    assert (used.call, used.until) == (2, "2026-09")  # latest month among the inputs
    assert used.based_on == (1,)  # the call the inputs came from
    # transitive: the mean can now be an input
    out = calc(s, operation="convert", values=[1188370.5], to_unit="m3/d")
    assert out["result"] == pytest.approx(188935.46, abs=0.01)


def test_schema_is_a_tool_definition():
    assert SCHEMA["name"] == "calculate"
    props = SCHEMA["input_schema"]["properties"]
    assert set(SCHEMA["input_schema"]["required"]) == {"operation", "values"}
    assert set(props["operation"]["enum"]) == {
        "sum", "mean", "difference", "ratio", "percent_change", "multiply", "convert"}
