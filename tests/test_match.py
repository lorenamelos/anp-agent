import json

import pytest

from anp_agent.guardrail.match import check
from anp_agent.guardrail.sources import build_sources

FIELD = {"field_group": "BÚZIOS", "unit": "bbl/d", "months": [
    {"month": "2026-08", "petroleum_bpd": 1179084.0, "active_wells": 36},
    {"month": "2026-09", "petroleum_bpd": 1197657.0, "active_wells": 37},
]}
TOP = {"latest_month": "2026-09", "window_months": 6, "groups": [
    {"field_group": "BACALHAU", "before_bpd": 32119.0, "recent_bpd": 99492.0,
     "change_bpd": 67373.0, "change_pct": 209.8},
    {"field_group": "RONCADOR", "before_bpd": 120000.0, "recent_bpd": 90000.0,
     "change_bpd": -30000.0, "change_pct": -25.0},
]}


def sources(question="Quanto Búzios produziu?", *results):
    calls = [{"name": "t", "args": {}, "result": json.dumps(r), "is_error": False}
             for r in results]
    return build_sources(question, calls)


def grounded(answer, *results, question="Quanto Búzios produziu?"):
    report = check(answer, sources(question, *results))
    return [v.grounded for v in report.verdicts]


@pytest.mark.parametrize("answer", [
    "Búzios produziu 1.197.657 bbl/d",       # exact, pt
    "Búzios produced 1,197,657 bbl/d",       # exact, en
    "cerca de 1,2 milhão de bbl/d",          # rounded
    "1,19 milhão de bbl/d",                  # truncated
    "1.198 mil bbl/d",                       # rounded with scale word
    "about 1.22 million bbl/d",              # within 2% with a hedge
    "190 mil m³/d",                          # conversion to m³/d (1,197,657 x 0.158987)
    "35.929.710 barris em setembro",         # monthly total: 1,197,657 x 30 days
])
def test_numbers_that_come_from_the_tools(answer):
    assert all(grounded(answer, FIELD))


@pytest.mark.parametrize("answer", [
    "Búzios produziu 1.250.000 bbl/d",       # no source
    "1,22 milhão de bbl/d",                  # 2% away, but no hedge
    "cerca de 1,3 milhão de bbl/d",          # more than 2% away even with a hedge
])
def test_numbers_without_a_source(answer):
    assert not all(grounded(answer, FIELD))


def test_monthly_total_uses_the_days_of_that_month():
    assert all(grounded("36.551.604 barris", FIELD))      # 1,179,084 x 31 (August)
    assert not all(grounded("35.372.520 barris", FIELD))  # 1,179,084 x 30: wrong month length


def test_direction_must_agree_with_the_sign():
    assert all(grounded("Roncador teve queda de 25%", TOP))
    assert all(grounded("Roncador: −25,0%", TOP))
    assert not all(grounded("Roncador cresceu 25%", TOP))
    assert all(grounded("Bacalhau cresceu 209,8%", TOP))
    assert all(grounded("Roncador variou 25%", TOP))  # no direction: absolute value


def test_ranges():
    # both endpoints from the tools
    assert all(grounded("entre 1,18 e 1,2 milhão de bbl/d", FIELD))
    # narrow range containing a source value
    assert all(grounded("entre 1,15 e 1,25 milhão de bbl/d", FIELD))
    # wide range that only contains a value: not grounded
    assert not all(grounded("entre 1 e 1,5 milhão de bbl/d", FIELD))


def test_dates():
    assert all(grounded("Em setembro de 2026", FIELD))
    assert not all(grounded("Em setembro de 2025", FIELD))
    assert not all(grounded("2024 foi recorde", FIELD))  # a year no tool returned


def test_question_numbers_are_grounded():
    q = "Quando Búzios passou de 1 milhão de bbl/d?"
    assert all(grounded("Búzios passou de 1 milhão de bbl/d em 2026-09", FIELD, question=q))


def test_counts_are_exempt_and_wells_are_checked():
    assert all(grounded("os 2 meses mostram 37 poços", FIELD))
    assert not all(grounded("Búzios tinha 41 poços", FIELD))


def test_derive_hook_is_used_only_when_nothing_else_matches():
    seen = []

    def derive(reading, unit, hedged):
        seen.append(reading.value)
        return "derived" if reading.value == 1188370.5 else None

    report = check("média de 1.188.370,5 bbl/d e 1.197.657 bbl/d",
                   sources("q", FIELD), derive=derive)
    assert report.ok
    assert report.verdicts[0].source == "derived"
    assert seen == [1188370.5]


def test_report_lists_ungrounded_claims():
    report = check("Búzios: 1.197.657 bbl/d; Mero: 777.777 bbl/d", sources("q", FIELD))
    assert not report.ok
    assert [v.claim.text for v in report.ungrounded] == ["777.777 bbl/d"]


def test_one_barrel_definition_is_grounded():
    assert all(grounded("Usei 1 bbl = 0,158987 m³ e 1 barril ≈ 0,159 m³", FIELD))


def test_policy_trailing_zeros_and_cap():
    from anp_agent.guardrail.match import Policy

    s = sources("q", {"unit": "bbl/d", "months": [{"month": "2026-09", "petroleum_bpd": 43210.0}]})
    strict = check("40.000 bbl/d", s)  # default: "40.000" is exact to the unit
    loose = check("40.000 bbl/d", s, policy=Policy(trailing_zeros=True))  # 35,000..50,000
    capped = check("40.000 bbl/d", s, policy=Policy(trailing_zeros=True, max_rel=0.05))
    assert loose.ok and not strict.ok and not capped.ok  # capped: at most 2,000 either way
    assert check("cerca de 43.000 bbl/d", s).ok  # a hedge still covers round numbers
    # the cap never goes below the last written digit: "0,5%" still covers 0.45..0.6
    s2 = sources("q", {"x_pct": 0.52})
    assert check("0,5%", s2, policy=Policy(trailing_zeros=True, max_rel=0.05)).ok


def test_day_constants_only_ground_numbers_without_unit():
    assert all(grounded("multipliquei por 31 dias", FIELD))
    assert not all(grounded("Búzios subiu 31%", FIELD))
    assert not all(grounded("Búzios tinha 30 poços", FIELD))
