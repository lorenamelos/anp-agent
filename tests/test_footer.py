import json

import pytest

from anp_agent.calculator import add_result, calculate
from anp_agent.footer import (
    budget_notice,
    describe_call,
    detect_language,
    finalize,
    recent_months,
    recent_notice,
    uses_recent_months,
)
from anp_agent.guardrail.match import check
from anp_agent.guardrail.redact import redact
from anp_agent.guardrail.sources import build_sources

BUZIOS = {"field_group": "BÚZIOS", "unit": "bbl/d", "months": [
    {"month": "2025-06", "petroleum_bpd": 806000.0},
    {"month": "2026-08", "petroleum_bpd": 1179084.0},
    {"month": "2026-09", "petroleum_bpd": 1197657.0},
]}
SEARCH = {"matches": [{"field_group": "BÚZIOS", "latest_month": "2026-09"}]}


def calls(*pairs):
    return [{"name": n, "args": a, "result": json.dumps(r), "is_error": False}
            for n, a, r in pairs]


SEARCH_CALL = ("search_field_groups", {"text": "buzios"}, SEARCH)
FIELD_CALL = ("field_group_production", {"field_group": "BÚZIOS", "start": "2025-06"}, BUZIOS)


@pytest.mark.parametrize("question, language", [
    ("Quanto Búzios produziu em setembro de 2026?", "pt"),
    ("Qual campo teve a maior queda?", "pt"),
    ("How much did Búzios produce in September 2026?", "en"),
    ("Which field had the largest drop?", "en"),
    ("Búzios 2026-09?", "pt"),  # no stop words: Portuguese by default
])
def test_detect_language(question, language):
    assert detect_language(question) == language


def test_describe_call():
    assert describe_call({"name": "field_group_production", "args": {
        "field_group": "BÚZIOS", "start": "2026-01", "end": "2026-09"}}, "pt") == \
        "field_group_production(BÚZIOS, 2026-01..2026-09)"
    assert describe_call({"name": "national_production", "args": {"start": "2025-06"}}, "en") == \
        "national_production(2025-06..)"
    assert describe_call({"name": "national_reconciliation", "args": {"details": True}}, "pt") == \
        "national_reconciliation(details=true)"
    assert describe_call({"name": "top_changes", "args": {}}, "pt") == "top_changes()"
    assert describe_call({"name": "calculate", "args": {
        "operation": "mean", "values": [1] * 9}}, "pt") == "calculate(mean, 9 valores)"
    assert describe_call({"name": "calculate", "args": {
        "operation": "convert", "values": [1]}}, "en") == "calculate(convert, 1 value)"


def test_budget_notice():
    refused = [{"name": "field_group_production", "args": {"field_group": "MERO"}},
               {"name": "field_group_production", "args": {"field_group": "TUPI"}}]
    assert budget_notice(refused, False, "pt") == (
        "Resposta limitada pelo orçamento por pergunta: 2 consultas não foram feitas "
        "(field_group_production para MERO, TUPI).")
    calc = [{"name": "calculate", "args": {"operation": "sum", "values": [1, 2]}}]
    assert budget_notice(refused[:1] + calc, True, "en") == (
        "Answer limited by the per-question budget: the token limit was reached; 1 query was "
        "not made (field_group_production for MERO) and 1 calculation was not made.")
    assert budget_notice([{"name": "top_changes", "args": {}}], False, "pt").endswith(
        "1 consulta não foi feita (top_changes).")
    assert budget_notice([], False, "pt") is None


def test_recent_months_and_notice():
    assert recent_months("2026-09") == ["2026-07", "2026-08", "2026-09"]
    assert recent_months("2027-01") == ["2026-11", "2026-12", "2027-01"]
    assert recent_notice("2026-09", "pt") == (
        "Dados de jul–set/2026 ainda podem ser revisados pela ANP.")
    assert recent_notice("2027-01", "en") == "Data for Nov 2026–Jan 2027 may still be revised by ANP."


def test_uses_recent_months():
    s = build_sources("q", calls(FIELD_CALL))
    assert uses_recent_months(check("Em setembro: 1.197.657 bbl/d.", s), "2026-09")
    assert not uses_recent_months(check("Em jun/2025: 806.000 bbl/d.", s), "2026-09")
    assert uses_recent_months(check("Dados até 2026-08.", s), "2026-09")  # a date counts too
    # an ungrounded number never triggers it
    assert not uses_recent_months(check("Em jun/2025: 806.000; hoje 1,5 milhão.", s), "2026-09")


def test_uses_recent_months_through_a_calculation():
    s = build_sources("q", calls(FIELD_CALL))
    args = {"operation": "mean", "values": [806000, 1197657], "unit": "bbl/d"}
    add_result(s, 2, args, calculate(args, s))
    assert uses_recent_months(check("Média de 1.001.828 bbl/d.", s), "2026-09")


def test_finalize_with_redaction():
    question = "Quanto Búzios produziu em setembro de 2026?"
    tool_calls = calls(SEARCH_CALL, FIELD_CALL)
    s = build_sources(question, tool_calls)
    draft = "Búzios produziu 1.197.657 bbl/d em setembro de 2026, cerca de 35% do pré-sal."
    report = check(draft, s)
    text, removed = redact(draft, report, "pt")
    out = finalize(text, language="pt", calls=tool_calls, report=report, removed=removed,
                   latest_month="2026-09")
    assert out == (
        "Búzios produziu 1.197.657 bbl/d em setembro de 2026, cerca de [não confirmado] do "
        "pré-sal.\n\n"
        "Dados de jul–set/2026 ainda podem ser revisados pela ANP.\n\n"
        "Fontes: field_group_production(BÚZIOS, 2025-06..)\n"
        'Removido por não ter origem confirmada: "35%"')


def test_finalize_budget_and_calculator_sources():
    tool_calls = calls(FIELD_CALL)
    s = build_sources("q", tool_calls)
    args = {"operation": "mean", "values": [1179084, 1197657], "unit": "bbl/d"}
    add_result(s, 2, args, calculate(args, s))
    tool_calls.append({"name": "calculate", "args": args})
    draft = "The mean was 1,188,370 bbl/d."
    out = finalize(draft, language="en", calls=tool_calls, report=check(draft, s),
                   not_executed=[{"name": "field_group_production",
                                  "args": {"field_group": "MERO"}}])
    assert out == (
        "The mean was 1,188,370 bbl/d.\n\n"
        "Answer limited by the per-question budget: 1 query was not made "
        "(field_group_production for MERO).\n\n"
        "Sources: field_group_production(BÚZIOS, 2025-06..); calculate(mean, 2 values)")


def test_finalize_without_numbers_has_no_footer():
    assert finalize("Não encontrei esse campo.", language="pt", calls=[],
                    report=check("Não encontrei esse campo.", build_sources("q", []))) == \
        "Não encontrei esse campo."
