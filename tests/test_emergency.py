import json
from pathlib import Path

import pytest

from anp_agent.calculator import add_result
from anp_agent.emergency import render
from anp_agent.graph import as_calls
from anp_agent.guardrail.match import check
from anp_agent.guardrail.sources import build_sources

FIXTURES = Path(__file__).parent / "fixtures"


def fixture(case):
    """Real records of the first agent-v1 run (a03 r0, a05 r0), both cut by the token limit."""
    return json.loads((FIXTURES / f"emergency_{case}.json").read_text())


def sources_of(fx):
    records = fx["tool_records"]
    s = build_sources(fx["question"], as_calls(records))
    for i, r in enumerate(records, start=1):
        if r["name"] == "calculate" and r["status"] == "ok":
            add_result(s, i, r["args"], r["structured"])
    return s


@pytest.mark.parametrize("case", ["a03", "a05"])
def test_every_number_is_grounded_and_no_raw_json(case):
    fx = fixture(case)
    text = render(fx["tool_records"], fx["not_executed"], fx["token_limited"], "pt")
    assert "{" not in text and '"' not in text
    report = check(text, sources_of(fx))
    assert report.ok, [v.claim.text for v in report.ungrounded]


def test_a03_layout():
    fx = fixture("a03")
    text = render(fx["tool_records"], fx["not_executed"], fx["token_limited"], "pt")
    lines = text.splitlines()
    assert lines[0].startswith("O limite por pergunta foi atingido antes de concluir a resposta.")
    assert "- field_group_production(Lula, 2025-01..2025-12): sem dados" in lines
    assert "  - TUPI: último mês 2026-09, produção no último mês 879.206 bbl/d" in lines
    assert "  - 2025-12: petróleo 903.995 bbl/d, poços ativos 64" in lines
    assert "- calculate(mean, 12 valores): 837.532 bbl/d" in lines
    assert lines[-2].startswith("Consultas feitas: field_group_production(Lula, 2025-01..2025-12);")
    assert lines[-1] == ("Ficaram de fora: calculate(percent_change, 2 valores); "
                         "a conclusão da resposta (limite de tokens)")


def test_a05_rankings_keep_the_top_and_list_what_was_left_out():
    fx = fixture("a05")
    text = render(fx["tool_records"], fx["not_executed"], fx["token_limited"], "pt")
    lines = text.splitlines()
    search = lines.index("- search_field_groups(a, limit=50):")
    assert lines[search + 1].startswith("  - BÚZIOS:")  # the top of the list, not the tail
    assert "  - (+38 linhas seguintes omitidas)" in lines
    assert "  - BACALHAU: antes 32.119 bbl/d, recente 99.492 bbl/d, variação 67.374 bbl/d, " \
           "variação 209,8%" in lines
    left = lines[-1]
    assert left.startswith("Ficaram de fora: field_group_production(JUBARTE, 2025-01..2025-12);")
    assert left.count("field_group_production(") == 10


def test_long_series_show_their_latest_months():
    months = [{"month": f"20{y:02d}-{m:02d}", "petroleum_bpd": 1000.0 * m}
              for y in (24, 25) for m in range(1, 13)]
    records = [{"call_id": "c", "name": "national_production", "args": {"start": "2024-01"},
                "result_text": "", "structured": {"unit": "bbl/d", "months": months},
                "status": "ok", "seconds": 0.0}]
    lines = render(records, [], False, "en").splitlines()
    assert lines[0].startswith("The per-question limit was reached")
    assert "  - (+12 earlier periods omitted)" in lines
    assert "  - 2025-12: petroleum 12,000 bbl/d" in lines and lines[-1] == "Left out: none"
