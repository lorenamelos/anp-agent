import json

from anp_agent.guardrail.match import check
from anp_agent.guardrail.redact import redact
from anp_agent.guardrail.sources import build_sources

BUZIOS = {"field_group": "BÚZIOS", "unit": "bbl/d", "months": [
    {"month": "2026-09", "petroleum_bpd": 1197657.0, "active_wells": 37}]}


def sources():
    calls = [{"name": "field_group_production", "args": {"field_group": "BÚZIOS"},
              "result": json.dumps(BUZIOS), "is_error": False}]
    return build_sources("Quanto Búzios produziu em setembro de 2026?", calls)


def test_replaces_only_the_number_span():
    answer = ("Búzios produziu 1.197.657 bbl/d em setembro de 2026, cerca de 35% do pré-sal, "
              "e chegou a 1,3 milhão de bbl/d no pico.")
    text, removed = redact(answer, check(answer, sources()), "pt")
    assert text == ("Búzios produziu 1.197.657 bbl/d em setembro de 2026, cerca de "
                    "[não confirmado] do pré-sal, e chegou a [não confirmado] no pico.")
    assert removed == ["35%", "1,3 milhão de bbl/d"]


def test_english_marker_and_nothing_to_remove():
    answer = "Búzios: 1,197,657 bbl/d in September 2026; about 40% of the total."
    text, removed = redact(answer, check(answer, sources()), "en")
    assert text == "Búzios: 1,197,657 bbl/d in September 2026; about [unconfirmed] of the total."
    assert removed == ["40%"]
    ok = "Búzios: 1,197,657 bbl/d."
    assert redact(ok, check(ok, sources()), "en") == (ok, [])


def test_redacted_text_passes_the_guardrail():
    answer = "Em 2019 Búzios produzia 600 mil bbl/d; em setembro de 2026, 1.197.657 bbl/d."
    text, removed = redact(answer, check(answer, sources()), "pt")
    assert removed == ["2019", "600 mil bbl/d"]
    assert check(text, sources()).ok
