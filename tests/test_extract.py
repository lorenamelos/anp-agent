import pytest

from anp_agent.guardrail.extract import (
    COUNT,
    DATE,
    PERCENT,
    QUANTITY,
    RANGE,
    Reading,
    extract,
    readings,
)


def values(claim):
    return sorted(r.value for r in claim.readings)


def only(text, kind=None):
    claims = [c for c in extract(text) if kind is None or c.kind == kind]
    assert len(claims) == 1, claims
    return claims[0]


@pytest.mark.parametrize("raw, expected", [
    ("1.197.657", [Reading(1197657, 1)]),
    ("1,197,657", [Reading(1197657, 1)]),
    ("0,517", [Reading(0.517, 0.001)]),
    ("26.3", [Reading(26.3, 0.1)]),
    ("1.749.888,5", [Reading(1749888.5, 0.1)]),
    ("1,749,888.5", [Reading(1749888.5, 0.1)]),
    ("37", [Reading(37, 1)]),
    ("9.000", [Reading(9000, 1000), Reading(9.0, 0.001)]),  # ambiguous: both readings
    ("1.056", [Reading(1056, 1), Reading(1.056, 0.001)]),
    ("1 197 657", [Reading(1197657, 1)]),
    ("1.2.3", []),  # not a number
])
def test_readings(raw, expected):
    got = readings(raw)
    assert [r.value for r in got] == pytest.approx([r.value for r in expected])
    assert [r.place for r in got] == pytest.approx([r.place for r in expected])


@pytest.mark.parametrize("text, value, place, unit", [
    ("cerca de 989 mil bbl/d", 989_000, 1_000, "bbl/d"),
    ("1,2 milhão de barris por dia", 1_200_000, 100_000, "bbl/d"),
    ("1,06 milhão de bbl/d", 1_060_000, 10_000, "bbl/d"),
    ("300 mil barris", 300_000, 100_000, "bbl"),
    ("190 mil m³/d", 190_000, 10_000, "m3/d"),  # trailing zero: rounded
    ("about 1.2 million barrels per day", 1_200_000, 100_000, "bbl/d"),
    ("37 poços", 37, 1, "wells"),
    ("27,4 mil b/d", 27_400, 100, "bbl/d"),
])
def test_quantities_with_scale_and_unit(text, value, place, unit):
    c = only(text)
    assert c.kind == QUANTITY and c.unit == unit
    assert c.readings[0].value == pytest.approx(value)
    assert c.readings[0].place == pytest.approx(place)


def test_percent_sign_and_direction():
    assert only("variação de −26,3%").direction == -1
    assert only("+209,8%").direction == 1
    assert only("uma queda de 26,3%").direction == -1
    assert only("Bacalhau cresceu 33,2%").direction == 1
    assert only("decline of 4.1%").direction == -1
    c = only("diferença de 0,517%")
    assert c.kind == PERCENT and c.unit == "%" and c.direction is None
    assert only("2,3 p.p.").unit == "pp"


def test_hedges():
    assert only("cerca de 268 mil bbl/d").hedged
    assert only("Búzios ~37% acima").hedged
    assert only("| ≈ 989.427 |").hedged
    assert not only("produziu 1.197.657 bbl/d").hedged


@pytest.mark.parametrize("text, year, month", [
    ("em 2026-09", 2026, 9),
    ("setembro de 2026", 2026, 9),
    ("jan/2005", 2005, 1),
    ("March 2008", 2008, 3),
    ("09/2026", 2026, 9),
    ("o ano de 2024", 2024, None),
])
def test_dates(text, year, month):
    c = only(text)
    assert (c.kind, c.year, c.month) == (DATE, year, month)


def test_date_spans_are_two_dates_not_a_range():
    claims = extract("de 2019-10..2026-09 e 2023–2025")
    assert [(c.kind, c.year) for c in claims] == [
        (DATE, 2019), (DATE, 2026), (DATE, 2023), (DATE, 2025)]


def test_markdown_table_and_bold():
    text = "| Junho | 1.028.843 | 761.700 |\n**Búzios produziu 1.197.657 bbl/d**"
    assert [values(c) for c in extract(text) if c.kind == QUANTITY] == [
        [1028843], [761.7, 761700], [1197657]]  # 761.700 is ambiguous


def test_skipped_tokens():
    text = "1. Búzios\n2) Mero\nP-70 e BÚZIOS_ECO, o 1º e o 2nd, versão v2026.09, q13"
    assert extract(text) == []


def test_small_bare_integers_are_counts():
    c = only("os 5 campos")
    assert c.kind == COUNT
    assert only("5 poços").kind == QUANTITY  # a unit makes it a quantity


@pytest.mark.parametrize("text, low, high, unit", [
    ("entre 1 e 1,1 milhão de bbl/d", 1_000_000, 1_100_000, "bbl/d"),
    ("between 900 and 950 thousand barrels per day", 900_000, 950_000, "bbl/d"),
    ("de 800 mil a 900 mil bbl/d", 800_000, 900_000, "bbl/d"),
    ("800–900 mil bbl/d", 800_000, 900_000, "bbl/d"),
    ("1,2 milhão ± 50 mil bbl/d", 1_150_000, 1_250_000, "bbl/d"),
    ("entre 10% e 12%", 10, 12, "%"),
])
def test_ranges(text, low, high, unit):
    c = only(text)
    assert c.kind == RANGE and c.unit == unit
    assert c.bounds[0][0].value == pytest.approx(low)
    assert c.bounds[1][0].value == pytest.approx(high)


def test_not_ranges():
    # a list joined by "e", and "de X para Y", are separate numbers
    assert [c.kind for c in extract("Búzios 37 poços e Mero 22 poços")] == [QUANTITY, QUANTITY]
    assert [c.kind for c in extract("caiu de 120 mil para 90 mil bbl/d")] == [QUANTITY, QUANTITY]


def test_offsets_point_into_the_original_answer():
    answer = "Em **setembro de 2026**, Búzios: **1.197.657 bbl/d**."
    for c in extract(answer):
        assert answer[c.start:c.end] == c.text
