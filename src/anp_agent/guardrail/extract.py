"""Find the numbers an answer states, as claims to be checked against the tool results.

Pure functions, no I/O. Handles Portuguese and English formats ("1.197.657" and "1,197,657",
"0,517%" and "0.517%"), scale words ("989 mil", "1,2 milhão"), units, dates, ranges and
the words that change how a number is checked (hedges such as "cerca de", directions such
as "queda de").

Precision: each reading carries `place`, the value of the last displayed digit after
scaling ("1,06 milhão" -> 10,000; "26,3%" -> 0.1). An integer of 1,000 or more read without
decimals takes its trailing zeros as rounding ("9.000" -> 1,000; "300 mil" -> 100,000), and
keeps the last digit actually written in `literal`; the matcher's policy picks one of the two.
"""

import re
from dataclasses import dataclass

QUANTITY, PERCENT, RANGE, DATE, COUNT = "quantity", "percent", "range", "date", "count"
YEARS = range(1990, 2101)  # a bare 4-digit integer in this range is a year


@dataclass(frozen=True)
class Reading:
    value: float  # absolute value after scaling; the sign is in Claim.direction
    place: float  # value of the last displayed digit, after scaling
    literal: float | None = None  # last digit actually written, when trailing zeros count


@dataclass
class Claim:
    text: str
    start: int
    end: int
    kind: str
    readings: tuple[Reading, ...] = ()  # quantity, percent, count
    unit: str | None = None  # bbl/d, m3/d, bbl, %, pp, wells
    hedged: bool = False
    direction: int | None = None  # +1 rise, -1 fall, None not stated
    bounds: tuple[tuple[Reading, ...], tuple[Reading, ...]] | None = None  # range
    year: int | None = None  # date
    month: int | None = None  # date, when stated
    raw: str = ""  # digits as written (internal, for ranges)
    scale: float = 1.0  # scale word factor (internal, for ranges)


SCALES = {
    "mil": 1e3, "k": 1e3, "thousand": 1e3,
    "milhão": 1e6, "milhao": 1e6, "milhões": 1e6, "milhoes": 1e6, "mi": 1e6,
    "million": 1e6, "millions": 1e6, "m": 1e6, "mn": 1e6,
    "bilhão": 1e9, "bilhao": 1e9, "bilhões": 1e9, "bilhoes": 1e9, "bi": 1e9,
    "billion": 1e9, "bn": 1e9,
}
UNITS = [  # (pattern, canonical unit), tried in this order
    (r"%|por\s+cento\b|percent\b", "%"),
    (r"p\.\s?p\.|pp\b|pontos\s+percentuais|percentage\s+points?", "pp"),
    ((r"(?:bbl|barris|barrels)\s*(?:/|por|per)\s*(?:d\b|dia\b|day\b)|b/d\b|bpd\b"
      r"|barris\s+diários"), "bbl/d"),
    ((r"(?:m³|m3)\s*/\s*(?:d\b|dia\b|day\b)|metros\s+cúbicos\s+por\s+dia"
      r"|cubic\s+met(?:er|re)s\s+per\s+day"), "m3/d"),
    (r"barris\b|barril\b|barrels?\b|bbl\b", "bbl"),
    (r"poços\b|(?:active\s+|producing\s+)?wells\b", "wells"),
]
NUM_RE = re.compile(r"(?<![\w.,/])([+\-−]?)(\d+(?:[.,  ]\d+)*)(?!\w)")
SCALE_RE = re.compile(
    r"\s*(milh(?:ão|ao|ões|oes)|bilh(?:ão|ao|ões|oes)|mil|thousand|millions?|billion|"
    r"mi|bi|bn|mn|k|M)\b"
)
UNIT_RE = re.compile(
    r"\s*(?:de\s+|of\s+)?(?:" + "|".join(f"({p})" for p, _ in UNITS) + ")", re.IGNORECASE
)
HEDGE_RE = re.compile(
    r"(?:cerca\s+de|aproximadamente|aprox\.|quase|em\s+torno\s+de|por\s+volta\s+de|"
    r"perto\s+de|about|around|approximately|approx\.|roughly|nearly|almost|~|≈)\s*$",
    re.IGNORECASE,
)
FALL = (r"queda|redução|reducao|recuo|declínio|declinio|retração|retracao|perda|"
        r"drop|decline|decrease|fall|loss")
RISE = r"alta|aumento|crescimento|expansão|expansao|ganho|increase|growth|rise|gain"
DIRECTION_RES = [
    re.compile(rf"\b(?:({FALL})|({RISE}))\s+(?:de|of)\s*$", re.IGNORECASE),
    re.compile(
        r"\b(?:(caiu|recuou|diminuiu|encolheu|fell|dropped|declined|decreased)|"
        r"(cresceu|subiu|aumentou|avançou|avancou|rose|grew|increased|climbed))\s*$",
        re.IGNORECASE,
    ),
]
MONTHS = {
    "janeiro": 1, "fevereiro": 2, "março": 3, "marco": 3, "abril": 4, "maio": 5, "junho": 6,
    "julho": 7, "agosto": 8, "setembro": 9, "outubro": 10, "novembro": 11, "dezembro": 12,
    "january": 1, "february": 2, "march": 3, "april": 4, "may": 5, "june": 6, "july": 7,
    "august": 8, "september": 9, "october": 10, "november": 11, "december": 12,
    "jan": 1, "fev": 2, "feb": 2, "mar": 3, "abr": 4, "apr": 4, "mai": 5, "jun": 6, "jul": 7,
    "ago": 8, "aug": 8, "set": 9, "sep": 9, "sept": 9, "out": 10, "oct": 10, "nov": 11,
    "dez": 12, "dec": 12,
}
MONTH_NAME = "|".join(sorted(MONTHS, key=len, reverse=True))
DATE_RES = [  # (pattern, group with the year, group with the month number or name)
    (re.compile(r"(?<!\w)((?:19|20)\d{2})-(0[1-9]|1[0-2])(?!\w)"), 1, 2),
    (re.compile(r"(?<![\w.,/])(0?[1-9]|1[0-2])/((?:19|20)\d{2})(?!\w)"), 2, 1),
    (re.compile(rf"\b({MONTH_NAME})\.?(?:\s+de\s+|\s*/\s*|\s+|,\s*)((?:19|20)\d{{2}})(?!\w)",
                re.IGNORECASE), 2, 1),
]
RANGE_OPEN_RE = re.compile(  # "entre X e Y", also "entre cerca de X e Y"
    r"\b(entre|between|de|from)\s+(?:cerca\s+de\s+|aproximadamente\s+|about\s+|around\s+"
    r"|approximately\s+|roughly\s+)?$",
    re.IGNORECASE,
)
RANGE_JOINS = {"entre": {"e"}, "between": {"and"}, "de": {"a", "até", "ate"}, "from": {"to"}}
BARE_JOINS = {"–", "to", "até", "ate"}  # "X–Y", "X to Y"; "X a Y" only with a unit after Y


def readings(raw: str, scale: float = 1.0) -> tuple[Reading, ...]:
    """All plausible values of a digit string; an ambiguous separator gives two readings."""
    s = raw.replace(" ", " ").replace(" ", " ")
    if " " in s:  # thin or non-breaking spaces only separate thousands
        return _int_reading(s.replace(" ", ""), scale)
    seps = [c for c in s if c in ".,"]
    if not seps:
        return _int_reading(s, scale)
    if len(set(seps)) == 2:  # "1,749,888.5" / "1.749.888,5": the last one is decimal
        dec = s[max(s.rfind("."), s.rfind(","))]
        return (_dec_reading(s, dec, scale),)
    groups = s.split(seps[0])
    if len(seps) > 1:  # "1.197.657": thousands, if every later group has 3 digits
        return _int_reading("".join(groups), scale) if all(
            len(g) == 3 for g in groups[1:]) else ()
    head, tail = groups
    if len(tail) == 3 and head != "0":  # "1.056" is 1056 or 1.056
        return _int_reading(head + tail, scale) + (_dec_reading(s, seps[0], scale),)
    return (_dec_reading(s, seps[0], scale),)


def _int_reading(digits: str, scale: float) -> tuple[Reading, ...]:
    value = int(digits) * scale
    place = scale
    if value >= 1000:
        zeros = len(digits) - len(digits.rstrip("0"))
        place = scale * 10 ** min(zeros, len(digits) - 1)
    return (Reading(value, place, scale if place != scale else None),)


def _dec_reading(s: str, dec: str, scale: float) -> Reading:
    other = "," if dec == "." else "."
    whole, frac = s.replace(other, "").split(dec)
    return Reading(float(f"{whole}.{frac}") * scale, 10.0 ** -len(frac) * scale)


def _direction(before: str, sign: str) -> int | None:
    if sign == "+":
        return 1
    if sign in ("-", "−"):
        return -1
    for pattern in DIRECTION_RES:
        m = pattern.search(before)
        if m:
            return -1 if m.group(1) else 1
    return None


def extract(answer: str) -> list[Claim]:
    """Claims in reading order. List markers and ordinals are skipped."""
    text = re.sub(r"[*`]", " ", answer)  # markdown emphasis; same length, so offsets hold
    dates: list[Claim] = []
    for pattern, gy, gm in DATE_RES:
        for m in pattern.finditer(text):
            month = m.group(gm)
            month = int(month) if month.isdigit() else MONTHS[month.lower()]
            dates.append(Claim(answer[m.start():m.end()], m.start(), m.end(), DATE,
                               year=int(m.group(gy)), month=month))
            text = text[:m.start()] + " " * (m.end() - m.start()) + text[m.end():]

    numbers = [c for m in NUM_RE.finditer(text) if (c := _number(answer, text, m))]
    return sorted(dates + _join_ranges(answer, text, numbers), key=lambda c: c.start)


def _number(answer: str, text: str, m: re.Match) -> Claim | None:
    sign, raw = m.group(1), m.group(2)
    start, end = m.start(), m.end()
    if not sign and start >= 2 and text[start - 1] == "-" and text[start - 2].isalpha():
        return None  # identifier such as P-70
    line_start = text.rfind("\n", 0, start) + 1
    if (not text[line_start:start].strip() and re.match(r"[.)]\s", text[end:end + 2])
            and raw.isdigit() and int(raw) < 100):
        return None  # list marker "1. " or "2) "
    scale = 1.0
    if sm := SCALE_RE.match(text, end):
        scale, end = SCALES[sm.group(1).lower()], sm.end()
    unit = None
    if um := UNIT_RE.match(text, end):
        unit = next(u for i, (_, u) in enumerate(UNITS) if um.group(i + 1))
        end = um.end()
    reads = readings(raw, scale)
    if not reads:
        return None
    bare = not sign and scale == 1.0 and unit is None and raw.isdigit()
    if bare and len(raw) == 4 and int(raw) in YEARS:
        return Claim(answer[start:end], start, end, DATE, year=int(raw), raw=raw)
    kind = PERCENT if unit in ("%", "pp") else QUANTITY
    if bare and int(raw) <= 12:
        kind = COUNT
    before = text[max(0, start - 30):start]
    return Claim(answer[start:end], start, end, kind, readings=reads, unit=unit,
                 hedged=bool(HEDGE_RE.search(before)), direction=_direction(before, sign),
                 raw=raw, scale=scale)


def _join_ranges(answer: str, text: str, numbers: list[Claim]) -> list[Claim]:
    """Join "entre X e Y", "de X a Y", "X–Y" and "X ± Y" into range claims."""
    out: list[Claim] = []
    i = 0
    while i < len(numbers):
        a, b = numbers[i], numbers[i + 1] if i + 1 < len(numbers) else None
        joined = _range(answer, text, a, b) if b else None
        out.append(joined or a)
        i += 2 if joined else 1
    return out


def _range(answer: str, text: str, a: Claim, b: Claim) -> Claim | None:
    numeric = (QUANTITY, PERCENT, COUNT)
    if a.kind not in numeric or b.kind not in numeric:
        return None
    if a.unit and b.unit and a.unit != b.unit:
        return None
    between = text[a.end:b.start].strip().lower()
    unit = b.unit or a.unit
    if between == "±":
        pairs = [(x, y) for x in a.readings for y in b.readings]
        return Claim(answer[a.start:b.end], a.start, b.end, RANGE, unit=unit,
                     hedged=a.hedged, bounds=(
                         tuple(Reading(x.value - y.value, max(x.place, y.place)) for x, y in pairs),
                         tuple(Reading(x.value + y.value, max(x.place, y.place)) for x, y in pairs)))
    window = max(0, a.start - 30)
    opener = RANGE_OPEN_RE.search(text[window:a.start])
    start = a.start
    if opener and between in RANGE_JOINS[opener.group(1).lower()]:
        start = window + opener.start()
    elif not (between in BARE_JOINS or (between == "a" and b.unit)):
        return None
    low = a.readings
    if a.scale == 1.0 and b.scale != 1.0:  # "entre 1 e 1,1 milhão": the scale applies to both
        low = readings(a.raw, b.scale)
    hedged = bool(HEDGE_RE.search(text[max(0, start - 30):start])) or a.hedged
    return Claim(answer[start:b.end], start, b.end, RANGE, unit=unit, hedged=hedged,
                 bounds=(low, b.readings))
