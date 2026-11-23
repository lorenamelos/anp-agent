"""Match the claims of an answer against its provenance sources (design, section 1.3).

A number is grounded when a source, possibly through a whitelisted unit conversion, rounds
or truncates to it at its displayed precision; hedge words widen that to 2%. A range is
grounded by both endpoints, or by containing a source when it is at most 10% wide. Dates
must appear in the sources; small bare counts are exempt.

"barris" without "por dia" is accepted against bbl/d sources: answers often drop "/d", and
whether the unit is right is the grader's question, not provenance.
"""

from collections.abc import Callable
from dataclasses import dataclass, field

from anp_agent.guardrail.extract import COUNT, DATE, RANGE, Claim, Reading, extract
from anp_agent.guardrail.sources import BBL_TO_M3, Sources

HEDGE_TOLERANCE = 0.02
RANGE_MAX_WIDTH = 0.10
COMPATIBLE = {("bbl", "bbl/d")}

# Optional extra grounding: (reading, unit, hedged) -> label of a derivation, or None.
Derive = Callable[[Reading, str | None, bool], str | None]


@dataclass(frozen=True)
class Candidate:
    value: float
    unit: str | None
    label: str


@dataclass
class Verdict:
    claim: Claim
    grounded: bool
    source: str | None  # label of the source, "exempt", or None


@dataclass
class Report:
    verdicts: list[Verdict]
    truncated: bool = False  # some tool result was not valid JSON (cut)
    ungrounded: list[Verdict] = field(init=False)

    def __post_init__(self):
        self.ungrounded = [v for v in self.verdicts if not v.grounded]

    @property
    def ok(self) -> bool:
        return not self.ungrounded


def candidates(sources: Sources) -> list[Candidate]:
    """Every source value, plus its whitelisted conversions."""
    out = []
    for s in sources.numbers:
        out.append(Candidate(s.value, s.unit, s.label))
        if s.unit == "bbl/d":
            out.append(Candidate(s.value * BBL_TO_M3, "m3/d", f"{s.label} x {BBL_TO_M3} m3/bbl"))
            if s.days:
                out.append(Candidate(s.value * s.days, "bbl", f"{s.label} x {s.days} days"))
    return out


def units_ok(claim_unit: str | None, source_unit: str | None) -> bool:
    if claim_unit is None or source_unit is None or claim_unit == source_unit:
        return True
    return (claim_unit, source_unit) in COMPATIBLE


def close(reading: Reading, source: float, hedged: bool) -> bool:
    """Does `source` display as `reading` (rounded or truncated), or within 2% if hedged?"""
    v, s, p = reading.value, abs(source), reading.place
    eps = 1e-9 * max(1.0, v)
    if abs(s - v) <= p / 2 + eps or 0 <= s - v < p - eps:
        return True
    return hedged and abs(s - v) <= HEDGE_TOLERANCE * s + eps


def find(reading: Reading, unit: str | None, hedged: bool, direction: int | None,
         cands: list[Candidate]) -> Candidate | None:
    for c in cands:
        if not units_ok(unit, c.unit) or not close(reading, c.value, hedged):
            continue
        if direction and c.value and (c.value > 0) != (direction > 0):
            continue  # a fall stated as a rise, or the reverse
        return c
    return None


def check(answer: str, sources: Sources, derive: Derive | None = None) -> Report:
    cands = candidates(sources)
    verdicts = [_verdict(c, sources, cands, derive) for c in extract(answer)]
    return Report(verdicts, truncated=bool(sources.truncated))


def ground_number(readings, unit, hedged, direction, cands, derive=None) -> str | None:
    """Label of the first source (or derivation) that grounds any reading, else None."""
    for r in readings:
        if c := find(r, unit, hedged, direction, cands):
            return c.label
    if derive:
        for r in readings:
            if label := derive(r, unit, hedged):
                return label
    return None


def _verdict(claim: Claim, sources: Sources, cands, derive) -> Verdict:
    if claim.kind == COUNT:
        return Verdict(claim, True, "exempt")
    if claim.kind == DATE:
        ok = ((claim.year, claim.month) in sources.months if claim.month
              else claim.year in sources.years)
        return Verdict(claim, ok, "date" if ok else None)
    if claim.kind == RANGE:
        return _range(claim, cands, derive)
    label = ground_number(claim.readings, claim.unit, claim.hedged, claim.direction, cands, derive)
    return Verdict(claim, label is not None, label)


def _range(claim: Claim, cands, derive) -> Verdict:
    low, high = claim.bounds
    a = ground_number(low, claim.unit, claim.hedged, None, cands, derive)
    b = ground_number(high, claim.unit, claim.hedged, None, cands, derive)
    if a and b:
        return Verdict(claim, True, f"{a} .. {b}")
    for lo in low:
        for hi in high:
            mid = (lo.value + hi.value) / 2
            if mid <= 0 or (hi.value - lo.value) / mid > RANGE_MAX_WIDTH:
                continue
            for c in cands:
                if units_ok(claim.unit, c.unit) and lo.value <= abs(c.value) <= hi.value:
                    return Verdict(claim, True, f"contains {c.label}")
    return Verdict(claim, False, None)
