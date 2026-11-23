"""Match the claims of an answer against its provenance sources (design, section 1.3).

A number is grounded when a source, possibly through a whitelisted unit conversion, rounds
or truncates to it at its displayed precision; hedge words widen that to 2%. A range is
grounded by both endpoints, or by containing a source when it is at most 10% wide. Dates
must appear in the sources; small bare counts are exempt.

"barris" without "por dia" is accepted against bbl/d sources: answers often drop "/d", and
whether the unit is right is the grader's question, not provenance.

`Policy` sets how loose the displayed precision is. By default the last digit actually written
sets it ("40.000" is exact to the unit; "cerca de 40.000" gets the 2% hedge). Optionally,
trailing zeros count as rounding ("40.000" -> within 5,000), with an optional cap relative to
the value that never goes below half of the last digit written. The replay measured both: the
default catches one more wrong number with no more false alarms (design, section 4.4).
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
class Policy:
    trailing_zeros: bool = False  # True: "9.000" is rounded to the thousand
    max_rel: float | None = None  # cap on the rounding tolerance, relative to the value


DEFAULT = Policy()


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


def close(reading: Reading, source: float, hedged: bool, policy: Policy = DEFAULT) -> bool:
    """Does `source` display as `reading` (rounded or truncated), or within 2% if hedged?"""
    v, s = reading.value, abs(source)
    written = reading.literal or reading.place  # last digit actually written
    p = reading.place if policy.trailing_zeros else written
    half, window = p / 2, p
    if policy.max_rel is not None:
        cap = policy.max_rel * v
        half, window = max(min(half, cap), written / 2), max(min(window, cap), written)
    eps = 1e-9 * max(1.0, v)
    if abs(s - v) <= half + eps or 0 <= s - v < window - eps:
        return True
    return hedged and abs(s - v) <= HEDGE_TOLERANCE * s + eps


def find(reading: Reading, unit: str | None, hedged: bool, direction: int | None,
         cands: list[Candidate], policy: Policy = DEFAULT) -> Candidate | None:
    for c in cands:
        if not units_ok(unit, c.unit) or not close(reading, c.value, hedged, policy):
            continue
        if direction and c.value and (c.value > 0) != (direction > 0):
            continue  # a fall stated as a rise, or the reverse
        return c
    return None


def check(answer: str, sources: Sources, derive: Derive | None = None,
          policy: Policy = DEFAULT) -> Report:
    cands = candidates(sources)
    verdicts = [_verdict(c, sources, cands, derive, policy) for c in extract(answer)]
    return Report(verdicts, truncated=bool(sources.truncated))


def ground_number(readings, unit, hedged, direction, cands, derive=None,
                  policy: Policy = DEFAULT) -> str | None:
    """Label of the first source (or derivation) that grounds any reading, else None."""
    for r in readings:
        if c := find(r, unit, hedged, direction, cands, policy):
            return c.label
    if derive:
        for r in readings:
            if label := derive(r, unit, hedged):
                return label
    return None


def _verdict(claim: Claim, sources: Sources, cands, derive, policy) -> Verdict:
    if claim.kind == COUNT:
        return Verdict(claim, True, "exempt")
    if claim.kind == DATE:
        ok = ((claim.year, claim.month) in sources.months if claim.month
              else claim.year in sources.years)
        return Verdict(claim, ok, "date" if ok else None)
    if claim.kind == RANGE:
        return _range(claim, cands, derive, policy)
    label = ground_number(claim.readings, claim.unit, claim.hedged, claim.direction, cands,
                          derive, policy)
    return Verdict(claim, label is not None, label)


def _range(claim: Claim, cands, derive, policy) -> Verdict:
    low, high = claim.bounds
    a = ground_number(low, claim.unit, claim.hedged, None, cands, derive, policy)
    b = ground_number(high, claim.unit, claim.hedged, None, cands, derive, policy)
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
