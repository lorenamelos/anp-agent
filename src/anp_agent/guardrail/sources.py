"""Provenance sources: the numbers and dates an answer may use for one question.

Built from the tool results of this question, the tool arguments, the question itself, the
server instructions and a few constants (design, section 1.3). Each tool number keeps its
path, a unit inferred from its key, and the month (or complete year) of its row, which the
matcher needs for bbl/d -> bbl conversions and R5 needs for recent months.
"""

import calendar
import json
import re
from dataclasses import dataclass, field

from anp_agent.guardrail.extract import COUNT, DATE, PERCENT, QUANTITY, RANGE, extract

BBL_TO_M3 = 0.158987  # 1 barrel = 0.158987 m³
CONSTANTS = [BBL_TO_M3, 28, 29, 30, 31, 365, 366]  # conversion factor, days in a period
MONTH_RE = re.compile(r"^((?:19|20)\d{2})-(0[1-9]|1[0-2])$")
KEY_NUMBER_RE = re.compile(r"(\d+)(pct)?")  # "months_outside_1pct" -> 1 %
PAIR_RE = re.compile(r'"(\w+)":\s*(?:"((?:[^"\\]|\\.)*)"|(-?\d+(?:\.\d+)?(?:[eE][+-]?\d+)?))')


@dataclass(frozen=True)
class Source:
    value: float  # signed, as in the tool result
    unit: str | None
    label: str  # e.g. "call 2: months[8].petroleum_bpd", "question"
    origin: str  # tool, args, question, instructions, constant
    month: str | None = None  # YYYY-MM of the row
    days: int | None = None  # days of the row's period (month, or complete year)
    series: str | None = None  # same key across the rows of one list, in order
    pos: int | None = None  # position in that series


@dataclass
class Sources:
    numbers: list[Source] = field(default_factory=list)
    months: set[tuple[int, int]] = field(default_factory=set)
    years: set[int] = field(default_factory=set)
    truncated: list[int] = field(default_factory=list)  # calls whose result was not JSON

    def add_constants(self) -> None:
        for c in CONSTANTS:
            self.numbers.append(Source(c, None, f"constant {c}", "constant"))

    def add_text(self, text: str, origin: str, label: str | None = None) -> None:
        """Numbers and dates written in free text (question, instructions, string fields)."""
        label = label or origin
        for c in extract(text):
            if c.kind == DATE:
                self.years.add(c.year)
                if c.month:
                    self.months.add((c.year, c.month))
            elif c.kind in (QUANTITY, PERCENT, COUNT):
                for r in c.readings:
                    value = -r.value if c.direction == -1 else r.value
                    self.numbers.append(Source(value, c.unit, label, origin))
            elif c.kind == RANGE:
                for bound in c.bounds:
                    for r in bound:
                        self.numbers.append(Source(r.value, c.unit, label, origin))

    def add_args(self, index: int, args: dict) -> None:
        for key, value in args.items():
            if isinstance(value, bool):
                continue
            if isinstance(value, int | float):
                self.numbers.append(Source(float(value), None, f"call {index}: args.{key}", "args"))
            elif isinstance(value, str):
                self._add_month(value)
                self.add_text(value, "args", f"call {index}: args.{key}")

    def add_result(self, index: int, result: dict | str) -> None:
        """A tool result, as a parsed dict or as the text the model saw."""
        if isinstance(result, str):
            try:
                result = json.loads(result)
            except json.JSONDecodeError:
                self.truncated.append(index)
                self._scan(index, result)
                return
        self._walk(index, result, "", {})

    def _walk(self, index: int, node, path: str, row: dict) -> None:
        if isinstance(node, dict):
            row = self._row_context(node) or row
            for key, value in node.items():
                self._key_number(index, key, path)
                self._walk(index, value, f"{path}.{key}" if path else key, row)
        elif isinstance(node, list):
            for i, item in enumerate(node):
                self._walk(index, item, f"{path}[{i}]", row)
        elif isinstance(node, str):
            self._add_month(node)
            self.add_text(node, "tool", f"call {index}: {path}")
        elif isinstance(node, int | float) and not isinstance(node, bool):
            key = path.rsplit(".", 1)[-1]
            if key == "year":
                self.years.add(int(node))
                return
            series = re.sub(r"\[\d+\]", "[]", path) if "[" in path else None
            m = re.findall(r"\[(\d+)\]", path)
            self.numbers.append(Source(
                float(node), unit_of(key), f"call {index}: {path}", "tool",
                month=row.get("month"), days=row.get("days"),
                series=f"call {index}: {series}" if series else None,
                pos=int(m[-1]) if m else None,
            ))

    def _row_context(self, node: dict) -> dict:
        month = node.get("month")
        if isinstance(month, str) and (m := MONTH_RE.match(month)):
            y, mo = int(m.group(1)), int(m.group(2))
            return {"month": month, "days": calendar.monthrange(y, mo)[1]}
        year = node.get("year")
        if isinstance(year, int) and node.get("complete_year") is True:
            return {"days": 366 if calendar.isleap(year) else 365}
        return {}

    def _key_number(self, index: int, key: str, path: str) -> None:
        for digits, pct in KEY_NUMBER_RE.findall(key):
            self.numbers.append(Source(float(digits), "%" if pct else None,
                                       f"call {index}: key {key}", "tool"))

    def _add_month(self, text: str) -> None:
        if m := MONTH_RE.match(text):
            self.years.add(int(m.group(1)))
            self.months.add((int(m.group(1)), int(m.group(2))))

    def _scan(self, index: int, text: str) -> None:
        """Fallback for a result cut mid-JSON: read "key": value pairs in order."""
        row: dict = {}
        positions: dict[str, int] = {}
        for key, string, number in PAIR_RE.findall(text):
            self._key_number(index, key, "")
            if string:
                if key == "month":
                    row = self._row_context({"month": string})
                self._add_month(string)
                self.add_text(string, "tool", f"call {index}: {key}")
            elif number:
                if key == "year":
                    self.years.add(int(float(number)))
                    continue
                pos = positions.get(key, 0)
                positions[key] = pos + 1
                self.numbers.append(Source(
                    float(number), unit_of(key), f"call {index}: {key}#{pos} (truncated)", "tool",
                    month=row.get("month"), days=row.get("days"),
                    series=f"call {index}: {key}", pos=pos,
                ))


def unit_of(key: str) -> str | None:
    if key.endswith("_bpd"):
        return "bbl/d"
    if key.endswith("_pct"):
        return "%"
    if key == "active_wells":
        return "wells"
    return None


def build_sources(question: str, calls: list[dict], instructions: str = "") -> Sources:
    """Sources for one question. Each call: {"name", "args", "result", "is_error"}."""
    s = Sources()
    s.add_constants()
    s.add_text(question, "question")
    if instructions:
        s.add_text(instructions, "instructions")
    for i, call in enumerate(calls, start=1):
        s.add_args(i, call.get("args") or {})
        if not call.get("is_error"):
            s.add_result(i, call["result"])
    for y, _ in s.months:
        s.years.add(y)
    return s
