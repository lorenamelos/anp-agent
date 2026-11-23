"""Provenance sources: the numbers and dates an answer may use for one question.

Built from the tool results of this question, the tool arguments, the question itself, the
server instructions and a few constants (design, section 1.3). Each tool number keeps its
path, a unit inferred from its key, the month (or complete year) of its row, which the
matcher needs for bbl/d -> bbl conversions, and the last month the value covers, which R5
needs for recent months (a month row: its month; a year row: December of that year; anything
under a top-level `latest_month`: that month).

Arguments include the defaults from the tool's input schema, as the server lists it, since
the tool ran with them. One tool-specific rule: the months of the two windows that
top_changes compares are derived from its latest_month and window_months.
"""

import calendar
import json
import re
from dataclasses import dataclass, field

from anp_agent.guardrail.extract import COUNT, DATE, PERCENT, QUANTITY, RANGE, extract

BBL_TO_M3 = 0.158987  # 1 barrel = 0.158987 m³
# Conversion factors and days in a period. Each has its own unit, so a day count never
# grounds "30%" or "31 poços"; claims without a unit still match them ("x 31 dias").
# "1 bbl" is the left side of "1 bbl = 0.158987 m³", and 6.2898 bbl is one m³.
CONSTANTS = [(BBL_TO_M3, "factor"), (1 / BBL_TO_M3, "bbl"), (1, "bbl"),
             *((d, "days") for d in (28, 29, 30, 31, 365, 366))]
MONTH_RE = re.compile(r"^((?:19|20)\d{2})-(0[1-9]|1[0-2])$")
KEY_NUMBER_RE = re.compile(r"(\d+)(pct)?")  # "months_outside_1pct" -> 1 %
CALL_RE = re.compile(r"^call (\d+): ")
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
    until: str | None = None  # YYYY-MM: last month the value covers, for R5
    based_on: tuple[int, ...] = ()  # calculator result: the calls its inputs came from

    @property
    def call(self) -> int | None:
        """1-based index of the call this number came from, if any."""
        return call_of(self.label)


@dataclass
class Sources:
    numbers: list[Source] = field(default_factory=list)
    months: set[tuple[int, int]] = field(default_factory=set)
    years: set[int] = field(default_factory=set)
    truncated: list[int] = field(default_factory=list)  # calls whose result was not JSON

    def add_constants(self) -> None:
        for value, unit in CONSTANTS:
            self.numbers.append(Source(value, unit, f"constant {value:g}", "constant"))

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

    def add_args(self, index: int, args: dict, schema: dict | None = None) -> None:
        """Arguments as the tool ran them: the call's own, then the schema's defaults."""
        defaults = {k: p["default"] for k, p in ((schema or {}).get("properties") or {}).items()
                    if p.get("default") is not None and k not in args}
        for key, value in [*args.items(), *defaults.items()]:
            label = f"call {index}: args.{key}" + (" (default)" if key in defaults else "")
            if isinstance(value, bool):
                continue
            if isinstance(value, int | float):
                self.numbers.append(Source(float(value), None, label, "args"))
            elif isinstance(value, str):
                self._add_month(value)
                self.add_text(value, "args", label)

    def add_top_changes_windows(self, result: dict) -> None:
        """The 2 x window_months months before and including latest_month."""
        latest, window = result.get("latest_month"), result.get("window_months")
        if not (isinstance(latest, str) and MONTH_RE.match(latest) and isinstance(window, int)):
            return
        y, m = int(latest[:4]), int(latest[5:])
        for back in range(2 * window):
            yy, mm = divmod(y * 12 + m - 1 - back, 12)
            self.months.add((yy, mm + 1))
            self.years.add(yy)

    def add_result(self, index: int, result: dict | str) -> dict | None:
        """A tool result, as a parsed dict or as the text the model saw. Returns the dict."""
        if isinstance(result, str):
            try:
                result = json.loads(result)
            except json.JSONDecodeError:
                self.truncated.append(index)
                self._scan(index, result)
                return None
        self._walk(index, result, "", {})
        return result if isinstance(result, dict) else None

    def _walk(self, index: int, node, path: str, row: dict) -> None:
        if isinstance(node, dict):
            latest = node.get("latest_month") if not path else None
            if isinstance(latest, str) and MONTH_RE.match(latest):
                row = {"until": latest}
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
                pos=int(m[-1]) if m else None, until=row.get("until"),
            ))

    def _row_context(self, node: dict) -> dict:
        month = node.get("month")
        if isinstance(month, str) and (m := MONTH_RE.match(month)):
            y, mo = int(m.group(1)), int(m.group(2))
            return {"month": month, "days": calendar.monthrange(y, mo)[1], "until": month}
        year = node.get("year")
        if isinstance(year, int) and not isinstance(year, bool) and 1900 <= year <= 2100:
            days = (366 if calendar.isleap(year) else 365) if node.get("complete_year") else None
            return {"days": days, "until": f"{year}-12"}
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
                    series=f"call {index}: {key}", pos=pos, until=row.get("until"),
                ))


def call_of(label: str) -> int | None:
    m = CALL_RE.match(label)
    return int(m.group(1)) if m else None


def unit_of(key: str) -> str | None:
    if key.endswith("_bpd"):
        return "bbl/d"
    if key.endswith("_pct"):
        return "%"
    if key == "active_wells":
        return "wells"
    return None


def build_sources(question: str, calls: list[dict], instructions: str = "",
                  schemas: dict[str, dict] | None = None) -> Sources:
    """Sources for one question. Each call: {"name", "args", "result", "is_error"};
    `schemas` maps a tool name to its input schema as the server lists it."""
    s = Sources()
    s.add_constants()
    s.add_text(question, "question")
    if instructions:
        s.add_text(instructions, "instructions")
    for i, call in enumerate(calls, start=1):
        s.add_args(i, call.get("args") or {}, (schemas or {}).get(call["name"]))
        if call.get("is_error"):
            continue
        parsed = s.add_result(i, call["result"])
        if call["name"] == "top_changes" and parsed:
            s.add_top_changes_windows(parsed)
    for y, _ in s.months:
        s.years.add(y)
    return s
