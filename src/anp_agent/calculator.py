"""The local `calculate` tool (design, section 1.4).

Deterministic arithmetic the model calls instead of computing in its head. Every input must
already be grounded in this question's sources (tool results, arguments, question,
instructions, constants, or earlier calculator results), so the calculator cannot turn a
remembered number into a grounded one. Inputs are matched at the digit the model wrote,
without taking trailing zeros as rounding and without hedge tolerance.

The cap on calls is enforced by the graph, not here.
"""

import calendar
import re

from anp_agent.guardrail.extract import Reading
from anp_agent.guardrail.match import Candidate, Policy, candidates, find
from anp_agent.guardrail.sources import BBL_TO_M3, Source, Sources

NAME = "calculate"
OPERATIONS = ("sum", "mean", "difference", "ratio", "percent_change", "multiply", "convert",
              "annual_volume")
MONTH_RE = re.compile(r"((?:19|20)\d{2})-(0[1-9]|1[0-2])")
UNITS = ("bbl/d", "m3/d", "bbl", "%", "wells")
INPUT_POLICY = Policy(trailing_zeros=False)

SCHEMA = {
    "name": NAME,
    "description": (
        "Exact arithmetic on numbers from this question's tool results. Use it for every "
        "number you derive (averages, totals, differences, shares, changes, unit "
        "conversions) instead of computing it yourself. Each value must be copied from a "
        "tool result (or an earlier calculate result); other values are rejected.\n"
        "- sum, mean: one or more values.\n"
        "- difference: [a, b] -> b - a.\n"
        "- ratio: [part, whole] -> 100 * part / whole, in %.\n"
        "- percent_change: [before, after] -> 100 * (after / before - 1), in %.\n"
        "- multiply: [a, b] -> a * b; with percent=true, a is a percentage (a% of b).\n"
        "- convert: [value in bbl/d] -> to_unit 'm3/d', or 'bbl' in a month (month='YYYY-MM') "
        "or in a year (year=YYYY).\n"
        "- annual_volume: the monthly bbl/d values of one year, with months=['YYYY-MM', ...] "
        "in the same order -> the year's volume in barrels (each value times the days of its "
        "month, summed). Each value must be the one the tool returned for that month. Use it "
        "for a yearly total in barrels, in one call."
    ),
    "input_schema": {
        "type": "object",
        "properties": {
            "operation": {"type": "string", "enum": list(OPERATIONS)},
            "values": {"type": "array", "items": {"type": "number"}, "minItems": 1},
            "unit": {"type": "string", "enum": list(UNITS),
                     "description": "Unit of the values (of b, for multiply)."},
            "to_unit": {"type": "string", "enum": ["m3/d", "bbl"]},
            "month": {"type": "string", "description": "YYYY-MM, for convert to bbl."},
            "months": {"type": "array", "items": {"type": "string"},
                       "description": "annual_volume: the month (YYYY-MM) of each value."},
            "year": {"type": "integer", "description": "For convert to bbl in a year."},
            "percent": {"type": "boolean", "description": "multiply: a is a percentage."},
        },
        "required": ["operation", "values"],
    },
}


class CalcError(ValueError):
    """Bad arguments or an ungrounded input; the message goes back to the model."""


def calculate(args: dict, sources: Sources) -> dict:
    """Run one calculation. Raises CalcError; never touches `sources`."""
    op = args.get("operation")
    if op not in OPERATIONS:
        raise CalcError(f"unknown operation {op!r}; use one of {', '.join(OPERATIONS)}")
    values = args.get("values")
    if (not isinstance(values, list) or not values
            or not all(isinstance(v, int | float) and not isinstance(v, bool) for v in values)):
        raise CalcError("values must be a non-empty list of numbers")
    unit = args.get("unit")
    if unit is not None and unit not in UNITS:
        raise CalcError(f"unknown unit {unit!r}; use one of {', '.join(UNITS)}")
    percent = bool(args.get("percent"))

    arity = {"difference": 2, "ratio": 2, "percent_change": 2, "multiply": 2, "convert": 1}
    if op in arity and len(values) != arity[op]:
        raise CalcError(f"{op} takes {arity[op]} value(s), got {len(values)}")
    cands = candidates(sources)
    if op == "annual_volume":
        months = _annual_months(args, values)
        for value, month in zip(values, months, strict=True):
            rows = [c for c in cands if c.month == month]
            if _input(value, "bbl/d", rows) is None:
                raise CalcError(
                    f"{_plain(value)} is not the {month} value in this question's tool results; "
                    "copy each month's bbl/d exactly as the tool returned it")
        result = sum(v * calendar.monthrange(int(m[:4]), int(m[5:]))[1]
                     for v, m in zip(values, months, strict=True))
        return {"operation": op, "inputs": values, "result": round(result, 6), "unit": "bbl"}
    units = _input_units(args)
    for value, u in zip(values, units, strict=True):
        if _input(value, u, cands) is None:
            raise CalcError(
                f"{_plain(value)} is not in this question's tool results; call a data tool first "
                "or copy the value exactly as the tool returned it")

    result, out_unit = _compute(op, values, unit, args, percent)
    return {"operation": op, "inputs": values, "result": round(result, 6), "unit": out_unit}


def _plain(value: float) -> str:
    """1200000.0 -> '1200000', 0.517 -> '0.517' (no scientific notation)."""
    return f"{value:.10f}".rstrip("0").rstrip(".")


def _annual_months(args: dict, values: list[float]) -> list[str]:
    """The months of an annual_volume call: one per value, distinct, all in one year."""
    months = args.get("months")
    if (not isinstance(months, list) or len(months) != len(values)
            or not all(isinstance(m, str) and MONTH_RE.fullmatch(m) for m in months)):
        raise CalcError("annual_volume needs months=['YYYY-MM', ...], one per value")
    if len(set(months)) != len(months) or len({m[:4] for m in months}) != 1:
        raise CalcError("annual_volume takes distinct months of a single year")
    return months


def _input_units(args: dict) -> list[str | None]:
    if args["operation"] == "annual_volume":
        return ["bbl/d"] * len(args["values"])
    units = [args.get("unit")] * len(args["values"])
    if args["operation"] == "multiply" and args.get("percent"):
        units[0] = "%"
    if args["operation"] == "convert":
        units[0] = "bbl/d"
    return units


def _input(value: float, unit: str | None, cands) -> Candidate | None:
    """The source of the value, at the last digit the model wrote, or None."""
    text = repr(abs(float(value)))
    if "e" in text:  # 1e-05 or 1e+16: no written digit to round at; compare exactly
        place = 0.0
    else:
        frac = text.split(".")[1]
        place = 1.0 if frac == "0" else 10.0 ** -len(frac)
    direction = -1 if value < 0 else None
    return find(Reading(abs(value), place), unit, False, direction, cands, INPUT_POLICY)


def _compute(op: str, v: list[float], unit: str | None, args: dict, percent: bool):
    if op == "sum":
        return sum(v), unit
    if op == "mean":
        return sum(v) / len(v), unit
    if op == "difference":
        return v[1] - v[0], "pp" if unit == "%" else unit
    if op in ("ratio", "percent_change"):
        if v[1 if op == "ratio" else 0] == 0:
            raise CalcError("division by zero")
        if op == "ratio":
            return 100 * v[0] / v[1], "%"
        return 100 * (v[1] / v[0] - 1), "%"
    if op == "multiply":
        return (v[0] / 100 if percent else v[0]) * v[1], unit
    return _convert(v[0], args)


def _convert(value: float, args: dict):
    to_unit, month, year = args.get("to_unit"), args.get("month"), args.get("year")
    if to_unit == "m3/d":
        return value * BBL_TO_M3, "m3/d"
    if to_unit != "bbl":
        raise CalcError("convert needs to_unit 'm3/d' or 'bbl'")
    if month is not None:
        m = MONTH_RE.fullmatch(str(month))
        if not m:
            raise CalcError("month must be YYYY-MM")
        return value * calendar.monthrange(int(m.group(1)), int(m.group(2)))[1], "bbl"
    if isinstance(year, int) and not isinstance(year, bool) and 1900 <= year <= 2100:
        return value * (366 if calendar.isleap(year) else 365), "bbl"
    raise CalcError("convert to bbl needs month='YYYY-MM' or year=YYYY")


def add_result(sources: Sources, index: int, args: dict, result: dict) -> None:
    """Make a calculator result a source for the guardrail and for later calculations. It
    covers up to the latest month any of its inputs covers (R5), and keeps the calls its
    inputs came from (for the sources footer)."""
    cands = candidates(sources)
    inputs = [_input(v, u, cands) for v, u in zip(result["inputs"], _input_units(args),
                                                     strict=True)]
    until = max((c.until for c in inputs if c and c.until), default=None)
    based_on = sorted({n for c in inputs if c for n in (c.call, *c.based_on) if n is not None})
    label = f"call {index}: calculate.{result['operation']}"
    sources.numbers.append(Source(result["result"], result["unit"], label, "calculator",
                                  until=until, based_on=tuple(based_on)))
