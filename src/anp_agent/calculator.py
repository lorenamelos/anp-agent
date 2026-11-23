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
from anp_agent.guardrail.match import Policy, candidates, find
from anp_agent.guardrail.sources import BBL_TO_M3, Source, Sources

NAME = "calculate"
OPERATIONS = ("sum", "mean", "difference", "ratio", "percent_change", "multiply", "convert")
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
        "or in a year (year=YYYY)."
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
    units = [unit] * len(values)
    if op == "multiply" and percent:
        units[0] = "%"
    if op == "convert":
        units[0] = "bbl/d"
    cands = candidates(sources)
    for value, u in zip(values, units, strict=True):
        if not _grounded(value, u, cands):
            raise CalcError(
                f"{_plain(value)} is not in this question's tool results; call a data tool first "
                "or copy the value exactly as the tool returned it")

    result, out_unit = _compute(op, values, unit, args, percent)
    return {"operation": op, "inputs": values, "result": round(result, 6), "unit": out_unit}


def _plain(value: float) -> str:
    """1200000.0 -> '1200000', 0.517 -> '0.517' (no scientific notation)."""
    return f"{value:.10f}".rstrip("0").rstrip(".")


def _grounded(value: float, unit: str | None, cands) -> bool:
    """Is the value, at the last digit the model wrote, in the sources?"""
    text = repr(abs(float(value)))
    if "e" in text:  # 1e-05 or 1e+16: no written digit to round at; compare exactly
        place = 0.0
    else:
        frac = text.split(".")[1]
        place = 1.0 if frac == "0" else 10.0 ** -len(frac)
    direction = -1 if value < 0 else None
    return find(Reading(abs(value), place), unit, False, direction, cands, INPUT_POLICY) is not None


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
        m = re.fullmatch(r"((?:19|20)\d{2})-(0[1-9]|1[0-2])", str(month))
        if not m:
            raise CalcError("month must be YYYY-MM")
        return value * calendar.monthrange(int(m.group(1)), int(m.group(2)))[1], "bbl"
    if isinstance(year, int) and not isinstance(year, bool) and 1900 <= year <= 2100:
        return value * (366 if calendar.isleap(year) else 365), "bbl"
    raise CalcError("convert to bbl needs month='YYYY-MM' or year=YYYY")


def add_result(sources: Sources, index: int, result: dict) -> None:
    """Make a calculator result a source for the guardrail and for later calculations."""
    label = f"call {index}: calculate.{result['operation']}"
    sources.numbers.append(Source(result["result"], result["unit"], label, "calculator"))
