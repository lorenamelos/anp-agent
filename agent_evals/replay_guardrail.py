"""Replay the R1 guardrail on the graded instrument-v3 answers of anp-mcp (no LLM calls).

For each answer in vendor/anp-mcp/evals/runs/*v3.jsonl, run the guardrail against that
answer's own question, tool calls and the server instructions, and compare its flags with
the grader's verdict (design, section 4.4). Two modes:

- base: sources only (tool results, arguments, question, instructions, constants);
- calc: also any number one calculator operation could produce from grounded inputs (sum or
  mean of a contiguous window of one series; difference, ratio or percent change of two
  same-unit values, which may be window aggregates; a percentage of a value; bbl/d to
  barrels in a year). An upper bound: it assumes the model would always call the calculator
  when it needed to. It says nothing about catches: with free windows and pairs, almost any
  percentage comes out of some combination.

A coincidence probe multiplies each stated number by 0.83 and 1.17 and counts how many
wrong values would still be grounded, per mode.

    uv run python -m agent_evals.replay_guardrail
    uv run python -m agent_evals.replay_guardrail --sensitivity   # tolerance variants

Writes agent_evals/results/replay_<time>.md (summary) and .csv (every flagged number).
"""

import argparse
import asyncio
import bisect
import csv
import json
import math
import re
from collections import Counter
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path

from anp_mcp.server import DATASET_REVISION, INSTRUCTIONS, mcp

from anp_agent.guardrail.extract import PERCENT, QUANTITY, Reading, extract
from anp_agent.guardrail.match import DEFAULT, Policy, candidates, check, close, ground_number
from anp_agent.guardrail.sources import Sources, build_sources

ROOT = Path(__file__).resolve().parent.parent
RUNS = ROOT / "vendor" / "anp-mcp" / "evals" / "runs"
OUT = ROOT / "agent_evals" / "results"
FACTORS = (0.83, 1.17)  # coincidence probe
MAX_SERIES = 60  # longer series (every month since 2005) get no window aggregates
CAUGHT = {"wrong_number", "ungrounded"}


@dataclass(frozen=True)
class Value:
    value: float
    unit: str | None
    label: str


def load() -> list[dict]:
    records = []
    for path in sorted(RUNS.glob("*v3.jsonl")):
        for line in path.read_text(encoding="utf-8").splitlines():
            r = json.loads(line)
            assert r["instrument"] == "v3", path
            assert r["dataset_revision"] == DATASET_REVISION, path
            r["file"] = path.name
            records.append(r)
    return records


class Calculator:
    """Every result of one calculator operation over grounded inputs (upper bound)."""

    def __init__(self, sources: Sources, policy: Policy = DEFAULT):
        self.policy = policy
        pool = [Value(s.value, s.unit, s.label) for s in sources.numbers
                if s.origin != "constant" and s.unit]
        series: dict[str, list] = {}
        for s in sources.numbers:
            if s.series and s.pos is not None and s.unit:
                series.setdefault(s.series, []).append(s)
        aggregates = []
        for name, items in series.items():
            items = sorted(items, key=lambda s: s.pos)
            if len(items) > MAX_SERIES:
                continue
            values = [s.value for s in items]
            for i in range(len(values)):
                total = 0.0
                for j in range(i, len(values)):
                    total += values[j]
                    n = j - i + 1
                    if n < 2:
                        continue
                    where = f"{name}[{i}:{j + 1}]"
                    aggregates.append(Value(total, items[0].unit, f"sum({where})"))
                    aggregates.append(Value(total / n, items[0].unit, f"mean({where})"))
        self.operands = sorted(pool + aggregates, key=lambda v: v.value)
        self.keys = [v.value for v in self.operands]
        self.aggregates = sorted(aggregates, key=lambda v: v.value)
        self.agg_keys = [v.value for v in self.aggregates]

    def _near(self, keys, items, lo, hi):
        return items[bisect.bisect_left(keys, lo):bisect.bisect_right(keys, hi)]

    def __call__(self, r: Reading, unit: str | None, hedged: bool) -> str | None:
        """Operands as a sensible calculator call would take them: both with a known unit,
        the same unit, and a result in the unit the answer states (or no stated unit)."""
        v = r.value
        slack = max(r.place, 0.02 * v if hedged else 0.0) + 1e-9
        # a window aggregate on its own
        for a in self._near(self.agg_keys, self.aggregates, v - slack, v + slack):
            if unit in (None, a.unit) and close(r, a.value, hedged, self.policy):
                return a.label
        for a in self.operands:
            # difference b - a, either order; a difference of percentages is in pp
            if unit in (None, a.unit) or (unit == "pp" and a.unit == "%"):
                for target in (a.value + v, a.value - v):
                    for b in self._near(self.keys, self.operands, target - slack, target + slack):
                        if b is not a and b.unit == a.unit and close(
                                r, b.value - a.value, hedged, self.policy):
                            return f"difference({b.label}, {a.label})"
            # a percentage of a value: a% x b
            if a.unit == "%" and a.value > 0 and unit != "%":
                target = v * 100 / a.value
                width = slack * 100 / a.value
                for b in self._near(self.keys, self.operands, target - width, target + width):
                    if b.unit not in ("%", None) and unit in (None, b.unit) and close(
                            r, a.value * b.value / 100, hedged, self.policy):
                        return f"multiply({a.label}, {b.label})"
            # bbl/d to barrels in a year
            if a.unit == "bbl/d" and unit in (None, "bbl"):
                for days in (365, 366):
                    if close(r, a.value * days, hedged, self.policy):
                        return f"convert({a.label}, year of {days} days)"
            if a.value <= 0 or unit != "%":
                continue
            # ratio 100 * b / a, and percent change 100 * (b / a - 1), of same-unit values
            width = a.value * slack / 100 + 1e-9
            for name, target in (("ratio", a.value * v / 100),
                                 ("percent_change", a.value * (1 + v / 100)),
                                 ("percent_change", a.value * (1 - v / 100))):
                for b in self._near(self.keys, self.operands, target - width, target + width):
                    if b is a or b.value <= 0 or b.unit != a.unit:
                        continue
                    result = 100 * b.value / a.value - (0 if name == "ratio" else 100)
                    if close(r, abs(result), hedged, self.policy):
                        return f"{name}({b.label}, {a.label})"
        return None


def probe(answer: str, sources: Sources, derive, policy: Policy) -> list[tuple[int, bool]]:
    """(digits written, grounded by chance) for each stated number scaled by FACTORS."""
    cands = candidates(sources)
    out = []
    for c in extract(answer):
        if c.kind not in (QUANTITY, PERCENT) or not c.readings[0].value:
            continue
        for f in FACTORS:
            wrong = tuple(Reading(r.value * f, r.place, r.literal) for r in c.readings)
            out.append((digits(c.readings[0]), ground_number(
                wrong, c.unit, c.hedged, None, cands, derive, policy) is not None))
    return out


def digits(r: Reading) -> int:
    """Significant digits written: "35%" -> 2, "1,06 milhão" -> 3, "40.000" -> 5."""
    written = r.literal or r.place
    return max(1, math.floor(math.log10(r.value) + 1e-9) - math.floor(math.log10(written)) + 1)


def coincidence(probes: list[tuple[int, bool]]) -> str:
    """Probe hit rate overall and by digits written (1, 2, 3, 4+)."""
    def rate(sub):
        return f"{sum(h for _, h in sub)}/{len(sub)} ({sum(h for _, h in sub) / max(1, len(sub)):.1%})"
    parts = [f"{label}: {rate([p for p in probes if lo <= p[0] <= hi])}"
             for label, lo, hi in (("1 digit", 1, 1), ("2", 2, 2), ("3", 3, 3), ("4+", 4, 99))]
    return f"{rate(probes)}; by digits written: " + ", ".join(parts)


def cited(number: str, reason: str) -> bool:
    """Does the grader's reason quote this number (same leading digits)?"""
    digits = re.sub(r"\D", "", re.split(r"\s", number.strip("+-−~ "))[0])
    return bool(digits) and digits in re.sub(r"[.,\s]", "", reason)


def group(r: dict) -> str:
    if r["grader_passed"]:
        return "passed"
    return "failed: wrong_number/ungrounded" if r["error_type"] in CAUGHT else "failed: other"


def server_schemas() -> dict[str, dict]:
    """Input schemas as the server lists them (the same listing a client receives)."""
    return {t.name: t.input_schema for t in asyncio.run(mcp.list_tools())}


def replay(records: list[dict], policy: Policy = DEFAULT):
    """(rows, probes, flagged numbers) for both modes."""
    schemas = server_schemas()
    rows, flagged_numbers = [], []
    probes = {m: [] for m in ("base", "calc")}
    for r in records:
        sources = build_sources(r["question"], r["calls"], INSTRUCTIONS, schemas)
        calc = Calculator(sources, policy)
        row = {"group": group(r), "model": r["model"], "truncated": bool(sources.truncated)}
        for mode, derive in (("base", None), ("calc", calc)):
            report = check(r["answer"], sources, derive, policy)
            row[mode] = not report.ok
            row[f"{mode}_n"] = len(report.ungrounded)
            row[f"{mode}_cited"] = any(cited(v.claim.text, r["reason"]) for v in report.ungrounded)
            for v in report.ungrounded:
                c = v.claim
                flagged_numbers.append({
                    "mode": mode, "file": r["file"], "case": r["case"], "repeat": r["repeat"],
                    "model": r["model"], "grader": row["group"], "error_type": r["error_type"],
                    "truncated": row["truncated"], "kind": c.kind, "number": c.text,
                    "context": r["answer"][max(0, c.start - 60):c.end + 30].replace("\n", " "),
                })
            if mode == "calc":
                row["derived"] = sum(1 for v in report.verdicts
                                     if v.grounded and v.source and v.source.startswith(
                                         ("sum(", "mean(", "difference(", "ratio(", "percent_",
                                          "multiply(", "convert(")))
            probes[mode] += probe(r["answer"], sources, derive, policy)
        rows.append(row)
    return rows, probes, flagged_numbers


def run() -> tuple[str, list[dict]]:
    rows, probes, flagged = replay(load())
    return summary(rows, probes, flagged), flagged


VARIANTS = [
    ("trailing zeros not rounding (default)", Policy()),
    ("trailing zeros as rounding", Policy(trailing_zeros=True)),
    ("trailing zeros as rounding, cap 5%", Policy(trailing_zeros=True, max_rel=0.05)),
    ("trailing zeros as rounding, cap 2%", Policy(trailing_zeros=True, max_rel=0.02)),
]


def sensitivity() -> str:
    """False alarms, catches and coincidence for each tolerance variant (complete results)."""
    records = load()
    lines = [("| tolerance | mode | false alarms (of 102) | catches (of 11) | cited | "
              "coincidence |"), "|---|---|---|---|---|---|"]
    for name, policy in VARIANTS:
        rows, probes, _ = replay(records, policy)
        clean = [r for r in rows if not r["truncated"]]
        passed = [r for r in clean if r["group"] == "passed"]
        caught = [r for r in clean if r["group"] == "failed: wrong_number/ungrounded"]
        for mode in ("base", "calc"):
            hits, total = sum(h for _, h in probes[mode]), len(probes[mode])
            lines.append(
                f"| {name} | {mode} | {sum(r[mode] for r in passed)} | "
                f"{sum(r[mode] for r in caught)} | {sum(r[f'{mode}_cited'] for r in caught)} | "
                f"{hits / max(1, total):.1%} |")
    return "\n".join(lines)


def table(rows: list[dict], mode: str) -> list[str]:
    groups = ["passed", "failed: wrong_number/ungrounded", "failed: other"]
    lines = ["| grader verdict | answers | flagged | not flagged | numbers flagged |",
             "|---|---|---|---|---|"]
    for g in groups:
        sub = [r for r in rows if r["group"] == g]
        flagged = sum(r[mode] for r in sub)
        lines.append(f"| {g} | {len(sub)} | {flagged} | {len(sub) - flagged} | "
                     f"{sum(r[f'{mode}_n'] for r in sub)} |")
    return lines


def rates(rows: list[dict], mode: str) -> str:
    passed = [r for r in rows if r["group"] == "passed"]
    caught = [r for r in rows if r["group"] == "failed: wrong_number/ungrounded"]
    fa = sum(r[mode] for r in passed)
    ca = sum(r[mode] for r in caught)
    ci = sum(r[f"{mode}_cited"] for r in caught)
    return (f"False alarms: **{fa}/{len(passed)}** grader-passed answers flagged "
            f"({fa / max(1, len(passed)):.0%}). Catches: **{ca}/{len(caught)}** "
            f"wrong_number/ungrounded answers flagged; in **{ci}** of them a flagged number "
            "is one the grader's reason cites.")


def summary(rows: list[dict], probes: dict, flagged: list[dict]) -> str:
    clean = [r for r in rows if not r["truncated"]]
    cut = [r for r in rows if r["truncated"]]
    models = ", ".join(f"{m}: {n}" for m, n in Counter(r["model"] for r in rows).items())
    out = [f"# Guardrail replay on instrument-v3 answers ({datetime.now(UTC):%Y-%m-%d %H:%M} UTC)",
           "", (f"{len(rows)} answers ({models}); {len(cut)} with a tool result cut at "
                "6,000 characters are reported separately."), ""]
    for mode, title in (("base", "Without calculator"), ("calc", "With calculator (upper bound)")):
        out += [f"## {title}", "", f"Answers with complete tool results ({len(clean)}):", ""]
        out += table(clean, mode) + ["", rates(clean, mode), ""]
        for model in sorted({r["model"] for r in clean}):
            out += [f"- {model}: " + rates([r for r in clean if r["model"] == model], mode)]
        out += ["", f"Answers with a cut tool result ({len(cut)}):", ""] + table(cut, mode)
        out += ["", ("Coincidence probe, wrong values (stated number x 0.83 or x 1.17) that "
                     f"would be grounded: {coincidence(probes[mode])}."), ""]
    out += [(f"Numbers grounded only through a calculator operation (calc mode): "
             f"{sum(r['derived'] for r in rows)}."), ""]
    kinds = Counter((f["mode"], f["kind"]) for f in flagged if not f["truncated"])
    out += ["Flagged numbers by kind (complete results): " + ", ".join(
        f"{m}/{k}: {n}" for (m, k), n in sorted(kinds.items())), ""]
    return "\n".join(out)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--sensitivity", action="store_true", help="tolerance variants only")
    if parser.parse_args().sensitivity:
        print(sensitivity())
        return
    text, flagged = run()
    OUT.mkdir(exist_ok=True)
    stamp = f"{datetime.now(UTC):%Y%m%dT%H%M%SZ}"
    (OUT / f"replay_{stamp}.md").write_text(text + "\n", encoding="utf-8")
    with (OUT / f"replay_{stamp}.csv").open("w", newline="", encoding="utf-8-sig") as f:
        w = csv.DictWriter(f, fieldnames=list(flagged[0]) if flagged else ["mode"])
        w.writeheader()
        w.writerows(flagged)
    print(text)
    print(f"Written: agent_evals/results/replay_{stamp}.md and .csv")


if __name__ == "__main__":
    main()
