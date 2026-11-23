"""Run the anp-mcp eval instrument (v3) on the agent (design, section 4.1).

Mirrors anp-mcp's `evals/run_evals.main()` and replaces only `ask()` with the agent graph:
the same cases, ground truth, grader (Sonnet 5.5), grounding rule and prices, imported
unchanged from the vendored anp-mcp at v0.1.0. Records keep every field of the harness
schema, so its `report.py` reads them as is, plus agent fields that it ignores.

    uv run python -m agent_evals.run_agent_evals --cases q01 q05 -r 1
    uv run python -m agent_evals.run_agent_evals --max-usd 0.90      # all cases, 3 repeats

Needs ANTHROPIC_API_KEY in .env. One anp-mcp server for the whole run. Stops before a new
answer once the run's cost reaches --max-usd. Results: agent_evals/results/<run>.jsonl.
"""

import argparse
import asyncio
import json
import logging
import os
import subprocess
import sys
import time
from datetime import UTC, datetime
from pathlib import Path

import yaml
from anthropic import AsyncAnthropic
from dotenv import load_dotenv

from anp_agent.budget import CALCULATOR
from anp_agent.config import Settings
from anp_agent.graph import AgentContext, answer
from anp_agent.llm import load_prices, make_llm
from anp_agent.mcp_client import AnpToolbox

ROOT = Path(__file__).resolve().parent.parent
VENDOR = ROOT / "vendor" / "anp-mcp"
OUT = ROOT / "agent_evals" / "results"
sys.path.insert(0, str(VENDOR))  # the vendored harness: `evals.*` (design, section 4.1)
from evals.ground_truth import compute
from evals.run_evals import INSTRUMENT, RESULT_LIMIT, Costs, grade

logging.getLogger("httpx2").setLevel(logging.WARNING)  # no per-request log lines (SDK 2.x)

TOKEN_KEYS = {"input": "input_tokens", "output": "output_tokens",
              "cache_read": "cache_read_tokens", "cache_write": "cache_write_tokens"}


def git_commit(path: Path) -> str:
    def git(*args):
        return subprocess.run(["git", "-C", str(path), *args], check=False, capture_output=True,
                              text=True).stdout.strip()
    commit = git("rev-parse", "--short", "HEAD") or "unknown"
    return commit + ("-dirty" if git("status", "--porcelain", "--untracked-files=no") else "")


def as_run(result: dict, seconds: float) -> dict:
    """The agent's result in the shape of the harness's `ask()`."""
    records = result.get("tool_records", [])
    calls = [{"name": r["name"], "args": r["args"], "result": r["result_text"][:RESULT_LIMIT],
              "is_error": r["status"] != "ok"} for r in records]
    tokens = {key: sum(e.get(kind, 0) for e in result.get("llm_log", []))
              for kind, key in TOKEN_KEYS.items()}
    return {"answer": result.get("answer") or result.get("error", ""), "calls": calls,
            "solver_usd": result.get("usd_used", 0.0), **tokens, "seconds": round(seconds, 1)}


def agent_fields(result: dict, agent_commit: str) -> dict:
    reports = result.get("reports", [])
    return {
        "status": result.get("status"), "error": result.get("error"),
        "regenerated": result.get("regen_count", 0) > 0,
        "ungrounded_before_regen": reports[0]["ungrounded"] if reports else [],
        "ungrounded_final": reports[-1]["ungrounded"] if reports else [],
        "removed": result.get("removed", []),
        "budget_reached": result.get("status") == "budget_reached",
        "token_limited": result.get("token_limited", False),
        "not_executed": result.get("not_executed", []),
        "tool_calls_used": result.get("tool_calls_used", 0),
        "calculator_calls_used": result.get("calculator_calls_used", 0),
        "llm_calls": result.get("llm_calls", 0), "tokens_used": result.get("tokens_used", 0),
        "llm_log": result.get("llm_log", []), "agent_commit": agent_commit,
    }


async def main(args) -> None:
    load_dotenv(ROOT / ".env")
    if not os.environ.get("ANTHROPIC_API_KEY"):
        sys.exit("ANTHROPIC_API_KEY missing: add it to .env at the repository root.")
    settings = Settings.from_env()
    prices = load_prices()
    for m in (settings.model, args.judge_model):
        if m not in prices:
            sys.exit(f"No price for {m} in the vendored pricing.yaml.")
    cases = yaml.safe_load((VENDOR / "evals" / "cases.yaml").read_text())
    if args.cases:
        cases = [c for c in cases if c["id"] in args.cases]
    truth = compute()
    server_revision = os.environ.get("ANP_MCP_REVISION", truth["dataset_revision"])
    if server_revision != truth["dataset_revision"]:  # design, risk 10
        sys.exit(f"dataset mismatch: server {server_revision}, truth {truth['dataset_revision']}")
    costs = Costs(prices)
    client = AsyncAnthropic()
    agent_commit, server_commit = git_commit(ROOT), git_commit(VENDOR)
    run_id = (f"{datetime.now(UTC):%Y%m%dT%H%M%SZ}_agent_{settings.model}"
              + (f"_{args.label}" if args.label else ""))
    OUT.mkdir(parents=True, exist_ok=True)
    out = OUT / f"{run_id}.jsonl"
    print(f"Agent {agent_commit}, anp-mcp {server_commit}, instrument {INSTRUMENT}, "
          f"{len(cases)} cases x {args.repeats}, solver {settings.model}, judge {args.judge_model}")

    async with AnpToolbox.start(settings) as toolbox:
        ctx = AgentContext(settings, toolbox, make_llm(settings), prices)
        tools = [t["name"] for t in ctx.tools]
        with out.open("w", encoding="utf-8") as f:
            for case in cases:
                rubric = case["rubric"].format(**truth)
                for rep in range(args.repeats):
                    if costs.total >= args.max_usd:
                        print(f"Stopped: budget of ${args.max_usd:.2f} reached.")
                        break
                    start = time.monotonic()
                    result = await answer(case["question"], ctx)
                    run = as_run(result, time.monotonic() - start)
                    _add_solver_cost(costs, settings.model, run)
                    verdict = await grade(client, args.judge_model, case, rubric, run, costs,
                                          toolbox.instructions)
                    # a data question must be answered from some anp-mcp tool; the
                    # calculator alone does not count (design, section 4.1)
                    called = {c["name"] for c in run["calls"]} - {CALCULATOR}
                    tool_ok = not case["expect_tools"] or bool(called)
                    unexpected = (sorted(called - set(case["expect_tools"]))
                                  if case["expect_tools"] else [])
                    passed = verdict["passed"] and tool_ok
                    record = {
                        "run_id": run_id, "instrument": INSTRUMENT, "server_commit": server_commit,
                        "label": args.label or "", "case": case["id"],
                        "category": case["category"], "repeat": rep, "model": settings.model,
                        "judge_model": args.judge_model,
                        "dataset_revision": truth["dataset_revision"],
                        "server_instructions": bool(toolbox.instructions),
                        "tools_available": tools, "question": case["question"],
                        "rubric": rubric, **run, "tool_ok": tool_ok,
                        "unexpected_tools": unexpected, **verdict,
                        "grader_passed": verdict["passed"], "passed": passed,
                        **agent_fields(result, agent_commit),
                    }
                    f.write(json.dumps(record, ensure_ascii=False, default=str) + "\n")
                    f.flush()
                    mark = "PASS" if passed else "FAIL"
                    print(f"{case['id']} r{rep} {mark:4} {verdict['error_type']:12} "
                          f"{record['status']:15} ${run['solver_usd'] + verdict['judge_usd']:.4f}"
                          f"  {verdict['reason'][:80]}")

    print(f"\nCost by model: {json.dumps(costs.by_model, indent=1)}")
    print(f"Total: ${costs.total:.4f}   Results: {out.relative_to(ROOT)}")
    print(f"Summary: PYTHONPATH=vendor/anp-mcp uv run python -m evals.report "
          f"{out.relative_to(ROOT)}")


def _add_solver_cost(costs: Costs, model: str, run: dict) -> None:
    """Book the agent's tokens in the harness's Costs, so its totals and --max-usd cover the
    solver as in run_evals (where ask() calls costs.add per response)."""
    class U:  # the attribute names Costs.add reads
        input_tokens = run["input_tokens"]
        output_tokens = run["output_tokens"]
        cache_read_input_tokens = run["cache_read_tokens"]
        cache_creation_input_tokens = run["cache_write_tokens"]
    costs.add(model, U)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--judge-model", default="claude-sonnet-5-5")
    parser.add_argument("-r", "--repeats", type=int, default=3)
    parser.add_argument("--cases", nargs="*", help="case ids, e.g. q01 q05")
    parser.add_argument("--max-usd", type=float, default=1.0, help="stop the run at this cost")
    parser.add_argument("--label", help="tag added to the run name")
    asyncio.run(main(parser.parse_args()))
