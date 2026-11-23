"""Command line: `anp-agent "pergunta"` (design, D13).

Prints the answer, with its notices and footer, on stdout. `-v` adds a summary of the run on
stderr (status, calls, tokens, USD); `--json` prints the whole result record instead, for
scripts and evals. Settings come from ANP_AGENT_* variables; the API key from
ANTHROPIC_API_KEY, read from `.env` in the current directory if it is not set.

Exit codes: 0 answered (also redacted or limited by the budget), 1 the question failed
(status error or tool_unavailable), 2 the agent could not start (settings, API key, server).
"""

import argparse
import asyncio
import json
import os
import sys
from typing import Any

from dotenv import load_dotenv

from anp_agent.config import ConfigError, Settings
from anp_agent.graph import AgentContext, answer
from anp_agent.llm import load_prices, make_llm
from anp_agent.mcp_client import AnpToolbox, ToolboxError

FAILED = ("error", "tool_unavailable")


def parse_args(argv: list[str] | None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog="anp-agent", description="Answer a question about Brazilian oil production "
        "(ANP data, through the anp-mcp server), with every number checked against its source.")
    parser.add_argument("question", help="the question, in Portuguese or English")
    out = parser.add_mutually_exclusive_group()
    out.add_argument("-v", "--verbose", action="store_true",
                     help="also print a summary of the run on stderr")
    out.add_argument("--json", action="store_true", help="print the result record as JSON")
    return parser.parse_args(argv)


async def run(question: str, settings: Settings) -> dict[str, Any]:
    async with AnpToolbox.start(settings) as toolbox:
        ctx = AgentContext(settings, toolbox, make_llm(settings), load_prices())
        return await answer(question, ctx)


def record(result: dict[str, Any], settings: Settings) -> dict[str, Any]:
    """The serialisable part of the final state (the messages stay out)."""
    keys = ("question", "language", "status", "error", "answer", "notices", "removed",
            "regen_count", "reports", "tool_calls_used", "calculator_calls_used",
            "not_executed", "llm_calls", "tokens_used", "usd_used", "llm_log", "tool_records")
    return {"model": settings.model, **{k: result[k] for k in keys if k in result}}


def summary(result: dict[str, Any]) -> str:
    calls = ", ".join(f"{r['name']}={r['status']}" for r in result.get("tool_records", []))
    return (f"status={result.get('status')} llm_calls={result.get('llm_calls', 0)} "
            f"tool_calls={result.get('tool_calls_used', 0)} "
            f"calculator_calls={result.get('calculator_calls_used', 0)} "
            f"regenerations={result.get('regen_count', 0)} tokens={result.get('tokens_used', 0)} "
            f"usd={result.get('usd_used', 0.0):.6f}" + (f"\ncalls: {calls}" if calls else ""))


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    load_dotenv(".env")
    try:
        settings = Settings.from_env()
    except ConfigError as e:
        print(f"anp-agent: {e}", file=sys.stderr)
        return 2
    if settings.provider == "anthropic" and not os.environ.get("ANTHROPIC_API_KEY"):
        print("anp-agent: ANTHROPIC_API_KEY is not set (add it to .env)", file=sys.stderr)
        return 2
    try:
        result = asyncio.run(run(args.question, settings))
    except ToolboxError as e:
        print(f"anp-agent: {e}", file=sys.stderr)
        return 2
    except KeyboardInterrupt:
        return 130
    if args.json:
        print(json.dumps(record(result, settings), ensure_ascii=False, indent=2))
    else:
        print(result.get("answer") or result.get("error", ""))
        if args.verbose:
            print(summary(result), file=sys.stderr)
    if result.get("status") in FAILED:
        if not args.json and result.get("error"):
            print(f"anp-agent: {result['error']}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
