# anp-agent

[![CI](https://github.com/lorenamelos/anp-agent/actions/workflows/ci.yml/badge.svg)](https://github.com/lorenamelos/anp-agent/actions/workflows/ci.yml)

A [LangGraph](https://github.com/langchain-ai/langgraph) agent that answers questions about
Brazilian oil production, with every number in the answer checked against the tool results
it came from.

The data comes only from the [anp-mcp](https://github.com/lorenamelos/anp-mcp) server, pinned
to `v0.1.0` and used as is. The model is Claude Haiku 5.5, behind a single configuration point.
Design and decisions: [docs/design.md](docs/design.md).

## What it does

```bash
anp-agent "Quanto Búzios produziu em setembro de 2026?"
```

The model calls the anp-mcp tools, and a local `calculate` tool for anything it derives
(averages, totals, shares, changes, unit conversions). Before the answer is shown, a
deterministic check (no LLM) extracts every number and date from it and looks for each one in
this question's tool results. A number with no source sends the draft back to the model once;
if it is still there, only that number is replaced with `[não confirmado]` and listed in the
footer.

The output looks like this (the format, not a recorded run):

```text
Em setembro de 2026, Búzios produziu 1.197.657 bbl/d, cerca de [não confirmado] do pré-sal.

Dados de jul–set/2026 ainda podem ser revisados pela ANP.

Fontes: search_field_groups(buzios); field_group_production(BÚZIOS, 2026-08..2026-09)
Removido por não ter origem confirmada: "35%"
```

The notices, the footer and the redaction marker are written by code, in the language of the
question (Portuguese or English).

## Rules

| Rule | What the agent does |
|---|---|
| **R1** Provenance | Every number must come from a tool result of this question, the question itself, the server instructions, a conversion constant, or a `calculate` result whose inputs passed the same check. Checked in code (`guardrail/`), not only asked for in the prompt. |
| **R2** History vs forecast | Out of scope in v1: there is no forecast tool, so forecasts are declined. |
| **R3** Tool gap | When no tool covers the question (gas, royalties, prices, forecasts), the model is told to say so instead of answering from memory. This rule is in the prompt and the server's instructions; R1 catches the numbers such an answer would invent, not the wording. |
| **R4** Budget | Per question: 6 anp-mcp calls, 4 calculator calls, 10 LLM calls and a hard limit of 40,000 tokens (input, output, cache read and cache write alike), projected before each LLM call. When a limit is hit, the model answers with what it has and a notice says what was not done. USD is recorded, never a limit. |
| **R5** Revisions | When the answer uses any of the dataset's last 3 months, a notice says ANP may still revise them. |

## How it works

```text
start ─> agent ⇄ tools              anp-mcp over stdio, and the local calculator
         agent ─> guardrail (R1) ─> finalize                notices and footer
                  guardrail ─> feedback ─> agent            one regeneration, if it fits
                  guardrail ─> redact ─> finalize           otherwise
         agent ─> finalize          no LLM call fits: a readable list of the values obtained
         tools ─> finalize          server unavailable: a fixed answer, no model numbers
```

| Module | Role |
|---|---|
| `graph.py` | Nodes, routing and the per-question run |
| `guardrail/` | `extract.py` (numbers, units, ranges, hedges, dates in pt-BR and en), `sources.py` (what an answer may cite), `match.py` (rounding-aware matching), `redact.py` |
| `calculator.py` | Exact arithmetic, including a year's volume from monthly rates; rejects any input that is not already in the sources |
| `mcp_client.py` | `AnpToolbox`: stdio client, version check, warm-up, timeouts, one restart |
| `budget.py`, `config.py` | Token accounting, the projection before each call, the call caps; settings |
| `llm.py` | The provider switch; usage from the raw API `usage` block |
| `footer.py`, `prompts.py`, `emergency.py` | Deterministic texts in pt/en; the system prompt; the answer when no LLM call fits |

## Quick start

Requires [uv](https://docs.astral.sh/uv/) and an Anthropic API key.

```bash
git clone --recurse-submodules https://github.com/lorenamelos/anp-agent.git
cd anp-agent
make setup                                         # submodule, virtual environment, dependencies
echo 'ANTHROPIC_API_KEY=...' > .env                # never committed (.gitignore)
uv run anp-agent "Quais os 5 campos que mais cresceram nos últimos 6 meses?"
uv run anp-agent -v "How much did Búzios produce in September 2026?"   # + run summary on stderr
uv run anp-agent --json "Bacalhau cresceu? Quanto?"                    # the whole result record
```

The first run downloads the anp-mcp dataset (version `v2026.09`) from the Hugging Face Hub.
Exit codes: 0 answered (also when redacted or limited by the budget), 1 the question failed,
2 the agent could not start.

### Settings

Environment variables, all optional:

| Variable | Default |
|---|---|
| `ANP_AGENT_MODEL` | `claude-haiku-5-5` |
| `ANP_AGENT_MAX_TOOL_CALLS` | 6 |
| `ANP_AGENT_MAX_CALCULATOR_CALLS` | 4 |
| `ANP_AGENT_MAX_TOKENS_PER_QUESTION` | 40000 |
| `ANP_AGENT_MAX_OUTPUT_TOKENS` | 2048 |
| `ANP_AGENT_MAX_LLM_CALLS` | 10 |
| `ANP_AGENT_TOOL_TIMEOUT_S`, `ANP_AGENT_QUESTION_TIMEOUT_S` | 20, 120 |

The anp-mcp server's own variables (`ANP_MCP_REVISION`, `ANP_MCP_DATA_DIR`) are passed through.

## Results

Final runs on one commit (`567640b`), Claude Haiku 5.5 as the solver, Sonnet 5.5 as the
grader, 3 repeats per question. Every failure was read by hand before it was counted (the
grader's labels are model output too; see below).

### Instrument v3 (anp-mcp's 21 questions)

The same questions, ground truth, grader and prices as anp-mcp's own Haiku v3 run, which used
the plain tool loop of its harness:

| | anp-mcp harness (v3) | agent |
|---|---|---|
| Pass rate | 87% (55/63) | 98% (62/63) |
| Failed for a wrong number | **5** | **0** (1 labelled `wrong_number`; its numbers are right) |
| Failed as `ungrounded` | 2 | 0 |
| Other failures | 1 `other` | 0 |
| Solver cost per answer | US$ 0.00061 | US$ 0.00068 (+11%) |
| Tokens per answer (mean / p95 / max) | 7,350 / 13,018 / 17,107 | 11,439 / 20,142 / 29,558 (+56% mean) |
| Seconds per answer | 5.3 | 5.4 |

**The main finding: no answer failed for a wrong number, against 5 in the harness run.** The
pass rates differ by 7 answers in 63; the 3 repeats of a question are not independent, so
read it as a direction, not a measured gain.

- The one failure (q10 r2, "Did Bacalhau grow? By how much?") has every number right: the
  error is a statement, "in absolute volume it was behind only Búzios", when the tool's own
  ranking put Mero ahead. The grader labelled it `wrong_number`; R1 checks numbers, not
  rankings.
- The guardrail flagged 1 draft; the regeneration fixed it. No number was redacted. The
  calculator was used in 17 answers (29 calls, none rejected). No answer hit a call cap or
  the token limit.

### Instrument agent-v1 (the agent's own 14 questions)

[agent_evals/cases_agent.yaml](agent_evals/cases_agent.yaml); not comparable with the table
above:

| Category | Cases | Passed |
|---|---|---|
| multi-step | a01–a04 | 8/12 |
| budget stress | a05 | 1/3 |
| refusals | a06–a09 | 12/12 |
| partial refusal | a10 | 2/3 |
| over-refusal check | a11 | 3/3 |
| derivation (Búzios in barrels per year) | a12 | 3/3 |
| regression (v3 q17, token limit) | a13 | 3/3 |
| tool gap (a field that does not exist) | a14 | 3/3 |
| **all** | | **35/42 (83%)** |

The 7 failures, read by hand. None is a wrong number:

- **a02 r0, r2** (the top 5 risers, then their wells): a regression from the rule that asks
  for `search_field_groups` before every lookup. The model searched the 5 names that
  `top_changes` had just returned, used the 6 calls, and had the 5 well lookups refused. r1
  did the same and passed, as the rubric accepts saying what was not checked. Before the
  rule, a02 passed 3/3 with `top_changes` and 5 lookups. See Known limits.
- **a01 r0** (Búzios' share of the national total): the calculator rejected the ratio four
  times because the model passed `unit: "%"`, the unit of the result, which the calculator
  reads as the unit of the inputs. The answer gave both inputs, right, and no share. Seven
  calculator rejections in the run, all this case of `ratio` or `percent_change`.
- **a03 r0** (did Lula produce more than Mero in 2025): the first draft was right and
  complete, but its difference (287 thousand bbl/d) did not come from `calculate`, whose 4
  calls were spent. The regeneration asked for one more calculation, which the cap refused,
  and the next LLM call did not fit in the 40,000 tokens, so the answer was the
  deterministic list of values, with no conclusion.
- **a05 r1, r2** (the 10 largest fields of 2025, month by month): no tool ranks by a past
  year within the call cap (Known limits).
- **a10 r2**: "the data covers oil and gas production"; the tools cover no gas. A statement,
  which R1 does not check.

Cost of the two final runs, grader included: US$ 0.54 (v3, 63 answers) and US$ 0.41
(agent-v1, 42 answers). Earlier runs, and the fixes made between them, are in
[docs/design.md](docs/design.md), section 4.5.

## Checking the check

Before any LLM spend, the guardrail was replayed on the 126 graded answers that anp-mcp's own
evals produced (instrument v3, Sonnet 5.5 and Haiku 5.5, 63 each), comparing its flags with
the grader's verdicts (`agent_evals/replay_guardrail.py`, no API calls). On the 114 answers
whose tool results were stored complete:

| | grader passed (102) | grader failed for a wrong or unsupported number (11) |
|---|---|---|
| guardrail flags something | 31 (false alarms) | 8 (catches; in 6 the flagged number is the one the grader cited) |

Most false alarms are correct numbers the old answers derived without a tool (averages,
shares, m³/d, monthly totals). That is what the `calculate` tool is for: with an upper-bound
emulation of it, 1 false alarm is left. Whether the model actually calls it is measured by
the live evals.

## Known limits

The provenance check (R1) has little power on low-precision numbers. With many numbers in
the tool results, a wrong value written with few significant digits often matches some other
real value by chance. In an offline probe on 126 graded answers (each stated number moved by
−17% and +17%), these wrong values were still accepted:

| significant digits written | wrong values accepted |
|---|---|
| 1 | 79% |
| 2 | 19% |
| 3 | 15% |
| 4 or more | 1% |

R1 also checks numbers only: a wrong entity, a wrong explanation, or a right number attached
to the wrong field or month passes it. Details: [docs/design.md](docs/design.md), section 4.4
and risks.

anp-mcp has no ranking by a past period: its lists order by the latest month or by a recent
change. A question such as "the 10 largest fields of 2025" would need the 2025 series of
more field groups than the 6-call cap allows, so the agent can only rank by a stated proxy.

Removing a number can leave a sentence that reads badly ("[não confirmado]" in place of a
range), most often in final mode, where there is no regeneration.

Found in the final runs, not fixed in this version (the published results measure this
commit):

- **The search-first rule costs calls on names a tool already returned** (a02). The prompt
  asks for `search_field_groups` before every `field_group_production`, which was meant for
  names typed by the user ("Lula" is in the TUPI group). Proposed: search only names that
  come from the question; names from a tool result are already exact.
- **`unit: "%"` on `ratio` or `percent_change` is rejected** with a message that says the
  value is not in the results (a01, a04). Proposed: take `%` there as the result's unit, or
  say in the error that the value was found in another unit.
- **A regeneration can be spent on a call the cap refuses** (a03). Proposed: when a cap is
  already reached, send the regeneration in final mode (no tools), so it ends in text.

## Development

```bash
make test    # offline: no network, no API
make lint
uv run pytest -m integration   # also the real anp-mcp server (local data, no API)
uv run pytest -m api -s        # one usage check against the API (needs .env; < US$ 0.001)
uv run python -m agent_evals.replay_guardrail   # the offline replay above
```

`vendor/anp-mcp` is a read-only git submodule at tag `v0.1.0`, used for the eval instrument
(cases, ground truth, grader, prices and the graded runs).

CI (GitHub Actions) runs `ruff` and the offline tests on every push and pull request, with no
API key.

## Licence

MIT.
