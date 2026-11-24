# Evaluation

Two tools, both reading the anp-mcp eval instrument from the `vendor/anp-mcp` submodule (tag
`v0.1.0`), which is used as is.

## Live evals: `run_agent_evals.py`

Runs anp-mcp's instrument **v3** on the agent: the same 21 cases, ground truth, grader
(Sonnet 5.5), grounding rule and prices as anp-mcp's own runs, so the results compare
directly with its Haiku v3 run. Only the system under test changes: the agent graph replaces
the harness's plain tool loop.

```bash
uv run python -m agent_evals.run_agent_evals --cases q01 q11 -r 1   # a pilot
uv run python -m agent_evals.run_agent_evals --max-usd 1.00         # 21 cases x 3
PYTHONPATH=vendor/anp-mcp uv run python -m evals.report agent_evals/results/<run>.jsonl
```

Needs `ANTHROPIC_API_KEY` in `.env`. `--max-usd` stops the run before a new answer once the
cost (agent and grader) reaches it.

**What the grader reads.** The model's text after redaction (`answer`), without the notices
and the sources footer. Those are written by code, not by the model, and no tool result states
them, so the grader's grounding rule fails an answer for a notice such as "data for Jul–Sep
2026 may still be revised by ANP" (it did in the pilot). anp-mcp's harness also graded only
the model's text. The full output, as a user sees it, is kept in `answer_full`. When there is
no model text (server unavailable, no LLM call fitted, an error), the answer as shown is
graded.

**Records** keep every field of the harness schema, so its `report.py` reads them as is, plus
agent fields: `answer_full`, `status`, `regenerated`, `draft_before_regen` (the first draft,
when the answer was regenerated), `ungrounded_before_regen`, `ungrounded_final`, `removed`,
`budget_reached`, `token_limited`, `not_executed`, `tool_calls_used`,
`calculator_calls_used`, `llm_calls`, `tokens_used`, `llm_log` (per LLM call: mode,
projected prompt, actual prompt, every token kind) and `agent_commit`.

A data question passes the harness's tool check only with an anp-mcp call; a calculator call
alone does not count.

## Instrument agent-v1

The agent's own cases (`cases_agent.yaml`, a01–a14), with the same runner, grader and
prices: multi-step questions (a share of the national total, a ranking followed by lookups,
an old field name, year over year), a question that needs more calls than the cap allows,
refusals (prices, a forecast, investment advice, another country), a partial refusal, an
over-refusal check, a derivation the guardrail must see grounded, a field that does not
exist, and v3 q17 again as a regression case. Expected values come from
`ground_truth_agent.py`, which imports anp-mcp's `ground_truth.py` (unchanged) and adds
keys, each a query on the same dataset files.

```bash
uv run python -m agent_evals.ground_truth_agent                  # the expected values
uv run python -m agent_evals.run_agent_evals --instrument agent-v1 --max-usd 0.90
```

As in anp-mcp: the cases are committed before a run, and a change in how answers are
checked bumps the name to agent-v2. Results of different instruments are never mixed in one
table.

## Offline replay: `replay_guardrail.py`

Runs the guardrail (R1) on anp-mcp's 126 graded v3 answers and compares its flags with the
grader's verdicts, with no API calls. See the main README and the design, section 4.4.

```bash
uv run python -m agent_evals.replay_guardrail
uv run python -m agent_evals.replay_guardrail --sensitivity
```

Results go to `agent_evals/results/` (not committed).
