# anp-agent: design

Status: **v0.1.0 implemented and evaluated: final runs of instruments v3 and agent-v1 on
`567640b` (section 4.5.3).** Decisions are recorded in section 6. The calculator (D1) was implemented after the
offline guardrail replay in section 4.4 was reviewed.

## 0. Scope and fixed constraints

- A LangGraph agent that answers questions about ANP oil production using the
  [anp-mcp](https://github.com/lorenamelos/anp-mcp) server, pinned to tag `v0.1.0`, as its
  only data source. anp-mcp is **not modified** to suit the agent.
- v1 covers history only (field group, national, top changes, reconciliation). Forecasts,
  royalties, gas, prices, investment advice and non-ANP data are refused, with a statement of
  what is not covered.
- LLM: Claude Haiku 5.5 through the Anthropic API, behind a single configuration point so
  the provider can be swapped (e.g. a local model later).
- Interface: command line only in v1 (`anp-agent "question"`), plus the eval runner.
- Stack: Python 3.13 (same as anp-mcp), uv, pytest, ruff (line length 100, as in anp-mcp).
  Licence: MIT.

Agent rules:

| Rule | Statement |
|---|---|
| **R1** Provenance | Every number in an answer comes from a tool call made for this question (or from the question itself). Checked by a verifier node, not only by the prompt. |
| **R2** History vs forecast | Never mixed; a forecast always carries its interval, model version and data cut-off. Applies once a forecast tool exists; not in v1. |
| **R3** Tool gap | If no tool covers it, say so; never fill the gap from the model's memory. |
| **R4** Budget | Per-question caps on tool calls and tokens; when a cap is hit, answer with what was fetched and say what is missing. |
| **R5** Revisions | When an answer uses the latest months, say they may still be revised by ANP. |

Facts from the anp-mcp code that shape this design:

| Fact | Consequence |
|---|---|
| 5 read-only tools; outputs are JSON with numbers already rounded (`round(x, 0)` for bbl/d, 1 decimal for `change_pct`, 3 for `gap_pct`) | Provenance can be checked against exact numeric leaves, but answers derive numbers (averages, m³/d, monthly totals, shares) that are **not** in any output |
| Every output carries a `unit` field or unit-bearing keys (`*_bpd`, `*_pct`, `active_wells`) | The verifier can infer the unit of each source number from its key |
| The server sends `instructions` at initialisation (scope, dataset version, "not covered") | The agent passes them to the model, as desktop clients and the eval harness do; they are also a valid provenance source (as in grader v3) |
| `serverInfo.version` reports the package version | The agent refuses to start against a server other than 0.1.0 |
| The harness (`evals/`) is in the repository but **not in the installed package** | Reuse needs the tagged source tree, not only the dependency (section 4) |
| Claude 5.5 models do not support forced `tool_choice` (noted in `run_evals.py`); `tool_choice: none` is not forced and works (checked, D21) | "Answer now without tools" keeps the tool definitions and sends `tool_choice: none` (D21), so the cached prefix (tools + system) stays the same |
| Haiku v3 runs: mean 7.3k tokens per answer, p95 13k, max 17k; mean 1.7 tool calls, max 6 | Grounds the budget in section 3 |

## 1. Graph

### 1.1 Nodes and edges

```text
                 ┌──────────────────────────────────────────┐
                 v                                          │
START ──> agent ──(tool calls?)──yes──> tools ──────────────┘
            │                             (anp-mcp tools and the local calculator,
            │ no (final draft)             each within its own cap; over-budget calls
            v                              get an error result and set final mode)
        guardrail (R1)
            │
            ├── all numbers grounded ───────────────────────────> finalize ──> END
            ├── ungrounded, regen not used, budget allows ──> agent (with feedback)
            └── ungrounded, otherwise ──────────────> redact ──> finalize ──> END
```

| Node | What it does | LLM? |
|---|---|---|
| `agent` | Calls the model with the system prompt, the server instructions, the MCP tool schemas and the calculator schema. In **final mode** (a cap reached, or the last allowed LLM call) it calls the model with `tool_choice: none` and, on the first final call, a note listing what was not fetched; a tool request that still comes back gets "not executed" results. Checks the token budget **before** each call (section 3). | yes |
| `tools` | Executes the tool calls of the last AI message, in order: anp-mcp tools through the MCP session, `calculate` locally (section 1.4). Each kind has its own cap. Every `tool_use` gets a `tool_result` (the API requires it): executed, error, timed out, or "not executed: budget reached". Appends a `ToolRecord` per call. | no |
| `guardrail` | R1: extracts the numbers from the draft and matches them against the provenance sources (section 1.3). Writes a `GuardrailReport`. | no |
| `redact` | Replaces each ungrounded number span (number, scale word and unit) with "[não confirmado]" / "[unconfirmed]" and records the removed text for the footer. Deterministic; no new LLM text, so nothing new to verify. | no |
| `finalize` | Appends, in the question's language: the R4 notice (cap reached, calls not made), the R5 notice (section 1.5), the list of removed values (if any) and the sources footer. Sets `status`. | no |

Routing functions:

- `after_agent`: tool calls in the last message → `tools`; otherwise → `guardrail`.
- `after_tools`: always → `agent` (the agent node decides normal vs final mode from the state).
- `after_guardrail`: `ok` → `finalize`; not ok and `regen_count == 0` and the token budget
  can pay one more call → `agent`; otherwise → `redact`.

Regeneration: the draft stays in the message list, followed by a user-role feedback message
naming each ungrounded number and the three ways out: fetch it with a tool, derive it with
the calculator, or remove it. Regeneration **may call tools**, within the same caps (the
counters are not reset). At most one regeneration per question.

LangGraph's `recursion_limit` is set as a backstop (e.g. 25 steps); the real limits are the
budgets in section 3.

**Footer** (built by `finalize` from the state, never by the model), in the language of the
question:

```text
Fontes: field_group_production(BÚZIOS, 2026-01..2026-09); calculate(mean, 9 valores)
Removido por não ter origem confirmada: "1,2 milhão de bbl/d"
```

Sources list every successful call whose output grounded at least one claim of the final
answer, with its arguments abbreviated (the field group or search text bare, `start..end`,
other arguments as `key=value`); a calculator result also brings in the calls its inputs came
from, transitively. The "removed" line appears only after redaction. An answer with no
grounded number from a call has no sources line.

Language: the system prompt tells the model to answer in the language of the question; the
deterministic texts (notices, footer, redaction marker) use a small pt/en detector on the
question (stop-word count), defaulting to Portuguese.

### 1.2 State

The state holds only serialisable per-question data. The MCP session, the LLM client and the
limits live in the run context (LangGraph `context` / `RunnableConfig`), not in the state.

```python
class ToolRecord(TypedDict):
    call_id: str
    name: str                 # anp-mcp tool name or "calculate"
    args: dict
    result_text: str          # what the model saw
    structured: dict | None   # parsed JSON (source of numbers for R1)
    status: Literal["ok", "tool_error", "timeout", "transport_error", "not_executed"]
    seconds: float

class Claim(TypedDict):
    text: str                 # span as written, e.g. "1,06 milhão de bbl/d"
    start: int; end: int      # offsets in the draft, for redaction
    kind: Literal["quantity", "percent", "range", "date", "count"]
    value: float | tuple[float, float]
    unit: str | None          # canonical: "bbl/d", "m3/d", "bbl", "%", "pp", "wells", None
    grounded: bool
    source: str | None        # e.g. "call 2: months[8].petroleum_bpd" or "question"
    source_until: str | None  # last month the source value covers (for R5, section 1.5)

class GuardrailReport(TypedDict):
    ok: bool
    claims: list[Claim]
    ungrounded: list[Claim]

class AgentState(TypedDict):
    question: str
    language: Literal["pt", "en"]
    messages: Annotated[list[AnyMessage], add_messages]
    tool_records: list[ToolRecord]
    # R4
    tool_calls_used: int      # anp-mcp calls
    calculator_calls_used: int
    tokens_used: int          # all billed tokens: input, output, cache read, cache write
    usd_used: float           # recorded only, never a limit
    llm_calls: int
    final_mode: bool
    not_executed: list[dict]  # calls refused by a cap: name + args
    # R1
    regen_count: int
    reports: list[GuardrailReport]   # one per draft checked
    removed: list[str]               # spans replaced by the redaction marker
    # Output
    answer: str
    status: Literal["answered", "redacted", "budget_reached", "tool_unavailable", "error"]
    notices: list[str]
```

### 1.3 R1 guardrail: where it runs and how it checks

**Where.** Once per final draft, after the model stops calling tools and before anything is
returned. It does not check intermediate turns (they are not shown to the user) or tool
arguments. It runs again on the regenerated draft; the second result is final.

**Provenance sources** (the set a number may come from), built from this question only:

1. Every numeric leaf in the `structured` output of each successful anp-mcp `ToolRecord`,
   with a path (`call 2: months[8].petroleum_bpd`), the row's month when it has one, and a
   unit inferred from the key or the record's `unit` field (`*_bpd` → bbl/d, `*_pct` → %,
   `active_wells` → wells, `months_*`/`year` → count). Numbers inside string fields (e.g.
   `what_it_shows`) are also taken, unit unknown.
2. Numbers in the tool **arguments** (e.g. `months=6`), so "nos últimos 6 meses" is grounded.
   Arguments include the defaults from the tool's `input_schema`, as the server lists it,
   since the tool ran with them (e.g. `start="2024-01"` of `field_group_production`).
3. Numbers in the **question**: any number the user wrote counts as grounded, whatever its
   kind (quantity, date, count), e.g. "1 milhão de barris" in "quando Búzios passou de
   1 milhão de barris por dia?".
4. Numbers in the **server instructions** (e.g. 2005, the dataset version, "two months"),
   matching grader v3, which accepts them as a source.
5. A short whitelist of constants, each with its own unit so that it grounds only numbers of
   that kind: the factor 0.158987 and "1 bbl" (both sides of "1 bbl = 0.158987 m³"),
   6.2898 bbl (one m³), and days in a period (28–31, 365, 366; they ground "× 31 dias" but
   never "31%" or "31 poços").
6. One tool-specific date rule: the months of the two windows `top_changes` compares are
   derived from its `latest_month` and `window_months` (the 2 × window months up to and
   including `latest_month`), so "abr–set/2026 vs out/2025–mar/2026" is grounded.
7. Outputs of the local **calculator** (section 1.4). Its inputs must themselves be grounded
   in 1–6 or in earlier calculator outputs (transitive provenance), so it cannot launder a
   number taken from memory.

**Extraction** (pure function, `text -> list[Claim]`):

1. Strip markdown emphasis while keeping a character-offset map back to the original text.
2. Find number tokens with a regex that accepts: sign; digits with `.`, `,`, space or thin
   space as thousands separators; `.` or `,` as decimal separator; scale words (`mil`,
   `milhão/milhões`, `mi`, `bilhão`, `k`, `thousand`, `million`, `M`, `bn`); percent
   (`%`, `por cento`, `percent`); percentage points (`p.p.`, `pp`, `pontos percentuais`).
3. Attach a unit from the text that follows (`bbl/d`, `b/d`, `bpd`, `barris por dia`,
   `barris/dia`, `m³/d`, `m3/d`, `metros cúbicos por dia`, `barris`, `bbl`, `poços`, `wells`).
4. Locale: an ambiguous token like `1.056` yields **both** readings (1056 and 1.056); the
   claim is grounded if either reading matches. Unambiguous tokens (`1.056.771`, `1,06`
   followed by `milhão`) yield one reading.
5. Classify:
   - **date**: `YYYY-MM`, month name + year, a 4-digit year 1990–2100 with no unit or scale.
     Grounded if the year/month appears in any source (tool output periods, args, question,
     instructions). This catches "2024 was a record" stated without a tool call.
   - **list marker / ordinal**: `1.` at line start, `1º`, `2nd` → ignored.
   - **count**: a bare integer ≤ 12 with no unit or scale (e.g. "5 campos", "3 meses") →
     exempt.
   - **range**: see below.
   - everything else → **quantity** or **percent**.

**Ranges** ("entre X e Y", "between X and Y", "de X a Y", "from X to Y", "X–Y", "X a Y",
"X ± Y"): a scale word or unit written only after Y applies to X too ("entre 1 e 1,1 milhão
de bbl/d" → 1,000,000 to 1,100,000 bbl/d). A hyphen between two `YYYY`/`YYYY-MM` tokens is a
date span, not a range. A range is grounded if **either**:

- both endpoints are grounded independently (e.g. min and max of a series, or a forecast
  interval's bounds once a forecast tool exists, R2); **or**
- it is an approximate statement about one value: some source value lies inside [X, Y] and
  the range is narrow, (Y − X) / midpoint ≤ 10%. Wider ranges that only contain a value are
  not grounded, because they say little and would let vague guesses pass.

`X ± Y` is checked as the range [X − Y, X + Y] with the same rules.

**Matching** a claim `c` (value `v`, unit `u`) against a source `s`:

1. **Units.** Compatible if equal, if either is unknown, or through a whitelisted conversion
   applied to the source: bbl/d → m³/d (× 0.158987); bbl/d → bbl in a month (× days of that
   row's month, read from the sibling `month` key); bbl/d → bbl in a year (× days of the
   year, only for `frequency='year'` rows with `complete_year: true`). Scale words
   (mil = 10³, milhão = 10⁶, …) are applied to the claim before comparing.
2. **Rounding.** Tolerance comes from what the claim displays: if the last displayed digit
   has place value `p` (for "1,06 milhão", p = 10,000; for "26,3%", p = 0.1; for
   "1.056.771", p = 1), the claim matches if `s` rounds **or truncates** to `v` at place `p`
   (|s − v| ≤ p/2, or 0 ≤ s − v < p for a positive truncation). Source numbers are already
   rounded by the server, so exact displays match exactly. Trailing zeros are written digits:
   "40.000" has p = 1, so a round number needs a hedge word ("cerca de 40.000", rule 3) or
   must be the exact value. The matcher's `Policy` can instead take the trailing zeros of an
   integer of 1,000 or more as rounding ("40.000" → p = 10,000), optionally capped relative
   to the value (never below the last written digit); the replay measured these variants and
   the default catches one more wrong number with no more false alarms (section 4.4, D19).
3. **Hedge words** ("cerca de", "aproximadamente", "quase", "about", "~", "around") widen the
   tolerance to max(rounding rule, 2% relative).
4. **Sign.** "+26,3%", "queda de 26,3%" and "−26,3%" all match a source of −26.3 or +26.3
   only when the direction word agrees with the sign of the source (a fall stated as a rise
   is wrong); without a direction word the absolute value is compared.
5. **Derived numbers** (averages, sums, shares, changes not returned by an anp-mcp tool) are
   grounded only through the calculator. The verifier does not search for derivations itself
   at run time.

Output: a `GuardrailReport`. `ok` is true when no quantity, percent, range or date claim is
ungrounded.

**What R1 does not cover**, stated so the metric is not over-read: non-numeric claims (wrong
entity, wrong grouping reason, "X is outside group Y"), and a correct number attached to the
wrong entity or month ("Mero produced 1,056,771" when that was Búzios). The second case can be
narrowed later by requiring the source row's `field_group`/`month` to appear near the claim;
not in v1. These stay with the eval grader.

### 1.4 Calculator tool (local)

A deterministic tool offered to the model next to the anp-mcp tools. It lives in this
repository; anp-mcp is not changed.

```text
calculate(operation, values, unit=None, to_unit=None, month=None, year=None, percent=False,
          months=None)
  operation: sum | mean | difference | ratio | percent_change | multiply | convert
             | annual_volume
  values:    list of numbers, each copied from a tool result of this question
  -> {"operation", "inputs", "result", "unit"}
```

- `mean` is the simple mean (as the eval ground truth computes yearly field averages);
  `difference` and `percent_change` take two values, (b − a) and 100·(b/a − 1); `ratio` takes
  [part, whole] and returns 100·part/whole in %; the difference of two percentages is in
  percentage points (`pp`), so "fell 2.3%" for a 2.3 pp change is flagged; `multiply` takes two values, and with `percent=True` the first one is a
  percentage (0.517 % of 1,749,888 bbl/d = 9,047 bbl/d); `convert` covers bbl/d → m³/d,
  bbl/d → bbl in `month` (days of that month) and bbl/d → bbl in `year` (365 or 366 days).
- **Input check at call time**: each input must match a source (rules of section 1.3) at the
  last digit the model wrote, with no hedge widening and never taking trailing zeros as
  rounding. An ungrounded input makes the call fail with a
  `tool_result` error naming the value, so the model learns immediately; the failed call
  still counts toward the calculator cap.
- Cap: **4 calculator calls per question**, separate from the 6 anp-mcp calls.
- Results are full precision; the answer may round them (rounding rule applies).

### 1.5 R5 notice

At start-up the toolbox reads the dataset's latest month (from the warm-up call, section 2).
If any grounded claim of the final answer comes from a value that covers one of the **last
3 months** of the dataset (ANP re-exports the two months before each release), or is a date
naming one of them, `finalize` adds the notice below. Each source value records the last
month it covers (`until`): a month row, its month; a year row, December of that year (so the
current, incomplete year counts); anything under a top-level `latest_month` (`top_changes`),
that month; a calculator result, the latest month among its inputs. The notice is one short
line, since most questions about current production trigger it: "Dados de jul–set/2026 ainda
podem ser revisados pela ANP." / "Data for Jul–Sep 2026 may still be revised by ANP."

## 2. Connection to anp-mcp

**Dependency.** `anp-mcp @ git+https://github.com/lorenamelos/anp-mcp@v0.1.0` in
`pyproject.toml`; `uv.lock` records the commit hash (`a313249`). The server runs from the
same virtual environment, so no network install happens at run time.

**Process and session.** A thin adapter of our own, `AnpToolbox`, wraps the official `mcp`
client (`stdio_client` + `ClientSession`), as `evals/run_evals.py` does. No
`langchain-mcp-adapters`.

```text
async with AnpToolbox.start(settings) as toolbox:     # one per process (CLI run, eval run)
    toolbox.instructions   # server instructions, for the system prompt
    toolbox.tool_schemas   # [{name, description, input_schema}] exactly as listed
    toolbox.latest_month   # dataset's latest month, for R5
    await toolbox.call(name, args) -> ToolRecord
```

- Launch: `StdioServerParameters(command=sys.executable, args=["-m", "anp_mcp.server"])`,
  environment inherited plus explicit `ANP_MCP_REVISION` only if set by the user. Server
  stderr goes to `logs/anp-mcp.stderr.log`, not to the terminal.
- `start()`: spawn, `initialize()` with a 30 s timeout, check `serverInfo.name == "anp-mcp"`
  and `serverInfo.version == "0.1.0"` (fail fast on mismatch), `list_tools()` once, keep the
  instructions. Tool schemas are passed to the model **as listed by the server**, with no
  hand-written wrappers, so the agent sees exactly what any MCP client sees.
- Warm-up: `start()` makes one `national_reconciliation` call outside any question budget.
  It downloads the dataset on first use (so the first question is not penalised) and gives
  `latest_month` for R5. Its output **never enters the model's context** and is not a
  provenance source: only `latest_month` is kept, for the R5 notice. If the model needs the
  reconciliation, it calls the tool itself, and that call counts toward the budget. CI can
  set `ANP_MCP_DATA_DIR` to a cached copy. The warm-up has its own 300 s timeout, since the
  first download takes far longer than a tool call; a failed warm-up is fatal at `start()`.
- Lifetime: one session per process, shared by all questions of that process. The graph gets
  the toolbox from the run context. One owner task opens the session, serves the calls in
  order from a queue and closes it, because the stdio transport's task groups must be entered
  and left by the same task, while graph nodes may call from other tasks. `AsyncExitStack` closes the session and terminates the
  child process on exit, including on cancellation and Ctrl-C.
- Calls run sequentially, in the order the model issued them (determinism over latency; the
  tools take milliseconds once the data is cached).

**Errors and timeouts.**

| Failure | Detection | Handling | Counts toward call cap |
|---|---|---|---|
| Server does not start / `initialize` times out / version mismatch | at `start()` | Fatal before any LLM call; non-zero exit; eval run aborted | n/a |
| Tool returns `isError` (e.g. bad arguments) | `result.isError` | `tool_result` with `is_error`, model may correct itself | yes |
| Tool call exceeds timeout (default 20 s) | `asyncio.timeout` around `call_tool` | `tool_result` "timed out", status `timeout`, no automatic retry | yes |
| Transport broken (child died, broken pipe, EOF) | exception from the session | Restart the server once and retry that call once (tools are read-only, so retry is safe). If it fails again: the record's status is `transport_error`, later calls fail fast with the same status, and the question's status is `tool_unavailable`, with a deterministic answer "could not reach the data" and no LLM-generated numbers | yes, once |
| Anthropic API 429 / 5xx | SDK | SDK retries (`max_retries=2`); then status `error` | no |
| Whole question exceeds wall-clock limit (default 120 s) | `asyncio.timeout` around the graph run | status `error`, partial state logged | n/a |

## 3. Budget control (R4)

**Where the limits live.** One `Settings` object (`src/anp_agent/config.py`), read from
environment variables (`ANP_AGENT_<SETTING>`, e.g. `ANP_AGENT_MAX_TOOL_CALLS=4`) with
defaults, which is also the single configuration point for the LLM provider and model. Passed
to the graph through the run context; never in the prompt. The API key stays in the SDK's own
variable (`ANTHROPIC_API_KEY`, from `.env`, never committed).

| Setting | Default | Note |
|---|---|---|
| `max_tool_calls` | 6 | anp-mcp calls; max observed in Haiku v3 runs is 6 |
| `max_calculator_calls` | 4 | separate cap |
| `max_tokens_per_question` | 40,000 | **hard** limit; all billed tokens count alike (input, output, cache read, cache write) |
| `max_output_tokens` per LLM call | 2,048 | same as the eval harness |
| `max_llm_calls` | 10 | loop backstop; the last allowed call is in final mode |
| `recursion_limit` | 25 | LangGraph steps, behind the limits above |
| `tool_timeout_s`, `question_timeout_s`, `server_init_timeout_s` | 20, 120, 30 | section 2 |

The token limit exists to stop loops and runaway context, not to measure cost, so the four
token kinds count alike. Cost in USD is computed per call with the harness's `Costs` (each
kind at its own price) and recorded in the state and the eval records. It is never a limit.

Basis for 40,000, counted the same way (input + output + cache read + cache write, summed
over all calls of one answer) on the instrument-v3 runs: Haiku mean 7,349, p95 13,018,
max 17,107; Sonnet mean 8,667, max 32,613. The Sonnet maximum is q17, which called
`national_reconciliation(details=True)` (every month, a long result that is re-sent on each
later turn). A question like that fits the limit but leaves no room for a regeneration, so
it goes to redaction instead, as section 1.1 specifies.

**Tool-call caps.** Enforced in the `tools` node, per individual call (one model turn can
request several), separately for anp-mcp calls and calculator calls. If a turn asks for
3 anp-mcp calls with 2 left, the first 2 run and the third gets "not executed: tool-call
budget reached"; the call is added to `not_executed` and `final_mode` is set. Next, the
`agent` node calls the model with `tool_choice: none` (D21), with a note listing what was not
fetched, and asks it to answer with what it has and say what is missing.

**Token limit.** Checked in the `agent` node **before** each LLM call, using a projection,
because a call cannot be stopped halfway:

```text
projected = tokens_used + (input tokens of the last call + tokens added since) + max_output_tokens
projected <= limit            -> normal call (final mode if a cap was reached or it is the
                                 last allowed LLM call)
projected >  limit            -> no LLM call: deterministic answer from the tool results
                                 obtained (emergency.py, below) and what was not done
```

**The emergency answer** (no LLM call fits; D25) is written by code, with no raw JSON: one
sentence saying the limit was reached; what the data obtained shows, as each successful
call's own values, labelled and with units in the question's language (a time series shows
its latest 12 rows, a ranking its top 12, with a count of the rows left out); then the
queries made and those left out (refused calls, and the conclusion when the token limit was
the cause). Every value is a tool result's leaf, so the guardrail grounds all of them (tested
on the real a03 and a05 records of the first agent-v1 run). The R4 notice is not added on top,
since the answer already lists what was left out.

Since final mode keeps the tool definitions (D21), a final call costs as much as a normal
one, so the token limit has no "final call without tools" step: when the next call does not
fit, no call is made. Regeneration is offered only when one more call fits.

The last call's prompt is known exactly from its `usage` (input + cache read + cache write,
plus its output, which the next call re-sends). Only the messages added since (tool results,
the regeneration feedback) and the first prompt are estimated, on the high side, at 1.8
characters per token plus 50 tokens per added message. Both come from the eval runs: the
pilot measured up to 66 tokens of fixed per-message cost the text does not show, and the
full v3 run about 1.93 characters per token on long numeric series (2.0, the first value,
was 509 tokens short on one answer): JSON full of digits takes more tokens per character than prose, and an
overestimate only moves a call into final mode earlier. The eval run records the estimate
next to the real input of the following call, to check that it stays on the high side.
Because the projection includes the full `max_output_tokens`, the 40,000-token limit is
not exceeded as long as that estimate holds. Token usage is read from each response's raw `usage` block, not from
LangChain's `usage_metadata`, which folds cache tokens into `input_tokens` and would
double-count in the cost formula shared with the harness. A test compares both against a
direct SDK call.

**What the user sees when a limit is hit.** The model's answer from final mode, plus a
deterministic notice from `finalize` (so the R4 message does not depend on the model
remembering it), e.g. "Resposta limitada pelo orçamento por pergunta: 2 consultas não foram
feitas (field_group_production para MERO, TUPI)." `status = "budget_reached"`.

**Interaction with R1.** Regeneration happens only if the token projection allows one more
call; otherwise the draft goes straight to `redact`. Calls made during regeneration use the
same counters. The budget is never extended to make room for the guardrail.

## 4. Evaluation

### 4.1 Reusing the anp-mcp instrument without changing it

The harness lives in `anp-mcp/evals/` and is not part of the installed package. anp-mcp is
added as a **git submodule at `vendor/anp-mcp`, checked out at tag v0.1.0**, read-only. A
test (and CI) asserts that the submodule is at the tag, has no local changes, and is the
same commit as the `anp-mcp` entry in `uv.lock`.

Reused **unchanged**, imported from `vendor/anp-mcp` (`evals.*` on `sys.path` only inside the
eval runner):

| From anp-mcp `evals/` | Used for |
|---|---|
| `cases.yaml` (21 cases) | the same questions and rubrics |
| `ground_truth.compute()` | expected values, from the same dataset version the server reads |
| `grade()`, `GRADER_PROMPT`, `GRADE_TOOL`, `INSTRUMENT = "v3"` | the same grader (Sonnet 5.5) and grounding rule |
| `Costs`, `pricing.yaml` | the same cost accounting |
| `report.py` | the same summary and CSV |
| `runs/*_v3.jsonl` (126 graded answers) | offline guardrail replay (4.4) |

Our runner, `agent_evals/run_agent_evals.py`, mirrors `run_evals.main()` but replaces only
`ask()` with a call to the agent graph, which returns the same dict (`answer`, `calls` with
`name/args/result/is_error`, `solver_usd`, token counts, `seconds`). Records keep every field
of the harness schema, so `report.py` reads them as is, plus agent fields that it ignores:
`status`, `regenerated`, `ungrounded_before_regen`, `ungrounded_final`, `removed`,
`budget_reached`, `tool_calls_used`, `calculator_calls_used`, `llm_calls`, `agent_commit`.
The package name is `agent_evals`, not `evals`, to avoid a clash with the vendored package.

The grader reads the model's text after redaction (`answer`), without the notices and the
footer: code writes those, no tool result states them, and the grader's grounding rule would
fail an answer for "data may still be revised by ANP" (the pilot run did, once in two). The
harness graded the model's text too. The whole answer as the user sees it is kept in
`answer_full`. When there is no model text (server unavailable, no LLM call fitted, an
error), the answer as shown is graded.

Calculator calls appear in `calls` like any other tool call, so the grader sees them in the
trace. The `tool_ok` check of the harness ("a data question must be answered from some
tool") counts anp-mcp calls only: a calculator call alone does not satisfy it.

Comparability: the 21 original cases run under instrument **v3**, so the agent is directly
comparable with the anp-mcp Haiku v3 run (87%, USD 0.0006 per answer). What differs is the
system under test (agent graph, its system prompt and the calculator, instead of the
harness's plain loop); grader, cases, ground truth and pricing are identical. If anp-mcp
later ships an instrument v4, the agent moves to it only by bumping the tag, in a commit of
its own, never mixing versions in one table.

New cases are a **separate instrument, `agent-v1`**, in `agent_evals/cases_agent.yaml` (same
schema, optional extra fields such as `expect_refusal: true`). Their expected values come
from `agent_evals/ground_truth_agent.py`, which **imports** `connect()` and `one()` from the
vendored `ground_truth.py` and adds new keys; nothing in the vendored file is edited. Rule
kept from anp-mcp: commit the instrument before running it, bump `agent-v1` → `agent-v2` when
the way answers are checked changes.

### 4.2 New question types: instrument agent-v1

The cases are in `agent_evals/cases_agent.yaml` (a01–a14), their expected values in
`agent_evals/ground_truth_agent.py`; the runner takes `--instrument agent-v1`. a13 repeats
v3 q17 unchanged, as a regression case for the `details=true` prompt change (section 4.5).
a14 was added after the first run: a field that does not exist ("Atlântida": no field or
group name matches it, accents ignored, while the same query finds ATLANTA and BÚZIOS), to
test R3. Adding a case does not change how the other cases are checked, so the instrument
stays agent-v1.
The types, with the first wording of each:

| Type | Example | What the rubric checks |
|---|---|---|
| Multi-step: field + national | "Que parcela da produção nacional Búzios representou em setembro de 2026?" | share within 1 pp; both inputs grounded |
| Multi-step: ranking + lookups | "Dos 5 campos que mais cresceram em 6 meses, qual tinha mais poços ativos no último mês?" | correct field; needs 1 + 5 calls, i.e. exactly the cap |
| Multi-step: old name + comparison | "Lula produziu mais que Mero em 2025?" | resolves Lula → TUPI via search, compares 2025 averages |
| Multi-step: year over year | "A produção nacional de 2025 foi maior que a de 2024? Em quanto?" | direction and difference within 2% |
| Budget stress | "Compare mês a mês em 2025 os 10 maiores campos" | answers partially, says which fields were not fetched, no invented numbers |
| Correct refusal | royalties, natural gas, oil prices, forecast for a field, forecast for Brazil (v1 has no forecast tool), "devo investir na Petrobras?", Norway's production | says it is not covered and why; no figure of the refused kind |
| Partial refusal | "Quanto Mero produziu em 2025 e quanto Maricá recebeu de royalties?" | answers the first part, refuses the second |
| Over-refusal check | "Qual a produção offshore do Brasil em 2025?" (looks unusual, is covered) | answers; refusing fails |
| Guardrail pressure | "Quanto Búzios produz por ano, em barris?" (needs a derivation) | grounded derivation or a clear statement; no ungrounded figure |

### 4.3 Metrics

| Metric | Source | Target |
|---|---|---|
| Pass rate (v3 cases; agent-v1 cases) | grader | ≥ anp-mcp Haiku v3 (87%) on v3 cases |
| Answers failed as `ungrounded` or `invented` | grader (independent of R1) | zero |
| Ungrounded numbers in final answers | guardrail report | zero by construction; reported to show the guardrail ran |
| Guardrail triggers before regeneration; regeneration success; redaction rate | agent fields | reported, no target |
| Correct refusals / over-refusals | grader on refusal cases | all / none |
| Cost per answer (solver, USD), tokens, tool and calculator calls per answer, seconds, budget-reached rate | records | reported next to anp-mcp Haiku v3 |

The guardrail's own count is not evidence of correctness (it measures itself); the grader's
`ungrounded`/`invented` count is the independent check.

**Failures are read by hand before they are reported.** The grader's verdict and its
`error_type` label are both model output. Example: agent-v1 a08 r0 (section 4.5.2) was
labelled `wrong_number`, yet no number in the answer was wrong; the grader's own reason
named a stray sentence and a statement. The verdict stood, the label was the grader's
error, and the results say so instead of counting a wrong number. A label is reported as
the grader gave it only after the answer, its tool results and the reason have been read.

### 4.4 Offline guardrail replay (gate for the calculator)

anp-mcp's `evals/runs/` holds 126 graded instrument-v3 answers (Sonnet and Haiku, 63 each),
with the question, every tool call and result, the answer and the grader's verdict. The
replay (`agent_evals/replay_guardrail.py`) runs the guardrail on each answer, with no LLM
calls, in two modes:

- **Without calculator**: sources 1–5 of section 1.3 only (what R1 accepts if derived numbers
  are not allowed at all).
- **With calculator (upper bound)**: the old answers never called a calculator, so this mode
  also accepts any number the calculator *could* have produced from grounded inputs: one
  operation (sum or mean over a contiguous run of one series, difference, ratio or percent
  change of two source values, a whitelisted conversion). This is the best case for the
  calculator, assuming the model would have called it every time it needed to; it also
  shows how often a number matches only by coincidence.

For each mode, against the grader's verdict:

| | grader passed | grader failed (`wrong_number`, `ungrounded`) | grader failed (other) |
|---|---|---|---|
| guardrail flags ≥ 1 number | **false alarm** | **catch** | reported |
| guardrail flags nothing | agreement | miss | reported |

Plus the list of every flagged number with its answer excerpt, for manual review: a flag on a
grader-passed answer is not automatically wrong (the grader can miss an unsupported number).

**Results** (114 answers with complete tool results; 102 grader-passed, 11 failed for
`wrong_number`/`ungrounded`):

| tolerance | mode | false alarms | catches | catches citing the grader's number | coincidence |
|---|---|---|---|---|---|
| trailing zeros not rounding (default, D19) | base | 31 | 8 | 6 | 11.8% |
| trailing zeros not rounding (default, D19) | calc | 1 | 0 | 0 | 50.4% |
| trailing zeros as rounding | base | 31 | 7 | 5 | 12.8% |
| trailing zeros as rounding, cap 5% | base | 31 | 7 | 5 | 12.2% |
| trailing zeros as rounding, cap 2% | base | 31 | 7 | 5 | 11.9% |

With the calculator variants (calc mode) every tolerance gives 1 false alarm and 0 catches.
The remaining calc false alarm ("I can show the series since 2024", in an answer that did
not call the tool whose default is 2024) is left to regeneration, with no specific rule
(D20). The calc mode's 50% coincidence is a property of the emulation (free windows and
pairs), not of the real calculator, which only adds the results the model asked for.

Coincidence probe: each stated quantity or percentage is multiplied by 0.83 and 1.17, and the
probe counts how many of these wrong values the base mode would still ground, by the number
of significant digits written (default tolerance):

| digits written | probes | grounded by chance |
|---|---|---|
| 1 | 102 | 79.4% |
| 2 | 624 | 19.1% |
| 3 | 532 | 14.7% |
| 4 or more | 1,196 | 1.0% |
| all | 2,454 | 11.8% |

This is a known limit of R1 (risk 2): a wrong number written with 1 to 3 significant digits
("35%", "1,06 milhão") has a real chance of matching some other value in the sources, and no
tolerance setting changes that; the probe gives the same picture under every variant above.

Caveat: stored tool results are cut at 6,000 characters; records where a result hit the cut
are reported separately, since a flag there may be an artefact. Stored results are text, so
the replay parses them as JSON; a result that does not parse is reported, not guessed.

### 4.5 Results: instrument v3 (2026-11-24)

Run `20261124T002620Z`, agent `63c01b9`, anp-mcp `a313249`, Haiku 5.5 solver, Sonnet 5.5
grader, 21 cases × 3, dataset v2026.09; compared with anp-mcp's Haiku v3 run.

| | anp-mcp harness (v3) | agent |
|---|---|---|
| Pass rate | 87% (55/63) | 90% (57/63) |
| Failed for a wrong number (`wrong_number`) | **5** | **0** |
| Failed as `ungrounded` | 2 | 3 (none of them a number) |
| Other failures | 1 `other` | 3 `incomplete` |
| Solver cost per answer | US$ 0.00061 | US$ 0.00087 (+43%) |
| Tokens per answer (mean / p95 / max) | 7,350 / 13,018 / 17,107 | 12,302 / 27,080 / 31,878 (+67% mean) |
| Seconds per answer | 5.3 | 6.3 (+1 s) |

- **Main finding: 0 failures for a wrong number, against 5.** The pass rates differ by 2
  answers in 63, not a significant difference.
- The cost of the change: +43% solver cost per answer, +67% tokens, +1 s.
- The 3 `ungrounded` failures are non-numeric statements, which confirms the known limit of
  R1 (risk 4).
- R4: q17 r0 and r2 hit the token limit (about 31.5k used) after the model asked for
  `national_reconciliation(details=True)`, whose result (about 11k tokens) is re-sent on
  every turn; no LLM call fitted and the deterministic answer was graded `incomplete`. A
  final call without tools would not have fitted either (about 46k projected), so D21 played
  no part. The system prompt now asks for `details=true` only when the question needs
  month-by-month values; q17 is a regression case in agent-v1 to measure the effect. q11
  reached the anp-mcp cap (r0) and the calculator cap (r1) and passed. No call was refused
  by a cap.
- R1: 4 drafts flagged, 4 regenerations, 3 fixed, 1 redaction (q07 r0, the answer passed).
  The q19 "since 2024" case (D20) was fixed by regeneration.
- Token projection: 125 follow-up calls, 4 below the actual prompt (q04 ×3 by 29 tokens,
  q21 r1 by 509 on a long monthly series), so the estimate moved from 2.0 to 1.8 characters
  per token.
- Cost: US$ 0.632 for the run (solver US$ 0.055, grader US$ 0.578), plus US$ 0.025 for the
  2-question pilot.

### 4.5.1 Results: instrument agent-v1, first run (2026-11-24)

Run `20261124T011041Z`, agent `fca876e` (includes the 1.8 characters per token and the
`details=true` prompt change), anp-mcp `a313249`, Haiku 5.5 solver, Sonnet 5.5 grader, 13
cases × 3, dataset v2026.09. Not comparable with the v3 results: a different instrument.

| Category | Cases | Passed |
|---|---|---|
| multi-step | a01–a04 | 10/12 (a03: 1/3) |
| budget stress | a05 | 1/3 |
| refusals | a06–a09 | 11/12 (a08: 2/3) |
| partial refusal | a10 | 3/3 |
| over-refusal check | a11 | 3/3 |
| derivation | a12 | 0/3 |
| regression (v3 q17) | a13 | 3/3 |
| **all** | | **31/39 (79%)** |

- **a13 (regression for v3 q17): 3/3, and no repeat asked for `details=true`**, against 2
  token-limit failures in 3 under v3; 6k–14k tokens per answer, against about 31.5k.
- **R4 was reached in 8 answers** (status `budget_reached`), all failed except one:
  - a12 ×3 (derivation): the calculator cap. The model converted several yearly averages to
    barrels one `calculate` call at a time, used the 4 calls and had more refused, then
    answered with daily rates only; graded `incomplete`. The question asks for one figure;
    the cap and the one-value `convert` meet here.
  - a03 r0, r1: the token limit, after 6 anp-mcp calls, 4 calculator calls and a
    regeneration (about 35k tokens); no LLM call fitted and the deterministic answer was
    graded `incomplete`.
  - a05 ×3 (budget stress, by design more calls than the cap): r2 passed (it named what it
    did not fetch); r0 hit the token limit; r1 named two unfetched "top 10" fields from
    memory (`wrong_entity`), a non-numeric claim R1 does not check.
- a08 r0: `ungrounded` for "the fastest-growing fields are largely operated by Petrobras",
  again a statement, not a number.
- No failure for a wrong or ungrounded number. R1 flagged 2 drafts; both regenerations
  fixed them; no redaction.
- Token projection: 81 follow-up calls, none below the actual prompt (minimum 42 above).
- Cost: US$ 0.4492 (solver US$ 0.046, grader US$ 0.403), US$ 0.0012 solver per answer,
  8.8 s per answer.

### 4.5.2 Results: instrument agent-v1, the affected cases after D22–D26 (2026-11-24)

Run `20261124T141010Z`, agent `26bb469`, the same setup as 4.5.1, cases a03, a05, a08, a12
and a14 (new) × 3. Partial: it measures the fixes on the cases that failed, not the
instrument.

| Case | First run (`fca876e`) | This run (`26bb469`) |
|---|---|---|
| a03 Lula vs Mero | 1/3 | 3/3 |
| a05 budget stress | 1/3 | 0/3 |
| a08 investment advice | 2/3 | 2/3 |
| a12 Búzios per year | 0/3 | 3/3 |
| a14 nonexistent field | (new) | 3/3 |
| **all** | 4/12 | **11/15** |

- **a12:** every repeat used `annual_volume` (D24), 1 or 2 calls, and stated the basis
  (2025, the last complete year, D26): 276.8 million barrels, the ground truth's 2025 total.
- **a03:** 3/3, 25k–36k tokens; two repeats reached the call caps (R4) and passed with the
  final answer. No answer fell to the emergency text.
- **a14:** 3/3. No figure for Atlântida; each repeat named ATLANTA as a different group and
  gave its figures labelled as ATLANTA's.
- **a08 r0, graded `wrong_number`; on inspection, no number in the answer is wrong.** Every
  figure (4.71 million, 4.62 million, 87 thousand, 3.45 million bbl/d, +36.6%) is in the
  tool results or a `calculate` result. R1 flagged "20" in the first draft (that draft is
  not stored, so where "20" came from is unknown); the regenerated answer opened with a
  sentence about the feedback itself ("I could not confirm the number "20"… so it was
  removed"), and redaction turned it into "…o número "[não confirmado]"…". The grader's
  reason cites that stray sentence and a statement ("onshore has been falling slightly
  since early 2025": 89,131 in 2025-01, a peak of 92,616 in 2025-09, 87,345 in 2026-09),
  not a number. The failure is fair; its label is the grader's error. Cause: the
  regeneration, which narrated the feedback, then the removal. Not a hole in R1.
  Fixed by D27 (the first draft is now kept in the records, `draft_before_regen`);
  measured in the final runs. An example of why failures are read by hand (4.3).
- **a05 0/3:** the 2025 top 10 is not available from the tools. `search_field_groups`
  orders by the latest month, `top_changes` by the change between two windows ending at
  the latest month, and `field_group_production` reads one group per call; ranking by 2025
  needs the 2025 series of more groups than the 6-call cap allows. r0 and r1 ranked by
  2026-09 and said so, so their "not fetched" lists named BACALHAU and PEREGRINO instead of
  SAPINHOÁ and RONCADOR (`wrong_entity`). r2 spent its calls on `top_changes` and one
  `search_field_groups` per name (D22) and fetched no 2025 series (`incomplete`). The
  rubric expects the unfetched part of the 2025 top 10, which the agent cannot know: an
  instrument issue, recorded as a known limit of the tool layer, not fixed here. For a
  future agent-v2: a rubric that accepts a ranking by a stated proxy, with what was not
  fetched by that proxy.
- Redaction left garbled text twice: a08 r0 (above) and a05 r0, where a range ("between
  800 and 900 thousand bbl/d", no tool value) was removed from a sentence in final mode, so
  with no regeneration ("TUPI ficou quase sempre [não confirmado]").
- R1 flagged 3 drafts: 1 fixed by a regeneration (a03 r2), 2 redacted (a08 r0, a05 r0).
- Token projection: 52 follow-up calls, none below the actual prompt (minimum 41 above).
- Cost: US$ 0.2166 (solver US$ 0.0305, grader US$ 0.1861), US$ 0.0020 solver per answer,
  12.8 s per answer. The cases are the heavy ones, so not comparable with 4.5.1's means.

### 4.5.3 Results: final runs, v3 and agent-v1 on one commit (2026-11-25)

The step planned before going public: both instruments on the same commit, after D22–D27
changed the prompts. These are the numbers in the README. Agent `567640b`, anp-mcp
`a313249`, Haiku 5.5 solver, Sonnet 5.5 grader, dataset v2026.09, 3 repeats. Every failure
was read by hand (4.3) before it was counted.

**v3** (`20261125T000219Z`): 62/63. 0 failures for a wrong number, 0 `ungrounded`.

- q10 r2, the one failure, labelled `wrong_number` by the grader: every number is right
  (+67,374 bbl/d, +209.8%, 32,119 → 99,492). The error is the statement "in absolute volume
  it was behind only Búzios", when the `top_changes` result in the same answer put Mero
  (+97,346) ahead; the grader also rejected "the highest percentage rise but Wahoo", which
  a top 10 by absolute change cannot support. A statement R1 does not check (D23 asks for
  it in the prompt only).
- R1 flagged 1 draft (q11 r2, three means computed by the model); the regeneration fixed it
  with `calculate`, and its text has no sentence about the note (D27; the only regenerated
  text of the two runs, so one observation). No redaction.
- Calculator in 17 answers, 29 calls, none rejected. No call refused by a cap, no token
  limit. Tokens 11,439 / 20,142 / 29,558 (mean / p95 / max), against 12,302 / 27,080 /
  31,878 at `63c01b9`; solver US$ 0.00068 per answer (+11% on the harness run), 5.4 s.
- Cost US$ 0.5413 (solver US$ 0.0429, grader US$ 0.4983).

**agent-v1** (`20261125T002230Z`, 14 cases): 35/42. By case: a01 2/3, a02 1/3, a03 2/3,
a05 1/3, a10 2/3, every other case 3/3. The 7 failures, by hand; none is a wrong number:

- **a02 r0, r2 (`incomplete`): a regression from D22.** All three repeats ran
  `top_changes(n=5)`, then `search_field_groups` on each of the 5 names it returned, used
  the 6 calls and had the 5 `field_group_production` lookups refused. At `fca876e`, before
  D22, all three ran `top_changes` and the 5 lookups and passed. r1 did the same as r0 and
  r2 and passed: the rubric accepts saying which fields were not checked, and the grader
  applied that to r1 only. Proposed for the next version: search only names that come from
  the question; names copied from a tool result are already exact.
- **a01 r0 (`incomplete`): a calculator defect.** `ratio` with `unit: "%"` was rejected
  four times: `unit` is read as the unit of the inputs, so 1,197,657 was looked for as a
  percentage, and the message said it was not in the tool results. All 7 calculator
  rejections of the run are this (a01 r0 ×4; a04 r2 ×3, which passed on the absolute
  difference alone, with no percentage).
  Proposed: take `%` as the result's unit on `ratio` and `percent_change`, or name the unit
  mismatch in the error.
- **a03 r0 (`incomplete`): a regeneration spent on a refused call.** The first draft
  (`draft_before_regen`) was right and complete; its "287 mil bbl/d" (837,532 − 550,143 =
  287,389) was computed by the model after `annual_volume` ×2 and `mean` ×2 had used the 4
  calculator calls. The regeneration asked for `calculate(difference)`, refused by the
  cap; the next LLM call did not fit in the 40,000 tokens (33,494 used), so the answer was
  the emergency text (D25). Proposed: when a cap is already reached, send the regeneration
  in final mode, so it ends in text.
- **a05 r1, r2 (`incomplete`):** the known limit (4.5.2): no tool ranks by 2025.
- **a10 r2 (`ungrounded`):** "the data covers oil and gas production"; the server
  instructions say gas is not covered. A statement.

Calculator in 20 answers, 39 calls, 7 rejected (above). R4 reached in 7 answers (a02 ×3,
a03 r0, a05 ×3); token limit in a03 r0 and a05 r2. Token projection: 94 follow-up calls,
none below the actual prompt (minimum 40 above); 116 in v3, minimum 49. Tokens 14,290 /
25,717 / 33,494; solver US$ 0.00102 per answer, 7.7 s. Cost US$ 0.4107 (solver US$ 0.0428,
grader US$ 0.3679).

The three defects are left in this version on purpose, so that the published results
measure the published code; they are listed in the README's known limits.

Future idea, not done: the cache breakpoint covers only the tools and the system prompt
(2,515 tokens), so the conversation history is re-sent uncached on every turn. A breakpoint
on the latest message would make the re-sent history a cache read (cheaper); it would not
change the token limit, which counts every kind alike.

### 4.6 Tests (pytest, no network by default)

- Extraction and matching: table-driven cases for pt-BR and en formats, scale words, units,
  ranges, hedges, signs, dates, list markers, numbers from the question.
- Calculator: operations, input-grounding check, cap.
- Graph routing: fake LLM and fake toolbox; budget edges (split turn, final mode, no-room
  deterministic answer), one regeneration then redaction, footer and notices.
- MCP adapter: integration test against the real server subprocess (marked `integration`,
  needs the dataset cache), including version check and a killed-child restart.
- Usage accounting: LangChain raw usage vs direct SDK call (marked `api`, opt-in).
- Submodule pin: `vendor/anp-mcp` at v0.1.0, clean, same commit as `uv.lock`.

## 5. Layout

```text
anp-agent/
  pyproject.toml, uv.lock, .python-version (3.13), Makefile, README.md, LICENSE (MIT)
  docs/design.md
  src/anp_agent/
    config.py        # Settings: provider, model, limits, timeouts
    llm.py           # make_llm(settings): the single provider switch
    mcp_client.py    # AnpToolbox
    state.py         # AgentState, ToolRecord, Claim, GuardrailReport
    graph.py         # nodes, routing, build_graph()
    budget.py        # projections and checks
    calculator.py    # calculate tool
    guardrail/       # extract.py, sources.py, match.py, redact.py
    footer.py        # notices, sources footer, pt/en texts
    prompts.py
    cli.py           # `anp-agent "pergunta"`
  agent_evals/       # replay_guardrail.py, run_agent_evals.py, cases_agent.yaml, ground_truth_agent.py
  vendor/anp-mcp/    # git submodule @ v0.1.0, read-only
  tests/
```

## 6. Risks and decisions

### Risks

1. **R1 false alarms on correct derived numbers** (averages, shares, m³/d, monthly totals),
   leading to redacted answers that were right. Measured by the replay in 4.4 before any
   LLM spend; reduced by the calculator.
2. **Coincidental grounding (known limit)**: R1 has little power on low-precision numbers.
   With many source numbers, a wrong value written with few significant digits often matches
   some other source value: in the replay's probe (section 4.4), 79% of wrong 1-digit values,
   19% of 2-digit and 15% of 3-digit values were grounded by chance, against 1% with 4 or more
   digits (12% overall). Tolerance settings do not change this. Mitigated only in part by
   tight rounding, no run-time search for derivations, and the sign/direction check; the
   eval grader stays the check for these numbers. Stated in the README.
3. **The model may not call the calculator** when it should, so the replay's upper bound is
   optimistic. The live evals measure the real rate (guardrail triggers before
   regeneration).
4. **R1 is numbers only.** Wrong entities, wrong grouping explanations and right number on
   the wrong row pass the guardrail. Two of Haiku's eight v3 failures are of this kind.
   Stated in the README; the grader stays the authority.
5. **pt/en number ambiguity** (`1.056`): handled by accepting either reading, which slightly
   loosens the check.
6. **Instrument drift**: the harness keeps evolving inside anp-mcp; the agent is tied to the
   v0.1.0 copy until the tag is bumped deliberately.
7. **Cost accounting differences** between LangChain and the raw SDK (cache tokens) could make
   cost comparisons with anp-mcp wrong; covered by the usage test.
8. **Frozen prices**: `pricing.yaml` at v0.1.0 cannot be edited here; if Anthropic prices
   change, USD columns become stale until a new anp-mcp tag.
9. **Regeneration cost**: a failing draft costs about one extra full call; bounded by one
   regeneration and the token limit.
10. **Dataset version mismatch** if `ANP_MCP_REVISION` is set differently for the server and
    the ground truth; the runner asserts they are equal.
11. **LangGraph / langchain-anthropic API churn**: versions pinned in `uv.lock`; the LLM is
    reached only through `llm.py`.

### Decisions

| # | Decision |
|---|---|
| D1 | Derived numbers are grounded through a local calculator whose inputs must be grounded. Implemented after the replay (4.4) is reviewed, with and without the calculator. |
| D2 | Calculator cap: 4 calls per question, separate from the 6 anp-mcp calls. |
| D3 | Hard limit of 40,000 tokens per question, all billed token kinds counted alike; USD recorded only. |
| D4 | Hedge words widen tolerance to 2%; a range is grounded by containment only if its width is ≤ 10% of its midpoint. |
| D5 | Bare integers ≤ 12 without unit are exempt; years and months must be grounded; any number in the question counts as grounded. |
| D6 | Redaction replaces only the number span with "[não confirmado]" / "[unconfirmed]"; the footer lists what was removed. |
| D7 | The answer ends with a sources footer built from the state. |
| D8 | The eval instrument is reused through a git submodule at `vendor/anp-mcp`, pinned to tag v0.1.0. |
| D9 | Own thin MCP adapter over the `mcp` SDK; no `langchain-mcp-adapters`. |
| D10 | Regeneration may call tools, within the same caps. |
| D11 | R5 notice when the answer uses any of the dataset's last 3 months. |
| D12 | The agent answers in the language of the question; deterministic texts follow it. |
| D13 | v1 interface: command line only. |
| D14 | Licence: MIT. |
| D15 | Calculator operations also include `multiply` (with a percentage option) and `convert` to barrels in a year. |
| D16 | "1 bbl" in "1 bbl = 0.158987 m³" is part of the conversion constant. |
| D17 | Tool arguments include the defaults of the tool's input schema as the server lists it. |
| D18 | `top_changes` window months are derived deterministically from `latest_month` and `window_months`. |
| D19 | Trailing zeros are written digits by default ("40.000" is exact to the unit); rounding them stays a configurable policy. |
| D20 | No specific rule for a number grounded only by the default of a tool that was not called; regeneration handles it. |
| D21 | Final mode keeps the tool definitions and sends `tool_choice: none`, instead of calling the model with no tools bound. Checked on 2026-11-23 with Haiku 5.5 (2 calls, US$ 0.000325): a request with `tool_use`/`tool_result` blocks and **no** tool definitions is accepted (HTTP 200, `req_011ChLjtHFkW65bxX7PXNnhf`), so dropping the tools was possible after all; `tool_choice: none` with the tools is accepted and the model answered in text, with no `tool_use` block (`req_011ChLjtQKQbk98ccM7uV8xj`). Kept anyway: the "fits only without tools" step would only matter within about 2,000 tokens of the 40,000 limit, and the worst Haiku answer measured was 17,107 tokens; one more code path, which also loses the cache, does not pay for so rare a case. To be revisited if the evals show questions near the limit. |
| D22 | The system prompt asks for `search_field_groups` before every `field_group_production` call (agent-v1 a03 lost 2 of its 6 calls to "Lula" and "Mero", which match no group name). |
| D23 | R3 reinforced in the prompt: nothing about fields, operators, rankings or groups that no tool result of the question states (five failures for such statements across the two runs: v3 q03, q10, q18; agent-v1 a05 r1, a08 r0). Still a prompt rule; R1 does not check statements. |
| D24 | Calculator operation `annual_volume`: a year's monthly bbl/d values with their months, each checked against the source row of that month, summed with each month's days, in barrels; one call for a yearly volume (agent-v1 a12 ran out of calculator calls converting year by year). |
| D25 | The emergency answer is readable and deterministic (section 3): no raw JSON, only the tools' values, then the queries made and left out. |
| D26 | Periods left open, in the prompt: a yearly figure without a year uses the last complete year, current production without a period uses the latest month, and the answer says which. A single "last complete year" rule would have broken v3 q06 and q15, whose rubrics expect the latest month. |
| D27 | The regeneration note ends with "write only the answer for the user; do not mention this note or the numbers removed" (agent-v1 a08 r0 opened its regenerated answer with a sentence about the removed number, which redaction then garbled). The eval records keep the first draft of a regenerated answer (`draft_before_regen`). |
