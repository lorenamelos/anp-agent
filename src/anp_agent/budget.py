"""R4 budget: token accounting, the projection before each LLM call, and the call caps
(design, section 3).

Pure functions over numbers; the graph keeps the counters in its state and calls these.

Tokens: every billed kind (input, output, cache read, cache write) counts alike toward the
hard per-question limit, read from each response's raw `usage` block. Before a call the
projection adds what the call can cost at most: its whole prompt plus `max_output_tokens`.
The prompt is known exactly for what the previous call already sent (all its input kinds
plus its output, which the next call re-sends); only the messages added since are estimated,
on the high side, from their length.
"""

import json
import math
from dataclasses import dataclass
from enum import Enum

from anp_agent.config import Settings

CALCULATOR = "calculate"
# Conservative: tool results are JSON full of digits and punctuation, which take more tokens
# per character than prose. Overestimating only moves a call into final mode earlier.
CHARS_PER_TOKEN = 2.0
# Each message added since the last call also carries fixed tokens (role, tool_result
# wrapper) that its text does not show; the first eval pilot measured up to 66 tokens below
# the actual prompt for two tool results without this margin.
MESSAGE_OVERHEAD = 50


class Mode(Enum):
    NORMAL = "normal"  # a call with tools bound
    FINAL = "final"  # a call without tools: answer with what was fetched
    NONE = "none"  # no LLM call fits: deterministic answer


@dataclass(frozen=True)
class Usage:
    input: int = 0
    output: int = 0
    cache_read: int = 0
    cache_write: int = 0

    @classmethod
    def from_raw(cls, usage: dict | None) -> "Usage":
        """From the raw Anthropic `usage` block (not LangChain's `usage_metadata`)."""
        u = usage or {}
        return cls(u.get("input_tokens") or 0, u.get("output_tokens") or 0,
                   u.get("cache_read_input_tokens") or 0, u.get("cache_creation_input_tokens") or 0)

    @property
    def prompt(self) -> int:
        """Every token the call was sent, cached or not."""
        return self.input + self.cache_read + self.cache_write

    @property
    def total(self) -> int:
        return self.prompt + self.output


def estimate_tokens(content) -> int:
    """High-side token estimate for text or any JSON-serialisable content."""
    text = content if isinstance(content, str) else json.dumps(content, ensure_ascii=False)
    return math.ceil(len(text) / CHARS_PER_TOKEN)


def next_prompt(last: Usage | None, added: int, initial: int) -> int:
    """Tokens the next call will send: the last call's prompt and output plus what was added
    since; before the first call, the estimate of the initial prompt."""
    if last is None:
        return initial
    return last.prompt + last.output + added


def plan_call(settings: Settings, *, tokens_used: int, llm_calls: int, prompt: int,
              tools_tokens: int, final_mode: bool = False) -> Mode:
    """Which LLM call, if any, fits. `prompt` includes the tool definitions, `tools_tokens`
    what a call without them would save (0 in the graph: final mode keeps them, D21)."""
    if llm_calls >= settings.max_llm_calls:
        return Mode.NONE
    room = settings.max_tokens_per_question - tokens_used - settings.max_output_tokens
    last_call = llm_calls + 1 == settings.max_llm_calls
    if prompt <= room and not final_mode and not last_call:
        return Mode.NORMAL
    if prompt - tools_tokens <= room:
        return Mode.FINAL
    return Mode.NONE


@dataclass(frozen=True)
class Admission:
    run: list[bool]  # per requested call, in order
    tool_calls_used: int
    calculator_calls_used: int

    @property
    def refused(self) -> bool:
        return not all(self.run)


def admit(settings: Settings, names: list[str], *, tool_calls_used: int,
          calculator_calls_used: int) -> Admission:
    """Admit the calls of one model turn in order, each kind against its own cap. A refused
    call gets "not executed" and moves the agent to final mode."""
    run, tools, calcs = [], tool_calls_used, calculator_calls_used
    for name in names:
        if name == CALCULATOR:
            ok = calcs < settings.max_calculator_calls
            calcs += ok
        else:
            ok = tools < settings.max_tool_calls
            tools += ok
        run.append(ok)
    return Admission(run, tools, calcs)
