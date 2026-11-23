"""Settings: provider, model, limits and timeouts (design, section 3).

The single configuration point. Read from environment variables (`ANP_AGENT_<FIELD>`, e.g.
`ANP_AGENT_MAX_TOOL_CALLS=4`) with the defaults below, passed to the graph through the run
context, never into the prompt. The API key is read by the provider SDK from its own
variable (`ANTHROPIC_API_KEY`), not stored here.
"""

import os
from collections.abc import Mapping
from dataclasses import dataclass, fields

PREFIX = "ANP_AGENT_"
PROVIDERS = ("anthropic",)


class ConfigError(ValueError):
    """A setting is missing, malformed or inconsistent."""


@dataclass(frozen=True)
class Settings:
    provider: str = "anthropic"
    model: str = "claude-haiku-5-5"
    # R4 budget, per question
    max_tool_calls: int = 6  # anp-mcp calls
    max_calculator_calls: int = 4
    max_tokens_per_question: int = 40_000  # hard; input, output, cache read, cache write alike
    max_output_tokens: int = 2_048  # per LLM call, as in the eval harness
    max_llm_calls: int = 10  # loop backstop
    recursion_limit: int = 25  # LangGraph steps, a backstop behind the limits above
    # timeouts, seconds
    tool_timeout_s: float = 20.0
    question_timeout_s: float = 120.0
    server_init_timeout_s: float = 30.0

    def __post_init__(self):
        if self.provider not in PROVIDERS:
            raise ConfigError(f"provider must be one of {', '.join(PROVIDERS)}")
        if not self.model:
            raise ConfigError("model must not be empty")
        for f in fields(self):
            value = getattr(self, f.name)
            if f.type in (int, float) and not value > 0:
                raise ConfigError(f"{f.name} must be positive, got {value}")
        if self.max_output_tokens >= self.max_tokens_per_question:
            raise ConfigError("max_output_tokens must be below max_tokens_per_question")

    @classmethod
    def from_env(cls, env: Mapping[str, str] = os.environ) -> "Settings":
        values = {}
        for f in fields(cls):
            raw = env.get(PREFIX + f.name.upper())
            if raw is None or raw == "":
                continue
            try:
                values[f.name] = f.type(raw.replace("_", "")) if f.type in (int, float) else raw
            except ValueError:
                raise ConfigError(
                    f"{PREFIX + f.name.upper()} must be a number, got {raw!r}") from None
        return cls(**values)
