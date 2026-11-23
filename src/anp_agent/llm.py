"""The single provider switch, and usage accounting (design, sections 0 and 3).

`make_llm(settings)` is the only place that names a provider class; the graph binds the
anp-mcp tool schemas (as the server lists them) and the calculator schema to it, or calls it
without tools in final mode. The system prompt carries a cache breakpoint, which covers the
tool definitions before it, as in the eval harness.

Usage comes from the raw Anthropic `usage` block that LangChain keeps in
`response_metadata["usage"]`, not from `usage_metadata`, which adds the cache tokens into
`input_tokens`. USD uses the harness's prices (anp-mcp `evals/pricing.yaml` at v0.1.0) and
the harness's formula: each token kind at its own price. It is recorded, never a limit.
"""

from pathlib import Path

import yaml
from langchain_core.language_models import BaseChatModel
from langchain_core.messages import AIMessage, SystemMessage

from anp_agent.budget import Usage
from anp_agent.config import Settings

PRICING = Path(__file__).resolve().parents[2] / "vendor" / "anp-mcp" / "evals" / "pricing.yaml"
CACHE = {"type": "ephemeral"}


def make_llm(settings: Settings) -> BaseChatModel:
    if settings.provider == "anthropic":
        from langchain_anthropic import ChatAnthropic

        # The API key is read by the SDK from ANTHROPIC_API_KEY. The SDK retries 429/5xx twice.
        return ChatAnthropic(model=settings.model, max_tokens=settings.max_output_tokens,
                             max_retries=2, timeout=settings.question_timeout_s)
    raise ValueError(f"unknown provider {settings.provider!r}")  # Settings validates it


def system_message(text: str) -> SystemMessage:
    """The system prompt, with a cache breakpoint that also covers the tool definitions."""
    return SystemMessage(content=[{"type": "text", "text": text, "cache_control": CACHE}])


def usage_of(message: AIMessage) -> Usage:
    return Usage.from_raw(message.response_metadata.get("usage"))


def load_prices(path: Path = PRICING) -> dict[str, dict[str, float]]:
    """USD per million tokens, by model; empty if the vendored file is not there."""
    return yaml.safe_load(path.read_text()) if path.exists() else {}


def usd(usage: Usage, model: str, prices: dict[str, dict[str, float]]) -> float | None:
    """Cost of one call, or None for a model without a price."""
    p = prices.get(model)
    if p is None:
        return None
    return (usage.input * p["input"] + usage.output * p["output"]
            + usage.cache_read * p["cache_read"] + usage.cache_write * p["cache_write"]) / 1e6
