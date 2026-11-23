import pytest

from anp_agent.config import ConfigError, Settings


def test_defaults_match_the_design():
    s = Settings.from_env({})
    assert (s.provider, s.model) == ("anthropic", "claude-haiku-5-5")
    assert (s.max_tool_calls, s.max_calculator_calls) == (6, 4)
    assert (s.max_tokens_per_question, s.max_output_tokens, s.max_llm_calls) == (40_000, 2_048, 10)
    assert (s.tool_timeout_s, s.question_timeout_s, s.server_init_timeout_s) == (20, 120, 30)


def test_from_env():
    s = Settings.from_env({"ANP_AGENT_MODEL": "claude-sonnet-5-5",
                           "ANP_AGENT_MAX_TOKENS_PER_QUESTION": "30_000",
                           "ANP_AGENT_TOOL_TIMEOUT_S": "7.5", "ANP_AGENT_MAX_TOOL_CALLS": "",
                           "OTHER": "x"})
    assert (s.model, s.max_tokens_per_question, s.tool_timeout_s, s.max_tool_calls) == \
        ("claude-sonnet-5-5", 30_000, 7.5, 6)


@pytest.mark.parametrize("env, message", [
    ({"ANP_AGENT_MAX_TOOL_CALLS": "six"}, "must be a number"),
    ({"ANP_AGENT_MAX_TOOL_CALLS": "0"}, "must be positive"),
    ({"ANP_AGENT_PROVIDER": "openai"}, "provider must be one of"),
    ({"ANP_AGENT_MAX_OUTPUT_TOKENS": "40000"}, "below max_tokens_per_question"),
])
def test_bad_settings(env, message):
    with pytest.raises(ConfigError, match=message):
        Settings.from_env(env)
