# anp-agent

A [LangGraph](https://github.com/langchain-ai/langgraph) agent that answers questions about
Brazilian oil production using the [anp-mcp](https://github.com/lorenamelos/anp-mcp) server
(pinned to `v0.1.0`) as its only data source, with a check that every number in an answer
came from a tool call.

Work in progress. Design: [docs/design.md](docs/design.md).

## Development

Requires [uv](https://docs.astral.sh/uv/).

```bash
git clone --recurse-submodules https://github.com/lorenamelos/anp-agent.git
cd anp-agent
make setup   # submodule, virtual environment, dependencies
make test
make lint
```

`vendor/anp-mcp` is a read-only git submodule at tag `v0.1.0`, used for the eval instrument.

## Licence

MIT.
