# anp-agent

A [LangGraph](https://github.com/langchain-ai/langgraph) agent that answers questions about
Brazilian oil production using the [anp-mcp](https://github.com/lorenamelos/anp-mcp) server
(pinned to `v0.1.0`) as its only data source, with a check that every number in an answer
came from a tool call.

Work in progress. Design: [docs/design.md](docs/design.md).

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
