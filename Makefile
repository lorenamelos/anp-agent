.PHONY: setup test lint

setup:
	git submodule update --init && uv venv --prompt anp-agent && uv sync

test:
	uv run pytest -q

lint:
	uv run ruff check .
