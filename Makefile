.PHONY: install check lint typecheck test index retrieval e2e report bench docker

install:
	uv sync

lint:
	uv run ruff check src tests
	uv run ruff format --check src tests

typecheck:
	uv run mypy

test:
	uv run pytest -q

# The full local check: what CI runs.
check: lint typecheck test

index:
	uv run afbench index

retrieval: index
	uv run afbench retrieval

e2e:
	uv run afbench e2e

report:
	uv run afbench report

# One command: index, retrieval eval, end-to-end eval on Groq, report.
bench: retrieval e2e report

docker:
	docker build -t agent-frameworks-bench .
