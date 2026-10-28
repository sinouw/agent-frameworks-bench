FROM python:3.12-slim

COPY --from=ghcr.io/astral-sh/uv:0.9 /uv /usr/local/bin/uv
WORKDIR /app
ENV UV_COMPILE_BYTECODE=1 UV_LINK_MODE=copy

COPY pyproject.toml uv.lock README.md ./
RUN uv sync --frozen --no-dev --no-install-project
COPY src ./src
COPY data/corpus ./data/corpus
COPY evals ./evals
RUN uv sync --frozen --no-dev

# Build the indexes at image build time (downloads the embedding models, no API key needed).
RUN uv run --no-sync afbench index

ENTRYPOINT ["uv", "run", "--no-sync", "afbench"]
CMD ["retrieval"]
