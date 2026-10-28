from __future__ import annotations

import json
from collections.abc import Callable
from pathlib import Path
from typing import Any

import httpx
import pytest

from afbench.common import Hit
from afbench.corpus import Document, write_metadata_db

CORPUS_DIR = Path(__file__).resolve().parents[1] / "data" / "corpus"


def completion(
    content: str | None = None, tool_calls: list[tuple[str, dict[str, Any]]] | None = None
) -> dict[str, Any]:
    """An OpenAI-format chat completion body."""
    message: dict[str, Any] = {"role": "assistant", "content": content}
    if tool_calls:
        message["tool_calls"] = [
            {"id": f"call_{i}", "type": "function", "function": {"name": name, "arguments": json.dumps(args)}}
            for i, (name, args) in enumerate(tool_calls)
        ]
    return {
        "id": "chatcmpl-test",
        "object": "chat.completion",
        "created": 0,
        "model": "test-model",
        "choices": [{"index": 0, "finish_reason": "tool_calls" if tool_calls else "stop", "message": message}],
        "usage": {"prompt_tokens": 100, "completion_tokens": 10, "total_tokens": 110},
    }


class ScriptedBackend:
    """A fake OpenAI-compatible server: replays completions in order and keeps every request."""

    def __init__(self, turns: list[dict[str, Any]]) -> None:
        self.turns = list(turns)
        self.requests: list[dict[str, Any]] = []

    def handler(self, request: httpx.Request) -> httpx.Response:
        self.requests.append(json.loads(request.content))
        body = self.turns.pop(0) if self.turns else completion("I don't know.")
        return httpx.Response(200, json=body)

    def transport(self) -> httpx.MockTransport:
        return httpx.MockTransport(self.handler)


class StubSearcher:
    name = "stub"

    def __init__(self, hits: list[Hit]) -> None:
        self.hits = hits
        self.queries: list[str] = []

    def search(self, query: str, k: int) -> list[Hit]:
        self.queries.append(query)
        return self.hits[:k]


PEP8_HIT = Hit(
    "pep-0008#3",
    "pep-0008",
    "PEP 8: Style Guide for Python Code",
    "Maximum Line Length",
    "Limit all lines to a maximum of 79 characters.",
    0.9,
)


@pytest.fixture
def metadata_db(tmp_path: Path) -> Path:
    db = tmp_path / "metadata.db"
    docs = [
        Document(
            "pep-0008",
            "PEP 8: Style",
            "word " * 100,
            {"PEP": "8", "Title": "Style", "Status": "Active", "Created": "05-Jul-2001", "Author": "A, B"},
        ),
        Document(
            "pep-0572",
            "PEP 572: Walrus",
            "word " * 50,
            {"PEP": "572", "Title": "Walrus", "Status": "Final", "Created": "28-Feb-2018", "Python-Version": "3.8"},
        ),
    ]
    write_metadata_db(docs, db)
    return db


@pytest.fixture
def stub_searcher() -> StubSearcher:
    return StubSearcher([PEP8_HIT])


SearchFactory = Callable[[], StubSearcher]
