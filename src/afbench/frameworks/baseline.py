"""Plain-SDK baseline: `qdrant-client` for hybrid retrieval and the `openai` client for a
hand-written tool-calling loop. This is the design of the reference agent
(github.com/sinouw/rag-agent-evals), trimmed to what the benchmark needs."""

from __future__ import annotations

import json
import uuid
from typing import Any

from openai import AsyncOpenAI
from pydantic import BaseModel, ValidationError
from qdrant_client import QdrantClient, models

from afbench.common import (
    CALCULATOR_DESCRIPTION,
    FINAL_STEP_NOTE,
    SEARCH_DESCRIPTION,
    SYSTEM_PROMPT,
    AgentResult,
    CalculatorArgs,
    Hit,
    SearchArgs,
    SqlArgs,
    SqlTool,
    ToolBox,
    hit_from_payload,
    sql_description,
)
from afbench.corpus import Chunk
from afbench.embeddings import DenseEmbedder, SparseEmbedder

DENSE, SPARSE = "dense", "bm25"


def build_index(
    client: QdrantClient, collection: str, chunks: list[Chunk], dense: DenseEmbedder, sparse: SparseEmbedder
) -> None:
    if client.collection_exists(collection):
        client.delete_collection(collection)
    client.create_collection(
        collection,
        vectors_config={DENSE: models.VectorParams(size=dense.dim, distance=models.Distance.COSINE)},
        sparse_vectors_config={SPARSE: models.SparseVectorParams(modifier=models.Modifier.IDF)},
    )
    texts = [c.embedding_text() for c in chunks]
    vectors = dense.embed_documents(texts)
    sparse_vectors = sparse.embed_documents(texts)
    points = [
        models.PointStruct(
            id=str(uuid.uuid5(uuid.NAMESPACE_URL, c.chunk_id)),
            vector={DENSE: v, SPARSE: models.SparseVector(indices=s[0], values=s[1])},
            payload=c.payload(),
        )
        for c, v, s in zip(chunks, vectors, sparse_vectors, strict=True)
    ]
    for i in range(0, len(points), 256):
        client.upsert(collection, points=points[i : i + 256])


class BaselineSearcher:
    """Dense + BM25 candidates fused with Reciprocal Rank Fusion inside Qdrant."""

    def __init__(
        self,
        client: QdrantClient,
        collection: str,
        dense: DenseEmbedder,
        sparse: SparseEmbedder,
        candidates: int = 20,
        name: str = "baseline",
    ) -> None:
        self.client, self.collection = client, collection
        self.dense, self.sparse = dense, sparse
        self.candidates = candidates
        self.name = name

    def search(self, query: str, k: int) -> list[Hit]:
        indices, values = self.sparse.embed_query(query)
        fetch = max(k, self.candidates)
        result = self.client.query_points(
            self.collection,
            prefetch=[
                models.Prefetch(query=self.dense.embed_query(query), using=DENSE, limit=fetch),
                models.Prefetch(query=models.SparseVector(indices=indices, values=values), using=SPARSE, limit=fetch),
            ],
            query=models.FusionQuery(fusion=models.Fusion.RRF),
            limit=k,
            with_payload=True,
        )
        return [hit_from_payload(p.payload or {}, p.score) for p in result.points]


def _schema(name: str, description: str, args: type[BaseModel]) -> dict[str, Any]:
    return {
        "type": "function",
        "function": {"name": name, "description": description, "parameters": args.model_json_schema()},
    }


class BaselineAgent:
    name = "baseline"

    def __init__(
        self, llm: AsyncOpenAI, model: str, searcher: Any, sql: SqlTool, max_steps: int = 5, top_k: int = 5
    ) -> None:
        self.llm, self.model = llm, model
        self.searcher, self.sql = searcher, sql
        self.max_steps, self.top_k = max_steps, top_k
        self.tools = [
            _schema("search_docs", SEARCH_DESCRIPTION, SearchArgs),
            _schema("sql_query", sql_description(sql.schema()), SqlArgs),
            _schema("calculator", CALCULATOR_DESCRIPTION, CalculatorArgs),
        ]

    async def answer(self, question: str) -> AgentResult:
        box = ToolBox(self.searcher, self.sql, self.top_k)
        messages: list[dict[str, Any]] = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": question},
        ]
        try:
            for step in range(1, self.max_steps + 1):
                last = step == self.max_steps
                if last:
                    messages.append({"role": "user", "content": FINAL_STEP_NOTE})
                kwargs: dict[str, Any] = {"model": self.model, "messages": messages, "temperature": 0.0}
                if not last:
                    kwargs["tools"] = self.tools
                response = await self.llm.chat.completions.create(**kwargs)
                message = response.choices[0].message
                if not message.tool_calls:
                    return box.result(message.content or "")
                messages.append(
                    {
                        "role": "assistant",
                        "content": message.content,
                        "tool_calls": [c.model_dump(include={"id", "type", "function"}) for c in message.tool_calls],
                    }
                )
                for call in message.tool_calls:
                    if call.type != "function":
                        continue
                    output = run_tool(box, call.function.name, call.function.arguments)
                    messages.append({"role": "tool", "tool_call_id": call.id, "content": output})
        except Exception as exc:  # provider or tool failure: report it as this question's error
            return box.result("", error=f"{type(exc).__name__}: {exc}"[:500])
        return box.result("", error="no answer within the step budget")


def run_tool(box: ToolBox, name: str, raw_arguments: str) -> str:
    try:
        arguments = json.loads(raw_arguments or "{}")
        if name == "search_docs":
            search = SearchArgs.model_validate(arguments)
            return box.search_docs(search.query, search.k)
        if name == "sql_query":
            return box.sql_query(SqlArgs.model_validate(arguments).sql)
        if name == "calculator":
            return box.calculator(CalculatorArgs.model_validate(arguments).expression)
    except (ValueError, ValidationError) as exc:
        return f"Error: invalid arguments for {name}: {exc}"
    return f"Error: unknown tool {name!r}."
