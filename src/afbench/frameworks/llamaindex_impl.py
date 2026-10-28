"""LlamaIndex implementation: `VectorStoreIndex` over `llama-index-vector-stores-qdrant`
in hybrid mode, and the workflow-based `FunctionAgent`."""

from __future__ import annotations

import uuid
from collections.abc import Callable
from typing import Any, cast

from llama_index.core import StorageContext, VectorStoreIndex
from llama_index.core.agent.workflow import AgentOutput, FunctionAgent
from llama_index.core.base.embeddings.base import BaseEmbedding
from llama_index.core.llms.function_calling import FunctionCallingLLM
from llama_index.core.schema import TextNode
from llama_index.core.tools import BaseTool, FunctionTool
from llama_index.core.vector_stores.types import VectorStoreQueryResult
from llama_index.vector_stores.qdrant import QdrantVectorStore
from pydantic import PrivateAttr
from qdrant_client import QdrantClient, models
from qdrant_client.hybrid.fusion import DEFAULT_RANKING_CONSTANT_K

from afbench.common import (
    CALCULATOR_DESCRIPTION,
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
from afbench.embeddings import DenseEmbedder, SparseEmbedder, SparseFn


class LlamaIndexDense(BaseEmbedding):
    """The shared embedder behind LlamaIndex's interface."""

    _embedder: Any = PrivateAttr()

    def __init__(self, embedder: DenseEmbedder, **kwargs: Any) -> None:
        super().__init__(model_name="shared", embed_batch_size=256, **kwargs)
        self._embedder = embedder

    def _get_text_embedding(self, text: str) -> list[float]:
        return list(self._embedder.embed_documents([text])[0])

    def _get_text_embeddings(self, texts: list[str]) -> list[list[float]]:
        return list(self._embedder.embed_documents(texts))

    def _get_query_embedding(self, query: str) -> list[float]:
        return list(self._embedder.embed_query(query))

    async def _aget_query_embedding(self, query: str) -> list[float]:
        return self._get_query_embedding(query)


def sparse_fns(embedder: SparseEmbedder, query_side: bool) -> tuple[SparseFn, SparseFn]:
    """(document fn, query fn) in the shape the Qdrant store expects. `query_side=False`
    reproduces the library's built-in fastembed encoder, which encodes queries with the
    document-side `embed` instead of `query_embed`."""

    def docs(texts: list[str]) -> tuple[list[list[int]], list[list[float]]]:
        vectors = embedder.embed_documents(texts)
        return [v[0] for v in vectors], [v[1] for v in vectors]

    def queries(texts: list[str]) -> tuple[list[list[int]], list[list[float]]]:
        encode = embedder.embed_query if query_side else embedder.embed_document_side
        vectors = [encode(t) for t in texts]
        return [v[0] for v in vectors], [v[1] for v in vectors]

    return docs, queries


def vector_store(
    client: QdrantClient,
    collection: str,
    sparse_doc_fn: SparseFn,
    sparse_query_fn: SparseFn,
    fusion: Callable[..., VectorStoreQueryResult] | None = None,
) -> QdrantVectorStore:
    """`fusion=None` is the library default: relative score fusion. The IDF modifier is what
    the store sets on its own for `fastembed_sparse_model="Qdrant/bm25"`; with custom sparse
    functions it would silently drop it, so it is set explicitly."""
    return QdrantVectorStore(
        collection,
        client=client,
        enable_hybrid=True,
        sparse_config=models.SparseVectorParams(index=models.SparseIndexParams(), modifier=models.Modifier.IDF),
        sparse_doc_fn=sparse_doc_fn,
        sparse_query_fn=sparse_query_fn,
        hybrid_fusion_fn=fusion,
    )


def build_index(chunks: list[Chunk], store: QdrantVectorStore, embed_model: BaseEmbedding) -> VectorStoreIndex:
    keys = list(chunks[0].payload()) if chunks else []
    nodes = [
        TextNode(
            id_=str(uuid.uuid5(uuid.NAMESPACE_URL, c.chunk_id)),
            text=c.embedding_text(),
            metadata=c.payload(),
            # Embed exactly the same text as the other implementations, without metadata lines.
            excluded_embed_metadata_keys=keys,
            excluded_llm_metadata_keys=keys,
        )
        for c in chunks
    ]
    storage = StorageContext.from_defaults(vector_store=store)
    return VectorStoreIndex(nodes, storage_context=storage, embed_model=embed_model)


def rrf_fusion(
    dense_result: VectorStoreQueryResult, sparse_result: VectorStoreQueryResult, alpha: float = 0.5, top_k: int = 2
) -> VectorStoreQueryResult:
    """Reciprocal Rank Fusion with Qdrant's ranking constant, so fusion matches the baseline.
    (The library default is relative score fusion.)"""
    scores: dict[str, float] = {}
    nodes = {}
    for result in (dense_result, sparse_result):
        for rank, node in enumerate(result.nodes or []):
            scores[node.node_id] = scores.get(node.node_id, 0.0) + 1.0 / (rank + DEFAULT_RANKING_CONSTANT_K)
            nodes[node.node_id] = node
    ranked = sorted(scores, key=lambda node_id: scores[node_id], reverse=True)[:top_k]
    return VectorStoreQueryResult(
        nodes=[nodes[i] for i in ranked], similarities=[scores[i] for i in ranked], ids=ranked
    )


class LlamaIndexSearcher:
    """`candidates=None` is the library default: `k` dense and `k` sparse candidates."""

    def __init__(self, index: VectorStoreIndex, candidates: int | None = None, name: str = "llamaindex") -> None:
        self.index = index
        self.candidates = candidates
        self.name = name

    def search(self, query: str, k: int) -> list[Hit]:
        fetch = max(k, self.candidates or 0)
        retriever = self.index.as_retriever(
            similarity_top_k=fetch, sparse_top_k=fetch, hybrid_top_k=k, vector_store_query_mode="hybrid"
        )
        return [hit_from_payload(n.node.metadata, n.score or 0.0) for n in retriever.retrieve(query)][:k]


class LlamaIndexAgent:
    name = "llamaindex"

    def __init__(
        self, llm: FunctionCallingLLM, searcher: Any, sql: SqlTool, max_steps: int = 5, top_k: int = 5
    ) -> None:
        self.llm, self.searcher, self.sql = llm, searcher, sql
        self.max_steps, self.top_k = max_steps, top_k
        self.sql_description = sql_description(sql.schema())

    async def answer(self, question: str) -> AgentResult:
        box = ToolBox(self.searcher, self.sql, self.top_k)
        tools: list[BaseTool | Callable[..., Any]] = [
            FunctionTool.from_defaults(
                box.search_docs, name="search_docs", description=SEARCH_DESCRIPTION, fn_schema=SearchArgs
            ),
            FunctionTool.from_defaults(
                box.sql_query, name="sql_query", description=self.sql_description, fn_schema=SqlArgs
            ),
            FunctionTool.from_defaults(
                box.calculator, name="calculator", description=CALCULATOR_DESCRIPTION, fn_schema=CalculatorArgs
            ),
        ]
        agent = FunctionAgent(
            tools=tools,
            llm=self.llm,
            system_prompt=SYSTEM_PROMPT,
            streaming=False,
            # On the last step, ask for an answer from what was gathered instead of raising.
            early_stopping_method="generate",
        )
        try:
            output = cast(AgentOutput, await agent.run(user_msg=question, max_iterations=self.max_steps))
        except Exception as exc:  # provider or tool failure: report it as this question's error
            return box.result("", error=f"{type(exc).__name__}: {exc}"[:500])
        return box.result(str(output.response.content or ""))
