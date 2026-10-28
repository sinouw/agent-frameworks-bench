"""LangChain + LangGraph implementation: `langchain-qdrant` hybrid retrieval and the
prebuilt tool-calling agent graph (`langchain.agents.create_agent`, LangGraph underneath)."""

from __future__ import annotations

import uuid
from typing import Any

from langchain.agents import create_agent
from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings
from langchain_core.language_models import BaseChatModel
from langchain_core.tools import StructuredTool
from langchain_qdrant import QdrantVectorStore, RetrievalMode, SparseEmbeddings, SparseVector
from langgraph.errors import GraphRecursionError
from qdrant_client import QdrantClient, models

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
from afbench.embeddings import DenseEmbedder, SparseEmbedder


class LangChainDense(Embeddings):
    """The shared embedder behind LangChain's interface (replaces `FastEmbedEmbeddings`,
    which lives in the sunset `langchain-community` package)."""

    def __init__(self, embedder: DenseEmbedder) -> None:
        self.embedder = embedder

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return self.embedder.embed_documents(texts)

    def embed_query(self, text: str) -> list[float]:
        return self.embedder.embed_query(text)


class LangChainSparse(SparseEmbeddings):
    """Same behaviour as `langchain_qdrant.FastEmbedSparse`: BM25 queries use `query_embed`."""

    def __init__(self, embedder: SparseEmbedder) -> None:
        self.embedder = embedder

    def embed_documents(self, texts: list[str]) -> list[SparseVector]:
        return [SparseVector(indices=i, values=v) for i, v in self.embedder.embed_documents(texts)]

    def embed_query(self, text: str) -> SparseVector:
        indices, values = self.embedder.embed_query(text)
        return SparseVector(indices=indices, values=values)


def build_index(
    chunks: list[Chunk],
    embedding: Embeddings,
    sparse: SparseEmbeddings,
    collection: str,
    idf: bool,
    **client_options: Any,
) -> QdrantVectorStore:
    """`idf=False` is the library default: the BM25 sparse vector is created without
    Qdrant's IDF modifier, which the `Qdrant/bm25` model expects."""
    documents = [Document(page_content=c.embedding_text(), metadata=c.payload()) for c in chunks]
    return QdrantVectorStore.from_documents(
        documents,
        embedding=embedding,
        sparse_embedding=sparse,
        retrieval_mode=RetrievalMode.HYBRID,
        collection_name=collection,
        ids=[str(uuid.uuid5(uuid.NAMESPACE_URL, c.chunk_id)) for c in chunks],
        sparse_vector_params={"modifier": models.Modifier.IDF} if idf else None,
        force_recreate=True,
        **client_options,
    )


def open_store(
    client: QdrantClient, collection: str, embedding: Embeddings, sparse: SparseEmbeddings
) -> QdrantVectorStore:
    return QdrantVectorStore(
        client, collection, embedding=embedding, sparse_embedding=sparse, retrieval_mode=RetrievalMode.HYBRID
    )


class LangChainSearcher:
    """`candidates=None` is the library default: Qdrant prefetches only `k` dense and `k`
    sparse candidates before RRF. Asking for `candidates` results and truncating to `k`
    gives the same candidate pool as the baseline."""

    def __init__(self, store: QdrantVectorStore, candidates: int | None = None, name: str = "langgraph") -> None:
        self.store = store
        self.candidates = candidates
        self.name = name

    def search(self, query: str, k: int) -> list[Hit]:
        fetch = max(k, self.candidates or 0)
        results = self.store.similarity_search_with_score(query, k=fetch)
        return [hit_from_payload(doc.metadata, score) for doc, score in results[:k]]


class LangGraphAgent:
    name = "langgraph"

    def __init__(self, llm: BaseChatModel, searcher: Any, sql: SqlTool, max_steps: int = 5, top_k: int = 5) -> None:
        self.llm, self.searcher, self.sql = llm, searcher, sql
        self.max_steps, self.top_k = max_steps, top_k
        self.sql_description = sql_description(sql.schema())

    async def answer(self, question: str) -> AgentResult:
        box = ToolBox(self.searcher, self.sql, self.top_k)
        tools = [
            StructuredTool.from_function(
                box.search_docs, name="search_docs", description=SEARCH_DESCRIPTION, args_schema=SearchArgs
            ),
            StructuredTool.from_function(
                box.sql_query, name="sql_query", description=self.sql_description, args_schema=SqlArgs
            ),
            StructuredTool.from_function(
                box.calculator, name="calculator", description=CALCULATOR_DESCRIPTION, args_schema=CalculatorArgs
            ),
        ]
        agent = create_agent(self.llm, tools, system_prompt=SYSTEM_PROMPT)
        try:
            # Each model call and each tool round is one graph step.
            state = await agent.ainvoke(
                {"messages": [{"role": "user", "content": question}]},
                config={"recursion_limit": 2 * self.max_steps},
            )
        except GraphRecursionError:
            return box.result("", error="no answer within the step budget")
        except Exception as exc:  # provider or tool failure: report it as this question's error
            return box.result("", error=f"{type(exc).__name__}: {exc}"[:500])
        return box.result(state["messages"][-1].text)
