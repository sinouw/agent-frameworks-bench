"""Wiring for real runs: builds each framework's index with its own Qdrant + fastembed
integration, opens the retrieval variants compared in the retrieval eval, and creates
the three agents on one metered HTTP transport.

Each framework gets its own embedded Qdrant directory under `data/index/`, because an
embedded Qdrant directory can only be opened by one client at a time.
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass
from typing import Any

from openai import AsyncOpenAI
from qdrant_client import QdrantClient

from afbench.common import QAAgent, Searcher, SqlTool
from afbench.config import Settings
from afbench.corpus import build_chunks, load_corpus, write_metadata_db
from afbench.embeddings import FastEmbedDense, FastEmbedSparse
from afbench.frameworks import baseline, langgraph_impl, llamaindex_impl
from afbench.transport import Meter, metered_client

FRAMEWORKS = ("baseline", "langgraph", "llamaindex")

# The variants used by the end-to-end agents: identical candidate pool, fusion and BM25
# weighting across frameworks. The defaults are measured too, in the retrieval eval.
MATCHED = {
    "baseline": "baseline",
    "langgraph": "langchain: IDF + 20 candidates",
    "llamaindex": "llamaindex: 20 candidates + BM25 query encoding + RRF",
}


def build_indexes(settings: Settings) -> dict[str, Any]:
    from langchain_community.embeddings import FastEmbedEmbeddings
    from langchain_qdrant import FastEmbedSparse as LCFastEmbedSparse
    from llama_index.embeddings.fastembed import FastEmbedEmbedding

    docs = load_corpus(settings.corpus_dir)
    chunks = build_chunks(docs)
    write_metadata_db(docs, settings.sqlite_path)
    stats: dict[str, Any] = {"documents": len(docs), "chunks": len(chunks), "seconds": {}}

    def timed(name: str, fn: Any) -> None:
        started = time.perf_counter()
        fn()
        stats["seconds"][name] = round(time.perf_counter() - started, 1)
        print(f"indexed {name} in {stats['seconds'][name]}s")

    def run_baseline() -> None:
        client = QdrantClient(path=str(settings.index_dir / "baseline"))
        dense, sparse = FastEmbedDense(settings.embedding_model), FastEmbedSparse(settings.sparse_model)
        baseline.build_index(client, "docs", chunks, dense, sparse)
        client.close()

    def run_langchain(collection: str, idf: bool) -> None:
        store = langgraph_impl.build_index(
            chunks,
            FastEmbedEmbeddings(model_name=settings.embedding_model),
            LCFastEmbedSparse(model_name=settings.sparse_model),
            collection,
            idf=idf,
            path=str(settings.index_dir / "langchain"),
        )
        store.client.close()

    def run_llamaindex() -> None:
        client = QdrantClient(path=str(settings.index_dir / "llamaindex"))
        store = llamaindex_impl.vector_store(client, "docs", sparse_model=settings.sparse_model)
        llamaindex_impl.build_index(chunks, store, FastEmbedEmbedding(model_name=settings.embedding_model))
        client.close()

    settings.index_dir.mkdir(parents=True, exist_ok=True)
    timed("baseline", run_baseline)
    timed("langchain", lambda: run_langchain("docs", idf=False))
    timed("langchain (IDF collection)", lambda: run_langchain("docs_idf", idf=True))
    timed("llamaindex", run_llamaindex)
    return stats


def open_searchers(settings: Settings) -> dict[str, Searcher]:
    """Every retrieval variant, keyed by a readable name."""
    from langchain_community.embeddings import FastEmbedEmbeddings
    from langchain_qdrant import FastEmbedSparse as LCFastEmbedSparse
    from llama_index.core import VectorStoreIndex
    from llama_index.embeddings.fastembed import FastEmbedEmbedding

    index_dir = settings.index_dir
    if not (index_dir / "llamaindex").exists():
        raise RuntimeError("No index yet: run `uv run afbench index` first.")

    bl_client = QdrantClient(path=str(index_dir / "baseline"))
    dense, sparse = FastEmbedDense(settings.embedding_model), FastEmbedSparse(settings.sparse_model)

    lc_client = QdrantClient(path=str(index_dir / "langchain"))
    lc_dense = FastEmbedEmbeddings(model_name=settings.embedding_model)
    lc_sparse = LCFastEmbedSparse(model_name=settings.sparse_model)
    lc_default = langgraph_impl.open_store(lc_client, "docs", lc_dense, lc_sparse)
    lc_idf = langgraph_impl.open_store(lc_client, "docs_idf", lc_dense, lc_sparse)

    li_client = QdrantClient(path=str(index_dir / "llamaindex"))
    li_embed = FastEmbedEmbedding(model_name=settings.embedding_model)

    def query_fn(texts: list[str]) -> tuple[list[list[int]], list[list[float]]]:
        vectors = [sparse.embed_query(t) for t in texts]
        return [v[0] for v in vectors], [v[1] for v in vectors]

    li_default = VectorStoreIndex.from_vector_store(
        llamaindex_impl.vector_store(li_client, "docs", sparse_model=settings.sparse_model), embed_model=li_embed
    )
    li_matched = VectorStoreIndex.from_vector_store(
        llamaindex_impl.vector_store(
            li_client,
            "docs",
            sparse_model=settings.sparse_model,
            sparse_query_fn=query_fn,
            fusion=llamaindex_impl.rrf_fusion,
        ),
        embed_model=li_embed,
    )
    li_query_only = VectorStoreIndex.from_vector_store(
        llamaindex_impl.vector_store(li_client, "docs", sparse_model=settings.sparse_model, sparse_query_fn=query_fn),
        embed_model=li_embed,
    )
    n = settings.candidates
    variants: list[Searcher] = [
        baseline.BaselineSearcher(bl_client, "docs", dense, sparse, candidates=n, name="baseline"),
        langgraph_impl.LangChainSearcher(lc_default, None, name="langchain: defaults"),
        langgraph_impl.LangChainSearcher(lc_idf, None, name="langchain: IDF"),
        langgraph_impl.LangChainSearcher(lc_idf, n, name=MATCHED["langgraph"]),
        llamaindex_impl.LlamaIndexSearcher(li_default, None, name="llamaindex: defaults"),
        llamaindex_impl.LlamaIndexSearcher(li_default, n, name="llamaindex: 20 candidates"),
        llamaindex_impl.LlamaIndexSearcher(li_query_only, n, name="llamaindex: 20 candidates + BM25 query encoding"),
        llamaindex_impl.LlamaIndexSearcher(li_matched, n, name=MATCHED["llamaindex"]),
    ]
    return {s.name: s for s in variants}


@dataclass
class Bench:
    meter: Meter
    agents: dict[str, QAAgent]
    judge: AsyncOpenAI


def build_bench(settings: Settings, searchers: dict[str, Searcher], frameworks: tuple[str, ...] = FRAMEWORKS) -> Bench:
    from langchain_openai import ChatOpenAI
    from llama_index.llms.openai_like import OpenAILike

    meter = Meter()
    key, url, model = settings.api_key(), settings.llm_base_url, settings.answer_model
    sql = SqlTool(settings.sqlite_path)
    common: dict[str, Any] = {"max_steps": settings.max_steps, "top_k": settings.top_k}
    agents: dict[str, QAAgent] = {}
    # SDK retries are off everywhere: the metered transport handles 429s identically for all.
    if "baseline" in frameworks:
        client = AsyncOpenAI(api_key=key, base_url=url, max_retries=0, http_client=metered_client(meter))
        agents["baseline"] = baseline.BaselineAgent(client, model, searchers[MATCHED["baseline"]], sql, **common)
    if "langgraph" in frameworks:
        chat = ChatOpenAI(
            model=model,
            api_key=key,  # type: ignore[arg-type]
            base_url=url,
            temperature=0.0,
            max_retries=0,
            http_async_client=metered_client(meter),
        )
        agents["langgraph"] = langgraph_impl.LangGraphAgent(chat, searchers[MATCHED["langgraph"]], sql, **common)
    if "llamaindex" in frameworks:
        llm = OpenAILike(
            model=model,
            api_key=key,
            api_base=url,
            temperature=0.0,
            max_retries=0,
            timeout=120,
            is_chat_model=True,
            is_function_calling_model=True,
            context_window=131_072,
            async_http_client=metered_client(meter),  # type: ignore[call-arg]
        )
        agents["llamaindex"] = llamaindex_impl.LlamaIndexAgent(llm, searchers[MATCHED["llamaindex"]], sql, **common)
    judge = AsyncOpenAI(api_key=key, base_url=url, max_retries=0, http_client=metered_client(meter))
    return Bench(meter, agents, judge)


def write_json(path: Any, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2, default=str) + "\n")
