"""Wiring for real runs: builds each framework's index through its own Qdrant integration,
opens the retrieval variants compared in the retrieval eval, and creates the three agents
on one metered HTTP transport.

The corpus is embedded once; each framework gets its own embedded Qdrant directory under
`data/index/` (an embedded Qdrant directory can only be opened by one client at a time).
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass
from typing import Any

import httpx
from openai import AsyncOpenAI
from qdrant_client import QdrantClient

from afbench.common import QAAgent, Searcher, SqlTool
from afbench.config import Settings
from afbench.corpus import build_chunks, load_corpus, write_metadata_db
from afbench.embeddings import (
    CachedEmbeddings,
    CachedSparse,
    FastEmbedDense,
    FastEmbedSparse,
)
from afbench.frameworks import baseline, langgraph_impl, llamaindex_impl
from afbench.transport import Meter, metered_client

FRAMEWORKS = ("baseline", "langgraph", "llamaindex")

# Groq's free tier caps tokens per model over a rolling 24 h window, shared here with other
# work on the same key. Short waits for old usage to age out are waited out (and recorded
# as waiting, not latency); longer ones stop the run, which then resumes on rerun.
DAILY_WAIT_S = 1800.0

# The variants used by the end-to-end agents: identical candidate pool, fusion and BM25
# weighting across frameworks. The defaults are measured too, in the retrieval eval.
MATCHED = {
    "baseline": "baseline",
    "langgraph": "langchain: IDF + 20 candidates",
    "llamaindex": "llamaindex: 20 candidates + BM25 query encoding + RRF",
}


def shared_embeddings(settings: Settings) -> tuple[CachedEmbeddings, CachedSparse]:
    """One dense and one sparse model for the whole process, with the document-vector cache."""
    cache = CachedEmbeddings(FastEmbedDense(settings.embedding_model), FastEmbedSparse(settings.sparse_model))
    cache.load(settings.index_dir / "embeddings")
    return cache, CachedSparse(cache)


def build_indexes(settings: Settings) -> dict[str, Any]:
    """Embed the corpus once, then let each framework write its own index from the cache.
    The per-framework seconds are therefore indexing overhead, not embedding time."""
    docs = load_corpus(settings.corpus_dir)
    chunks = build_chunks(docs)
    write_metadata_db(docs, settings.sqlite_path)
    stats: dict[str, Any] = {"documents": len(docs), "chunks": len(chunks), "seconds": {}}

    def timed(name: str, fn: Any) -> None:
        started = time.perf_counter()
        fn()
        stats["seconds"][name] = round(time.perf_counter() - started, 2)
        print(f"{name}: {stats['seconds'][name]}s", flush=True)

    settings.index_dir.mkdir(parents=True, exist_ok=True)
    cache, sparse = shared_embeddings(settings)
    cached = (settings.index_dir / "embeddings" / "index.json").exists()
    label = "embed corpus (loaded from cache)" if cached else "embed corpus once (dense + BM25, shared)"
    timed(label, lambda: cache.warm([c.embedding_text() for c in chunks]))
    cache.save(settings.index_dir / "embeddings")

    def run_baseline() -> None:
        client = QdrantClient(path=str(settings.index_dir / "baseline"))
        baseline.build_index(client, "docs", chunks, cache, sparse)
        client.close()

    def run_langchain() -> None:
        for collection, idf in (("docs", False), ("docs_idf", True)):
            store = langgraph_impl.build_index(
                chunks,
                langgraph_impl.LangChainDense(cache),
                langgraph_impl.LangChainSparse(sparse),
                collection,
                idf=idf,
                path=str(settings.index_dir / "langchain"),
            )
            store.client.close()

    def run_llamaindex() -> None:
        client = QdrantClient(path=str(settings.index_dir / "llamaindex"))
        doc_fn, query_fn = llamaindex_impl.sparse_fns(sparse, query_side=True)
        store = llamaindex_impl.vector_store(client, "docs", doc_fn, query_fn)
        llamaindex_impl.build_index(chunks, store, llamaindex_impl.LlamaIndexDense(cache))
        client.close()

    timed("baseline: write index", run_baseline)
    timed("langchain: write index (x2: default and IDF collections)", run_langchain)
    timed("llamaindex: write index", run_llamaindex)
    return stats


def open_searchers(settings: Settings) -> dict[str, Searcher]:
    """Every retrieval variant, keyed by a readable name. All share one embedding model."""
    from llama_index.core import VectorStoreIndex

    index_dir = settings.index_dir
    if not (index_dir / "llamaindex").exists():
        raise RuntimeError("No index yet: run `uv run afbench index` first.")
    cache, sparse = shared_embeddings(settings)

    bl_client = QdrantClient(path=str(index_dir / "baseline"))
    lc_client = QdrantClient(path=str(index_dir / "langchain"))
    lc_dense, lc_sparse = langgraph_impl.LangChainDense(cache), langgraph_impl.LangChainSparse(sparse)
    lc_default = langgraph_impl.open_store(lc_client, "docs", lc_dense, lc_sparse)
    lc_idf = langgraph_impl.open_store(lc_client, "docs_idf", lc_dense, lc_sparse)

    li_client = QdrantClient(path=str(index_dir / "llamaindex"))
    li_embed = llamaindex_impl.LlamaIndexDense(cache)

    def li_index(query_side: bool, rrf: bool) -> VectorStoreIndex:
        doc_fn, query_fn = llamaindex_impl.sparse_fns(sparse, query_side)
        fusion = llamaindex_impl.rrf_fusion if rrf else None
        store = llamaindex_impl.vector_store(li_client, "docs", doc_fn, query_fn, fusion)
        return VectorStoreIndex.from_vector_store(store, embed_model=li_embed)

    n = settings.candidates
    variants: list[Searcher] = [
        baseline.BaselineSearcher(bl_client, "docs", cache, sparse, candidates=n, name="baseline"),
        langgraph_impl.LangChainSearcher(lc_default, None, name="langchain: defaults"),
        langgraph_impl.LangChainSearcher(lc_idf, None, name="langchain: IDF"),
        langgraph_impl.LangChainSearcher(lc_idf, n, name=MATCHED["langgraph"]),
        llamaindex_impl.LlamaIndexSearcher(li_index(False, False), None, name="llamaindex: defaults"),
        llamaindex_impl.LlamaIndexSearcher(li_index(False, False), n, name="llamaindex: 20 candidates"),
        llamaindex_impl.LlamaIndexSearcher(
            li_index(True, False), n, name="llamaindex: 20 candidates + BM25 query encoding"
        ),
        llamaindex_impl.LlamaIndexSearcher(li_index(True, True), n, name=MATCHED["llamaindex"]),
    ]
    return {s.name: s for s in variants}


@dataclass
class Bench:
    meter: Meter
    agents: dict[str, QAAgent]
    judge: AsyncOpenAI


def build_bench(
    settings: Settings,
    searchers: dict[str, Searcher],
    frameworks: tuple[str, ...] = FRAMEWORKS,
    inner: httpx.AsyncBaseTransport | None = None,
) -> Bench:
    """`inner` replaces the real HTTP transport (tests use a scripted backend)."""
    from langchain_openai import ChatOpenAI
    from llama_index.llms.openai_like import OpenAILike

    meter = Meter()
    key, url, model = settings.api_key(), settings.llm_base_url, settings.answer_model
    sql = SqlTool(settings.sqlite_path)
    common: dict[str, Any] = {"max_steps": settings.max_steps, "top_k": settings.top_k}
    agents: dict[str, QAAgent] = {}
    # SDK retries are off everywhere: the metered transport handles 429s identically for all.
    if "baseline" in frameworks:
        client = AsyncOpenAI(
            api_key=key,
            base_url=url,
            max_retries=0,
            http_client=metered_client(meter, inner, daily_wait_s=DAILY_WAIT_S),
        )
        agents["baseline"] = baseline.BaselineAgent(client, model, searchers[MATCHED["baseline"]], sql, **common)
    if "langgraph" in frameworks:
        chat = ChatOpenAI(
            model=model,
            api_key=key,  # type: ignore[arg-type]
            base_url=url,
            temperature=0.0,
            max_retries=0,
            http_async_client=metered_client(meter, inner, daily_wait_s=DAILY_WAIT_S),
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
            async_http_client=metered_client(meter, inner, daily_wait_s=DAILY_WAIT_S),  # type: ignore[call-arg]
        )
        agents["llamaindex"] = llamaindex_impl.LlamaIndexAgent(llm, searchers[MATCHED["llamaindex"]], sql, **common)
    judge = AsyncOpenAI(
        api_key=key, base_url=url, max_retries=0, http_client=metered_client(meter, inner, daily_wait_s=DAILY_WAIT_S)
    )
    return Bench(meter, agents, judge)


def write_json(path: Any, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2, default=str) + "\n")
