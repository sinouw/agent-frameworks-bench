"""Each framework's indexing + hybrid search, on in-memory Qdrant with hashing embedders."""

from typing import Any

import pytest
from llama_index.core import VectorStoreIndex
from qdrant_client import QdrantClient

from afbench.corpus import build_chunks, load_corpus
from afbench.embeddings import (
    CachedEmbeddings,
    CachedSparse,
    HashEmbedder,
    HashSparseEmbedder,
)
from afbench.frameworks import baseline, langgraph_impl, llamaindex_impl
from tests.conftest import CORPUS_DIR

DENSE, SPARSE = HashEmbedder(), HashSparseEmbedder()


@pytest.fixture(scope="module")
def chunks():
    return build_chunks(load_corpus(CORPUS_DIR))


def _searchers(chunks) -> dict[str, Any]:
    client = QdrantClient(":memory:")
    baseline.build_index(client, "docs", chunks, DENSE, SPARSE)
    lc_store = langgraph_impl.build_index(
        chunks,
        langgraph_impl.LangChainDense(DENSE),
        langgraph_impl.LangChainSparse(SPARSE),
        "docs",
        idf=True,
        location=":memory:",
    )
    li_client = QdrantClient(":memory:")
    doc_fn, query_fn = llamaindex_impl.sparse_fns(SPARSE, query_side=True)
    li_index = llamaindex_impl.build_index(
        chunks,
        llamaindex_impl.vector_store(li_client, "docs", doc_fn, query_fn),
        llamaindex_impl.LlamaIndexDense(DENSE),
    )
    rrf_store = llamaindex_impl.vector_store(li_client, "docs", doc_fn, query_fn, fusion=llamaindex_impl.rrf_fusion)
    return {
        "baseline": baseline.BaselineSearcher(client, "docs", DENSE, SPARSE),
        "langchain": langgraph_impl.LangChainSearcher(lc_store, candidates=20),
        "llamaindex": llamaindex_impl.LlamaIndexSearcher(li_index),
        "llamaindex-rrf": llamaindex_impl.LlamaIndexSearcher(
            VectorStoreIndex.from_vector_store(rrf_store, embed_model=llamaindex_impl.LlamaIndexDense(DENSE)),
            candidates=20,
        ),
    }


def test_every_implementation_finds_the_answer_chunk(chunks):
    for name, searcher in _searchers(chunks).items():
        hits = searcher.search("EXTERNALLY-MANAGED marker file pip", 5)
        assert len(hits) == 5, name
        assert hits[0].doc_id == "pep-0668", name
        assert any("EXTERNALLY-MANAGED" in h.text for h in hits), name
        # Payloads round-trip to the same Hit shape, with the raw chunk body as text.
        assert not hits[0].text.startswith("PEP 668"), name


def test_rrf_fusion_ranks_by_reciprocal_rank():
    from llama_index.core.schema import TextNode
    from llama_index.core.vector_stores.types import VectorStoreQueryResult

    a, b, c = (TextNode(id_=x, text=x) for x in "abc")
    dense = VectorStoreQueryResult(nodes=[a, b], similarities=[0.9, 0.8], ids=["a", "b"])
    sparse = VectorStoreQueryResult(nodes=[b, c], similarities=[12.0, 3.0], ids=["b", "c"])
    fused = llamaindex_impl.rrf_fusion(dense, sparse, top_k=2)
    assert fused.ids == ["b", "a"]  # b appears in both lists


def test_embedding_cache_round_trips_and_is_used(tmp_path):
    class Counting(HashEmbedder):
        calls = 0

        def embed_documents(self, texts):
            Counting.calls += len(texts)
            return super().embed_documents(texts)

    cache = CachedEmbeddings(Counting(), SPARSE)
    vectors = cache.embed_documents(["alpha beta", "gamma"])
    cache.embed_documents(["gamma", "alpha beta"])
    assert Counting.calls == 2  # the second call is served from the cache
    cache.save(tmp_path)

    reloaded = CachedEmbeddings(Counting(), SPARSE)
    assert reloaded.load(tmp_path)
    assert reloaded.embed_documents(["alpha beta"])[0] == pytest.approx(vectors[0])
    assert CachedSparse(reloaded).embed_documents(["gamma"]) == SPARSE.embed_documents(["gamma"])
    assert Counting.calls == 2


def test_llamaindex_default_encodes_queries_document_side():
    _, default_query = llamaindex_impl.sparse_fns(SPARSE, query_side=False)
    _, query_side = llamaindex_impl.sparse_fns(SPARSE, query_side=True)
    assert default_query(["pip pip"])[1] == [[2.0]]
    assert query_side(["pip pip"])[1] == [[1.0]]
