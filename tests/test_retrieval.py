"""Each framework's indexing + hybrid search, on in-memory Qdrant with hashing embedders."""

from typing import Any

import pytest
from langchain_core.embeddings import Embeddings
from langchain_qdrant import SparseEmbeddings, SparseVector
from llama_index.core.base.embeddings.base import BaseEmbedding
from qdrant_client import QdrantClient

from afbench.corpus import build_chunks, load_corpus
from afbench.embeddings import HashEmbedder, HashSparseEmbedder
from afbench.frameworks import baseline, langgraph_impl, llamaindex_impl
from tests.conftest import CORPUS_DIR

DENSE, SPARSE = HashEmbedder(), HashSparseEmbedder()


class LCDense(Embeddings):
    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return DENSE.embed_documents(texts)

    def embed_query(self, text: str) -> list[float]:
        return DENSE.embed_query(text)


class LCSparse(SparseEmbeddings):
    def embed_documents(self, texts: list[str]) -> list[SparseVector]:
        return [SparseVector(indices=i, values=v) for i, v in SPARSE.embed_documents(texts)]

    def embed_query(self, text: str) -> SparseVector:
        i, v = SPARSE.embed_query(text)
        return SparseVector(indices=i, values=v)


class LIDense(BaseEmbedding):
    def _get_text_embedding(self, text: str) -> list[float]:
        return DENSE.embed_query(text)

    def _get_query_embedding(self, query: str) -> list[float]:
        return DENSE.embed_query(query)

    async def _aget_query_embedding(self, query: str) -> list[float]:
        return DENSE.embed_query(query)


def li_sparse(texts: list[str]) -> tuple[list[list[int]], list[list[float]]]:
    vectors = SPARSE.embed_documents(texts)
    return [v[0] for v in vectors], [v[1] for v in vectors]


@pytest.fixture(scope="module")
def chunks():
    return build_chunks(load_corpus(CORPUS_DIR))


def _searchers(chunks) -> dict[str, Any]:
    client = QdrantClient(":memory:")
    baseline.build_index(client, "docs", chunks, DENSE, SPARSE)
    lc_store = langgraph_impl.build_index(chunks, LCDense(), LCSparse(), "docs", idf=True, location=":memory:")
    li_client = QdrantClient(":memory:")
    store = llamaindex_impl.vector_store(li_client, "docs", sparse_doc_fn=li_sparse, sparse_query_fn=li_sparse)
    li_index = llamaindex_impl.build_index(chunks, store, LIDense(embed_batch_size=256))
    rrf_store = llamaindex_impl.vector_store(
        li_client, "docs", sparse_doc_fn=li_sparse, sparse_query_fn=li_sparse, fusion=llamaindex_impl.rrf_fusion
    )
    from llama_index.core import VectorStoreIndex

    return {
        "baseline": baseline.BaselineSearcher(client, "docs", DENSE, SPARSE),
        "langchain": langgraph_impl.LangChainSearcher(lc_store, candidates=20),
        "llamaindex": llamaindex_impl.LlamaIndexSearcher(li_index),
        "llamaindex-rrf": llamaindex_impl.LlamaIndexSearcher(
            VectorStoreIndex.from_vector_store(rrf_store, embed_model=LIDense()), candidates=20
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
