"""Embedding backends shared by the three implementations.

Production runs use fastembed (`bge-small-en-v1.5` dense + `Qdrant/bm25` sparse), each
framework through its own fastembed integration. Tests use the deterministic hashing
embedders below so they run offline without model downloads.
"""

from __future__ import annotations

import hashlib
import math
import re
from typing import Protocol

SparseVector = tuple[list[int], list[float]]


class DenseEmbedder(Protocol):
    dim: int

    def embed_documents(self, texts: list[str]) -> list[list[float]]: ...

    def embed_query(self, text: str) -> list[float]: ...


class SparseEmbedder(Protocol):
    def embed_documents(self, texts: list[str]) -> list[SparseVector]: ...

    def embed_query(self, text: str) -> SparseVector: ...


def _tokens(text: str) -> list[str]:
    return re.findall(r"[a-z0-9_]+", text.lower())


def _bucket(token: str, size: int) -> int:
    return int.from_bytes(hashlib.blake2b(token.encode(), digest_size=8).digest()[:4], "big") % size


class HashEmbedder:
    """Deterministic bag-of-words hashing embedder for offline tests."""

    def __init__(self, dim: int = 256) -> None:
        self.dim = dim

    def _embed(self, text: str) -> list[float]:
        vec = [0.0] * self.dim
        for token in _tokens(text):
            vec[_bucket(token, self.dim)] += 1.0
        norm = math.sqrt(sum(v * v for v in vec)) or 1.0
        return [v / norm for v in vec]

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return [self._embed(t) for t in texts]

    def embed_query(self, text: str) -> list[float]:
        return self._embed(text)


class HashSparseEmbedder:
    """Term-frequency sparse vectors over hashed tokens, for offline tests."""

    def _embed(self, text: str) -> SparseVector:
        counts: dict[int, float] = {}
        for token in _tokens(text):
            idx = _bucket(token, 2**20)
            counts[idx] = counts.get(idx, 0.0) + 1.0
        return list(counts), list(counts.values())

    def embed_documents(self, texts: list[str]) -> list[SparseVector]:
        return [self._embed(t) for t in texts]

    def embed_query(self, text: str) -> SparseVector:
        return self._embed(text)


class FastEmbedDense:
    def __init__(self, model_name: str) -> None:
        from fastembed import TextEmbedding

        self._model = TextEmbedding(model_name)
        self.dim = len(next(iter(self._model.embed(["dimension probe"]))))

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return [v.tolist() for v in self._model.embed(texts, batch_size=32)]

    def embed_query(self, text: str) -> list[float]:
        # query_embed applies the BGE query instruction prefix.
        return [float(x) for x in next(iter(self._model.query_embed(text)))]


class FastEmbedSparse:
    def __init__(self, model_name: str) -> None:
        from fastembed import SparseTextEmbedding

        self._model = SparseTextEmbedding(model_name)

    def embed_documents(self, texts: list[str]) -> list[SparseVector]:
        return [(v.indices.tolist(), v.values.tolist()) for v in self._model.embed(texts, batch_size=64)]

    def embed_query(self, text: str) -> SparseVector:
        # For BM25, query_embed weights each query term once instead of applying document TF.
        v = next(iter(self._model.query_embed(text)))
        return v.indices.tolist(), v.values.tolist()
