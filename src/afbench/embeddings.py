"""Embeddings, computed once and shared by all three implementations.

The corpus is embedded a single time (fastembed `bge-small-en-v1.5` dense + `Qdrant/bm25`
sparse, both ONNX on CPU) and cached under `data/index/embeddings/`. Each framework then
indexes from that cache through a small adapter for its own embedding interface (in its
own module), and at query time all three share one model instance. Vectors are identical
by construction and the laptop holds one copy of each model.

Hashing embedders at the bottom keep the tests offline.
"""

from __future__ import annotations

import hashlib
import json
import math
import re
from collections.abc import Callable
from pathlib import Path
from typing import Protocol

import numpy as np

SparseVector = tuple[list[int], list[float]]
SparseFn = Callable[[list[str]], tuple[list[list[int]], list[list[float]]]]


class DenseEmbedder(Protocol):
    dim: int

    def embed_documents(self, texts: list[str]) -> list[list[float]]: ...

    def embed_query(self, text: str) -> list[float]: ...


class SparseEmbedder(Protocol):
    def embed_documents(self, texts: list[str]) -> list[SparseVector]: ...

    def embed_query(self, text: str) -> SparseVector: ...

    def embed_document_side(self, text: str) -> SparseVector: ...


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

    def embed_document_side(self, text: str) -> SparseVector:
        return self.embed_documents([text])[0]


def _key(text: str) -> str:
    return hashlib.sha256(text.encode()).hexdigest()


class CachedEmbeddings:
    """Document vectors computed once and stored on disk; misses fall back to the model."""

    def __init__(self, dense: DenseEmbedder, sparse: SparseEmbedder) -> None:
        self.dense, self.sparse = dense, sparse
        self.dim = dense.dim
        self._dense: dict[str, list[float]] = {}
        self._sparse: dict[str, SparseVector] = {}

    def warm(self, texts: list[str]) -> None:
        missing = [t for t in dict.fromkeys(texts) if _key(t) not in self._dense]
        if missing:
            for text, d, s in zip(
                missing, self.dense.embed_documents(missing), self.sparse.embed_documents(missing), strict=True
            ):
                self._dense[_key(text)], self._sparse[_key(text)] = d, s

    def save(self, directory: Path) -> None:
        directory.mkdir(parents=True, exist_ok=True)
        keys = list(self._dense)
        np.save(directory / "dense.npy", np.array([self._dense[k] for k in keys], dtype=np.float32))
        (directory / "index.json").write_text(json.dumps({"keys": keys, "sparse": [self._sparse[k] for k in keys]}))

    def load(self, directory: Path) -> bool:
        if not (directory / "index.json").exists():
            return False
        meta = json.loads((directory / "index.json").read_text())
        dense = np.load(directory / "dense.npy")
        for i, key in enumerate(meta["keys"]):
            self._dense[key] = dense[i].tolist()
            indices, values = meta["sparse"][i]
            self._sparse[key] = (indices, values)
        return True

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        self.warm(texts)
        return [self._dense[_key(t)] for t in texts]

    def embed_sparse_documents(self, texts: list[str]) -> list[SparseVector]:
        self.warm(texts)
        return [self._sparse[_key(t)] for t in texts]

    def embed_query(self, text: str) -> list[float]:
        return self.dense.embed_query(text)


class CachedSparse:
    """The sparse half of `CachedEmbeddings`, as a `SparseEmbedder`."""

    def __init__(self, cache: CachedEmbeddings) -> None:
        self.cache = cache

    def embed_documents(self, texts: list[str]) -> list[SparseVector]:
        return self.cache.embed_sparse_documents(texts)

    def embed_query(self, text: str) -> SparseVector:
        return self.cache.sparse.embed_query(text)

    def embed_document_side(self, text: str) -> SparseVector:
        return self.cache.sparse.embed_document_side(text)


# ---------------------------------------------------------------- offline test embedders


def _tokens(text: str) -> list[str]:
    return re.findall(r"[a-z0-9_]+", text.lower())


def _bucket(token: str, size: int) -> int:
    return int.from_bytes(hashlib.blake2b(token.encode(), digest_size=8).digest()[:4], "big") % size


class HashEmbedder:
    """Deterministic bag-of-words hashing embedder."""

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
    """Term-frequency sparse vectors over hashed tokens; queries weight each term once."""

    def _embed(self, text: str, unique: bool = False) -> SparseVector:
        counts: dict[int, float] = {}
        for token in _tokens(text):
            idx = _bucket(token, 2**20)
            counts[idx] = 1.0 if unique else counts.get(idx, 0.0) + 1.0
        return list(counts), list(counts.values())

    def embed_documents(self, texts: list[str]) -> list[SparseVector]:
        return [self._embed(t) for t in texts]

    def embed_query(self, text: str) -> SparseVector:
        return self._embed(text, unique=True)

    def embed_document_side(self, text: str) -> SparseVector:
        return self._embed(text)
