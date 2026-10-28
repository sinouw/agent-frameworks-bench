from afbench.corpus import (
    Document,
    build_chunks,
    chunk_document,
    document_card,
    load_corpus,
    pack,
    parse_header_block,
    split_sections,
)
from tests.conftest import CORPUS_DIR

RST = """PEP: 999
Title: Example
Author: Ada <ada@example.com>, Bob
Status: Final
Created: 01-Jan-2020

Abstract
========

First paragraph.

Second paragraph.

Details
-------

More text here.
"""


def test_header_block_and_sections():
    headers, body = parse_header_block(RST)
    assert headers["Title"] == "Example" and headers["Status"] == "Final"
    assert split_sections(body) == [
        ("Abstract", "First paragraph.\n\nSecond paragraph."),
        ("Details", "More text here."),
    ]


def test_text_without_headers_is_left_alone():
    assert parse_header_block("Just prose.\n\nMore.") == ({}, "Just prose.\n\nMore.")


def test_pack_respects_size_and_overlap():
    chunks = pack(["a" * 50, "b" * 50, "c" * 50], size=120, overlap=20)
    assert all(len(c) <= 120 for c in chunks)
    assert len(chunks) == 2
    assert pack(["x" * 300], size=100, overlap=10)[0] == "x" * 100


def test_chunks_cards_and_embedding_text():
    headers, body = parse_header_block(RST)
    doc = Document("pep-0999", "PEP 999: Example", body, headers)
    chunks = chunk_document(doc)
    assert chunks[0].chunk_id == "pep-0999#0" and chunks[0].section == "Abstract"
    assert chunks[0].embedding_text().startswith("PEP 999: Example — Abstract\n\n")
    card = document_card(doc)
    assert card is not None and card.kind == "card"
    assert "Author: Ada, Bob" in card.text  # e-mail addresses dropped


def test_real_corpus_loads():
    docs = load_corpus(CORPUS_DIR)
    assert len(docs) == 25
    chunks = build_chunks(docs)
    assert sum(1 for c in chunks if c.kind == "card") == 25
    assert len({c.chunk_id for c in chunks}) == len(chunks)
