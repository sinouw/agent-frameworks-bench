"""Load the corpus, split it into section-aware chunks, and build the SQLite metadata table.

This is shared by all three implementations, so every framework indexes exactly the same
chunks with exactly the same embedding text. Chunking is deliberately not delegated to a
framework's splitter: the comparison is about agents and retrieval, not about splitters.
"""

from __future__ import annotations

import re
import sqlite3
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path

_HEADER_LINE = re.compile(r"^([A-Z][A-Za-z-]+):\s*(.*)$")
_RST_UNDERLINE = re.compile(r"^([=\-~^\"'`#*+])\1{2,}\s*$")
_MD_HEADING = re.compile(r"^(#{1,6})\s+(.+?)\s*#*\s*$")

SUPPORTED_SUFFIXES = {".md", ".rst", ".txt"}
CARD_FIELDS = ("Title", "Author", "Sponsor", "PEP-Delegate", "Status", "Type", "Topic", "Created", "Python-Version")


@dataclass
class Document:
    doc_id: str
    title: str
    text: str
    metadata: dict[str, str] = field(default_factory=dict)


@dataclass(frozen=True)
class Chunk:
    chunk_id: str
    doc_id: str
    title: str
    section: str
    text: str
    kind: str = "text"  # "text" for body chunks, "card" for the per-document metadata card

    def embedding_text(self) -> str:
        """What gets embedded: the chunk prefixed with its document and section titles."""
        heading = f"{self.title} — {self.section}" if self.section else self.title
        return f"{heading}\n\n{self.text}"

    def payload(self) -> dict[str, str]:
        return {
            "chunk_id": self.chunk_id,
            "doc_id": self.doc_id,
            "title": self.title,
            "section": self.section,
            "body": self.text,
            "kind": self.kind,
        }


def parse_header_block(text: str) -> tuple[dict[str, str], str]:
    """Parse an RFC 822-style `Key: value` block at the top of a file (PEP headers)."""
    lines = text.splitlines()
    headers: dict[str, str] = {}
    key: str | None = None
    end = len(lines)
    for i, line in enumerate(lines):
        if not line.strip():
            end = i
            break
        match = _HEADER_LINE.match(line)
        if match:
            key = match.group(1)
            headers[key] = match.group(2).strip()
        elif key and line[:1].isspace():
            headers[key] += " " + line.strip()
        else:
            return {}, text
    if len(headers) < 2:
        return {}, text
    return headers, "\n".join(lines[end + 1 :])


def load_document(path: Path) -> Document:
    if path.suffix.lower() not in SUPPORTED_SUFFIXES:
        raise ValueError(f"Unsupported file type: {path.name}")
    headers, body = parse_header_block(path.read_text(encoding="utf-8"))
    title = headers.get("Title") or path.stem
    if "PEP" in headers:
        title = f"PEP {headers['PEP']}: {title}"
    return Document(path.stem, title, body, headers)


def load_corpus(corpus_dir: Path) -> list[Document]:
    paths = sorted(p for p in corpus_dir.rglob("*") if p.suffix.lower() in SUPPORTED_SUFFIXES)
    return [load_document(p) for p in paths]


def split_sections(text: str) -> list[tuple[str, str]]:
    """Split RST or Markdown text into (heading, body) sections."""
    lines = text.splitlines()
    sections: list[tuple[str, str]] = []
    heading = ""
    buf: list[str] = []

    def flush() -> None:
        body = "\n".join(buf).strip()
        if body or heading:
            sections.append((heading, body))

    i = 0
    while i < len(lines):
        line = lines[i]
        nxt = lines[i + 1] if i + 1 < len(lines) else ""
        md = _MD_HEADING.match(line)
        if md:
            flush()
            heading, buf = md.group(2), []
        elif line.strip() and _RST_UNDERLINE.match(nxt) and len(nxt.strip()) >= len(line.strip()) - 2:
            flush()
            heading, buf = line.strip(), []
            i += 1  # skip the underline
        elif _RST_UNDERLINE.match(line) and not buf:
            pass  # overline of an RST title
        else:
            buf.append(line)
        i += 1
    flush()
    return sections


def pack(paragraphs: list[str], size: int, overlap: int) -> list[str]:
    """Greedily pack paragraphs into chunks of ~`size` chars, carrying `overlap` chars forward."""
    chunks: list[str] = []
    current = ""
    for para in paragraphs:
        while len(para) > size:  # hard-split oversized paragraphs
            head, para = para[:size], para[size - overlap :]
            if current:
                chunks.append(current)
                current = ""
            chunks.append(head)
        candidate = f"{current}\n\n{para}" if current else para
        if len(candidate) <= size:
            current = candidate
            continue
        chunks.append(current)
        tail = current[-overlap:] if overlap else ""
        cut = tail.find(" ")
        tail = tail[cut + 1 :] if cut != -1 else tail
        current = f"{tail}\n\n{para}" if tail and len(tail) + len(para) + 2 <= size else para
    if current.strip():
        chunks.append(current)
    return chunks


def chunk_document(doc: Document, size: int = 1200, overlap: int = 200) -> list[Chunk]:
    chunks: list[Chunk] = []
    for section, body in split_sections(doc.text):
        if not body:
            continue
        paragraphs = [p.strip() for p in re.split(r"\n\s*\n", body) if p.strip()]
        for piece in pack(paragraphs, size, overlap):
            chunks.append(Chunk(f"{doc.doc_id}#{len(chunks)}", doc.doc_id, doc.title, section, piece))
    return chunks


def document_card(doc: Document) -> Chunk | None:
    """One searchable chunk per document with its header fields (authors, status, dates).
    Body chunks never contain these, so "who wrote X?" needs the card to be answerable."""
    lines = []
    for name in CARD_FIELDS:
        value = doc.metadata.get(name)
        if value:
            lines.append(f"{name}: {re.sub(r'\s*<[^>]*>', '', value).strip()}")
    if len(lines) < 2:
        return None
    return Chunk(f"{doc.doc_id}#card", doc.doc_id, doc.title, "Document metadata", "\n".join(lines), kind="card")


def build_chunks(docs: list[Document], size: int = 1200, overlap: int = 200) -> list[Chunk]:
    chunks = [c for doc in docs for c in chunk_document(doc, size, overlap)]
    return chunks + [card for doc in docs if (card := document_card(doc))]


def _iso_date(value: str | None) -> str | None:
    if not value:
        return None
    for fmt in ("%d-%b-%Y", "%Y-%m-%d"):
        try:
            return datetime.strptime(value.strip(), fmt).date().isoformat()
        except ValueError:
            continue
    return None


def write_metadata_db(docs: list[Document], db_path: Path) -> None:
    db_path.parent.mkdir(parents=True, exist_ok=True)
    tmp = db_path.with_suffix(".tmp")
    tmp.unlink(missing_ok=True)
    with sqlite3.connect(tmp) as conn:
        conn.execute(
            """CREATE TABLE documents (
                doc_id TEXT PRIMARY KEY,
                pep_number INTEGER,
                title TEXT NOT NULL,
                status TEXT,
                type TEXT,
                created DATE,
                python_version TEXT,
                author_count INTEGER,
                word_count INTEGER NOT NULL
            )"""
        )
        for doc in docs:
            meta = doc.metadata
            authors = meta.get("Author")
            conn.execute(
                "INSERT INTO documents VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)",
                (
                    doc.doc_id,
                    int(meta["PEP"]) if meta.get("PEP", "").isdigit() else None,
                    meta.get("Title") or doc.title,
                    meta.get("Status"),
                    meta.get("Type"),
                    _iso_date(meta.get("Created")),
                    meta.get("Python-Version"),
                    len([a for a in authors.split(",") if a.strip()]) if authors else None,
                    len(doc.text.split()),
                ),
            )
    conn.close()
    tmp.replace(db_path)
