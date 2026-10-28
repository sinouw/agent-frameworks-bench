"""Logical lines of code: non-blank lines that are not comments or docstrings."""

from __future__ import annotations

import io
import tokenize
from pathlib import Path


def logical_lines(source: str) -> int:
    lines: set[int] = set()
    previous = tokenize.INDENT
    for tok in tokenize.generate_tokens(io.StringIO(source).readline):
        if tok.type in (tokenize.COMMENT, tokenize.NL, tokenize.NEWLINE, tokenize.INDENT, tokenize.DEDENT):
            if tok.type != tokenize.COMMENT:
                previous = tok.type
            continue
        if tok.type == tokenize.ENDMARKER:
            break
        # A string that starts a statement on its own is a docstring (or a bare string): skip it.
        if tok.type == tokenize.STRING and previous in (tokenize.INDENT, tokenize.DEDENT, tokenize.NEWLINE):
            previous = tokenize.STRING
            continue
        if previous == tokenize.STRING and tok.type == tokenize.STRING:
            continue
        previous = tok.type
        lines.update(range(tok.start[0], tok.end[0] + 1))
    return len(lines)


def count_files(paths: list[Path]) -> dict[str, int]:
    return {p.name: logical_lines(p.read_text()) for p in paths}
