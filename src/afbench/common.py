"""The contract all three implementations share: system prompt, tool names, descriptions and
argument schemas, tool behaviour (calculator, read-only SQL, search-result formatting),
citation handling, and the result type the eval harness consumes.

Each framework wraps these same functions in its own tool abstraction, so differences in
the results come from orchestration and retrieval, not from different tools or prompts.
"""

from __future__ import annotations

import ast
import math
import operator
import re
import sqlite3
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Protocol

from pydantic import BaseModel, Field

SYSTEM_PROMPT = """You answer questions about a collection of Python Enhancement Proposals (PEPs), \
the design documents that describe new Python features, processes and conventions.

Tools:
- search_docs: find passages in the documents. Search before answering any content question; \
search again with different wording if the first results don't contain the answer.
- sql_query: metadata across documents (counts, creation dates, statuses, types, Python versions, word counts).
- calculator: any arithmetic, including date differences. Never do arithmetic in your head: \
whenever the answer is a number you derive (a difference, sum, average, count of years), \
compute it with the calculator, even if it looks trivial.

Rules:
- Base every factual statement on tool results, and cite the passages you used inline with \
their source ids in square brackets, e.g. "... introduced in Python 3.8 [S2]." Only search_docs \
results have source ids; facts from sql_query or the calculator need no citation.
- If the tools don't give you the answer, say that the documents don't say. Do not guess.
- Be concise: a direct answer first, then brief supporting detail."""

FINAL_STEP_NOTE = (
    "Tool budget exhausted. Answer now from the results above, or say the documents don't contain the answer."
)

SEARCH_DESCRIPTION = (
    "Semantic search over the document corpus. Returns the most relevant passages, "
    "each labelled with a source id like [S1] to cite in the answer."
)
CALCULATOR_DESCRIPTION = "Evaluate an arithmetic expression, e.g. '(2023 - 2001) * 12'. Use for any arithmetic."


def sql_description(schema: str) -> str:
    return (
        "Run a read-only SQLite SELECT over the document metadata table. Use it for "
        "questions about many documents at once (counts, dates, statuses, versions, "
        "lengths). Schema:\n" + schema
    )


class SearchArgs(BaseModel):
    query: str = Field(description="A focused search query.")
    k: int = Field(default=5, description="Number of passages (1-10, default 5).")


class SqlArgs(BaseModel):
    sql: str = Field(description="A single SELECT statement.")


class CalculatorArgs(BaseModel):
    expression: str = Field(description="An arithmetic expression.")


# ---------------------------------------------------------------- retrieval contract


@dataclass
class Hit:
    chunk_id: str
    doc_id: str
    title: str
    section: str
    text: str
    score: float


class Searcher(Protocol):
    """A framework's retriever, reduced to the one call the agent and the retrieval eval need."""

    name: str

    def search(self, query: str, k: int) -> list[Hit]: ...


def hit_from_payload(payload: dict[str, Any], score: float) -> Hit:
    return Hit(
        payload["chunk_id"], payload["doc_id"], payload["title"], payload.get("section") or "", payload["body"], score
    )


class SourceRegistry:
    """Numbers retrieved chunks [S1], [S2], ... across all searches in one question, so
    citations in the answer resolve back to chunks."""

    def __init__(self) -> None:
        self.by_label: dict[str, Hit] = {}
        self._by_chunk: dict[str, str] = {}

    def label(self, hit: Hit) -> str:
        if hit.chunk_id not in self._by_chunk:
            label = f"S{len(self._by_chunk) + 1}"
            self._by_chunk[hit.chunk_id] = label
            self.by_label[label] = hit
        return self._by_chunk[hit.chunk_id]


def format_hits(hits: list[Hit], registry: SourceRegistry) -> str:
    if not hits:
        return "No matching passages found."
    blocks = []
    for hit in hits:
        location = f" › {hit.section}" if hit.section else ""
        blocks.append(f"[{registry.label(hit)}] {hit.title}{location}\n{hit.text}")
    return "\n\n---\n\n".join(blocks)


def clamp_k(k: int | None, default: int = 5) -> int:
    return max(1, min(int(k or default), 10))


# ---------------------------------------------------------------- calculator

_BIN_OPS: dict[type[ast.operator], Callable[[Any, Any], Any]] = {
    ast.Add: operator.add,
    ast.Sub: operator.sub,
    ast.Mult: operator.mul,
    ast.Div: operator.truediv,
    ast.FloorDiv: operator.floordiv,
    ast.Mod: operator.mod,
    ast.Pow: operator.pow,
}
_UNARY_OPS: dict[type[ast.unaryop], Callable[[Any], Any]] = {ast.UAdd: operator.pos, ast.USub: operator.neg}
_FUNCS: dict[str, Callable[..., Any]] = {"abs": abs, "round": round, "min": min, "max": max, "sqrt": math.sqrt}


class CalculatorError(ValueError):
    pass


def safe_eval(expression: str) -> float | int:
    """Arithmetic without `eval`: numbers, + - * / // % **, parentheses and a few functions."""
    if len(expression) > 200:
        raise CalculatorError("expression too long")
    try:
        tree = ast.parse(expression, mode="eval")
    except SyntaxError as exc:
        raise CalculatorError(f"invalid expression: {exc.msg}") from exc

    def ev(node: ast.AST) -> Any:
        if isinstance(node, ast.Expression):
            return ev(node.body)
        if isinstance(node, ast.Constant) and isinstance(node.value, int | float) and not isinstance(node.value, bool):
            return node.value
        if isinstance(node, ast.BinOp) and type(node.op) in _BIN_OPS:
            left, right = ev(node.left), ev(node.right)
            if isinstance(node.op, ast.Pow) and abs(right) > 100:
                raise CalculatorError("exponent too large")
            return _BIN_OPS[type(node.op)](left, right)
        if isinstance(node, ast.UnaryOp) and type(node.op) in _UNARY_OPS:
            return _UNARY_OPS[type(node.op)](ev(node.operand))
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Name) and node.func.id in _FUNCS:
            if node.keywords:
                raise CalculatorError("keyword arguments are not supported")
            return _FUNCS[node.func.id](*[ev(a) for a in node.args])
        raise CalculatorError(f"unsupported syntax: {type(node).__name__}")

    try:
        value = ev(tree)
    except ZeroDivisionError as exc:
        raise CalculatorError("division by zero") from exc
    if not isinstance(value, int | float):
        raise CalculatorError("result is not a number")
    return value


def calculator(expression: str) -> str:
    try:
        value = safe_eval(expression)
    except CalculatorError as exc:
        return f"Error: {exc}"
    if isinstance(value, float) and value.is_integer():
        value = int(value)
    return f"{expression} = {value}"


# ---------------------------------------------------------------- SQL


def _read_only_authorizer(action: int, *_args: Any) -> int:
    allowed = {sqlite3.SQLITE_SELECT, sqlite3.SQLITE_READ, sqlite3.SQLITE_FUNCTION}
    return sqlite3.SQLITE_OK if action in allowed else sqlite3.SQLITE_DENY


class SqlTool:
    """Read-only SELECT access to the document metadata table."""

    max_rows = 50

    def __init__(self, db_path: Path) -> None:
        self.db_path = db_path

    def _connect(self) -> sqlite3.Connection:
        # mode=ro opens the file read-only at the OS level; the authorizer is a second guard.
        conn = sqlite3.connect(f"file:{self.db_path}?mode=ro", uri=True)
        conn.set_authorizer(_read_only_authorizer)
        return conn

    def schema(self) -> str:
        conn = self._connect()
        try:
            rows = conn.execute("SELECT sql FROM sqlite_master WHERE type = 'table'").fetchall()
        finally:
            conn.close()
        return "\n".join(r[0] for r in rows)

    def run(self, sql: str) -> str:
        statement = sql.strip().rstrip(";").strip()
        if ";" in statement:
            return "Error: only a single statement is allowed."
        if not statement.lower().startswith(("select", "with")):
            return "Error: only SELECT queries are allowed."
        conn = self._connect()
        try:
            cursor = conn.execute(statement)
            columns = [d[0] for d in cursor.description or []]
            rows = cursor.fetchmany(self.max_rows + 1)
        except sqlite3.Error as exc:
            return f"SQL error: {exc}"
        finally:
            conn.close()
        lines = [" | ".join(columns)]
        lines += [" | ".join("NULL" if v is None else str(v) for v in row) for row in rows[: self.max_rows]]
        if len(rows) > self.max_rows:
            lines.append(f"(truncated to {self.max_rows} rows)")
        return "\n".join(lines)


# ---------------------------------------------------------------- results

# Provider errors can echo account ids or (masked) keys back; keep them out of stored results.
_SECRET_LIKE = re.compile(r"\b(?:gsk_|sk-|AIza)[A-Za-z0-9_\-*]{6,}|\b(?:org|proj|user)_[A-Za-z0-9]{8,}")


def redact(text: str) -> str:
    return _SECRET_LIKE.sub("[redacted]", text)


_CITATION_GROUP = re.compile(r"\[([^\[\]]+)\]")
_LABEL = re.compile(r"\bS\d+\b")


def normalize_citations(answer: str) -> str:
    """gpt-oss sometimes writes fullwidth brackets: 【S1】 -> [S1]."""
    return answer.replace("【", "[").replace("】", "]")


def extract_citation_labels(answer: str) -> list[str]:
    """Labels cited in an answer, in order of first appearance. Handles [S1] and [S1, S3]."""
    seen: list[str] = []
    for group in _CITATION_GROUP.findall(answer):
        for label in _LABEL.findall(group):
            if label not in seen:
                seen.append(label)
    return seen


@dataclass
class ToolUse:
    name: str
    arguments: dict[str, Any]
    output: str


@dataclass
class AgentResult:
    answer: str
    tools: list[ToolUse] = field(default_factory=list)
    cited: list[Hit] = field(default_factory=list)
    error: str | None = None


class QAAgent(Protocol):
    name: str

    async def answer(self, question: str) -> AgentResult: ...


class ToolBox:
    """Per-question tool state: one SourceRegistry, a log of every call, and the three tools.
    Every framework adapter calls these methods from its own tool wrappers."""

    def __init__(self, searcher: Searcher, sql: SqlTool, top_k: int = 5) -> None:
        self.searcher = searcher
        self.sql = sql
        self.top_k = top_k
        self.registry = SourceRegistry()
        self.log: list[ToolUse] = []

    def search_docs(self, query: str, k: int | None = None) -> str:
        query = (query or "").strip()
        if not query:
            output = "Error: 'query' is required."
        else:
            output = format_hits(self.searcher.search(query, clamp_k(k, self.top_k)), self.registry)
        self.log.append(ToolUse("search_docs", {"query": query, "k": k}, output))
        return output

    def sql_query(self, sql: str) -> str:
        output = self.sql.run(sql or "")
        self.log.append(ToolUse("sql_query", {"sql": sql}, output))
        return output

    def calculator(self, expression: str) -> str:
        output = calculator(expression or "")
        self.log.append(ToolUse("calculator", {"expression": expression}, output))
        return output

    def result(self, answer: str, error: str | None = None) -> AgentResult:
        if error:
            error = redact(error)
        answer = normalize_citations(answer.strip())
        cited = [
            self.registry.by_label[label]
            for label in extract_citation_labels(answer)
            if label in self.registry.by_label
        ]
        return AgentResult(answer, list(self.log), cited, error)
