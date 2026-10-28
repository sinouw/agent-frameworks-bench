"""The eval suite, identical for all three implementations.

- Retrieval (no API key): doc Hit@1/3/5, MRR, and answer-chunk Hit@5 / MRR for every
  retrieval variant over all 65 golden questions.
- End to end (LLM): each question goes to each framework in turn (the order rotates per
  question), then a judge from a different model family grades correctness and
  faithfulness. Tool selection, citations, tokens, latency and back-off come from the
  metered transport, so they are measured the same way for every framework.

Answering and judging are separate resumable passes: rows are appended to
`results/e2e/<run>.jsonl` as they finish, and the latest row per (framework, question)
wins. A provider quota that runs out mid-run therefore costs nothing already done.
"""

from __future__ import annotations

import json
import re
import statistics
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from openai import AsyncOpenAI

from afbench.common import AgentResult, QAAgent, Searcher
from afbench.transport import Meter

JUDGE_PROMPT = """You are grading a question-answering agent's answer to a question about a document collection.

Question: {question}

Reference answer (ground truth): {reference}

Agent's answer: {answer}

Context the agent retrieved with its tools:
<context>
{context}
</context>

Grade two things independently:
1. correct: does the agent's answer agree in substance with the reference answer? Extra correct
   detail is fine; missing the key fact, or contradicting it, is not. If the reference says the
   documents do not contain the answer, the agent is correct only if it declines or says the
   documents don't say, rather than inventing an answer.
2. faithful: is every factual claim in the agent's answer supported by the context above?
   An answer that only says the information isn't available is faithful.

Reply with JSON only: {{"correct": true|false, "faithful": true|false, "reason": "<one sentence>"}}"""


@dataclass
class GoldenItem:
    id: str
    category: str
    question: str
    reference_answer: str
    expected_docs: list[str] = field(default_factory=list)
    evidence: str | None = None
    expected_tool: str | None = None
    difficulty: str = "core"

    @property
    def tool(self) -> str:
        """The tool a correct run must call: SQL/calculator where the golden set says so,
        otherwise document search."""
        return self.expected_tool or "search_docs"


def load_golden(path: Path) -> list[GoldenItem]:
    return [GoldenItem(**json.loads(line)) for line in path.read_text().splitlines() if line.strip()]


def select_subset(items: list[GoldenItem], lookups_per_difficulty: int = 3, per_category: int = 2) -> list[GoldenItem]:
    """A fixed, rule-based subset for the rate-limited end-to-end run: the first
    `lookups_per_difficulty` lookup questions of each difficulty and the first
    `per_category` questions of every other category, in golden-file order."""
    taken: dict[tuple[str, str], int] = {}
    subset = []
    for item in items:
        key = (item.category, item.difficulty) if item.category == "lookup" else (item.category, "")
        limit = lookups_per_difficulty if item.category == "lookup" else per_category
        if taken.get(key, 0) < limit:
            taken[key] = taken.get(key, 0) + 1
            subset.append(item)
    return subset


# ---------------------------------------------------------------- retrieval


def _normalize(text: str) -> str:
    return re.sub(r"\s+", " ", text).lower()


def retrieval_rows(searcher: Searcher, items: list[GoldenItem], k: int = 5) -> list[dict[str, Any]]:
    rows = []
    for item in items:
        if not item.expected_docs:
            continue
        started = time.perf_counter()
        hits = searcher.search(item.question, k)
        latency_ms = (time.perf_counter() - started) * 1000
        docs = [h.doc_id for h in hits]
        rank = next((n for n, d in enumerate(docs, 1) if d in item.expected_docs), None)
        evidence_rank = None
        if item.evidence:
            needle = _normalize(item.evidence)
            evidence_rank = next(
                (n for n, h in enumerate(hits, 1) if h.doc_id in item.expected_docs and needle in _normalize(h.text)),
                None,
            )
        rows.append(
            {
                "id": item.id,
                "difficulty": item.difficulty,
                "rank": rank,
                "evidence_rank": evidence_rank,
                "has_evidence": bool(item.evidence),
                "latency_ms": round(latency_ms, 1),
            }
        )
    return rows


def retrieval_summary(rows: list[dict[str, Any]]) -> dict[str, Any]:
    n = len(rows) or 1
    ev = [r for r in rows if r["has_evidence"]]
    n_ev = len(ev) or 1
    return {
        "questions": len(rows),
        "hit_at_1": sum(1 for r in rows if r["rank"] == 1) / n,
        "hit_at_5": sum(1 for r in rows if r["rank"]) / n,
        "mrr": sum(1 / r["rank"] for r in rows if r["rank"]) / n,
        "evidence_questions": len(ev),
        "evidence_hit_at_5": sum(1 for r in ev if r["evidence_rank"]) / n_ev,
        "evidence_mrr": sum(1 / r["evidence_rank"] for r in ev if r["evidence_rank"]) / n_ev,
        "p50_latency_ms": round(statistics.median(r["latency_ms"] for r in rows), 1) if rows else 0.0,
    }


def retrieval_eval(searchers: dict[str, Searcher], items: list[GoldenItem], k: int = 5) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for name, searcher in searchers.items():
        searcher.search("warm-up query", k)  # load models before timing
        rows = retrieval_rows(searcher, items, k)
        out[name] = {
            "summary": retrieval_summary(rows),
            "by_difficulty": {
                d: retrieval_summary([r for r in rows if r["difficulty"] == d]) for d in ("core", "hard")
            },
            "rows": rows,
        }
        s = out[name]["summary"]
        print(
            f"  {name:<58} hit@1 {s['hit_at_1']:.2f}  chunk hit@5 {s['evidence_hit_at_5']:.2f}  chunk MRR {s['evidence_mrr']:.2f}"
        )
    return out


# ---------------------------------------------------------------- judge


def parse_judgement(text: str) -> dict[str, Any]:
    text = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL)
    start, end = text.find("{"), text.rfind("}")
    try:
        data = json.loads(text[start : end + 1])
    except ValueError:
        return {"correct": None, "faithful": None, "reason": f"unparseable judge output: {text[:200]}"}
    return {
        "correct": bool(data.get("correct")),
        "faithful": bool(data.get("faithful")),
        "reason": str(data.get("reason", "")),
    }


async def judge_answer(
    judge: AsyncOpenAI, model: str, item: GoldenItem, answer: str, tools: list[dict[str, Any]]
) -> dict[str, Any]:
    context = "\n\n".join(f"[{t['name']}]\n{t['output']}" for t in tools)[:16_000] or "(no tool calls)"
    prompt = JUDGE_PROMPT.format(
        question=item.question, reference=item.reference_answer, answer=answer or "(empty)", context=context
    )
    response = await judge.chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": prompt}],
        temperature=0.0,
        response_format={"type": "json_object"},
    )
    return parse_judgement(response.choices[0].message.content or "")


# ---------------------------------------------------------------- end to end


def agent_row(
    item: GoldenItem, framework: str, result: AgentResult, meter: Meter, model: str, wall_ms: float
) -> dict[str, Any]:
    calls = meter.for_model(model)
    wait_ms = (meter.throttle_s + meter.backoff_s) * 1000
    http_ms = sum(c.http_ms for c in calls)
    cited_docs = [h.doc_id for h in result.cited]
    tools_used = [t.name for t in result.tools]
    latency_ms = wall_ms - wait_ms
    return {
        "framework": framework,
        "id": item.id,
        "category": item.category,
        "difficulty": item.difficulty,
        "question": item.question,
        "answer": result.answer,
        "error": result.error,
        "tools": [{"name": t.name, "arguments": t.arguments, "output": t.output} for t in result.tools],
        "cited_docs": cited_docs,
        "citation_precision": (
            sum(1 for d in cited_docs if d in item.expected_docs) / len(cited_docs)
            if cited_docs and item.expected_docs
            else None
        ),
        "expected_tool": item.tool,
        "tool_ok": item.tool in tools_used,
        "llm_calls": len(calls),
        "input_tokens": sum(c.input_tokens for c in calls),
        "output_tokens": sum(c.output_tokens for c in calls),
        "request_chars": sum(c.request_chars for c in calls),
        "wall_ms": round(wall_ms),
        "wait_ms": round(wait_ms),  # pacing + 429 back-off, excluded from latency
        "latency_ms": round(latency_ms),
        "llm_http_ms": round(http_ms),
        "overhead_ms": round(latency_ms - http_ms),  # everything that isn't waiting on the LLM
        "rate_limited": meter.rate_limited,
    }


def load_rows(path: Path) -> dict[tuple[str, str], dict[str, Any]]:
    """Latest row per (framework, question id)."""
    rows: dict[tuple[str, str], dict[str, Any]] = {}
    if path.exists():
        for line in path.read_text().splitlines():
            if line.strip():
                row = json.loads(line)
                rows[(row["framework"], row["id"])] = row
    return rows


def _answered(row: dict[str, Any] | None) -> bool:
    return bool(row) and not (row or {}).get("error")


def _final(row: dict[str, Any] | None) -> bool:
    """Answered, or failed for a reason that is part of the result (step budget, request too
    large). Only rate-limit failures are retried on resume."""
    return bool(row) and not str((row or {}).get("error") or "").startswith("RateLimitError")


def _done(row: dict[str, Any] | None) -> bool:
    return _answered(row) and (row or {}).get("correct") is not None


def _append(path: Path, row: dict[str, Any]) -> None:
    with path.open("a") as fh:
        fh.write(json.dumps(row) + "\n")


async def answer_all(
    agents: dict[str, QAAgent], meter: Meter, items: list[GoldenItem], answer_model: str, out_path: Path
) -> bool:
    """Answer every (question, framework) pair not already answered, one at a time. The
    framework order rotates per question. Returns False if the answer model's daily quota
    stopped the run; rerunning resumes."""
    out_path.parent.mkdir(parents=True, exist_ok=True)
    existing = load_rows(out_path)
    names = list(agents)
    for index, item in enumerate(items):
        order = names[index % len(names) :] + names[: index % len(names)]
        for framework in order:
            if _final(existing.get((framework, item.id))):
                continue
            meter.reset()
            started = time.perf_counter()
            result = await agents[framework].answer(item.question)
            wall_ms = (time.perf_counter() - started) * 1000
            if meter.quota_exhausted:
                print(f"daily quota of {answer_model} exhausted during {framework} {item.id}; rerun later to resume.")
                return False
            row = agent_row(item, framework, result, meter, answer_model, wall_ms)
            row.update({"correct": None, "faithful": None, "reason": None})
            if row["error"]:
                row.update({"correct": False, "reason": "agent error"})
            _append(out_path, row)
            existing[(framework, item.id)] = row
            print(
                f"  {'ERR' if row['error'] else 'ok '} {framework:<10} {item.id} {item.category:<13} "
                f"{row['latency_ms']:>6} ms (+{row['wait_ms']} ms waiting) {row['input_tokens']}/{row['output_tokens']} tok"
            )
    return True


async def judge_all(
    judge: AsyncOpenAI, meter: Meter, items: list[GoldenItem], judge_model: str, out_path: Path
) -> bool:
    """Grade every answered row that has no verdict yet. Returns False if the judge's quota
    stopped the pass; rerunning resumes."""
    by_id = {i.id: i for i in items}
    for (framework, qid), row in load_rows(out_path).items():
        if not _answered(row) or _done(row) or qid not in by_id:
            continue
        meter.reset()
        try:
            verdict = await judge_answer(judge, judge_model, by_id[qid], row["answer"], row["tools"])
        except Exception as exc:  # leave the verdict empty for a later pass
            print(f"  judge failed on {framework} {qid}: {type(exc).__name__}; rerun later to resume.")
            return False
        calls = meter.for_model(judge_model)
        row.update(verdict)
        row["judge_input_tokens"] = sum(c.input_tokens for c in calls)
        row["judge_output_tokens"] = sum(c.output_tokens for c in calls)
        row["judge_wait_ms"] = round((meter.throttle_s + meter.backoff_s) * 1000)
        _append(out_path, row)
        print(f"  {'ok ' if row['correct'] else 'x  '} {framework:<10} {qid} {verdict['reason'][:90]}")
    return True


# ---------------------------------------------------------------- summary


def _rate(rows: list[dict[str, Any]], key: str) -> float | None:
    values = [r[key] for r in rows if r.get(key) is not None]
    return sum(1 for v in values if v) / len(values) if values else None


def _mean(rows: list[dict[str, Any]], key: str) -> float | None:
    values = [r[key] for r in rows if r.get(key) is not None]
    return statistics.fmean(values) if values else None


def percentile(values: list[float], pct: float) -> float:
    ordered = sorted(values)
    return ordered[min(len(ordered) - 1, round(pct * (len(ordered) - 1)))]


def summarize(rows: list[dict[str, Any]]) -> dict[str, Any]:
    ok = [r for r in rows if not r.get("error")]
    lat = [r["latency_ms"] for r in ok]
    over = [r["overhead_ms"] for r in ok]
    return {
        "questions": len(rows),
        "judged": sum(1 for r in rows if r.get("correct") is not None and not r.get("error")),
        "correctness": _rate(rows, "correct"),
        "faithfulness": _rate(ok, "faithful"),
        "citation_precision": _mean(ok, "citation_precision"),
        "tool_selection": _rate(rows, "tool_ok"),
        "tool_selection_sql_calc": _rate([r for r in rows if r["expected_tool"] != "search_docs"], "tool_ok"),
        "errors": len(rows) - len(ok),
        "p50_latency_ms": round(percentile(lat, 0.5)) if lat else None,
        "p95_latency_ms": round(percentile(lat, 0.95)) if lat else None,
        "p50_overhead_ms": round(percentile(over, 0.5)) if over else None,
        "avg_llm_calls": _mean(ok, "llm_calls"),
        "avg_input_tokens": round(_mean(ok, "input_tokens") or 0),
        "avg_output_tokens": round(_mean(ok, "output_tokens") or 0),
        "avg_request_chars": round(_mean(ok, "request_chars") or 0),
        "total_wait_s": round(sum(r["wait_ms"] for r in rows) / 1000, 1),
        "rate_limited": sum(r["rate_limited"] for r in rows),
    }


EFFICIENCY_KEYS = (
    "p50_latency_ms",
    "p95_latency_ms",
    "p50_overhead_ms",
    "avg_llm_calls",
    "avg_input_tokens",
    "avg_output_tokens",
    "avg_request_chars",
)


def summarize_run(rows: dict[tuple[str, str], dict[str, Any]], frameworks: list[str]) -> dict[str, Any]:
    """Only questions with a final row for every framework count, so all columns cover the
    same questions. Correctness and faithfulness cover the judged rows (agent errors count as
    incorrect); until the judge pass has run they are reported as pending."""
    ids = sorted({qid for _, qid in rows})
    paired = [q for q in ids if all(_final(rows.get((f, q))) for f in frameworks)]
    # Latency, tokens and cost compare like with like: only questions every framework answered
    # without an error (a failed run that stopped early would otherwise look cheap).
    common = [q for q in paired if all(not rows[(f, q)].get("error") for f in frameworks)]
    out: dict[str, Any] = {"paired_questions": paired, "efficiency_questions": common, "frameworks": {}}
    for f in frameworks:
        frows = [rows[(f, q)] for q in paired]
        cats: dict[str, list[dict[str, Any]]] = {}
        for r in frows:
            cats.setdefault(r["category"], []).append(r)
        efficiency = summarize([rows[(f, q)] for q in common])
        out["frameworks"][f] = {
            **summarize(frows),
            **{k: efficiency[k] for k in EFFICIENCY_KEYS},
            "by_category": {c: summarize(g) for c, g in sorted(cats.items())},
            "by_difficulty": {d: summarize([r for r in frows if r["difficulty"] == d]) for d in ("core", "hard")},
        }
    return out
