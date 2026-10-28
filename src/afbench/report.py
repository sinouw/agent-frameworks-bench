"""Turns the committed raw results into `results/summary.{json,md}`."""

from __future__ import annotations

import json
from typing import Any

from afbench import evals
from afbench.config import Settings
from afbench.stack import FRAMEWORKS, MATCHED

# USD per 1M tokens (input, output), from console.groq.com/docs/models, read 2026-10-28.
PRICES_PER_MTOK: dict[str, tuple[float, float]] = {
    "openai/gpt-oss-20b": (0.075, 0.30),
    "qwen/qwen3.8-27b": (0.80, 4.00),
}


def cost_per_question(model: str, input_tokens: float, output_tokens: float) -> float | None:
    price = PRICES_PER_MTOK.get(model)
    if price is None:
        return None
    return (input_tokens * price[0] + output_tokens * price[1]) / 1_000_000


def _load(path: Any) -> Any:
    return json.loads(path.read_text()) if path.exists() else None


def build(settings: Settings, run: str) -> dict[str, Any]:
    results = settings.results_dir
    rows = evals.load_rows(results / "e2e" / f"{run}.jsonl")
    e2e = evals.summarize_run(rows, list(FRAMEWORKS)) if rows else None
    if e2e:
        for summary in e2e["frameworks"].values():
            summary["avg_cost_usd"] = cost_per_question(
                settings.answer_model, summary["avg_input_tokens"], summary["avg_output_tokens"]
            )
    retrieval = _load(results / "retrieval.json")
    return {
        "answer_model": settings.answer_model,
        "judge_model": settings.judge_model,
        "matched_retrieval": MATCHED,
        "e2e": e2e,
        "retrieval": {name: {k: v for k, v in r.items() if k != "rows"} for name, r in (retrieval or {}).items()},
        "loc": _load(results / "loc.json"),
        "index": _load(results / "index.json"),
    }


def _pct(value: float | None) -> str:
    return "—" if value is None else f"{value * 100:.0f}%"


def _num(value: float | None, digits: int = 0) -> str:
    return "—" if value is None else f"{value:,.{digits}f}"


def markdown(summary: dict[str, Any]) -> str:
    lines = [f"# Results\n\nAnswer model `{summary['answer_model']}`, judge `{summary['judge_model']}`.\n"]
    e2e = summary.get("e2e")
    if e2e:
        n = len(e2e["paired_questions"])
        lines += [
            f"## End to end ({n} questions, each answered by all three)\n",
            "| Framework | Correct | Faithful | Citation precision | Tool selection | p50 / p95 latency | "
            "p50 non-LLM time | LLM calls | Tokens in / out | Prompt chars sent | Cost / question | Errors | 429s |",
            "|---|---|---|---|---|---|---|---|---|---|---|---|---|",
        ]
        for name, s in e2e["frameworks"].items():
            cost = "—" if s.get("avg_cost_usd") is None else f"${s['avg_cost_usd']:.5f}"
            lines.append(
                f"| {name} | {_pct(s['correctness'])} | {_pct(s['faithfulness'])} | {_pct(s['citation_precision'])} "
                f"| {_pct(s['tool_selection'])} | {_num(s['p50_latency_ms'])} / {_num(s['p95_latency_ms'])} ms "
                f"| {_num(s['p50_overhead_ms'])} ms | {_num(s['avg_llm_calls'], 1)} "
                f"| {_num(s['avg_input_tokens'])} / {_num(s['avg_output_tokens'])} | {_num(s['avg_request_chars'])} "
                f"| {cost} | {s['errors']} | {s['rate_limited']} |"
            )
        lines += [
            "",
            "### By category\n",
            "| Category | " + " | ".join(e2e["frameworks"]) + " |",
            "|---|" + "---|" * len(e2e["frameworks"]),
        ]
        categories = sorted({c for s in e2e["frameworks"].values() for c in s["by_category"]})
        for cat in categories:
            cells = []
            for s in e2e["frameworks"].values():
                b = s["by_category"].get(cat)
                cells.append(
                    "—" if b is None else f"{_pct(b['correctness'])} correct, {_pct(b['tool_selection'])} tool"
                )
            count = next(iter(e2e["frameworks"].values()))["by_category"][cat]["questions"]
            lines.append(f"| {cat} ({count}) | " + " | ".join(cells) + " |")
        lines.append("")
    if summary.get("retrieval"):
        lines += [
            "## Retrieval (all golden questions, no LLM)\n",
            "| Variant | Doc Hit@1 | Doc Hit@5 | Doc MRR | Answer-chunk Hit@5 | Answer-chunk MRR | Chunk MRR core / hard | p50 latency |",
            "|---|---|---|---|---|---|---|---|",
        ]
        for name, r in summary["retrieval"].items():
            s, d = r["summary"], r["by_difficulty"]
            lines.append(
                f"| {name} | {_pct(s['hit_at_1'])} | {_pct(s['hit_at_5'])} | {s['mrr']:.2f} | {_pct(s['evidence_hit_at_5'])} "
                f"| {s['evidence_mrr']:.2f} | {d['core']['evidence_mrr']:.2f} / {d['hard']['evidence_mrr']:.2f} "
                f"| {s['p50_latency_ms']:.0f} ms |"
            )
        lines.append("")
    if summary.get("loc"):
        lines += [
            "## Lines of code (logical lines: no blanks, comments or docstrings)\n",
            "| Part | Lines |",
            "|---|---|",
        ]
        for name, files in summary["loc"].items():
            lines.append(f"| {name} | {sum(files.values())} |")
        lines.append("")
    if summary.get("index"):
        idx = summary["index"]
        lines += [
            f"## Indexing ({idx['documents']} documents, {idx['chunks']} chunks incl. metadata cards)\n",
            "| Implementation | Seconds |",
            "|---|---|",
            *[f"| {k} | {v} |" for k, v in idx["seconds"].items()],
            "",
        ]
    return "\n".join(lines)
