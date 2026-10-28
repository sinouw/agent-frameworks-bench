"""Quota planning for Groq's free tier, which caps each model's tokens per rolling 24 hours.

- `estimate` works out what a resumed run still has to spend, per model, from the rows
  already recorded, so the budget is written down before any call is made.
- `probe` sends one tiny request to a model and reads its rate-limit headers, so a pass
  whose daily window is still spent waits (or stops) instead of failing on its first
  real request.
"""

from __future__ import annotations

import asyncio
from dataclasses import dataclass
from typing import Any

import httpx

from afbench import evals
from afbench.evals import GoldenItem
from afbench.transport import DAILY_QUOTA, retry_after

# Groq free tier, per model (console.groq.com rate limits, observed 2026-10-28).
DAILY_TOKENS = 200_000
REQUEST_TOKENS = 8_000  # tokens-per-minute; a single request above it gets HTTP 413
CHARS_PER_TOKEN = 3.5  # conservative for English prose and JSON (more tokens, not fewer)
JUDGE_OUTPUT_TOKENS = 1_000  # verdict plus any reasoning the judge emits; deliberately high


def _tokens(row: dict[str, Any]) -> int:
    return int(row.get("input_tokens") or 0) + int(row.get("output_tokens") or 0)


def _answer_estimate(item: GoldenItem, rows: dict[tuple[str, str], dict[str, Any]]) -> int:
    """The most expensive recorded run of the same question by another framework, else of the
    same category and difficulty, else of any question."""
    recorded = [r for r in rows.values() if _tokens(r)]
    for same in (
        [r for r in recorded if r["id"] == item.id],
        [r for r in recorded if (r["category"], r["difficulty"]) == (item.category, item.difficulty)],
        recorded,
    ):
        if same:
            return max(_tokens(r) for r in same)
    return REQUEST_TOKENS


def _judge_tokens(item: GoldenItem, row: dict[str, Any]) -> int:
    prompt = evals.judge_prompt(item, row.get("answer") or "", row.get("tools") or [])
    return round(len(prompt) / CHARS_PER_TOKEN) + JUDGE_OUTPUT_TOKENS


def estimate(
    items: list[GoldenItem],
    rows: dict[tuple[str, str], dict[str, Any]],
    frameworks: tuple[str, ...],
    answer_model: str,
    judge_model: str,
) -> dict[str, Any]:
    by_id = {i.id: i for i in items}
    pending = [(f, i) for i in items for f in frameworks if not evals.final(rows.get((f, i.id)))]
    answer_tokens = sum(_answer_estimate(i, rows) for _, i in pending)

    to_judge = [(k, r) for k, r in rows.items() if k[1] in by_id and evals.answered(r) and not evals.done(r)]
    judged = [_judge_tokens(by_id[qid], r) for (_, qid), r in to_judge]
    # Answers not given yet will be judged too; assume the dearest grading seen so far.
    unseen = max(judged) if judged else REQUEST_TOKENS
    judge_tokens = sum(judged) + unseen * len(pending)
    judge_largest = max([*judged, unseen])
    return {
        "answer": {
            "model": answer_model,
            "pending_runs": [f"{f}:{i.id}" for f, i in pending],
            "estimated_tokens": answer_tokens,
        },
        "judge": {
            "model": judge_model,
            "calls": len(judged) + len(pending),
            "estimated_tokens": judge_tokens,
            "largest_request_tokens": judge_largest,
        },
        "limits": {"daily_tokens_per_model": DAILY_TOKENS, "request_tokens": REQUEST_TOKENS},
        "fits_one_daily_window": answer_tokens <= DAILY_TOKENS and judge_tokens <= DAILY_TOKENS,
        "assumptions": [
            "each pending answer costs as much as the most expensive recorded run of the same question "
            "(else of the same category and difficulty)",
            f"judge prompt tokens = prompt characters / {CHARS_PER_TOKEN}, plus {JUDGE_OUTPUT_TOKENS} output tokens",
            "answers still to be given are judged at the cost of the dearest judge prompt seen so far",
        ],
    }


@dataclass
class Probe:
    model: str
    status: int
    remaining_tokens: str | None
    remaining_requests: str | None
    reset_tokens: str | None
    reset_requests: str | None
    daily_exhausted: bool
    retry_s: float


async def probe(client: httpx.AsyncClient, base_url: str, api_key: str, model: str) -> Probe:
    """One request with a few tokens of prompt and output; never retried."""
    response = await client.post(
        f"{base_url.rstrip('/')}/chat/completions",
        headers={"Authorization": f"Bearer {api_key}"},
        json={"model": model, "messages": [{"role": "user", "content": "Reply with OK."}], "max_tokens": 8},
    )
    h = response.headers
    exhausted = response.status_code == 429 and bool(DAILY_QUOTA.search(response.text))
    return Probe(
        model=model,
        status=response.status_code,
        remaining_tokens=h.get("x-ratelimit-remaining-tokens"),
        remaining_requests=h.get("x-ratelimit-remaining-requests"),
        reset_tokens=h.get("x-ratelimit-reset-tokens"),
        reset_requests=h.get("x-ratelimit-reset-requests"),
        daily_exhausted=exhausted,
        retry_s=retry_after(h, response.text) if response.status_code == 429 else 0.0,
    )


def describe(p: Probe) -> str:
    if p.status == 200:
        return (
            f"{p.model}: available ({p.remaining_tokens} tokens this minute, "
            f"{p.remaining_requests} requests left, requests reset in {p.reset_requests})"
        )
    if p.daily_exhausted:
        return f"{p.model}: daily window spent, frees in about {round(p.retry_s / 60)} min"
    return f"{p.model}: HTTP {p.status}"


async def wait_for_quota(
    base_url: str, api_key: str, model: str, max_wait_s: float, sleep: Any = asyncio.sleep
) -> Probe:
    """Probe until the model's daily window has room, sleeping for the provider's stated
    retry time between probes, for at most `max_wait_s` in total."""
    waited = 0.0
    async with httpx.AsyncClient(timeout=60) as client:
        while True:
            result = await probe(client, base_url, api_key, model)
            print(f"  probe {describe(result)}")
            if result.status == 200 or not result.daily_exhausted or waited + result.retry_s > max_wait_s:
                return result
            waited += result.retry_s
            await sleep(result.retry_s)
