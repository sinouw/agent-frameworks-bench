import httpx
import pytest

from afbench import evals, quota

ITEMS = [
    evals.GoldenItem("q1", "lookup", "What is the line limit?", "79 characters", ["pep-0008"], None),
    evals.GoldenItem("q2", "lookup", "Another question?", "An answer", ["pep-0020"], None),
]


def _row(framework: str, qid: str, tokens: int, **extra):
    return {
        "framework": framework,
        "id": qid,
        "category": "lookup",
        "difficulty": "core",
        "answer": "79 characters [S1]",
        "error": None,
        "tools": [{"name": "search_docs", "arguments": {}, "output": "x" * 700}],
        "input_tokens": tokens,
        "output_tokens": 0,
        "correct": None,
        **extra,
    }


def test_estimate_counts_pending_answers_and_unjudged_rows():
    rows = {
        ("a", "q1"): _row("a", "q1", 1000),
        ("b", "q1"): _row("b", "q1", 3000, correct=True),  # already judged
        ("a", "q2"): _row("a", "q2", 2000),
    }
    plan = quota.estimate(ITEMS, rows, ("a", "b"), "answer-model", "judge-model")
    assert plan["answer"]["pending_runs"] == ["b:q2"]
    assert plan["answer"]["estimated_tokens"] == 2000  # the same question's recorded run
    # Two answered rows still to judge, plus the pending one at the dearest judged cost.
    assert plan["judge"]["calls"] == 3
    one = quota._judge_tokens(ITEMS[0], rows[("a", "q1")])
    assert one > quota.JUDGE_OUTPUT_TOKENS + 200
    assert plan["judge"]["estimated_tokens"] >= 3 * quota.JUDGE_OUTPUT_TOKENS
    assert plan["fits_one_daily_window"] is True


async def test_wait_for_quota_sleeps_for_the_stated_retry_time(monkeypatch):
    responses = [
        httpx.Response(429, json={"error": {"message": "tokens per day (TPD) reached. Please try again in 2m0s."}}),
        httpx.Response(200, headers={"x-ratelimit-remaining-tokens": "7990"}, json={}),
    ]
    sleeps: list[float] = []

    async def sleep(seconds: float) -> None:
        sleeps.append(seconds)

    original = httpx.AsyncClient

    def client(**kwargs):
        return original(transport=httpx.MockTransport(lambda r: responses.pop(0)), **kwargs)

    monkeypatch.setattr(quota.httpx, "AsyncClient", client)
    result = await quota.wait_for_quota("https://llm.test", "key", "m", max_wait_s=600, sleep=sleep)
    assert result.status == 200 and result.remaining_tokens == "7990"
    assert sleeps == [pytest.approx(120.25)]


async def test_wait_for_quota_gives_up_past_the_limit(monkeypatch):
    error = {"error": {"message": "tokens per day (TPD) reached. Please try again in 3h0m0s."}}
    original = httpx.AsyncClient

    def client(**kwargs):
        return original(transport=httpx.MockTransport(lambda r: httpx.Response(429, json=error)), **kwargs)

    monkeypatch.setattr(quota.httpx, "AsyncClient", client)
    result = await quota.wait_for_quota("https://llm.test", "key", "m", max_wait_s=1800)
    assert result.daily_exhausted and result.retry_s > 10_000
    assert "frees in about 180 min" in quota.describe(result)
