import json

import httpx
import pytest

from afbench.transport import Meter, MeteredTransport, ResponseCache, parse_duration, retry_after
from tests.conftest import completion


class FakeClock:
    def __init__(self) -> None:
        self.now = 0.0
        self.sleeps: list[float] = []

    def __call__(self) -> float:
        return self.now

    async def sleep(self, seconds: float) -> None:
        self.sleeps.append(seconds)
        self.now += seconds


def _client(handler, meter, clock):
    transport = MeteredTransport(meter, httpx.MockTransport(handler), sleep=clock.sleep, clock=clock)
    return httpx.AsyncClient(transport=transport, base_url="https://llm.test")


BODY = {"model": "m", "messages": [{"role": "user", "content": "hi"}], "tools": [{}, {}]}


def test_durations():
    assert parse_duration("577ms") == pytest.approx(0.577)
    assert parse_duration("1m26.4s") == pytest.approx(86.4)
    assert retry_after(httpx.Headers({"retry-after": "3"}), "Please try again in 7.5s.") == pytest.approx(7.75)
    assert retry_after(httpx.Headers(), "Please try again in 300ms. Need more tokens?") == pytest.approx(1.25)
    assert retry_after(httpx.Headers(), "Please try again in 8m55.2s. Need more") == pytest.approx(535.45)


async def test_records_usage_and_request_shape():
    meter, clock = Meter(), FakeClock()
    async with _client(lambda r: httpx.Response(200, json=completion("ok")), meter, clock) as client:
        response = await client.post("/chat/completions", json=BODY)
    assert response.json()["choices"][0]["message"]["content"] == "ok"
    [call] = meter.calls
    assert (call.model, call.input_tokens, call.output_tokens, call.messages, call.tools) == ("m", 100, 10, 1, 2)


async def test_429_backoff_is_waited_and_recorded_separately():
    meter, clock = Meter(), FakeClock()
    responses = [
        httpx.Response(429, headers={"retry-after": "2"}, json={"error": {"message": "Please try again in 4s"}}),
        httpx.Response(200, json=completion("ok")),
    ]
    async with _client(lambda r: responses.pop(0), meter, clock) as client:
        response = await client.post("/chat/completions", json=BODY)
    assert response.status_code == 200
    assert meter.rate_limited == 1
    assert meter.backoff_s == pytest.approx(4.25)
    assert len(meter.calls) == 1


async def test_daily_quota_is_not_retried():
    meter, clock = Meter(), FakeClock()
    error = {"error": {"message": "Rate limit reached on tokens per day (TPD): Limit 200000"}}
    async with _client(lambda r: httpx.Response(429, json=error), meter, clock) as client:
        response = await client.post("/chat/completions", json=BODY)
    assert response.status_code == 429
    assert meter.quota_exhausted and "TPD" in meter.quota_exhausted
    assert clock.sleeps == []


async def test_paces_requests_against_the_token_budget():
    meter, clock = Meter(), FakeClock()
    headers = {"x-ratelimit-limit-tokens": "6000", "x-ratelimit-remaining-tokens": "0"}
    async with _client(lambda r: httpx.Response(200, headers=headers, json=completion("ok")), meter, clock) as client:
        await client.post("/chat/completions", json=BODY)
        await client.post("/chat/completions", content=json.dumps(BODY))
    # Second request waits for ~1,000+ tokens to refill at 100 tokens/s.
    assert meter.throttle_s > 10
    meter.reset()
    assert meter.throttle_s == 0 and meter.calls == []


async def test_short_daily_quota_waits_are_waited_out_when_allowed():
    meter, clock = Meter(), FakeClock()
    error = {"error": {"message": "tokens per day (TPD): Limit 200000. Please try again in 8m55.2s."}}
    responses = [httpx.Response(429, json=error), httpx.Response(200, json=completion("ok"))]
    transport = MeteredTransport(
        meter, httpx.MockTransport(lambda r: responses.pop(0)), daily_wait_s=1800, sleep=clock.sleep, clock=clock
    )
    async with httpx.AsyncClient(transport=transport, base_url="https://llm.test") as client:
        response = await client.post("/chat/completions", json=BODY)
    assert response.status_code == 200
    assert meter.quota_exhausted is None
    assert meter.backoff_s == pytest.approx(535.45)


async def test_waits_for_the_requests_budget_to_reset():
    meter, clock = Meter(), FakeClock()
    headers = {"x-ratelimit-remaining-requests": "0", "x-ratelimit-reset-requests": "1m30s"}
    async with _client(lambda r: httpx.Response(200, headers=headers, json=completion("ok")), meter, clock) as client:
        await client.post("/chat/completions", json=BODY)
        await client.post("/chat/completions", json=BODY)
    assert meter.throttle_s == pytest.approx(90)


async def test_cached_responses_are_replayed_without_a_request(tmp_path):
    sent: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        sent.append(request)
        return httpx.Response(200, json=completion("ok"))

    for _ in range(2):
        meter, clock = Meter(), FakeClock()
        transport = MeteredTransport(
            meter, httpx.MockTransport(handler), sleep=clock.sleep, clock=clock, cache=ResponseCache(tmp_path)
        )
        async with httpx.AsyncClient(transport=transport, base_url="https://llm.test") as client:
            response = await client.post("/chat/completions", json=BODY)
            other = await client.post("/chat/completions", json={**BODY, "model": "n"})
        assert response.json()["choices"][0]["message"]["content"] == "ok"
        assert other.status_code == 200
    assert len(sent) == 2  # the second pass replayed both
    assert [c.cached for c in meter.calls] == [True, True]
    assert meter.calls[0].input_tokens == 100


async def test_errors_are_not_cached(tmp_path):
    meter, clock = Meter(), FakeClock()
    transport = MeteredTransport(
        meter,
        httpx.MockTransport(lambda r: httpx.Response(413, json={"error": {"message": "too large"}})),
        sleep=clock.sleep,
        clock=clock,
        cache=ResponseCache(tmp_path),
    )
    async with httpx.AsyncClient(transport=transport, base_url="https://llm.test") as client:
        await client.post("/chat/completions", json=BODY)
    assert list(tmp_path.iterdir()) == []
