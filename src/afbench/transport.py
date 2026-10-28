"""A metering httpx transport shared by all three implementations.

Every framework here talks to the LLM through the official `openai` Python client, and
all of them accept a custom `httpx.AsyncClient`. Plugging this transport into each one
gives framework-independent measurements of what actually went over the wire:

- one record per chat completion: model, prompt/completion tokens (from the provider's
  `usage` field, not from the framework's own accounting), request size, number of
  messages and tools sent, and HTTP time;
- rate limiting handled identically for everyone: SDK retries are disabled, and this
  transport waits out HTTP 429s and proactively paces requests against the provider's
  tokens-per-minute budget. That waiting is recorded separately (`throttle_s`,
  `backoff_s`) so it can be subtracted from latency.

The meter is per process and the eval runs questions one at a time, so `Meter.reset()`
between questions is enough; this is not designed for concurrent questions.
"""

from __future__ import annotations

import asyncio
import contextlib
import json
import re
import time
from collections.abc import Awaitable, Callable
from dataclasses import dataclass, field
from typing import Any

import httpx

_DAILY_QUOTA = re.compile(r"per day|\(TPD\)|\(RPD\)", re.IGNORECASE)
_TRY_AGAIN = re.compile(r"try again in ([\d.hms]+)", re.IGNORECASE)
_DURATION_PART = re.compile(r"([\d.]+)(ms|h|m|s)")


def parse_duration(text: str) -> float:
    """Groq-style durations: '577ms', '7.66s', '1m26.4s', '2h3m'. Returns seconds."""
    total = 0.0
    for value, unit in _DURATION_PART.findall(text or ""):
        total += float(value) * {"ms": 0.001, "s": 1.0, "m": 60.0, "h": 3600.0}[unit]
    return total


@dataclass
class CallRecord:
    model: str
    status: int
    input_tokens: int
    output_tokens: int
    http_ms: float
    request_chars: int
    messages: int
    tools: int


@dataclass
class Meter:
    calls: list[CallRecord] = field(default_factory=list)
    throttle_s: float = 0.0  # proactive pacing against the tokens-per-minute budget
    backoff_s: float = 0.0  # waiting after HTTP 429 responses
    rate_limited: int = 0  # HTTP 429 responses received
    quota_exhausted: str | None = None  # set when the provider's daily cap is hit

    def reset(self) -> None:
        self.calls = []
        self.throttle_s = 0.0
        self.backoff_s = 0.0
        self.rate_limited = 0

    def for_model(self, model: str) -> list[CallRecord]:
        return [c for c in self.calls if c.model == model]


@dataclass
class _Budget:
    limit: int = 0
    remaining: int = 0
    seen_at: float = 0.0


class MeteredTransport(httpx.AsyncBaseTransport):
    def __init__(
        self,
        meter: Meter,
        inner: httpx.AsyncBaseTransport | None = None,
        max_wait_s: float = 300.0,
        daily_wait_s: float = 0.0,
        sleep: Callable[[float], Awaitable[None]] = asyncio.sleep,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        self.meter = meter
        self.inner = inner or httpx.AsyncHTTPTransport()
        self.max_wait_s = max_wait_s
        # A rolling daily token window frees up as old usage ages out. Waits up to this long
        # are waited out (the judge pass uses this); longer ones stop the run.
        self.daily_wait_s = daily_wait_s
        self._sleep = sleep
        self._clock = clock
        self._budgets: dict[str, _Budget] = {}

    async def handle_async_request(self, request: httpx.Request) -> httpx.Response:
        body = await request.aread()
        payload = _json(body)
        model = str(payload.get("model", "?"))
        waited = 0.0
        while True:
            pause = self._pacing_delay(model, len(body))
            if pause > 0:
                self.meter.throttle_s += pause
                await self._sleep(pause)
            started = self._clock()
            raw = await self.inner.handle_async_request(request)
            content = await raw.aread()
            http_ms = (self._clock() - started) * 1000
            response = _rebuild(raw, content, request)
            self._update_budget(model, response.headers)
            if response.status_code != 429:
                self._record(model, payload, len(body), response, http_ms)
                return response

            self.meter.rate_limited += 1
            text = content.decode("utf-8", "replace")
            wait = retry_after(response.headers, text)
            daily = bool(_DAILY_QUOTA.search(text))
            if daily and wait > self.daily_wait_s:
                self.meter.quota_exhausted = text[:300]
                return response
            if waited + wait > max(self.max_wait_s, self.daily_wait_s):
                return response
            waited += wait
            self.meter.backoff_s += wait
            await self._sleep(wait)

    def _pacing_delay(self, model: str, request_bytes: int) -> float:
        """Wait until the token bucket can plausibly fit this request plus its completion."""
        budget = self._budgets.get(model)
        if budget is None or budget.limit <= 0:
            return 0.0
        rate = budget.limit / 60.0  # tokens refilled per second
        available = min(budget.limit, budget.remaining + (self._clock() - budget.seen_at) * rate)
        needed = min(budget.limit, request_bytes / 3.5 + 1000)  # rough prompt estimate + completion headroom
        return max(0.0, (needed - available) / rate)

    def _update_budget(self, model: str, headers: httpx.Headers) -> None:
        try:
            limit = int(headers["x-ratelimit-limit-tokens"])
            remaining = int(headers["x-ratelimit-remaining-tokens"])
        except (KeyError, ValueError):
            return
        self._budgets[model] = _Budget(limit, remaining, self._clock())

    def _record(
        self, model: str, payload: dict[str, Any], request_bytes: int, response: httpx.Response, http_ms: float
    ) -> None:
        usage = _json(response.content).get("usage") or {}
        self.meter.calls.append(
            CallRecord(
                model=model,
                status=response.status_code,
                input_tokens=int(usage.get("prompt_tokens") or 0),
                output_tokens=int(usage.get("completion_tokens") or 0),
                http_ms=round(http_ms, 1),
                request_chars=request_bytes,
                messages=len(payload.get("messages") or []),
                tools=len(payload.get("tools") or []),
            )
        )

    async def aclose(self) -> None:
        await self.inner.aclose()


def retry_after(headers: httpx.Headers, body: str) -> float:
    """Seconds to wait after a 429: the larger of `retry-after` and the body's 'try again in'."""
    wait = 0.0
    with contextlib.suppress(ValueError):
        wait = float(headers.get("retry-after") or 0)
    if match := _TRY_AGAIN.search(body):
        wait = max(wait, parse_duration(match.group(1)))
    return max(wait, 1.0) + 0.25


def _json(data: bytes) -> dict[str, Any]:
    try:
        value = json.loads(data or b"{}")
    except ValueError:
        return {}
    return value if isinstance(value, dict) else {}


def _rebuild(raw: httpx.Response, content: bytes, request: httpx.Request) -> httpx.Response:
    # The body is already decoded; drop encoding headers so httpx doesn't decode it twice.
    headers = [(k, v) for k, v in raw.headers.multi_items() if k.lower() not in {"content-encoding", "content-length"}]
    return httpx.Response(raw.status_code, headers=headers, content=content, request=request)


def metered_client(
    meter: Meter, inner: httpx.AsyncBaseTransport | None = None, daily_wait_s: float = 0.0
) -> httpx.AsyncClient:
    return httpx.AsyncClient(transport=MeteredTransport(meter, inner, daily_wait_s=daily_wait_s), timeout=120)
