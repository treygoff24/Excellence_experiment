from __future__ import annotations

from types import SimpleNamespace

import pytest

from backends.anthropic.rate_limiter import AnthropicBatchLimiter


def _make_client(pages: list) -> SimpleNamespace:
    class _Batches:
        def __init__(self, response_pages: list):
            self._pages = response_pages
            self.call_count = 0

        def list(self, **kwargs):  # type: ignore[no-untyped-def]
            idx = min(self.call_count, len(self._pages) - 1)
            self.call_count += 1
            return self._pages[idx]

    return SimpleNamespace(messages=SimpleNamespace(batches=_Batches(pages)))


def test_limiter_waits_for_queue_capacity() -> None:
    batches = [
        SimpleNamespace(
            data=[
                SimpleNamespace(
                    processing_status="in_progress",
                    request_counts={"processing": 85_000},
                )
            ],
            has_more=False,
        ),
        SimpleNamespace(
            data=[
                SimpleNamespace(
                    processing_status="in_progress",
                    request_counts={"processing": 70_000},
                )
            ],
            has_more=False,
        ),
    ]
    client = _make_client(batches)
    timeline = {"now": 0.0, "sleeps": []}

    def fake_time() -> float:
        return timeline["now"]

    def fake_sleep(seconds: float) -> None:
        timeline["sleeps"].append(seconds)
        timeline["now"] += seconds

    limiter = AnthropicBatchLimiter(
        processing_limit=100_000,
        safety_margin=0.1,
        queue_poll_seconds=5.0,
        rpm_limit=None,
        max_retries=3,
        retry_after_fallback=10.0,
        time_fn=fake_time,
        sleep_fn=fake_sleep,
    )

    limiter.before_submit(client, request_count=15_000)

    assert timeline["sleeps"] == [5.0]
    assert client.messages.batches.call_count == 2


def test_limiter_enforces_rpm_window() -> None:
    client = SimpleNamespace()  # unused by RPM limiter
    clock = {"now": 0.0, "sleeps": []}

    def fake_time() -> float:
        return clock["now"]

    def fake_sleep(seconds: float) -> None:
        clock["sleeps"].append(seconds)
        clock["now"] += seconds

    limiter = AnthropicBatchLimiter(
        processing_limit=None,
        safety_margin=0.1,
        queue_poll_seconds=5.0,
        rpm_limit=2_000,
        max_retries=3,
        retry_after_fallback=10.0,
        time_fn=fake_time,
        sleep_fn=fake_sleep,
    )
    # Register 1,800 requests at t=0, simulate current time just under 60s later.
    limiter.register_batch("existing", 1_800)
    clock["now"] = 59.0

    limiter.before_submit(client, request_count=400)

    assert clock["sleeps"] == [1.0]
    # After waiting, the limiter should allow registration of the new batch.
    limiter.register_batch("new", 400)
    assert limiter._pending_batches["new"] == 400


def test_limiter_retry_delay_honors_retry_after(monkeypatch: pytest.MonkeyPatch) -> None:
    limiter = AnthropicBatchLimiter(
        processing_limit=None,
        safety_margin=0.1,
        queue_poll_seconds=5.0,
        rpm_limit=None,
        max_retries=3,
        retry_after_fallback=4.0,
    )
    monkeypatch.setattr("backends.anthropic.rate_limiter.random.uniform", lambda a, b: a)

    class FakeError(Exception):
        def __init__(self, retry_after: str | None):
            headers = {}
            if retry_after is not None:
                headers["Retry-After"] = retry_after
            self.response = SimpleNamespace(headers=headers)
            self.body = {"error": {"type": "rate_limit_error"}}

    delay = limiter.compute_retry_delay(FakeError("2"), attempt=0)
    assert delay >= 4.0  # fallback dominates

    delay2 = limiter.compute_retry_delay(FakeError("8"), attempt=1)
    assert delay2 >= 8.0  # larger retry-after respected

    with pytest.raises(FakeError):
        limiter.compute_retry_delay(FakeError(None), attempt=3)
