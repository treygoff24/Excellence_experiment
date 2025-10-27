from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping

import pytest

from backends.anthropic.build_requests import RequestPayload
from backends.anthropic.start_message_batch import start_message_batch


@dataclass
class _FakeBatches:
    captured_kwargs: Mapping[str, Any] | None = None

    def create(self, **kwargs: Any) -> dict[str, Any]:
        self.captured_kwargs = kwargs
        return {"id": "batch_test"}


@dataclass
class _FakeMessages:
    batches: _FakeBatches


@dataclass
class _FakeClient:
    messages: _FakeMessages


def _make_payload(custom_id: str = "req-1") -> RequestPayload:
    return RequestPayload(
        custom_id=custom_id,
        params={"model": "claude-3-5-haiku-20241022", "temperature": 1.0, "messages": [{"role": "user", "content": "hi"}]},
        orig_custom_id=custom_id,
        metadata={},
    )


def test_start_message_batch_rejects_completion_window() -> None:
    batches = _FakeBatches()
    client = _FakeClient(messages=_FakeMessages(batches=batches))

    with pytest.raises(ValueError, match="completion_window"):
        start_message_batch(
            client,
            [_make_payload()],
            batch_params={"completion_window": "24h"},
        )


def test_start_message_batch_merges_extra_body_overrides() -> None:
    batches = _FakeBatches()
    client = _FakeClient(messages=_FakeMessages(batches=batches))

    start_message_batch(
        client,
        [_make_payload()],
        batch_params={"extra_body": {"top_k": 32}, "max_tokens": 256},
    )

    assert batches.captured_kwargs is not None
    assert batches.captured_kwargs.get("extra_body") == {"top_k": 32, "max_tokens": 256}


def test_start_message_batch_rejects_non_mapping_extra_body() -> None:
    batches = _FakeBatches()
    client = _FakeClient(messages=_FakeMessages(batches=batches))

    with pytest.raises(TypeError):
        start_message_batch(
            client,
            [_make_payload()],
            batch_params={"extra_body": 5},
        )
