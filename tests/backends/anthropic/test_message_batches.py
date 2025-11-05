from __future__ import annotations

import csv
import json
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

from backends.anthropic.adapter import AnthropicBatchAdapter
from backends.anthropic.build_requests import MAX_REQUESTS_PER_BATCH, build_message_requests, write_requests_preview
from backends.anthropic.normalize_to_openai import normalize_jsonl
from backends.anthropic.poll_and_stream import poll_until_complete, stream_results_to_jsonl
from fireworks.parse_results import process_results


def _jsonl_write(path: Path, rows: list[dict]) -> None:
    with path.open("w", encoding="utf-8") as fout:
        for row in rows:
            fout.write(json.dumps(row) + "\n")


def test_build_message_requests_basic(tmp_path: Path) -> None:
    src = tmp_path / "shard.jsonl"
    _jsonl_write(
        src,
        [
            {
                "custom_id": "dataset|item|control|0.0|0|open",
                "body": {
                    "messages": [
                        {"role": "system", "content": "You are a helpful assistant."},
                        {"role": "user", "content": "Outline selective risk."},
                    ],
                    "stop": ["</end>"],
                },
            }
        ],
    )
    requests = build_message_requests(
        src_path=str(src),
        model="claude-sonnet-3.5",
        temperature=0.4,
        top_p=0.95,
        top_k=50,
        max_new_tokens={"open_book": 1536, "default": 1024},
        metadata={"trial": "slug-123"},
    )
    assert len(requests) == 1
    req = requests[0]
    assert req.custom_id == "dataset_item_control_0_0_0_open"
    assert req.orig_custom_id == "dataset|item|control|0.0|0|open"
    assert req.params["model"] == "claude-sonnet-3.5"
    system_blocks = req.params["system"]
    assert isinstance(system_blocks, list)
    assert system_blocks[0]["text"] == "You are a helpful assistant."
    assert system_blocks[0]["cache_control"]["type"] == "ephemeral"
    assert req.params["messages"][0]["role"] == "user"
    user_blocks = req.params["messages"][0]["content"]
    assert isinstance(user_blocks, list)
    assert user_blocks[0]["text"] == "Outline selective risk."
    assert req.params["stop_sequences"] == ["</end>"]
    assert req.params["max_tokens"] == 1536
    assert req.metadata["trial"] == "slug-123"

    preview = tmp_path / "preview.jsonl"
    written = write_requests_preview(requests, str(preview))
    assert written == 1
    dumped = preview.read_text(encoding="utf-8").strip().splitlines()
    assert json.loads(dumped[0])["custom_id"] == req.custom_id


def test_build_message_requests_respects_cache_ttl(tmp_path: Path) -> None:
    src = tmp_path / "shard_ttl.jsonl"
    _jsonl_write(
        src,
        [
            {
                "custom_id": "dataset|item|control|0.0|0|open",
                "body": {
                    "messages": [
                        {
                            "role": "system",
                            "content": [
                                {"type": "text", "text": "System prompt.", "cache_control": {"type": "ephemeral", "ttl": "5m"}},
                            ],
                        },
                        {"role": "user", "content": "Question?"},
                    ]
                },
            }
        ],
    )
    requests = build_message_requests(
        src_path=str(src),
        model="claude-sonnet-3.5",
        temperature=0.4,
        top_p=None,
        top_k=None,
        max_new_tokens={"default": 512},
        cache_control={"enable_system_cache": True, "ttl": "1h"},
    )
    assert len(requests) == 1
    system_blocks = requests[0].params["system"]
    assert system_blocks[0]["cache_control"]["ttl"] == "1h"


def test_build_message_requests_preserves_existing_when_no_override(tmp_path: Path) -> None:
    src = tmp_path / "shard_existing.jsonl"
    _jsonl_write(
        src,
        [
            {
                "custom_id": "dataset|item|control|0.0|0|open",
                "body": {
                    "messages": [
                        {
                            "role": "system",
                            "content": [
                                {"type": "text", "text": "System prompt.", "cache_control": {"type": "ephemeral", "ttl": "5m"}},
                            ],
                        },
                        {"role": "user", "content": "Question?"},
                    ]
                },
            }
        ],
    )
    requests = build_message_requests(
        src_path=str(src),
        model="claude-sonnet-3.5",
        temperature=0.4,
        top_p=None,
        top_k=None,
        max_new_tokens={"default": 512},
    )
    assert len(requests) == 1
    system_blocks = requests[0].params["system"]
    assert system_blocks[0]["cache_control"]["ttl"] == "5m"


def test_build_message_requests_skips_cache_when_disabled(tmp_path: Path) -> None:
    src = tmp_path / "shard_nocache.jsonl"
    _jsonl_write(
        src,
        [
            {
                "custom_id": "dataset|item|control|0.0|0|open",
                "body": {
                    "messages": [
                        {
                            "role": "system",
                            "content": [
                                {"type": "text", "text": "Short.", "cache_control": {"type": "ephemeral", "ttl": "5m"}},
                            ],
                        },
                        {"role": "user", "content": "Hi"},
                    ]
                },
            }
        ],
    )
    requests = build_message_requests(
        src_path=str(src),
        model="claude-sonnet-3.5",
        temperature=0.1,
        top_p=None,
        top_k=None,
        max_new_tokens={"default": 128},
        cache_control={"enable_system_cache": False},
    )
    assert len(requests) == 1
    system_blocks = requests[0].params["system"]
    assert isinstance(system_blocks, list)
    assert "cache_control" not in system_blocks[0]


def test_build_message_requests_enforces_limit(tmp_path: Path) -> None:
    src = tmp_path / "shard_many.jsonl"
    rows = []
    for idx in range(MAX_REQUESTS_PER_BATCH + 1):
        rows.append(
            {
                "custom_id": f"dset|{idx}|control|0.0|0|open",
                "body": {
                    "messages": [
                        {"role": "system", "content": "System."},
                        {"role": "user", "content": f"Question {idx}?"},
                    ]
                },
            }
        )
    _jsonl_write(src, rows)
    with pytest.raises(ValueError):
        build_message_requests(
            src_path=str(src),
            model="claude-3-opus",
            temperature=0.1,
            top_p=None,
            top_k=None,
            max_new_tokens={"open_book": 2048},
        )


def test_build_message_requests_rejects_non_unit_temp_with_thinking(tmp_path: Path) -> None:
    src = tmp_path / "shard_thinking.jsonl"
    _jsonl_write(
        src,
        [
            {
                "custom_id": "dataset|item|control|0.5|0|open",
                "body": {
                    "messages": [
                        {"role": "system", "content": "You are a helpful assistant."},
                        {"role": "user", "content": "Hello?"},
                    ]
                },
            }
        ],
    )
    with pytest.raises(ValueError, match="temperature=0.5"):
        build_message_requests(
            src_path=str(src),
            model="claude-sonnet-3.5",
            temperature=0.5,
            top_p=None,
            top_k=None,
            max_new_tokens={"default": 512},
            request_overrides={"thinking": {"type": "enabled", "budget_tokens": 128}},
        )


def test_build_message_requests_allows_thinking_with_unit_temp(tmp_path: Path) -> None:
    src = tmp_path / "shard_thinking_allowed.jsonl"
    _jsonl_write(
        src,
        [
            {
                "custom_id": "dataset|item|control|1.0|0|open",
                "body": {
                    "messages": [
                        {"role": "user", "content": "Hi"},
                    ]
                },
            }
        ],
    )
    requests = build_message_requests(
        src_path=str(src),
        model="claude-sonnet-3.5",
        temperature=1.0,
        top_p=None,
        top_k=None,
        max_new_tokens={"default": 256},
        request_overrides={"thinking": {"type": "enabled", "budget_tokens": 64}},
    )
    assert len(requests) == 1
    assert requests[0].params["temperature"] == 1.0
    assert requests[0].params["thinking"]["type"] == "enabled"


def test_build_message_requests_rejects_override_temperature_with_thinking(tmp_path: Path) -> None:
    src = tmp_path / "shard_thinking_override.jsonl"
    _jsonl_write(
        src,
        [
            {
                "custom_id": "dataset|item|control|1.0|0|open",
                "body": {
                    "messages": [
                        {"role": "user", "content": "Hi"},
                    ]
                },
            }
        ],
    )
    with pytest.raises(ValueError, match="temperature=0.3"):
        build_message_requests(
            src_path=str(src),
            model="claude-sonnet-3.5",
            temperature=1.0,
            top_p=None,
            top_k=None,
            max_new_tokens={"default": 256},
            request_overrides={
                "thinking": {"type": "enabled"},
                "temperature": 0.3,
            },
        )


def test_normalize_jsonl_handles_success_and_failure(tmp_path: Path) -> None:
    src = tmp_path / "raw.jsonl"
    _jsonl_write(
        src,
        [
            {
                "id": "msgres_1",
                "custom_id": "dataset|item|treatment|0.0|0|open",
                "processing_status": "ended",
                "result": {
                    "type": "succeeded",
                    "message": {
                        "id": "msg_1",
                        "type": "message",
                        "role": "assistant",
                        "content": [
                            {"type": "text", "text": "Hello"},
                            {"type": "text", "text": " world!"},
                        ],
                        "usage": {"input_tokens": 42, "output_tokens": 128},
                        "stop_reason": "end_turn",
                        "model": "claude-3.5-sonnet",
                    },
                },
            },
            {
                "id": "msgres_2",
                "custom_id": "dataset|item|control|0.0|0|open",
                "processing_status": "ended",
                "result": {
                    "type": "errored",
                    "error": {"type": "invalid_request", "message": "Bad prompt"},
                },
            },
        ],
    )
    dest = tmp_path / "normalized.jsonl"
    count = normalize_jsonl(str(src), str(dest))
    assert count == 2
    lines = dest.read_text(encoding="utf-8").strip().splitlines()
    first = json.loads(lines[0])
    assert first["custom_id"] == "dataset|item|treatment|0.0|0|open"
    body = first["response"]["body"]
    assert body["choices"][0]["message"]["content"] == "Hello world!"
    usage = first["response"]["usage"]
    assert usage["prompt_tokens"] == 42
    assert usage["completion_tokens"] == 128
    assert usage["total_tokens"] == 170

    second = json.loads(lines[1])
    assert second["error"]["message"] == "Bad prompt"
    assert second["response"]["status"] == "errored"

    out_csv = tmp_path / "predictions.csv"
    process_results(str(dest), str(out_csv))
    with out_csv.open("r", encoding="utf-8", newline="") as fcsv:
        rows = list(csv.DictReader(fcsv))
    assert len(rows) == count
    success_row = rows[0]
    assert success_row["custom_id"] == "dataset|item|treatment|0.0|0|open"
    assert success_row["response_text"] == "Hello world!"
    assert success_row["finish_reason"] == "end_turn"
    assert success_row["request_id"] == "msg_1"
    # CSV writer stringifies integers; validate tokens survived normalization
    assert success_row["prompt_tokens"] == "42"
    assert success_row["completion_tokens"] == "128"


class _StubResults:
    def __init__(self, responses: list[SimpleNamespace], rows: list[dict]):
        self._responses = responses
        self._rows = rows
        self._retrieve_calls = 0

    def retrieve(self, batch_id: str) -> SimpleNamespace:
        if self._retrieve_calls >= len(self._responses):
            return self._responses[-1]
        resp = self._responses[self._retrieve_calls]
        self._retrieve_calls += 1
        return resp

    def results(self, batch_id: str):
        for row in self._rows:
            yield SimpleNamespace(model_dump_json=lambda data=row: json.dumps(data))


def test_poll_and_stream_writes_results(tmp_path: Path) -> None:
    responses = [
        SimpleNamespace(processing_status="in_progress"),
        SimpleNamespace(processing_status="ended", request_counts={"succeeded": 1}),
    ]
    rows = [
        {
            "custom_id": "dataset|item|control|0.0|0|open",
            "result": {"type": "succeeded", "message": {"content": "ok"}},
        }
    ]
    stub_batches = _StubResults(responses, rows)
    client = SimpleNamespace(messages=SimpleNamespace(batches=stub_batches))

    job = poll_until_complete(client, "batch_123", poll_seconds=0.0)
    assert job.processing_status == "ended"

    out_path = tmp_path / "out.jsonl"
    count = stream_results_to_jsonl(client, "batch_123", str(out_path))
    assert count == 1
    written = out_path.read_text(encoding="utf-8").strip()
    assert "custom_id" in written


class _StubLimiter:
    def __init__(self):
        self.before_calls: list[tuple[Any, int]] = []
        self.register_calls: list[tuple[str, int]] = []
        self.retry_calls: list[tuple[int, Any]] = []
        self.completed: list[str] = []

    def before_submit(self, client: Any, request_count: int) -> None:
        self.before_calls.append((client, request_count))

    def register_batch(self, batch_id: str | None, request_count: int) -> None:
        if batch_id:
            self.register_calls.append((batch_id, request_count))

    def compute_retry_delay(self, error: Any, attempt: int) -> float:
        self.retry_calls.append((attempt, error))
        return 0.5

    def mark_batch_complete(self, batch: Any) -> None:
        batch_id = getattr(batch, "id", None) or getattr(batch, "batch_id", None)
        if batch_id:
            self.completed.append(batch_id)


def test_adapter_retries_and_registers_batches(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    src = tmp_path / "batch.jsonl"
    _jsonl_write(
        src,
        [
            {
                "custom_id": "item|0",
                "body": {
                    "messages": [
                        {"role": "system", "content": "You are concise."},
                        {"role": "user", "content": "Hi"},
                    ]
                },
            }
        ],
    )

    class FakeRateLimitError(Exception):
        status_code = 429

        def __init__(self):
            self.response = SimpleNamespace(headers={"Retry-After": "1"})
            self.body = {"error": {"type": "rate_limit_error"}}

    class _Batches:
        def __init__(self):
            self.calls = 0

        def create(self, **kwargs):
            self.calls += 1
            if self.calls == 1:
                raise FakeRateLimitError()
            return SimpleNamespace(
                id="msgbatch_123",
                processing_status="in_progress",
                request_counts={"processing": len(kwargs.get("requests", []))},
            )

    client = SimpleNamespace(messages=SimpleNamespace(batches=_Batches()))

    cfg = {
        "model_id": "claude-test",
        "paths": {"batch_inputs_dir": str(tmp_path)},
        "max_new_tokens": {"open_book": 128, "closed_book": 128},
        "top_p": 1.0,
        "top_k": 50,
        "provider": {"name": "anthropic", "model": "claude-test"},
    }
    adapter = AnthropicBatchAdapter(cfg, client=client)
    adapter._anthropic = SimpleNamespace(RateLimitError=FakeRateLimitError)
    limiter = _StubLimiter()
    adapter._limiter = limiter

    sleep_calls: list[float] = []
    monkeypatch.setattr("backends.anthropic.adapter.time.sleep", lambda seconds: sleep_calls.append(seconds))

    artifact = SimpleNamespace(
        condition="control",
        temp=0.0,
        mode="producer",
        batch_id=None,
        extra={
            "model_id": "claude-test",
            "source_jsonl": str(src),
            "prompt_set": "default",
            "part_suffix": "",
        },
    )

    result = adapter.submit(trial_slug="trial-1", artifacts=artifact, dry_run=False)

    assert result.batch_id == "msgbatch_123"
    assert result.extra.get("rate_limit_retries") == 1
    assert sleep_calls == [0.5]
    assert len(limiter.before_calls) == 2  # one per attempt
    assert limiter.register_calls == [("msgbatch_123", 1)]


def test_adapter_poll_marks_completion(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    results_dir = tmp_path / "results"
    results_dir.mkdir()

    class _Results:
        def retrieve(self, batch_id: str):
            return SimpleNamespace(
                id=batch_id,
                processing_status="ended",
                request_counts={"processing": 0},
            )

        def results(self, batch_id: str):
            row = {
                "custom_id": "item|0",
                "result": {"type": "succeeded", "message": {"content": "ok"}},
            }
            yield SimpleNamespace(model_dump_json=lambda data=None, row=row: json.dumps(row))

    client = SimpleNamespace(messages=SimpleNamespace(batches=_Results()))

    cfg = {
        "model_id": "claude-test",
        "paths": {"batch_inputs_dir": str(tmp_path)},
        "max_new_tokens": {"open_book": 128, "closed_book": 128},
        "top_p": 1.0,
        "top_k": 50,
        "provider": {"name": "anthropic", "model": "claude-test", "poll_seconds": 0.0},
    }
    adapter = AnthropicBatchAdapter(cfg, client=client)
    limiter = _StubLimiter()
    adapter._limiter = limiter

    artifact = SimpleNamespace(
        condition="control",
        temp=0.0,
        mode="producer",
        batch_id="msgbatch_456",
        extra={},
    )

    updated = adapter.poll(results_dir=str(results_dir), artifact=artifact, dry_run=False)
    assert updated.extra["normalized_rows"] == 1
    assert limiter.completed == ["msgbatch_456"]
