from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from backends.anthropic.build_requests import MAX_REQUESTS_PER_BATCH, build_message_requests, write_requests_preview
from backends.anthropic.normalize_to_openai import normalize_jsonl
from backends.anthropic.poll_and_stream import poll_until_complete, stream_results_to_jsonl


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
    assert req.custom_id == "dataset|item|control|0.0|0|open"
    assert req.params["model"] == "claude-sonnet-3.5"
    assert req.params["system"] == "You are a helpful assistant."
    assert req.params["messages"][0]["role"] == "user"
    assert req.params["messages"][0]["content"] == "Outline selective risk."
    assert req.params["stop_sequences"] == ["</end>"]
    assert req.params["max_tokens"] == 1536
    assert req.params["metadata"]["trial"] == "slug-123"

    preview = tmp_path / "preview.jsonl"
    written = write_requests_preview(requests, str(preview))
    assert written == 1
    dumped = preview.read_text(encoding="utf-8").strip().splitlines()
    assert json.loads(dumped[0])["custom_id"] == req.custom_id


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
