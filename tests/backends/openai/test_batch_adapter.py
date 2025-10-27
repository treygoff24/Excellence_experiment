from __future__ import annotations

import csv
import json
import os
from pathlib import Path
from types import SimpleNamespace

import pytest

from backends.openai.build_inputs import build_batch_requests
from backends.openai.normalize import normalize_jsonl
from backends.openai.poll_and_download import download_and_extract, poll_until_complete
from fireworks.parse_results import process_results


def _read_jsonl(path: Path) -> list[dict]:
    with open(path, "r", encoding="utf-8") as fin:
        return [json.loads(line) for line in fin if line.strip()]


def test_build_batch_requests_responses(tmp_path: Path) -> None:
    src = tmp_path / "shard.jsonl"
    src.write_text(
        json.dumps(
            {
                "custom_id": "dataset|item|control|0.0|0|open",
                "body": {
                    "messages": [
                        {"role": "system", "content": "System prompt."},
                        {"role": "user", "content": "User question?"},
                    ],
                    "stop": ["</end>"],
                },
            }
        )
        + "\n",
        encoding="utf-8",
    )
    dest = tmp_path / "requests.jsonl"
    count = build_batch_requests(
        src_path=str(src),
        dest_path=str(dest),
        model="gpt-4.1-mini",
        temperature=0.0,
        top_p=0.9,
        max_new_tokens={"open_book": 2048, "closed_book": 1024},
        endpoint="/v1/responses",
        metadata={"trial": "slug-123", "condition": "control"},
    )
    assert count == 1
    record = _read_jsonl(dest)[0]
    assert record["method"] == "POST"
    assert record["url"] == "/v1/responses"
    body = record["body"]
    assert body["model"] == "gpt-4.1-mini"
    assert body["input"][0]["role"] == "system"
    assert body["input"][1]["role"] == "user"
    assert body["metadata"]["condition"] == "control"
    assert body["max_output_tokens"] == 2048
    assert body["stop"] == ["</end>"]


def test_build_batch_requests_rejects_thinking_without_budget(tmp_path: Path) -> None:
    src = tmp_path / "shard.jsonl"
    src.write_text(
        json.dumps(
            {
                "custom_id": "dataset|item|control|0.0|0|open",
                "body": {
                    "messages": [
                        {"role": "system", "content": "System prompt."},
                        {"role": "user", "content": "User question?"},
                    ]
                },
            }
        )
        + "\n",
        encoding="utf-8",
    )
    dest = tmp_path / "requests.jsonl"
    with pytest.raises(ValueError):
        build_batch_requests(
            src_path=str(src),
            dest_path=str(dest),
            model="gpt-4.1-mini",
            temperature=0.0,
            top_p=None,
            max_new_tokens=None,
            endpoint="/v1/responses",
            metadata=None,
            request_overrides={"thinking": {"type": "enabled", "budget_tokens": 0}},
        )


def test_build_batch_requests_accepts_thinking_with_budget(tmp_path: Path) -> None:
    src = tmp_path / "shard.jsonl"
    src.write_text(
        json.dumps(
            {
                "custom_id": "dataset|item|control|0.0|0|open",
                "body": {
                    "messages": [
                        {"role": "system", "content": "System prompt."},
                        {"role": "user", "content": "User question?"},
                    ]
                },
            }
        )
        + "\n",
        encoding="utf-8",
    )
    dest = tmp_path / "requests.jsonl"
    count = build_batch_requests(
        src_path=str(src),
        dest_path=str(dest),
        model="gpt-4.1-mini",
        temperature=0.0,
        top_p=None,
        max_new_tokens=None,
        endpoint="/v1/responses",
        metadata=None,
        request_overrides={"thinking": {"type": "enabled", "budget_tokens": 128}},
    )
    assert count == 1
    record = _read_jsonl(dest)[0]
    thinking = record["body"].get("thinking")
    assert thinking is not None
    assert thinking["budget_tokens"] == 128


class _StubBatches:
    def __init__(self, responses: list[SimpleNamespace]) -> None:
        self._responses = list(responses)
        self._index = 0

    def retrieve(self, batch_id: str) -> SimpleNamespace:
        if self._index >= len(self._responses):
            raise RuntimeError("retrieve called too many times")
        resp = self._responses[self._index]
        self._index += 1
        return resp


def test_poll_until_complete_success() -> None:
    responses = [
        SimpleNamespace(status="validating"),
        SimpleNamespace(status="in_progress"),
        SimpleNamespace(status="completed", output_file_id="file-123"),
    ]
    client = SimpleNamespace(batches=_StubBatches(responses))
    job = poll_until_complete(client, "batch-1", poll_seconds=0.0)
    assert job.status == "completed"
    assert job.output_file_id == "file-123"


def test_poll_until_complete_failure() -> None:
    responses = [
        SimpleNamespace(status="validating"),
        SimpleNamespace(status="failed", errors={"message": "boom"}),
    ]
    client = SimpleNamespace(batches=_StubBatches(responses))
    with pytest.raises(RuntimeError):
        poll_until_complete(client, "batch-err", poll_seconds=0.0)


class _ContentProxy:
    def __init__(self, src_path: Path) -> None:
        self._src_path = src_path

    def write_to_file(self, dest_path: str) -> None:
        os.makedirs(os.path.dirname(dest_path), exist_ok=True)
        with open(self._src_path, "rb") as src, open(dest_path, "wb") as dst:
            dst.write(src.read())


def test_download_and_normalize_results(tmp_path: Path) -> None:
    fixture = Path("tests/fixtures/backends/openai/sample_batch_output.zip")
    client = SimpleNamespace(files=SimpleNamespace(content=lambda file_id: _ContentProxy(fixture)))
    extracted = download_and_extract(client, output_file_id="out-1", out_dir=str(tmp_path))
    assert len(extracted) == 1
    normalized = tmp_path / "normalized.jsonl"
    count = normalize_jsonl(extracted, str(normalized), endpoint="/v1/responses")
    assert count == 2

    manifest_path = tmp_path / "trial_manifest.json"
    manifest_path.write_text(json.dumps({"control_registry": {}, "prompts": {}}), encoding="utf-8")
    predictions = tmp_path / "predictions.csv"
    process_results(str(normalized), str(predictions))
    with open(predictions, "r", encoding="utf-8") as fin:
        rows = list(csv.DictReader(fin))
    assert {row["condition"] for row in rows} == {"control", "treatment"}
    texts = {row["response_text"] for row in rows}
    assert texts == {"Control answer.", "Treatment answer."}
