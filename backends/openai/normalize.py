from __future__ import annotations

import json
from typing import Any, Iterable, Optional


class OpenAIBatchError(RuntimeError):
    """Represents a failure within an OpenAI batch output record."""


def _collect_output_text(output: Iterable[Any]) -> tuple[str, Optional[str]]:
    fragments: list[str] = []
    finish_reason: Optional[str] = None
    for item in output or []:
        if not isinstance(item, dict):
            continue
        item_type = item.get("type")
        if item_type == "message":
            for content in item.get("content") or []:
                if isinstance(content, dict) and content.get("type") in {"output_text", "text"}:
                    fragments.append(content.get("text") or "")
            fr = item.get("finish_reason") or item.get("status")
            if fr:
                finish_reason = fr
        elif item_type in {"output_text", "text"}:
            fragments.append(item.get("text") or "")
            fr = item.get("finish_reason")
            if fr:
                finish_reason = fr
    return "".join(fragments).strip(), finish_reason


def _normalize_responses_body(resp: dict[str, Any]) -> dict[str, Any]:
    aggregated, finish_reason = _collect_output_text(resp.get("output") or [])
    if not aggregated:
        text = resp.get("output_text") or resp.get("text")
        if isinstance(text, str):
            aggregated = text.strip()
    usage = resp.get("usage") or {}
    body = {
        "choices": [
            {
                "message": {"role": "assistant", "content": aggregated},
                "finish_reason": finish_reason or resp.get("status") or "stop",
            }
        ],
        "usage": usage,
        "id": resp.get("id"),
        "model": resp.get("model"),
    }
    return body


def _normalize_chat_completions_body(resp: dict[str, Any]) -> dict[str, Any]:
    choices = resp.get("choices") or []
    usage = resp.get("usage") or {}
    body = {
        "choices": choices,
        "usage": usage,
        "id": resp.get("id"),
        "model": resp.get("model"),
    }
    return body


def normalize_record(record: dict[str, Any], *, endpoint: str) -> dict[str, Any]:
    if not isinstance(record, dict):
        raise OpenAIBatchError("Batch record must be a JSON object.")
    if record.get("error"):
        raise OpenAIBatchError(f"Batch record for custom_id={record.get('custom_id')} returned error: {record['error']}")
    response = record.get("response")
    if not isinstance(response, dict):
        raise OpenAIBatchError(f"Missing response payload for custom_id={record.get('custom_id')}")

    if endpoint == "/v1/responses" or ("output" in response and endpoint != "/v1/chat/completions"):
        body = _normalize_responses_body(response)
    else:
        body = _normalize_chat_completions_body(response)

    usage = body.get("usage") or response.get("usage") or {}
    body["usage"] = usage

    normalized = dict(record)
    normalized["response"] = {
        "body": body,
        "usage": usage,
        "id": response.get("id"),
        "model": response.get("model"),
        "raw": response,
    }
    return normalized


def normalize_jsonl(
    src_paths: Iterable[str],
    dest_path: str,
    *,
    endpoint: str,
) -> int:
    """Normalize raw OpenAI batch output JSONL files into parser-compatible shape."""
    count = 0
    with open(dest_path, "w", encoding="utf-8") as fout:
        for path in src_paths:
            with open(path, "r", encoding="utf-8") as fin:
                for line in fin:
                    s = line.strip()
                    if not s:
                        continue
                    record = json.loads(s)
                    normalized = normalize_record(record, endpoint=endpoint)
                    fout.write(json.dumps(normalized, ensure_ascii=False) + "\n")
                    count += 1
    return count
