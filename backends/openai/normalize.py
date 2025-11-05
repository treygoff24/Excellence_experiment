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
    source = resp.get("body") if isinstance(resp.get("body"), dict) else resp
    aggregated, finish_reason = _collect_output_text(source.get("output") or [])
    if not aggregated:
        text = source.get("output_text") or source.get("text")
        if isinstance(text, str):
            aggregated = text.strip()
    usage = source.get("usage") or resp.get("usage") or {}
    finish = finish_reason or source.get("status") or resp.get("status") or "stop"
    normalized_body: dict[str, Any] = {
        "choices": [
            {
                "message": {"role": "assistant", "content": aggregated},
                "finish_reason": finish,
            }
        ],
        "usage": usage,
        "id": source.get("id") or resp.get("id"),
        "model": source.get("model") or resp.get("model"),
    }
    if isinstance(source.get("reasoning"), dict):
        normalized_body["reasoning"] = source["reasoning"]
    return normalized_body


def _normalize_chat_completions_body(resp: dict[str, Any]) -> dict[str, Any]:
    source = resp.get("body") if isinstance(resp.get("body"), dict) else resp
    choices = source.get("choices") or resp.get("choices") or []
    usage = source.get("usage") or resp.get("usage") or {}
    normalized_body = {
        "choices": choices,
        "usage": usage,
        "id": source.get("id") or resp.get("id"),
        "model": source.get("model") or resp.get("model"),
    }
    return normalized_body


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

    source = response.get("body") if isinstance(response.get("body"), dict) else response
    normalized = dict(record)
    normalized["response"] = {
        "body": body,
        "usage": usage,
        "id": body.get("id") or source.get("id"),
        "model": body.get("model") or source.get("model"),
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
