from __future__ import annotations

import json
import os
from typing import Any, Dict, Iterable, Optional


def _iter_jsonl(path: str) -> Iterable[dict[str, Any]]:
    with open(path, "r", encoding="utf-8") as fin:
        for line in fin:
            s = line.strip()
            if not s:
                continue
            try:
                payload = json.loads(s)
            except json.JSONDecodeError:
                continue
            if isinstance(payload, dict):
                yield payload


def _collect_text(content: Any) -> str:
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        chunks: list[str] = []
        for item in content:
            if isinstance(item, dict) and item.get("type") == "text":
                text = item.get("text")
                if text:
                    chunks.append(str(text))
            elif isinstance(item, str):
                chunks.append(item)
        return "".join(chunks)
    return "" if content is None else str(content)


def _map_usage(data: Optional[dict[str, Any]]) -> dict[str, int]:
    usage = {}
    if not data:
        return usage
    prompt = data.get("input_tokens")
    completion = data.get("output_tokens")
    if prompt is not None:
        usage["prompt_tokens"] = int(prompt)
    if completion is not None:
        usage["completion_tokens"] = int(completion)
    if usage:
        usage["total_tokens"] = usage.get("prompt_tokens", 0) + usage.get("completion_tokens", 0)
    return usage


def _normalize_success(row: dict[str, Any], result: dict[str, Any]) -> dict[str, Any]:
    message = result.get("message") or {}
    text = _collect_text(message.get("content"))
    usage = _map_usage(message.get("usage"))
    finish_reason = message.get("stop_reason") or message.get("stop_sequence") or "stop"

    body: dict[str, Any] = {
        "id": message.get("id"),
        "model": message.get("model"),
        "type": message.get("type"),
        "message": message,
        "choices": [
            {
                "index": 0,
                "message": {"role": message.get("role", "assistant"), "content": text},
                "finish_reason": finish_reason,
            }
        ],
    }
    if usage:
        body["usage"] = usage

    response: dict[str, Any] = {
        "status": "succeeded",
        "body": body,
        "provider": {"name": "anthropic"},
    }
    if usage:
        response["usage"] = usage
    request_id = message.get("id") or row.get("id")
    if request_id:
        response["request_id"] = request_id
    return response


def _normalize_failure(result_type: str, result: dict[str, Any]) -> tuple[dict[str, Any], dict[str, Any]]:
    error = result.get("error") or {}
    if not error:
        error = {"type": result_type, "message": result.get("error_message") or "Anthropic batch request failed."}
    body = {"error": error, "status": result_type}
    response = {
        "status": result_type,
        "body": body,
        "error": error,
        "provider": {"name": "anthropic"},
    }
    return response, error


def normalize_jsonl(src_path: str, dest_path: str) -> int:
    """Normalize Anthropic Message Batch results into OpenAI-compatible JSONL."""

    os.makedirs(os.path.dirname(dest_path) or ".", exist_ok=True)
    count = 0

    with open(dest_path, "w", encoding="utf-8") as fout:
        for row in _iter_jsonl(src_path):
            custom_id = row.get("custom_id") or row.get("customId")
            if not custom_id:
                continue
            request_info = row.get("request") or {}
            if isinstance(request_info, dict):
                meta = request_info.get("metadata")
                if isinstance(meta, dict):
                    orig_custom_id = meta.get("orig_custom_id") or meta.get("orig-custom-id")
                    if orig_custom_id:
                        custom_id = orig_custom_id
            result = row.get("result") or {}
            result_type = str(result.get("type") or "").lower()
            record: Dict[str, Any] = {"custom_id": str(custom_id)}

            if result_type == "succeeded":
                response = _normalize_success(row, result)
                record["response"] = response
            else:
                response, error = _normalize_failure(result_type or "errored", result)
                record["response"] = response
                record["error"] = error

            # Include raw result metadata to aid manifest debugging
            if row.get("id"):
                record.setdefault("response", {}).setdefault("body", {})["id"] = row["id"]
            if "processing_status" in row:
                record.setdefault("response", {}).setdefault("body", {})["processing_status"] = row["processing_status"]
            fout.write(json.dumps(record, ensure_ascii=False) + "\n")
            count += 1
    return count
