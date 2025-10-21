from __future__ import annotations

import json
import os
from typing import Any, Dict, Iterable, Optional


SUPPORTED_ENDPOINTS = {"/v1/responses", "/v1/chat/completions"}


def _load_rows(path: str) -> Iterable[dict[str, Any]]:
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


def _normalize_metadata(metadata: Optional[dict[str, Any]]) -> dict[str, str]:
    normalized: dict[str, str] = {}
    if not metadata:
        return normalized
    for key, value in metadata.items():
        if key is None:
            continue
        normalized[str(key)] = "" if value is None else str(value)
    return normalized


def _max_tokens_lookup(max_new_tokens: Optional[dict[str, Any] | Any]) -> dict[str, int]:
    if max_new_tokens is None:
        return {}
    data: Any
    model_dump = getattr(max_new_tokens, "model_dump", None)
    if callable(model_dump):
        data = model_dump()
    else:
        as_dict = getattr(max_new_tokens, "dict", None)
        if callable(as_dict):
            data = as_dict()
        elif isinstance(max_new_tokens, dict):
            data = dict(max_new_tokens)
        else:
            return {}
    if not isinstance(data, dict):
        try:
            data = dict(data)
        except Exception:
            return {}
    lookup: dict[str, int] = {}
    for key, value in data.items():
        if value is None:
            continue
        try:
            lookup[str(key)] = int(value)
        except (TypeError, ValueError):
            continue
    return lookup


def _resolve_token_limit(custom_id: str, tokens: dict[str, int]) -> Optional[int]:
    if not custom_id or not tokens:
        return None
    try:
        parts = custom_id.split("|")
        typ = parts[-1]
    except Exception:
        typ = None
    if typ == "open":
        for key in ("open_book", "open", "default"):
            if key in tokens:
                return tokens[key]
    elif typ == "closed":
        for key in ("closed_book", "closed", "default"):
            if key in tokens:
                return tokens[key]
    return tokens.get("default")


def _coerce_messages(messages: Any) -> list[dict[str, Any]]:
    if isinstance(messages, list):
        result: list[dict[str, Any]] = []
        for msg in messages:
            if isinstance(msg, dict):
                role = msg.get("role")
                content = msg.get("content")
                if isinstance(content, (str, list)):
                    result.append({"role": role, "content": content})
                else:
                    result.append({"role": role, "content": str(content) if content is not None else ""})
        return result
    return []


def build_batch_requests(
    *,
    src_path: str,
    dest_path: str,
    model: str,
    temperature: float,
    top_p: Optional[float],
    top_k: Optional[int],
    max_new_tokens: Optional[dict[str, Any] | Any],
    endpoint: str = "/v1/responses",
    metadata: Optional[dict[str, Any]] = None,
    request_overrides: Optional[dict[str, Any]] = None,
) -> int:
    """Convert prepared shard rows into OpenAI Batch JSONL records.

    Returns the number of records written to ``dest_path``.
    """

    if endpoint not in SUPPORTED_ENDPOINTS:
        raise ValueError(f"Unsupported OpenAI endpoint '{endpoint}'. Expected one of {sorted(SUPPORTED_ENDPOINTS)}.")

    token_lookup = _max_tokens_lookup(max_new_tokens)
    string_metadata = _normalize_metadata(metadata)
    overrides = dict(request_overrides or {})

    out_dir = os.path.dirname(dest_path) or "."
    os.makedirs(out_dir, exist_ok=True)
    written = 0

    with open(dest_path, "w", encoding="utf-8") as fout:
        for row in _load_rows(src_path):
            custom_id = row.get("custom_id")
            base_body = row.get("body") or {}
            if not custom_id or "messages" not in base_body:
                continue
            messages = _coerce_messages(base_body.get("messages"))
            if not messages:
                continue
            stop = base_body.get("stop")

            request_body: Dict[str, Any] = {
                "model": model,
                "temperature": float(temperature),
            }
            if top_p is not None:
                request_body["top_p"] = float(top_p)
            if top_k is not None:
                request_body["top_k"] = int(top_k)
            if stop:
                request_body["stop"] = stop

            token_limit = _resolve_token_limit(custom_id, token_lookup)
            if token_limit is not None:
                if endpoint == "/v1/responses":
                    request_body["max_output_tokens"] = int(token_limit)
                else:
                    request_body["max_tokens"] = int(token_limit)

            if endpoint == "/v1/responses":
                request_body["input"] = messages
            else:
                request_body["messages"] = messages

            if string_metadata:
                request_body["metadata"] = string_metadata

            if overrides:
                request_body.update(overrides)

            record = {
                "custom_id": custom_id,
                "method": "POST",
                "url": endpoint,
                "body": request_body,
            }

            fout.write(json.dumps(record, ensure_ascii=False) + "\n")
            written += 1

    return written
