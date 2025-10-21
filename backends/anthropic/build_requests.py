from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any, Iterable, Optional, Sequence, Tuple

MAX_REQUESTS_PER_BATCH = 10_000


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


def _coerce_stop_sequences(stop: Any) -> list[str]:
    if not stop:
        return []
    if isinstance(stop, (list, tuple, set)):
        return [str(item) for item in stop if item is not None]
    return [str(stop)]


def _coerce_text(content: Any) -> str:
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts: list[str] = []
        for item in content:
            if isinstance(item, dict):
                text = item.get("text")
                if text is not None:
                    parts.append(str(text))
                    continue
            if isinstance(item, str):
                parts.append(item)
        return "\n".join(part for part in parts if part)
    return "" if content is None else str(content)


def _extract_system_and_messages(messages: Sequence[dict[str, Any]]) -> Tuple[Optional[str], list[dict[str, Any]]]:
    system_prompt: Optional[str] = None
    normalized: list[dict[str, Any]] = []
    for msg in messages:
        if not isinstance(msg, dict):
            continue
        role = str(msg.get("role") or "").strip().lower()
        content = _coerce_text(msg.get("content"))
        if role == "system" and system_prompt is None:
            system_prompt = content
            continue
        if not role:
            role = "user"
        normalized.append({"role": role, "content": content})
    return system_prompt, normalized


@dataclass
class RequestPayload:
    custom_id: str
    params: dict[str, Any]


def build_message_requests(
    *,
    src_path: str,
    model: str,
    temperature: float,
    top_p: Optional[float],
    top_k: Optional[int],
    max_new_tokens: Optional[dict[str, Any] | Any],
    metadata: Optional[dict[str, Any]] = None,
    request_overrides: Optional[dict[str, Any]] = None,
) -> list[RequestPayload]:
    """Convert shard rows into Anthropic Message Batch request payloads."""

    token_lookup = _max_tokens_lookup(max_new_tokens)
    meta = _normalize_metadata(metadata)
    overrides = dict(request_overrides or {})

    requests: list[RequestPayload] = []
    for row in _load_rows(src_path):
        custom_id = row.get("custom_id")
        base_body = row.get("body") or {}
        if not custom_id or "messages" not in base_body:
            continue
        raw_messages = base_body.get("messages")
        if not isinstance(raw_messages, list):
            continue
        system_prompt, conversation = _extract_system_and_messages(raw_messages)
        if not conversation:
            continue

        params: dict[str, Any] = {
            "model": model,
            "temperature": float(temperature),
            "messages": conversation,
        }
        if top_p is not None:
            params["top_p"] = float(top_p)
        if top_k is not None:
            params["top_k"] = int(top_k)
        if system_prompt:
            params["system"] = system_prompt

        stop_sequences = _coerce_stop_sequences(base_body.get("stop"))
        if stop_sequences:
            params["stop_sequences"] = stop_sequences

        token_limit = _resolve_token_limit(str(custom_id), token_lookup)
        if token_limit is None:
            # Fallback to any configured value or default to 1024 tokens
            fallback = token_lookup.get("default")
            if fallback is None and token_lookup:
                fallback = next(iter(token_lookup.values()))
            token_limit = fallback if fallback is not None else 1024
        params["max_tokens"] = int(token_limit)

        if meta:
            params["metadata"] = meta
        if overrides:
            params.update(overrides)

        requests.append(RequestPayload(custom_id=str(custom_id), params=params))

    if len(requests) > MAX_REQUESTS_PER_BATCH:
        raise ValueError(
            f"Anthropic Message Batches support up to {MAX_REQUESTS_PER_BATCH} requests per job; "
            f"got {len(requests)} for shard {src_path}"
        )
    return requests


def write_requests_preview(requests: Iterable[RequestPayload], dest_path: str) -> int:
    """Persist a JSONL preview of Anthropic requests for inspection."""

    count = 0
    data = list(requests)
    if not dest_path:
        return len(data)
    from pathlib import Path

    path = Path(dest_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as fout:
        for req in data:
            fout.write(json.dumps({"custom_id": req.custom_id, "params": req.params}, ensure_ascii=False) + "\n")
            count += 1
    return count
