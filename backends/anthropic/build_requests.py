from __future__ import annotations

import hashlib
import json
import re
from dataclasses import dataclass
from typing import Any, Iterable, Optional, Sequence, Tuple

MAX_REQUESTS_PER_BATCH = 10_000
_SAFE_ID_PATTERN = re.compile(r"[^A-Za-z0-9_-]")
_TEMPERATURE_TOLERANCE = 1e-6


def _is_thinking_enabled(value: Any) -> bool:
    if value is None:
        return False
    if isinstance(value, dict):
        for key in ("type", "mode", "state"):
            setting = value.get(key)
            if isinstance(setting, str) and setting.lower() == "enabled":
                return True
        enabled_flag = value.get("enabled")
        if isinstance(enabled_flag, bool):
            return enabled_flag
        return False
    if hasattr(value, "type"):
        setting = getattr(value, "type")
        if isinstance(setting, str) and setting.lower() == "enabled":
            return True
    if isinstance(value, str):
        return value.lower() == "enabled"
    return False


def _ensure_thinking_temperature(temperature: Any, *, context: str) -> None:
    try:
        temp_val = float(temperature)
    except (TypeError, ValueError):
        raise ValueError(
            f"Anthropic thinking requires temperature=1. Unable to interpret temperature {temperature!r} for {context}."
        ) from None
    if abs(temp_val - 1.0) > _TEMPERATURE_TOLERANCE:
        raise ValueError(
            f"Anthropic thinking requires temperature=1. Received temperature={temp_val} for {context}. "
            "Update the eval configuration or disable thinking before retrying."
        )


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


def _ensure_block_list(content: Any) -> list[dict[str, Any]]:
    if isinstance(content, list):
        blocks: list[dict[str, Any]] = []
        for item in content:
            if isinstance(item, dict):
                block = dict(item)
                b_type = str(block.get("type") or "text")
                block["type"] = b_type
                if "text" in block:
                    block["text"] = "" if block["text"] is None else str(block["text"])
                blocks.append(block)
            elif isinstance(item, str):
                blocks.append({"type": "text", "text": item})
        return blocks
    if isinstance(content, str):
        return [{"type": "text", "text": content}]
    if content is None:
        return []
    return [{"type": "text", "text": str(content)}]


def _flatten_blocks(blocks: Sequence[dict[str, Any]]) -> str:
    parts: list[str] = []
    for block in blocks:
        if not isinstance(block, dict):
            continue
        text = block.get("text")
        if text:
            parts.append(str(text))
    return "\n".join(parts)


def _extract_system_and_messages(messages: Sequence[dict[str, Any]]) -> Tuple[Optional[list[dict[str, Any]]], list[dict[str, Any]]]:
    system_blocks: Optional[list[dict[str, Any]]] = None
    normalized: list[dict[str, Any]] = []
    for msg in messages:
        if not isinstance(msg, dict):
            continue
        role = str(msg.get("role") or "").strip().lower()
        content_blocks = _ensure_block_list(msg.get("content"))
        if role == "system" and system_blocks is None:
            if content_blocks:
                system_blocks = content_blocks
            continue
        if not role:
            role = "user"
        if not content_blocks:
            # Skip empty messages to avoid API errors
            continue
        normalized.append({"role": role, "content": content_blocks})
    return system_blocks, normalized


def _apply_cache_control(
    blocks: list[dict[str, Any]],
    cache_control: Optional[dict[str, Any]],
    *,
    explicit: bool,
) -> list[dict[str, Any]]:
    if not blocks:
        return blocks
    has_existing = any(isinstance(block, dict) and block.get("cache_control") for block in blocks)
    if cache_control is None:
        if not explicit:
            return blocks
        cleared: list[dict[str, Any]] = []
        for block in blocks:
            if not isinstance(block, dict):
                cleared.append(block)
                continue
            new_block = dict(block)
            new_block.pop("cache_control", None)
            cleared.append(new_block)
        return cleared

    if not explicit and has_existing:
        return blocks

    payload = dict(cache_control)
    updated: list[dict[str, Any]] = []
    applied = False
    for block in blocks:
        if not isinstance(block, dict):
            updated.append(block)
            continue
        new_block = dict(block)
        new_block.pop("cache_control", None)
        if not applied and new_block.get("type", "text") == "text":
            new_block["cache_control"] = dict(payload)
            applied = True
        updated.append(new_block)
    if not applied:
        updated.append({"type": "text", "text": "", "cache_control": dict(payload)})
    return updated


def _normalize_cache_settings(settings: Optional[dict[str, Any]]) -> tuple[Optional[dict[str, Any]], bool]:
    if settings is None:
        return {"type": "ephemeral"}, False
    enabled = settings.get("enable_system_cache")
    if enabled is None:
        enabled = True
    if not bool(enabled):
        return None, True
    cache_type = str(settings.get("type") or "ephemeral").strip() or "ephemeral"
    payload: dict[str, Any] = {"type": cache_type}
    ttl = settings.get("ttl")
    if ttl:
        payload["ttl"] = str(ttl)
    return payload, True


@dataclass
class RequestPayload:
    custom_id: str
    params: dict[str, Any]
    orig_custom_id: str
    metadata: dict[str, Any]


def _sanitize_custom_id(raw: str, used: set[str]) -> str:
    if not raw:
        raw = "request"
    safe = _SAFE_ID_PATTERN.sub("_", raw)
    if not safe:
        safe = hashlib.sha256(raw.encode("utf-8")).hexdigest()
    if len(safe) > 64:
        digest = hashlib.sha256(raw.encode("utf-8")).hexdigest()
        safe = f"{safe[:32]}_{digest[:16]}"
    safe = safe[:64]
    if safe not in used:
        used.add(safe)
        return safe
    counter = 1
    base = safe.rstrip("_")
    while True:
        suffix = f"_{counter}"
        if base:
            candidate = (base[: max(0, 64 - len(suffix))] + suffix)
        else:
            candidate = hashlib.sha256(f"{raw}:{counter}".encode("utf-8")).hexdigest()[:64]
        if candidate not in used:
            used.add(candidate)
            return candidate
        counter += 1


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
    cache_control: Optional[dict[str, Any]] = None,
) -> list[RequestPayload]:
    """Convert shard rows into Anthropic Message Batch request payloads."""

    token_lookup = _max_tokens_lookup(max_new_tokens)
    meta = _normalize_metadata(metadata)
    overrides = dict(request_overrides or {})
    cache_payload, cache_explicit = _normalize_cache_settings(cache_control)
    used_custom_ids: set[str] = set()

    if _is_thinking_enabled(overrides.get("thinking")):
        _ensure_thinking_temperature(temperature, context=f"shard {src_path} (request overrides)")

    requests: list[RequestPayload] = []
    for row in _load_rows(src_path):
        custom_id = row.get("custom_id")
        base_body = row.get("body") or {}
        if not custom_id or "messages" not in base_body:
            continue
        orig_custom_id = str(custom_id)
        raw_messages = base_body.get("messages")
        if not isinstance(raw_messages, list):
            continue
        system_blocks, conversation = _extract_system_and_messages(raw_messages)
        if not conversation:
            continue
        processed_system_blocks = _apply_cache_control(system_blocks or [], cache_payload, explicit=cache_explicit)

        safe_custom_id = _sanitize_custom_id(orig_custom_id, used_custom_ids)
        params: dict[str, Any] = {
            "model": model,
            "temperature": float(temperature),
            "messages": conversation,
        }
        if top_p is not None:
            params["top_p"] = float(top_p)
        if top_k is not None:
            params["top_k"] = int(top_k)
        if processed_system_blocks:
            params["system"] = processed_system_blocks

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

        request_meta = dict(meta)
        request_meta["orig_custom_id"] = orig_custom_id
        if overrides:
            params.update(overrides)

        if _is_thinking_enabled(params.get("thinking")):
            context = f"custom_id={orig_custom_id} (shard {src_path})"
            _ensure_thinking_temperature(params.get("temperature"), context=context)

        requests.append(
            RequestPayload(
                custom_id=safe_custom_id,
                params=params,
                orig_custom_id=orig_custom_id,
                metadata=request_meta,
            )
        )

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
            payload = {
                "custom_id": req.custom_id,
                "params": req.params,
                "orig_custom_id": req.orig_custom_id,
            }
            if req.metadata:
                payload["metadata"] = req.metadata
            fout.write(json.dumps(payload, ensure_ascii=False) + "\n")
            count += 1
    return count
