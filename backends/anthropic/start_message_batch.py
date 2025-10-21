from __future__ import annotations

from typing import Any, Mapping, Sequence

from .build_requests import MAX_REQUESTS_PER_BATCH, RequestPayload


def _to_request_mapping(item: RequestPayload | Mapping[str, Any]) -> dict[str, Any]:
    if isinstance(item, RequestPayload):
        return {"custom_id": item.custom_id, "params": dict(item.params)}
    if isinstance(item, Mapping):
        if "custom_id" not in item or "params" not in item:
            raise ValueError("Anthropic request payloads must include 'custom_id' and 'params'.")
        return {"custom_id": str(item["custom_id"]), "params": dict(item["params"])}
    raise TypeError("Unsupported request payload type for Anthropic batch submission.")


def start_message_batch(
    client: Any,
    requests: Sequence[RequestPayload | Mapping[str, Any]],
    *,
    metadata: Mapping[str, Any] | None = None,
    batch_params: Mapping[str, Any] | None = None,
) -> Any:
    """Submit an Anthropic Message Batch job using the provided client."""

    if not requests:
        raise ValueError("Anthropic Message Batches require at least one request.")
    if len(requests) > MAX_REQUESTS_PER_BATCH:
        raise ValueError(
            f"Anthropic Message Batches support up to {MAX_REQUESTS_PER_BATCH} requests; received {len(requests)}."
        )

    payload = [_to_request_mapping(item) for item in requests]
    create_kwargs = dict(batch_params or {})
    if metadata:
        combined_meta = dict(create_kwargs.get("metadata") or {})
        for key, value in metadata.items():
            if key is None:
                continue
            combined_meta[str(key)] = value
        create_kwargs["metadata"] = combined_meta
    create_kwargs["requests"] = payload

    messages_iface = getattr(client, "messages", None)
    if messages_iface is None:
        raise AttributeError("Anthropic client is missing the 'messages' attribute.")
    batches_iface = getattr(messages_iface, "batches", None)
    if batches_iface is None:
        raise AttributeError("Anthropic client is missing the 'messages.batches' attribute.")
    create_fn = getattr(batches_iface, "create", None)
    if not callable(create_fn):
        raise AttributeError("Anthropic client does not expose messages.batches.create.")
    return create_fn(**create_kwargs)
