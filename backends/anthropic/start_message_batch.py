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
    create_kwargs: dict[str, Any] = {}
    extra_body: dict[str, Any] = {}

    if metadata:
        # The Anthropic batches API currently rejects top-level metadata fields.
        # Keep the parameter for future compatibility and to match adapter usage.
        pass

    if batch_params:
        for key, value in batch_params.items():
            if key == "requests":
                raise ValueError("Anthropic batch params cannot include 'requests'.")
            if key == "completion_window":
                if value is None:
                    continue
                raise ValueError("Anthropic Message Batches no longer accept 'completion_window'. Remove it from batch_params.")
            if key == "extra_body":
                if not isinstance(value, Mapping):
                    raise TypeError("Anthropic batch 'extra_body' overrides must be a mapping.")
                extra_body.update(dict(value))
                continue
            if key in {"extra_headers", "extra_query", "timeout"}:
                create_kwargs[key] = value
                continue
            if key in {"metadata"}:
                # The public API does not currently accept this field; ignore for forward compatibility.
                continue
            extra_body[key] = value

    if extra_body:
        existing_extra_body = create_kwargs.get("extra_body")
        if existing_extra_body:
            if not isinstance(existing_extra_body, Mapping):
                raise TypeError("Anthropic batch params 'extra_body' must be a mapping when provided.")
            merged = dict(existing_extra_body)
            merged.update(extra_body)
            create_kwargs["extra_body"] = merged
        else:
            create_kwargs["extra_body"] = extra_body

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
