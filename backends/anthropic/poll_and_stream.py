from __future__ import annotations

import json
import os
import time
from typing import Any, Callable, Iterable, cast


def _get_processing_status(batch: Any) -> str | None:
    if batch is None:
        return None
    if hasattr(batch, "processing_status"):
        return getattr(batch, "processing_status")
    if isinstance(batch, dict):
        value = batch.get("processing_status")
        if isinstance(value, str):
            return value
    return None


def poll_until_complete(
    client: Any,
    batch_id: str,
    *,
    poll_seconds: float = 30.0,
    timeout_seconds: float | None = None,
) -> Any:
    """Poll the Anthropic batch until a terminal status is reached."""

    if not batch_id:
        raise ValueError("Anthropic batch polling requires a batch_id.")
    messages_iface = getattr(client, "messages", None)
    if messages_iface is None:
        raise AttributeError("Anthropic client is missing the 'messages' attribute.")
    batches_iface = getattr(messages_iface, "batches", None)
    if batches_iface is None:
        raise AttributeError("Anthropic client is missing the 'messages.batches' attribute.")
    retrieve_fn = getattr(batches_iface, "retrieve", None)
    if not callable(retrieve_fn):
        raise AttributeError("Anthropic client does not expose messages.batches.retrieve.")

    poll_seconds = max(0.0, float(poll_seconds))
    start = time.time()

    while True:
        batch = retrieve_fn(batch_id)
        status = (_get_processing_status(batch) or "").lower()
        if status in {"ended", "failed", "canceled", "cancelled"}:
            return batch
        if timeout_seconds is not None and (time.time() - start) > timeout_seconds:
            raise TimeoutError(f"Timed out waiting for Anthropic batch {batch_id} to complete.")
        time.sleep(poll_seconds)


def _serialize_result_item(item: Any) -> str:
    if item is None:
        return json.dumps({})
    dump_json = getattr(item, "model_dump_json", None)
    if callable(dump_json):
        return str(dump_json())
    dump = getattr(item, "model_dump", None)
    if callable(dump):
        return json.dumps(dump(), ensure_ascii=False)
    if isinstance(item, (bytes, bytearray)):
        try:
            return item.decode("utf-8")
        except Exception:
            return json.dumps({"raw": list(item)})
    if isinstance(item, str):
        return item
    if isinstance(item, dict):
        return json.dumps(item, ensure_ascii=False)
    return json.dumps(item, ensure_ascii=False)


def stream_results_to_jsonl(
    client: Any,
    batch_id: str,
    dest_path: str,
) -> int:
    """Stream Anthropic batch results to a JSONL file and return the row count."""

    if not batch_id:
        raise ValueError("Anthropic batch results streaming requires a batch_id.")
    messages_iface = getattr(client, "messages", None)
    if messages_iface is None:
        raise AttributeError("Anthropic client is missing the 'messages' attribute.")
    batches_iface = getattr(messages_iface, "batches", None)
    if batches_iface is None:
        raise AttributeError("Anthropic client is missing the 'messages.batches' attribute.")
    results_iter = getattr(batches_iface, "results", None)
    if not callable(results_iter):
        raise AttributeError("Anthropic client does not expose messages.batches.results.")

    os.makedirs(os.path.dirname(dest_path) or ".", exist_ok=True)
    count = 0
    with open(dest_path, "w", encoding="utf-8") as fout:
        iter_results = cast(Callable[[str], Iterable[Any]], results_iter)
        for item in iter_results(batch_id):
            serialized = _serialize_result_item(item)
            if not serialized:
                continue
            fout.write(serialized.rstrip() + "\n")
            count += 1
    return count
