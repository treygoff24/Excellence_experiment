from __future__ import annotations

from typing import Any, Dict, Optional, Tuple


def _stringify_metadata(metadata: Optional[dict[str, Any]]) -> dict[str, str]:
    if not metadata:
        return {}
    out: dict[str, str] = {}
    for key, value in metadata.items():
        if key is None:
            continue
        out[str(key)] = "" if value is None else str(value)
    return out


def upload_batch_file(client: Any, path: str, *, purpose: str = "batch") -> str:
    with open(path, "rb") as fin:
        file_obj = client.files.create(file=fin, purpose=purpose)  # type: ignore[attr-defined]
    file_id = getattr(file_obj, "id", None)
    if not file_id:
        raise RuntimeError("OpenAI file upload did not return an id.")
    return str(file_id)


def start_batch_job(
    client: Any,
    *,
    request_path: str,
    endpoint: str,
    completion_window: str,
    metadata: Optional[dict[str, Any]] = None,
    batch_params: Optional[dict[str, Any]] = None,
) -> Tuple[str, Any]:
    """Upload ``request_path`` and start an OpenAI Batch job.

    Returns ``(input_file_id, batch_job)``.
    """

    input_file_id = upload_batch_file(client, request_path)
    params: Dict[str, Any] = {
        "input_file_id": input_file_id,
        "endpoint": endpoint,
        "completion_window": completion_window,
    }
    meta = _stringify_metadata(metadata)
    if meta:
        params["metadata"] = meta
    if batch_params:
        params.update(batch_params)
    job = client.batches.create(**params)  # type: ignore[arg-type,attr-defined]
    batch_id = getattr(job, "id", None)
    if not batch_id:
        raise RuntimeError("OpenAI batch creation did not return an id.")
    return input_file_id, job
