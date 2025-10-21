from __future__ import annotations

import os
import shutil
import time
from typing import Any, Callable, Optional

from fireworks.poll_and_download import _try_extract_jsonls  # type: ignore


TERMINAL_SUCCESS = {"completed"}
TERMINAL_FAILURE = {"failed", "expired", "cancelled"}
TERMINAL_PROGRESS = {"in_progress", "finalizing", "validating", "cancelling"}


def poll_until_complete(
    client: Any,
    batch_id: str,
    *,
    poll_seconds: float = 30.0,
    status_callback: Optional[Callable[[str], None]] = None,
    stop_event: Optional[Any] = None,
) -> Any:
    """Poll ``client.batches.retrieve`` until the batch reaches a terminal state."""

    last_status: Optional[str] = None
    while True:
        if stop_event is not None:
            should_stop = getattr(stop_event, "is_set", None)
            if callable(should_stop) and should_stop():
                raise RuntimeError("Polling interrupted by stop event.")
        job = client.batches.retrieve(batch_id)  # type: ignore[attr-defined]
        status = getattr(job, "status", None)
        if status_callback and status != last_status and status is not None:
            status_callback(str(status))
        if status in TERMINAL_SUCCESS:
            return job
        if status in TERMINAL_FAILURE:
            errors = getattr(job, "errors", None)
            raise RuntimeError(f"OpenAI batch {batch_id} ended with status '{status}': {errors}")
        if status not in TERMINAL_PROGRESS:
            # Unknown status – treat like failure to avoid infinite loop
            raise RuntimeError(f"OpenAI batch {batch_id} entered unexpected status '{status}'.")
        last_status = status
        if poll_seconds > 0:
            time.sleep(poll_seconds)


def _write_content_to_path(content: Any, dest_path: str) -> None:
    if hasattr(content, "write_to_file"):
        content.write_to_file(dest_path)
        return
    data = getattr(content, "content", None)
    if isinstance(data, bytes):
        with open(dest_path, "wb") as fout:
            fout.write(data)
        return
    if hasattr(content, "read"):
        with open(dest_path, "wb") as fout:
            chunk = content.read()
            if isinstance(chunk, bytes):
                fout.write(chunk)
                return
    raise RuntimeError("Unsupported content object returned from OpenAI files.content")


def download_and_extract(
    client: Any,
    *,
    output_file_id: str,
    out_dir: str,
) -> list[str]:
    """Download ``output_file_id`` and extract JSONL payloads into ``out_dir``."""

    os.makedirs(out_dir, exist_ok=True)
    tmp_path = os.path.join(out_dir, f"{output_file_id}.download")
    content = client.files.content(output_file_id)  # type: ignore[attr-defined]
    _write_content_to_path(content, tmp_path)
    extracted = _try_extract_jsonls(tmp_path, out_dir)
    if extracted:
        try:
            os.remove(tmp_path)
        except OSError:
            pass
        return extracted
    final_path = os.path.join(out_dir, f"{output_file_id}.jsonl")
    if os.path.abspath(tmp_path) != os.path.abspath(final_path):
        shutil.move(tmp_path, final_path)
    return [final_path]
