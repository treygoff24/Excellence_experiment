from __future__ import annotations

# Thin adapter that re-exports Fireworks polling and download helpers

from fireworks.poll_and_download import (  # type: ignore
    poll_until_done as _poll_until_done,
    get_dataset as _get_dataset,
    try_download_external_url as _try_download_external_url,
    _try_extract_jsonls as __try_extract_jsonls,
)


def poll_until_done(account_id: str, job_name: str, *, poll_seconds: int = 30) -> dict:
    return _poll_until_done(account_id, job_name, poll_seconds=poll_seconds)


def get_dataset(account_id: str, dataset_id: str) -> dict:
    return _get_dataset(account_id, dataset_id)


def try_download_external_url(url: str, out_dir: str) -> str:
    return _try_download_external_url(url, out_dir)


def _try_extract_jsonls(bundle_path: str, out_dir: str) -> list[str]:
    return __try_extract_jsonls(bundle_path, out_dir)

