from __future__ import annotations

# Thin adapter that re-exports Fireworks dataset upload helpers

from fireworks.upload_dataset import (  # type: ignore
    create_dataset as _create_dataset,
    upload_dataset_file as _upload_dataset_file,
)


def create_dataset(display_name: str, account_id: str) -> str:
    return _create_dataset(display_name, account_id)


def upload_dataset_file(
    account_id: str,
    dataset_id: str,
    local_path: str,
    *,
    filename: str | None = None,
) -> None:
    _upload_dataset_file(account_id, dataset_id, local_path, filename=filename)

