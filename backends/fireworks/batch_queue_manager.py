from __future__ import annotations

# Thin adapter that re-exports Fireworks queue manager and helpers

from fireworks.batch_queue_manager import (  # type: ignore
    QueueManager as _QueueManager,
    JobInfo as _JobInfo,
    upload_datasets as _upload_datasets,
)


class QueueManager(_QueueManager):
    pass


class JobInfo(_JobInfo):
    pass


def upload_datasets(
    account_id: str,
    dataset_files: list[tuple[int, str]],
    base_name: str,
    temp_label: str,
    condition: str,
) -> list[tuple[int, str]]:
    return _upload_datasets(account_id, dataset_files, base_name, temp_label, condition)

