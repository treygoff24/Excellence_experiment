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


def create_queue(
    *,
    account_id: str,
    model_id: str,
    config: dict,
    max_concurrent: int,
    temp_label: str,
    temperature: float,
    condition: str,
    run_id: str,
    stop_event: object | None = None,
) -> QueueManager:
    """Factory adapter to satisfy the BatchExecutor.create_queue surface.

    Returns a QueueManager configured with the provided arguments.
    """
    return QueueManager(
        account_id=account_id,
        model_id=model_id,
        config=config,
        max_concurrent=max_concurrent,
        temp_label=temp_label,
        temperature=temperature,
        condition=condition,
        run_id=run_id,
        stop_event=stop_event,
    )
