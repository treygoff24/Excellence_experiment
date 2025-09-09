from __future__ import annotations

from typing import Any, Iterable, Optional, Protocol, Tuple


class InferenceClient(Protocol):
    """Minimal synchronous inference client interface.

    Implementations should provide one of messages or prompt. Return a dict
    containing at least the generated text and a finish reason. Optional usage
    accounting fields may be present.
    """

    def generate(
        self,
        *,
        messages: Optional[list[dict[str, str]]] = None,
        prompt: Optional[str] = None,
        model: str = "",
        params: Optional[dict[str, Any]] = None,
    ) -> dict[str, Any]:
        """Run a single completion call and return a result mapping.

        Expected keys: {"text", "finish_reason"}. Optional: {"usage", "request_id", "latency_s"}.
        """
        ...


class BatchExecutor(Protocol):
    """Protocol for batch execution on JSONL inputs.

    This surface is intentionally minimal for orchestration. Implementations
    encapsulate splitting, submission, polling, and downloading as needed.
    """

    def split_jsonl(
        self,
        src_path: str,
        out_dir: str,
        base_prefix: str,
        *,
        parts: Optional[int] = None,
        lines_per_part: Optional[int] = None,
        limit_items: Optional[int] = None,
    ) -> list[Tuple[int, str]]:
        """Split an input JSONL into parts; returns [(part_number, path)]."""
        ...

    def upload_datasets(
        self,
        account_id: str,
        dataset_files: list[Tuple[int, str]],
        base_name: str,
        temp_label: str,
        condition: str,
    ) -> list[Tuple[int, str]]:
        """Upload part files and return [(part_number, dataset_id)]."""
        ...

    def create_queue(
        self,
        *,
        account_id: str,
        model_id: str,
        config: dict,
        max_concurrent: int,
        temp_label: str,
        temperature: float,
        condition: str,
        run_id: str,
        stop_event: Optional[object] = None,
    ) -> Any:
        """Return an object with add_job(part_num, path, dataset_id) and run_queue(results_dir)."""
        ...

