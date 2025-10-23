from __future__ import annotations

import logging
import random
import time
from collections import deque
from email.utils import parsedate_to_datetime
from typing import Any, Deque, Dict, Iterable, Optional, Tuple

logger = logging.getLogger(__name__)


def _as_dict(obj: Any) -> dict[str, Any]:
    if obj is None:
        return {}
    if isinstance(obj, dict):
        return obj
    for attr in ("model_dump", "dict"):
        fn = getattr(obj, attr, None)
        if callable(fn):
            try:
                data = fn()
            except Exception:  # pragma: no cover - defensive
                continue
            if isinstance(data, dict):
                return data
    return {}


def _get_processing_status(batch: Any) -> str | None:
    if batch is None:
        return None
    if isinstance(batch, dict):
        status = batch.get("processing_status")
        return str(status) if status is not None else None
    status = getattr(batch, "processing_status", None)
    return str(status) if status is not None else None


def _get_request_counts(batch: Any) -> dict[str, int]:
    if batch is None:
        return {}
    if isinstance(batch, dict):
        counts = batch.get("request_counts") or {}
    else:
        counts = getattr(batch, "request_counts", {}) or {}
    if hasattr(counts, "model_dump"):
        try:
            counts = counts.model_dump()
        except Exception:  # pragma: no cover - defensive
            counts = {}
    if isinstance(counts, dict):
        out: dict[str, int] = {}
        for key, value in counts.items():
            try:
                out[str(key)] = int(value)
            except Exception:
                continue
        return out
    return {}


def _extract_retry_after_seconds(error: Any) -> float | None:
    headers = None
    response = getattr(error, "response", None)
    if response is not None:
        headers = getattr(response, "headers", None)
    if isinstance(headers, dict):
        for key in ("Retry-After", "retry-after"):
            if key in headers:
                raw = headers[key]
                if raw is None:
                    continue
                try:
                    return float(raw)
                except Exception:
                    try:
                        dt = parsedate_to_datetime(str(raw))
                        return max(0.0, (dt.timestamp() - time.time()))
                    except Exception:
                        continue
    retry_after = getattr(error, "retry_after", None)
    if retry_after is not None:
        try:
            return float(retry_after)
        except Exception:
            pass
    body = getattr(error, "body", None)
    data = _as_dict(body)
    if data:
        err = data.get("error") or {}
        if isinstance(err, dict):
            val = err.get("retry_after")
            if val is not None:
                try:
                    return float(val)
                except Exception:
                    return None
    return None


class AnthropicBatchLimiter:
    """Throttle Anthropic Message Batch submissions against queue and RPM limits."""

    _ACTIVE_STATUSES = {"in_progress", "processing", "queued", "running", "starting"}

    def __init__(
        self,
        *,
        processing_limit: Optional[int],
        safety_margin: float,
        queue_poll_seconds: float,
        rpm_limit: Optional[int],
        max_retries: int,
        retry_after_fallback: float,
        time_fn: callable = time.monotonic,
        sleep_fn: callable = time.sleep,
    ) -> None:
        self.processing_limit = processing_limit
        self.target_processing = None
        if processing_limit is not None:
            self.target_processing = int(processing_limit * (1.0 - safety_margin))
            if self.target_processing <= 0:
                self.target_processing = max(1, processing_limit)
        self.queue_poll_seconds = max(1.0, float(queue_poll_seconds))
        self.rpm_limit = rpm_limit
        self.max_retries = max(1, max_retries)
        self.retry_after_fallback = max(1.0, float(retry_after_fallback))
        self._time = time_fn
        self._sleep = sleep_fn
        self._request_window: Deque[Tuple[float, int]] = deque()
        self._pending_batches: Dict[str, int] = {}
        self._list_page_size = 100
        self._list_max_pages = 5

    # Internal helpers -----------------------------------------------------

    def _prune_window(self, now: float) -> None:
        cutoff = now - 60.0
        while self._request_window and self._request_window[0][0] <= cutoff:
            self._request_window.popleft()

    def _current_request_total(self, now: float) -> int:
        self._prune_window(now)
        return sum(count for _, count in self._request_window)

    def _iter_batches(self, client: Any) -> Iterable[Any]:
        messages_iface = getattr(client, "messages", None)
        if messages_iface is None:  # pragma: no cover - defensive
            return []
        batches_iface = getattr(messages_iface, "batches", None)
        if batches_iface is None:  # pragma: no cover - defensive
            return []
        list_fn = getattr(batches_iface, "list", None)
        if not callable(list_fn):
            return []

        cursor = None
        yielded = 0
        for _ in range(self._list_max_pages):
            kwargs = {"limit": self._list_page_size}
            if cursor:
                kwargs["after"] = cursor
            try:
                page = list_fn(**kwargs)
            except Exception as exc:  # pragma: no cover - network dependent
                logger.debug("Anthropic batch list failed: %s", exc)
                return []
            data = []
            if hasattr(page, "data"):
                data = getattr(page, "data") or []
            elif isinstance(page, dict):
                data = page.get("data") or []
            elif isinstance(page, list):
                data = page
            if not isinstance(data, list):
                break
            for item in data:
                yield item
                yielded += 1
            next_cursor = None
            for attr in ("last_id", "after", "next_page_token", "next"):
                value = getattr(page, attr, None)
                if value is None and isinstance(page, dict):
                    value = page.get(attr)
                if isinstance(value, str) and value:
                    next_cursor = value
                    break
            cursor = next_cursor
            has_more = getattr(page, "has_more", None)
            if isinstance(has_more, bool) and not has_more:
                break
            if not cursor:
                break
        if yielded == 0:
            return []

    def _sum_processing(self, client: Any) -> Optional[int]:
        if self.processing_limit is None:
            return None
        total = 0
        found = False
        for batch in self._iter_batches(client):
            found = True
            status = (_get_processing_status(batch) or "").lower()
            if status not in self._ACTIVE_STATUSES:
                continue
            counts = _get_request_counts(batch)
            total += int(counts.get("processing", 0))
        if found:
            return total
        if self._pending_batches:
            return sum(self._pending_batches.values())
        return None

    # Public API -----------------------------------------------------------

    def before_submit(self, client: Any, request_count: int) -> None:
        """Block until both RPM and queue capacity allow submitting a batch."""
        if request_count <= 0:
            return
        self._enforce_rpm_window(request_count)
        self._enforce_queue_capacity(client, request_count)

    def register_batch(self, batch_id: str | None, request_count: int) -> None:
        now = self._time()
        self._prune_window(now)
        self._request_window.append((now, max(0, int(request_count))))
        if batch_id:
            self._pending_batches[str(batch_id)] = max(0, int(request_count))

    def mark_batch_complete(self, batch: Any) -> None:
        batch_id = None
        if isinstance(batch, dict):
            batch_id = batch.get("id") or batch.get("batch_id")
        else:
            batch_id = getattr(batch, "id", None) or getattr(batch, "batch_id", None)
        if batch_id:
            self._pending_batches.pop(str(batch_id), None)

    def compute_retry_delay(self, error: Any, attempt: int) -> float:
        if attempt >= self.max_retries:
            raise error
        retry_after = _extract_retry_after_seconds(error)
        fallback = self.retry_after_fallback * (2 ** attempt)
        delay = max(retry_after or 0.0, fallback)
        jitter = min(delay * 0.1, 5.0)
        return delay + random.uniform(0.0, jitter)

    # Internal throttling -------------------------------------------------

    def _enforce_rpm_window(self, request_count: int) -> None:
        if self.rpm_limit is None or request_count <= 0:
            return
        while True:
            now = self._time()
            current = self._current_request_total(now)
            if current + request_count <= self.rpm_limit:
                return
            oldest_time, _ = self._request_window[0]
            wait_seconds = max(0.0, 60.0 - (now - oldest_time))
            wait_seconds = max(wait_seconds, 1.0)
            logger.info(
                "Anthropic batch RPM limit reached (%s requests queued in last minute); sleeping %.1fs",
                current,
                wait_seconds,
            )
            self._sleep(wait_seconds)

    def _enforce_queue_capacity(self, client: Any, request_count: int) -> None:
        if self.target_processing is None or request_count <= 0:
            return
        while True:
            total = self._sum_processing(client)
            if total is None:
                return
            projected = total + request_count
            if projected <= self.target_processing:
                return
            wait_seconds = self.queue_poll_seconds
            logger.info(
                "Anthropic processing queue at %s requests; target %s. Sleeping %.1fs before submitting.",
                total,
                self.target_processing,
                wait_seconds,
            )
            self._sleep(wait_seconds)
