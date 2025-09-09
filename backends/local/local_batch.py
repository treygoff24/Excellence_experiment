from __future__ import annotations

import json
import os
import time
import uuid
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Optional, Tuple

from backends.interfaces import InferenceClient


def _count_lines(path: str) -> int:
    n = 0
    try:
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    n += 1
    except Exception:
        return 0
    return n


def _iter_jsonl(path: str) -> Iterable[dict]:
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            s = line.strip()
            if not s:
                continue
            try:
                yield json.loads(s)
            except Exception:
                # Skip malformed lines but continue processing
                continue


def _parse_custom_id(cid: str) -> Tuple[str, str, str, float, int, str]:
    parts = str(cid).split("|")
    if len(parts) != 6:
        raise ValueError(f"Bad custom_id: {cid}")
    dataset, item_id, condition, temp_str, sample_idx, typ = parts
    return dataset, item_id, condition, float(temp_str), int(sample_idx), typ


def _ensure_dir(p: str) -> None:
    os.makedirs(p, exist_ok=True)


def _safe_atomic_write_jsonl(out_path: str, rows: List[dict]) -> None:
    # Write to a temp file first to avoid partial/corrupt outputs
    _ensure_dir(os.path.dirname(out_path))
    tmp = out_path + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
        f.flush()
        os.fsync(f.fileno())
    # Atomic rename
    if os.name == "nt":
        if os.path.exists(out_path):
            os.remove(out_path)
    os.replace(tmp, out_path)


def _build_client(cfg: dict) -> InferenceClient:
    engine = (cfg.get("local_engine") or "").strip().lower()
    model = cfg.get("local_model")
    if engine == "ollama":
        from backends.local.ollama_client import OllamaClient

        base = cfg.get("local_endpoint") or "http://127.0.0.1:11434"
        return OllamaClient(base_url=str(base), model=model)
    elif engine == "llama_cpp":
        from backends.local.llama_cpp_client import LlamaCppClient

        if not model:
            raise ValueError("Config.local_model must point to a GGUF path for llama_cpp")
        return LlamaCppClient(model_path=str(model))
    else:
        raise SystemExit("Unsupported or missing local_engine. Set config.local_engine to 'ollama' or 'llama_cpp'.")


def split_jsonl(
    src_path: str,
    out_dir: str,
    base_prefix: str,
    *,
    parts: Optional[int] = None,
    lines_per_part: Optional[int] = None,
    limit_items: Optional[int] = None,
) -> list[Tuple[int, str]]:
    """Split a JSONL into parts; returns [(part_number, path)].

    Mirrors the behavior used by the Fireworks path, but kept lightweight
    and isolated for local execution.
    """
    _ensure_dir(out_dir)
    # Read all lines up to limit
    items: List[str] = []
    with open(src_path, "r", encoding="utf-8") as f:
        for line in f:
            s = line.strip()
            if not s:
                continue
            items.append(s)
            if limit_items is not None and len(items) >= int(limit_items):
                break

    if not items:
        return []

    if lines_per_part and lines_per_part > 0:
        chunk = int(lines_per_part)
    elif parts and parts > 0:
        chunk = max(1, (len(items) + int(parts) - 1) // int(parts))
    else:
        # Default single part
        chunk = len(items)

    outputs: list[Tuple[int, str]] = []
    part_no = 1
    for i in range(0, len(items), chunk):
        shard = items[i : i + chunk]
        out_path = os.path.join(out_dir, f"{base_prefix}_p{part_no:02d}.jsonl")
        with open(out_path, "w", encoding="utf-8") as fo:
            for s in shard:
                fo.write(s + "\n")
        outputs.append((part_no, out_path))
        part_no += 1
    return outputs


@dataclass
class _Job:
    part_number: int
    input_path: str
    dataset_id: Optional[str] = None  # Unused for local, kept for API parity


class LocalQueueManager:
    """Local work queue that executes per-part JSONL using a local engine.

    API mirrors the subset used by scripts.run_all for the Fireworks path.
    Each part is processed sequentially; per-item requests within a part are
    executed with a bounded thread pool.
    """

    def __init__(
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
    ) -> None:
        self.account_id = account_id
        self.model_id = model_id
        self.cfg = config
        # Hard cap at 2 unless explicitly higher
        self.max_concurrent = max(1, min(int(max_concurrent or 1), 2))
        self.temp_label = temp_label
        self.temperature = float(temperature)
        self.condition = condition
        self.run_id = run_id
        self.stop_event = stop_event
        self.jobs: list[_Job] = []
        # Lazily initialized client per run
        self._client: Optional[InferenceClient] = None

    # Compatibility: used by run_all to listen to progress; optional here
    progress_cb: Optional[Any] = None

    @property
    def client(self) -> InferenceClient:
        if self._client is None:
            self._client = _build_client(self.cfg)
        return self._client

    def add_job(self, part_number: int, input_path: str, dataset_id: Optional[str] = None) -> None:
        self.jobs.append(_Job(part_number=part_number, input_path=input_path, dataset_id=dataset_id))

    # -----------------------------
    # Core execution
    # -----------------------------
    def run_queue(self, results_dir: str) -> None:
        for job in sorted(self.jobs, key=lambda j: j.part_number):
            if self.stop_event and getattr(self.stop_event, "is_set", lambda: False)():
                break
            self._run_one_part(job, results_dir)

    def _run_one_part(self, job: _Job, results_dir: str) -> None:
        # Output location mirrors Fireworks per-part directory
        group_key = f"t{self.temp_label}_{self.condition}"
        job_key = f"{group_key}_p{job.part_number:02d}"
        job_dir = os.path.join(results_dir, job_key)
        _ensure_dir(job_dir)
        out_jsonl = os.path.join(job_dir, "results.jsonl")
        err_jsonl = os.path.join(job_dir, "errors.jsonl")
        state_path = os.path.join(job_dir, "state.json")

        # Idempotent resume: if out count matches in count, skip
        in_count = _count_lines(job.input_path)
        out_count = _count_lines(out_jsonl)
        if in_count > 0 and out_count == in_count:
            # Already completed
            return

        # Read inputs
        rows = list(_iter_jsonl(job.input_path))
        if not rows:
            # Write empty output atomically
            _safe_atomic_write_jsonl(out_jsonl, [])
            return

        # Prepare per-item tasks
        tasks: list[Tuple[int, dict]] = []  # (index, input_row)
        for idx, r in enumerate(rows):
            tasks.append((idx, r))

        # Run with bounded concurrency; collect results in-order
        results_buf: list[Optional[dict]] = [None] * len(tasks)
        errors: list[dict] = []

        def _run_one(idx: int, row: dict) -> Tuple[int, dict]:
            cid = row.get("custom_id") or row.get("customId")
            body = row.get("body") or {}
            messages = body.get("messages")
            prompt = body.get("prompt")
            stop_list = body.get("stop") or (self.cfg.get("stop") or [])
            # Choose max tokens based on type in custom_id
            try:
                _ds, _iid, _cond, _t, _k, typ = _parse_custom_id(cid)
            except Exception:
                typ = "closed"
            mx_cfg = self.cfg.get("max_new_tokens", {}) or {}
            max_new = int(mx_cfg.get("open_book", 1024) if typ == "open" else mx_cfg.get("closed_book", 1024))
            params = {
                "temperature": float(self.temperature),
                "top_p": self.cfg.get("top_p"),
                "top_k": self.cfg.get("top_k"),
                "max_new_tokens": max_new,
                "stop": stop_list,
            }

            # Retries with simple backoff
            last_exc: Optional[Exception] = None
            for attempt in range(3):
                t0 = time.time()
                try:
                    resp = self.client.generate(
                        messages=messages if messages is not None else None,
                        prompt=prompt if (messages is None and prompt is not None) else None,
                        model=str(self.cfg.get("local_model") or self.model_id),
                        params=params,
                    )
                    txt = (resp.get("text") or "").strip()
                    finish = resp.get("finish_reason") or "stop"
                    usage = resp.get("usage") or {}
                    req_id = resp.get("request_id") or str(uuid.uuid4())
                    latency = float(resp.get("latency_s") or (time.time() - t0))
                    out_obj = {
                        "custom_id": cid,
                        "response": {
                            # Body compatible with fireworks.parse_results expectations
                            "body": {
                                "choices": [
                                    {
                                        "message": {"content": txt},
                                        "finish_reason": finish,
                                    }
                                ],
                                "usage": usage,
                                "id": req_id,
                            },
                            "usage": usage,
                            "request_id": req_id,
                            "latency_s": latency,
                        },
                    }
                    return idx, out_obj
                except Exception as e:
                    last_exc = e
                    # Backoff: 0.5s, 1.0s, 2.0s
                    time.sleep(0.5 * (2 ** attempt))
            # On failure after retries, emit an error-shaped record
            err_obj = {
                "custom_id": cid,
                "error": str(last_exc) if last_exc else "unknown",
            }
            errors.append(err_obj)
            # Still emit a response record with finish_reason=error to keep counts stable
            fail_obj = {
                "custom_id": cid,
                "response": {
                    "body": {
                        "choices": [
                            {
                                "message": {"content": ""},
                                "finish_reason": "error",
                            }
                        ],
                        "usage": {},
                        "id": str(uuid.uuid4()),
                    },
                    "usage": {},
                    "request_id": str(uuid.uuid4()),
                    "latency_s": 0.0,
                },
            }
            return idx, fail_obj

        with ThreadPoolExecutor(max_workers=self.max_concurrent) as ex:
            futs = {ex.submit(_run_one, i, r): i for i, r in tasks}
            for fut in as_completed(futs):
                i, obj = fut.result()
                results_buf[i] = obj

        # Verify all results generated
        out_rows: List[dict] = [x for x in results_buf if x is not None]
        if len(out_rows) != len(rows):
            # Fallback: drop Nones and continue; caller will see mismatch
            out_rows = [x for x in out_rows if x is not None]

        # Write outputs atomically
        _safe_atomic_write_jsonl(out_jsonl, out_rows)
        if errors:
            _safe_atomic_write_jsonl(err_jsonl, errors)

        # Write lightweight state for debugging
        try:
            with open(state_path, "w", encoding="utf-8") as fs:
                json.dump(
                    {
                        "input_path": os.path.relpath(job.input_path, results_dir),
                        "output_path": os.path.relpath(out_jsonl, results_dir),
                        "errors_path": os.path.relpath(err_jsonl, results_dir),
                        "items_in": int(len(rows)),
                        "items_out": int(len(out_rows)),
                        "errors": int(len(errors)),
                        "max_concurrent": int(self.max_concurrent),
                        "temperature": float(self.temperature),
                        "condition": str(self.condition),
                    },
                    fs,
                    indent=2,
                )
        except Exception:
            pass


def upload_datasets(
    account_id: str,
    dataset_files: list[Tuple[int, str]],
    base_name: str,
    temp_label: str,
    condition: str,
) -> list[Tuple[int, str]]:
    """Local path passthrough for API parity with Fireworks executor.

    Returns the same list but with the path echoed as a "dataset id".
    """
    return [(n, p) for (n, p) in dataset_files]


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
    stop_event: Optional[object] = None,
) -> LocalQueueManager:
    return LocalQueueManager(
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

