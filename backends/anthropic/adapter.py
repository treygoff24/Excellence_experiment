from __future__ import annotations

import glob
import json
import os
import time
from datetime import datetime, timezone
from typing import Any, Mapping, Optional

from backends.anthropic.build_requests import build_message_requests, write_requests_preview
from backends.anthropic.normalize_to_openai import normalize_jsonl
from backends.anthropic.poll_and_stream import poll_until_complete, stream_results_to_jsonl
from backends.anthropic.rate_limiter import AnthropicBatchLimiter
from backends.anthropic.start_message_batch import start_message_batch
from fireworks.parse_results import process_results


def _utc_now_iso() -> str:
    return datetime.now(tz=timezone.utc).isoformat().replace("+00:00", "Z")


def _format_temp_label(temp: float) -> str:
    value = f"{float(temp):.1f}"
    return "0" if value == "0.0" else value.replace(".", "")


def _part_suffix_fragment(extra: Mapping[str, Any] | None) -> str:
    if not extra:
        return ""
    fragment = extra.get("part_suffix")
    if fragment:
        return str(fragment)
    part_index = extra.get("part_index")
    if part_index is None:
        return ""
    try:
        idx = int(part_index)
    except (TypeError, ValueError):
        return ""
    return "" if idx <= 0 else f"_p{idx + 1:02d}"


def _remap_custom_ids(jsonl_path: str, map_path: str) -> None:
    if not jsonl_path or not map_path:
        return
    try:
        with open(map_path, "r", encoding="utf-8") as fin:
            mapping = json.load(fin)
    except Exception:
        return
    if not isinstance(mapping, dict):
        return
    normalized_map = {str(k): str(v) for k, v in mapping.items() if k is not None and v is not None}
    if not normalized_map:
        return
    tmp_path = f"{jsonl_path}.tmp"
    try:
        with open(jsonl_path, "r", encoding="utf-8") as src, open(tmp_path, "w", encoding="utf-8") as dst:
            for line in src:
                s = line.strip()
                if not s:
                    continue
                try:
                    obj = json.loads(s)
                except json.JSONDecodeError:
                    dst.write(line)
                    continue
                provider_id = obj.get("custom_id") or obj.get("customId")
                if provider_id and provider_id in normalized_map:
                    obj.setdefault("provider_custom_id", provider_id)
                    obj["custom_id"] = normalized_map[provider_id]
                dst.write(json.dumps(obj, ensure_ascii=False) + "\n")
        os.replace(tmp_path, jsonl_path)
    except Exception:
        try:
            os.remove(tmp_path)
        except OSError:
            pass

def _stringify_dict(data: Optional[dict[str, Any]]) -> dict[str, Any]:
    if not data:
        return {}
    out: dict[str, Any] = {}
    for key, value in data.items():
        if key is None:
            continue
        out[str(key)] = value
    return out


def _maybe_model_dump(obj: Any) -> Any:
    if hasattr(obj, "model_dump"):
        return obj.model_dump()
    if hasattr(obj, "dict"):
        return obj.dict()
    return obj


class AnthropicBatchAdapter:
    backend = "anthropic"

    def __init__(self, cfg: dict[str, Any], *, client: Any | None = None) -> None:
        self.cfg = cfg
        provider_cfg = (
            cfg.get("provider")
            or cfg.get("anthropic")
            or {}
        )
        self.poll_seconds = float(provider_cfg.get("poll_seconds") or 30.0)
        self.batch_params = dict(provider_cfg.get("batch_params") or {})
        self.request_overrides = dict(provider_cfg.get("request_overrides") or {})
        self.request_metadata = dict(provider_cfg.get("request_metadata") or {})
        self.batch_metadata = dict(provider_cfg.get("batch_metadata") or provider_cfg.get("job_metadata") or {})
        self.client_options = dict(provider_cfg.get("client_options") or {})
        self.provider_cfg = provider_cfg
        self.rate_limits_cfg = dict(provider_cfg.get("rate_limits") or {})
        self._client = client
        self._limiter: AnthropicBatchLimiter | None = None
        self._anthropic: Any | None = None

    def _ensure_client(self) -> Any:
        if self._client is not None:
            return self._client
        try:
            import anthropic  # type: ignore[import-not-found]
        except ImportError as exc:  # pragma: no cover - depends on environment
            raise RuntimeError(
                "Anthropic SDK is required for Anthropic backend integration. Install the 'anthropic' package."
            ) from exc
        client_ctor = getattr(anthropic, "Anthropic", None)
        if client_ctor is None or not callable(client_ctor):
            raise RuntimeError("Anthropic SDK does not expose the expected Anthropic client constructor.")
        self._client = client_ctor(**self.client_options)
        self._anthropic = anthropic
        return self._client

    def _ensure_limiter(self, client: Any) -> AnthropicBatchLimiter | None:
        if self._limiter is not None:
            return self._limiter
        cfg = self.rate_limits_cfg or {}
        queue_limit = cfg["processing_queue_limit"] if "processing_queue_limit" in cfg else 100_000
        rpm_limit = cfg["batch_requests_per_minute"] if "batch_requests_per_minute" in cfg else 2_000
        safety_margin = cfg.get("processing_queue_safety_margin", 0.1)
        queue_poll_seconds = cfg.get("processing_queue_poll_seconds", 10.0)
        max_retries = cfg.get("max_rate_limit_retries", 5)
        retry_fallback = cfg.get("retry_after_fallback_seconds", 30.0)
        # Allow disabling by explicitly setting values to null in config
        queue_limit_int = int(queue_limit) if queue_limit is not None else None
        rpm_limit_int = int(rpm_limit) if rpm_limit is not None else None
        if queue_limit_int is None and rpm_limit_int is None:
            return None
        self._limiter = AnthropicBatchLimiter(
            processing_limit=queue_limit_int,
            safety_margin=float(safety_margin),
            queue_poll_seconds=float(queue_poll_seconds),
            rpm_limit=rpm_limit_int,
            max_retries=int(max_retries),
            retry_after_fallback=float(retry_fallback),
        )
        return self._limiter

    def _default_batch_dir(self, artifact_extra: dict[str, Any]) -> str:
        if artifact_extra.get("provider_batch_dir"):
            return artifact_extra["provider_batch_dir"]
        run_root = artifact_extra.get("run_root")
        if run_root:
            path = os.path.join(run_root, "batch_inputs", self.backend)
        else:
            base = (self.cfg.get("paths") or {}).get("batch_inputs_dir") or "data/batch_inputs"
            path = os.path.join(base, self.backend)
        os.makedirs(path, exist_ok=True)
        return path

    def _prepare_request_metadata(self, trial_slug: str, artifact: Any) -> dict[str, Any]:
        data = {
            "trial": trial_slug,
            "condition": artifact.condition,
            "mode": artifact.mode,
            "temperature": f"{float(artifact.temp):.1f}",
        }
        data.update(self.request_metadata)
        return _stringify_dict(data)

    def _prepare_batch_metadata(self, trial_slug: str, artifact: Any) -> dict[str, Any]:
        if not self.batch_metadata:
            return {"trial": trial_slug, "condition": artifact.condition}
        metadata = _stringify_dict(self.batch_metadata)
        metadata.setdefault("trial", trial_slug)
        metadata.setdefault("condition", artifact.condition)
        metadata.setdefault("temperature", f"{float(artifact.temp):.2f}")
        return metadata

    def submit(
        self,
        *,
        trial_slug: str,
        artifacts: Any,
        dry_run: bool,
    ) -> Any:
        extra = artifacts.extra
        part_suffix = _part_suffix_fragment(extra)
        if artifacts.mode == "reuse":
            artifacts.batch_id = artifacts.batch_id or (
                f"reuse-{self.backend}-{trial_slug}-{artifacts.condition}-t{_format_temp_label(artifacts.temp)}{part_suffix}"
            )
            artifacts.extra.setdefault("reuse", True)
            artifacts.extra.setdefault("submitted_at", _utc_now_iso())
            return artifacts

        source_path = extra.get("source_jsonl")
        if not source_path or not os.path.isfile(source_path):
            raise FileNotFoundError(f"Missing batch shard for Anthropic submission: {source_path}")

        batch_dir = self._default_batch_dir(extra)
        temp_label = _format_temp_label(artifacts.temp)
        request_preview_path = os.path.join(
            batch_dir,
            f"{trial_slug}_{artifacts.condition}_t{temp_label}{part_suffix}_requests.jsonl",
        )

        model_id = extra.get("model_id") or self.cfg.get("model_id")
        if not model_id:
            raise ValueError("Anthropic adapter requires a model_id in config or artifact metadata.")
        top_p = extra.get("top_p", self.cfg.get("top_p"))
        top_k = extra.get("top_k", self.cfg.get("top_k"))
        max_new_tokens = extra.get("max_new_tokens") or self.cfg.get("max_new_tokens")

        request_metadata = self._prepare_request_metadata(trial_slug, artifacts)
        requests = build_message_requests(
            src_path=source_path,
            model=model_id,
            temperature=float(artifacts.temp),
            top_p=top_p,
            top_k=top_k,
            max_new_tokens=max_new_tokens,
            metadata=request_metadata,
            request_overrides=self.request_overrides or None,
        )
        write_requests_preview(requests, request_preview_path)
        id_map_path = os.path.join(
            batch_dir,
            f"{trial_slug}_{artifacts.condition}_t{temp_label}{part_suffix}_id_map.json",
        )
        try:
            with open(id_map_path, "w", encoding="utf-8") as fout:
                json.dump({req.custom_id: req.orig_custom_id for req in requests}, fout, ensure_ascii=False, indent=2)
        except Exception:
            id_map_path = None

        request_count = len(requests)
        artifacts.extra["request_count"] = request_count
        artifacts.extra["request_preview"] = request_preview_path
        artifacts.extra["submitted_at"] = _utc_now_iso()
        if id_map_path:
            artifacts.extra["custom_id_map_path"] = id_map_path

        if dry_run:
            artifacts.batch_id = f"dry-{self.backend}-{trial_slug}-{artifacts.condition}-t{temp_label}{part_suffix}"
            return artifacts

        client = self._ensure_client()
        limiter = self._ensure_limiter(client)
        batch_metadata = self._prepare_batch_metadata(trial_slug, artifacts)
        attempt = 0
        while True:
            if limiter:
                limiter.before_submit(client, request_count)
            try:
                batch = start_message_batch(
                    client,
                    requests,
                    metadata=batch_metadata,
                    batch_params=self.batch_params or None,
                )
                break
            except Exception as exc:
                if not limiter or not self._is_rate_limit_error(exc):
                    raise
                delay = limiter.compute_retry_delay(exc, attempt)
                attempt += 1
                print(
                    f"Anthropic batch rate limited (attempt {attempt}); retrying in {delay:.1f}s.",
                )
                time.sleep(delay)
        batch_id = getattr(batch, "id", None) or getattr(batch, "batch_id", None) or (batch.get("id") if isinstance(batch, dict) else None)
        if not batch_id:
            raise RuntimeError("Anthropic batch submission did not return a batch id.")
        if limiter:
            limiter.register_batch(batch_id, request_count)
        artifacts.batch_id = batch_id
        artifacts.extra["batch_metadata"] = batch_metadata
        artifacts.extra["batch_status"] = getattr(batch, "processing_status", None) or (
            batch.get("processing_status") if isinstance(batch, dict) else None
        )
        artifacts.extra["request_counts"] = _maybe_model_dump(getattr(batch, "request_counts", None))
        if attempt:
            artifacts.extra["rate_limit_retries"] = attempt
        return artifacts

    def poll(
        self,
        *,
        results_dir: str,
        artifact: Any,
        dry_run: bool,
    ) -> Any:
        temp_label = _format_temp_label(artifact.temp)
        part_suffix = _part_suffix_fragment(getattr(artifact, "extra", {}))
        if artifact.mode == "reuse":
            artifact.extra.setdefault("reuse", True)
            artifact.extra.setdefault("poll_completed_at", _utc_now_iso())
            return artifact

        if dry_run:
            placeholder = os.path.join(results_dir, f"{artifact.condition}_t{temp_label}{part_suffix}_anthropic_dry.jsonl")
            os.makedirs(os.path.dirname(placeholder), exist_ok=True)
            if not os.path.isfile(placeholder):
                with open(placeholder, "w", encoding="utf-8") as fout:
                    fout.write(
                        json.dumps(
                            {
                                "custom_id": f"{artifact.condition}|dry|{_format_temp_label(artifact.temp)}|0",
                                "response": {
                                    "status": "succeeded",
                                    "body": {
                                        "choices": [
                                            {
                                                "index": 0,
                                                "message": {
                                                    "role": "assistant",
                                                    "content": "dry-run placeholder",
                                                },
                                                "finish_reason": "stop",
                                            }
                                        ],
                                        "usage": {"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2},
                                    },
                                    "usage": {"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2},
                                    "provider": {"name": "anthropic"},
                                },
                            }
                        )
                        + "\n"
                    )
            artifact.results_uri = placeholder
            artifact.extra["poll_completed_at"] = _utc_now_iso()
            artifact.extra["dry_run"] = True
            return artifact

        if not artifact.batch_id:
            raise ValueError("Cannot poll Anthropic batch without batch_id.")

        client = self._ensure_client()
        limiter = self._ensure_limiter(client)
        batch = poll_until_complete(client, artifact.batch_id, poll_seconds=self.poll_seconds)
        if limiter:
            limiter.mark_batch_complete(batch)
        status = getattr(batch, "processing_status", None) or (
            batch.get("processing_status") if isinstance(batch, dict) else None
        )

        raw_path = os.path.join(results_dir, f"{artifact.condition}_t{temp_label}{part_suffix}_anthropic_raw.jsonl")
        normalized_path = os.path.join(results_dir, f"{artifact.condition}_t{temp_label}{part_suffix}_results.jsonl")

        streamed = stream_results_to_jsonl(client, artifact.batch_id, raw_path)
        normalized = normalize_jsonl(raw_path, normalized_path)
        id_map_path = getattr(artifact, "extra", {}).get("custom_id_map_path")
        if id_map_path:
            _remap_custom_ids(normalized_path, id_map_path)

        artifact.results_uri = normalized_path
        artifact.extra["batch_status"] = status
        artifact.extra["raw_results_path"] = raw_path
        artifact.extra["normalized_path"] = normalized_path
        artifact.extra["streamed_rows"] = streamed
        artifact.extra["normalized_rows"] = normalized
        artifact.extra["request_counts"] = _maybe_model_dump(getattr(batch, "request_counts", None))
        artifact.extra["poll_completed_at"] = _utc_now_iso()
        return artifact

    def _combine_results(self, results_dir: str) -> str | None:
        normalized_paths = sorted(glob.glob(os.path.join(results_dir, "*_results.jsonl")))
        if not normalized_paths:
            return None
        combined = os.path.join(results_dir, "results_combined.jsonl")
        with open(combined, "w", encoding="utf-8") as fout:
            for path in normalized_paths:
                with open(path, "r", encoding="utf-8") as fin:
                    for line in fin:
                        if line.strip():
                            fout.write(line.rstrip() + "\n")
        return combined

    def parse(
        self,
        *,
        results_dir: str,
        artifact: Any,
        dry_run: bool,
    ) -> Any:
        if artifact.mode == "reuse":
            artifact.extra.setdefault("parsed_at", _utc_now_iso())
            return artifact

        if dry_run:
            artifact.extra["parsed_at"] = _utc_now_iso()
            return artifact

        combined_path = self._combine_results(results_dir)
        if not combined_path:
            raise RuntimeError("No Anthropic normalized results available to parse.")
        predictions_csv = os.path.join(results_dir, "predictions.csv")
        process_results(combined_path, predictions_csv)
        artifact.extra["predictions_csv"] = predictions_csv
        artifact.extra["combined_results_path"] = combined_path
        artifact.extra["parsed_at"] = _utc_now_iso()
        return artifact

    def _is_rate_limit_error(self, exc: Exception) -> bool:
        if self._anthropic is None:
            try:
                import anthropic  # type: ignore[import-not-found]
            except Exception:  # pragma: no cover - optional dependency
                anthropic = None  # type: ignore[assignment]
            else:
                self._anthropic = anthropic
        rate_limit_cls = None
        if self._anthropic is not None:
            rate_limit_cls = getattr(self._anthropic, "RateLimitError", None)
        if rate_limit_cls is not None and isinstance(exc, rate_limit_cls):
            return True
        status_code = getattr(exc, "status_code", None)
        if status_code == 429:
            return True
        if exc.__class__.__name__.lower() == "ratelimiterror":
            return True
        body = getattr(exc, "body", None)
        data: dict[str, Any] = {}
        if isinstance(body, dict):
            data = body
        elif hasattr(body, "model_dump"):
            try:
                dumped = body.model_dump()
                if isinstance(dumped, dict):
                    data = dumped
            except Exception:
                data = {}
        error_obj = data.get("error") if isinstance(data, dict) else None
        if isinstance(error_obj, dict) and error_obj.get("type") == "rate_limit_error":
            return True
        return False
