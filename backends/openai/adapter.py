from __future__ import annotations

import glob
import json
import os
from datetime import datetime, timezone
from typing import Any, Mapping, Optional

import logging

from backends.openai.build_inputs import build_batch_requests, ensure_thinking_budget
from backends.openai.normalize import normalize_jsonl
from backends.openai.poll_and_download import download_and_extract, poll_until_complete
from backends.openai.start_batch_job import start_batch_job
from fireworks.parse_results import process_results


logger = logging.getLogger(__name__)


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


def _coerce_max_tokens(data: Any) -> dict[str, int]:
    if data is None:
        return {}
    if hasattr(data, "model_dump"):
        return {k: int(v) for k, v in data.model_dump().items() if v is not None}
    if hasattr(data, "dict"):
        return {k: int(v) for k, v in data.dict().items() if v is not None}
    if isinstance(data, dict):
        try:
            return {str(k): int(v) for k, v in data.items() if v is not None}
        except Exception:
            return {}
    return {}


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


class OpenAIBatchAdapter:
    backend = "openai"

    def __init__(self, cfg: dict[str, Any], *, client: Any | None = None) -> None:
        self.cfg = cfg
        provider_cfg = cfg.get("provider") or {}
        batch_cfg = provider_cfg.get("batch") or {}
        self.endpoint = batch_cfg.get("endpoint") or provider_cfg.get("endpoint") or "/v1/responses"
        completion_window = batch_cfg.get("completion_window") or provider_cfg.get("completion_window")
        self.completion_window = completion_window or "24h"
        poll_seconds = provider_cfg.get("poll_seconds")
        if poll_seconds is None:
            poll_seconds = batch_cfg.get("poll_seconds")
        self.poll_seconds = float(poll_seconds) if poll_seconds is not None else 30.0
        self.batch_params = dict(batch_cfg.get("batch_params") or {})
        self.batch_params.update(provider_cfg.get("batch_params") or {})
        self.request_overrides = dict(batch_cfg.get("request_overrides") or {})
        self.request_overrides.update(provider_cfg.get("request_overrides") or {})
        self.request_metadata = dict(batch_cfg.get("request_metadata") or {})
        self.request_metadata.update(provider_cfg.get("request_metadata") or {})
        job_metadata = batch_cfg.get("metadata") or batch_cfg.get("job_metadata")
        if job_metadata is None:
            job_metadata = provider_cfg.get("batch_metadata") or provider_cfg.get("job_metadata")
        self.job_metadata = dict(job_metadata or {})
        self.provider_cfg = provider_cfg
        self.batch_cfg = batch_cfg
        ensure_thinking_budget(self.request_overrides, context="provider.request_overrides")
        ensure_thinking_budget(self.batch_params, context="provider.batch_params")
        allow_temperature = provider_cfg.get("allow_temperature")
        self.allow_temperature = True if allow_temperature is None else bool(allow_temperature)
        batch_allow_temperature = self.batch_cfg.get("allow_temperature")
        if batch_allow_temperature is not None:
            self.allow_temperature = bool(batch_allow_temperature)
        allow_top_p = provider_cfg.get("allow_top_p")
        self.allow_top_p = True if allow_top_p is None else bool(allow_top_p)
        batch_allow_top_p = self.batch_cfg.get("allow_top_p")
        if batch_allow_top_p is not None:
            self.allow_top_p = bool(batch_allow_top_p)

        if not self.allow_temperature:
            self.request_overrides.pop("temperature", None)
            self.batch_params.pop("temperature", None)
        if not self.allow_top_p:
            self.request_overrides.pop("top_p", None)
            self.batch_params.pop("top_p", None)

        reasoning_cfg = self.request_overrides.get("reasoning") or self.batch_params.get("reasoning")
        if reasoning_cfg and self.endpoint != "/v1/responses":
            raise ValueError(
                "OpenAI reasoning runs must target the /v1/responses endpoint. "
                f"Configured endpoint {self.endpoint!r} is incompatible with reasoning overrides."
            )
        self._client = client

    def _ensure_client(self) -> Any:
        if self._client is not None:
            return self._client
        from openai import OpenAI

        client_options: dict[str, Any] = {}
        client_options.update(self.batch_cfg.get("client_options") or {})
        client_options.update(self.provider_cfg.get("client_options") or {})
        self._client = OpenAI(**client_options)
        return self._client

    def _default_batch_dir(self, artifact_extra: dict[str, Any]) -> str:
        if artifact_extra.get("provider_batch_dir"):
            return artifact_extra["provider_batch_dir"]
        run_root = artifact_extra.get("run_root")
        if run_root:
            path = os.path.join(run_root, "batch_inputs")
        else:
            base = (self.cfg.get("paths") or {}).get("batch_inputs_dir") or "data/batch_inputs"
            path = os.path.join(base, "openai")
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
            logger.info(
                "Reusing OpenAI batch artifacts for trial %s (%s, temp=%s, batch_id=%s)",
                trial_slug,
                artifacts.condition,
                artifacts.temp,
                artifacts.batch_id,
            )
            artifacts.batch_id = artifacts.batch_id or (
                f"reuse-openai-{trial_slug}-{artifacts.condition}-t{_format_temp_label(artifacts.temp)}{part_suffix}"
            )
            artifacts.extra.setdefault("reuse", True)
            artifacts.extra.setdefault("submitted_at", _utc_now_iso())
            return artifacts

        source_path = extra.get("source_jsonl")
        if not source_path or not os.path.isfile(source_path):
            raise FileNotFoundError(f"Missing batch shard for OpenAI submission: {source_path}")

        batch_dir = self._default_batch_dir(extra)
        temp_label = _format_temp_label(artifacts.temp)
        request_path = os.path.join(batch_dir, f"{trial_slug}_{artifacts.condition}_t{temp_label}{part_suffix}.jsonl")

        model_id = extra.get("model_id") or self.cfg.get("model_id")
        if not model_id:
            raise ValueError("OpenAI adapter requires a model_id in config or artifact metadata.")
        top_p = extra.get("top_p", self.cfg.get("top_p"))
        if not self.allow_top_p:
            top_p = None
        max_tokens = _coerce_max_tokens(extra.get("max_new_tokens") or self.cfg.get("max_new_tokens"))
        endpoint = extra.get("endpoint") or self.endpoint

        request_count = build_batch_requests(
            src_path=source_path,
            dest_path=request_path,
            model=model_id,
            temperature=float(artifacts.temp),
            top_p=top_p,
            max_new_tokens=max_tokens,
            endpoint=endpoint,
            metadata=self._prepare_request_metadata(trial_slug, artifacts),
            request_overrides=self.request_overrides or None,
            allow_temperature=self.allow_temperature,
            allow_top_p=self.allow_top_p,
        )
        artifacts.extra["request_count"] = request_count
        artifacts.extra["request_path"] = request_path
        artifacts.extra["endpoint"] = endpoint
        artifacts.extra["submitted_at"] = _utc_now_iso()
        logger.info(
            "Prepared OpenAI batch payload for trial %s (%s, temp=%s): %s requests written to %s (endpoint=%s, model=%s)",
            trial_slug,
            artifacts.condition,
            artifacts.temp,
            request_count,
            request_path,
            endpoint,
            model_id,
        )

        if dry_run:
            logger.info(
                "Dry-run mode active; skipping OpenAI batch submission for trial %s (%s, temp=%s).",
                trial_slug,
                artifacts.condition,
                artifacts.temp,
            )
            artifacts.batch_id = f"dry-openai-{trial_slug}-{artifacts.condition}-t{temp_label}{part_suffix}"
            return artifacts

        client = self._ensure_client()
        job_metadata = dict(self.job_metadata)
        job_metadata.setdefault("trial", trial_slug)
        job_metadata.setdefault("condition", artifacts.condition)
        job_metadata.setdefault("temperature", f"{float(artifacts.temp):.2f}")
        try:
            logger.info(
                "Submitting OpenAI batch job for trial %s (%s, temp=%s); completion_window=%s, batch_params=%s",
                trial_slug,
                artifacts.condition,
                artifacts.temp,
                self.completion_window,
                sorted(self.batch_params.keys()),
            )
            input_file_id, job = start_batch_job(
                client,
                request_path=request_path,
                endpoint=endpoint,
                completion_window=self.completion_window,
                metadata=job_metadata,
                batch_params=self.batch_params or None,
            )
        except Exception:
            logger.exception(
                "Failed to submit OpenAI batch job for trial %s (%s, temp=%s).",
                trial_slug,
                artifacts.condition,
                artifacts.temp,
            )
            raise
        artifacts.batch_id = getattr(job, "id", None)
        artifacts.extra["input_file_id"] = input_file_id
        artifacts.extra["job_status"] = getattr(job, "status", None)
        artifacts.extra["job_metadata"] = job_metadata
        artifacts.extra["submitted_at"] = _utc_now_iso()
        logger.info(
            "OpenAI batch job created for trial %s (%s, temp=%s): batch_id=%s, status=%s, input_file_id=%s",
            trial_slug,
            artifacts.condition,
            artifacts.temp,
            artifacts.batch_id,
            artifacts.extra["job_status"],
            input_file_id,
        )
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
            logger.info(
                "Skipping OpenAI poll for reused artifact (batch_id=%s, condition=%s, temp=%s).",
                artifact.batch_id,
                artifact.condition,
                artifact.temp,
            )
            artifact.extra.setdefault("reuse", True)
            artifact.extra.setdefault("poll_completed_at", _utc_now_iso())
            return artifact

        if dry_run:
            placeholder = os.path.join(results_dir, f"{artifact.condition}_t{temp_label}{part_suffix}_dry.jsonl")
            os.makedirs(os.path.dirname(placeholder), exist_ok=True)
            if not os.path.isfile(placeholder):
                with open(placeholder, "w", encoding="utf-8") as fout:
                    fout.write(
                        json.dumps(
                            {
                                "custom_id": f"dry|{artifact.condition}|{artifact.temp}",
                                "response": {
                                    "body": {
                                        "choices": [
                                            {
                                                "message": {"role": "assistant", "content": "dry-run placeholder"},
                                                "finish_reason": "stop",
                                            }
                                        ],
                                        "usage": {"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2},
                                    },
                                    "usage": {"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2},
                                },
                            }
                        )
                        + "\n"
                    )
            artifact.results_uri = placeholder
            artifact.extra["poll_completed_at"] = _utc_now_iso()
            logger.info(
                "Dry-run polling placeholder written for condition=%s temp=%s at %s.",
                artifact.condition,
                artifact.temp,
                placeholder,
            )
            return artifact

        if not artifact.batch_id:
            raise ValueError("Cannot poll OpenAI batch without batch_id.")

        client = self._ensure_client()
        logger.info(
            "Polling OpenAI batch %s for condition=%s temp=%s (poll interval %.1fs).",
            artifact.batch_id,
            artifact.condition,
            artifact.temp,
            self.poll_seconds,
        )

        def _status_logger(status: str) -> None:
            logger.info("OpenAI batch %s transitioned to status='%s'.", artifact.batch_id, status)

        try:
            job = poll_until_complete(
                client,
                artifact.batch_id,
                poll_seconds=self.poll_seconds,
                status_callback=_status_logger,
            )
        except RuntimeError:
            logger.exception(
                "OpenAI batch %s failed while polling (condition=%s, temp=%s).",
                artifact.batch_id,
                artifact.condition,
                artifact.temp,
            )
            raise
        artifact.extra["job_status"] = getattr(job, "status", None)
        artifact.extra["request_counts"] = _maybe_model_dump(getattr(job, "request_counts", None))
        artifact.extra["usage"] = _maybe_model_dump(getattr(job, "usage", None))
        output_file_id = getattr(job, "output_file_id", None)
        if not output_file_id:
            logger.error(
                "OpenAI batch %s completed without output_file_id (status=%s).",
                artifact.batch_id,
                artifact.extra.get("job_status"),
            )
            raise RuntimeError(f"OpenAI batch {artifact.batch_id} completed without output_file_id.")

        endpoint = artifact.extra.get("endpoint") or self.endpoint
        logger.info(
            "OpenAI batch %s completed with output_file_id=%s; downloading to %s.",
            artifact.batch_id,
            output_file_id,
            results_dir,
        )
        extracted = download_and_extract(client, output_file_id=output_file_id, out_dir=results_dir)
        normalized_path = os.path.join(results_dir, f"{artifact.condition}_t{temp_label}{part_suffix}_results.jsonl")
        normalize_jsonl(extracted, normalized_path, endpoint=endpoint)
        logger.info(
            "Normalized OpenAI results for batch %s written to %s (source files=%s).",
            artifact.batch_id,
            normalized_path,
            [os.path.basename(p) for p in extracted],
        )

        artifact.output_file_id = output_file_id
        artifact.results_uri = normalized_path
        artifact.extra["normalized_path"] = normalized_path
        artifact.extra["extracted_files"] = extracted
        artifact.extra["poll_completed_at"] = _utc_now_iso()
        return artifact

    def _combine_results(self, results_dir: str) -> Optional[str]:
        normalized_paths = sorted(glob.glob(os.path.join(results_dir, "*_results.jsonl")))
        if not normalized_paths:
            return None
        combined = os.path.join(results_dir, "results.jsonl")
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
        temp_label = _format_temp_label(artifact.temp)
        part_suffix = _part_suffix_fragment(getattr(artifact, "extra", {}))
        if artifact.mode == "reuse":
            artifact.extra.setdefault("parsed_at", _utc_now_iso())
            return artifact

        if dry_run:
            artifact.output_file_id = artifact.output_file_id or (
                f"dry-output-openai-{artifact.condition}-t{temp_label}{part_suffix}"
            )
            artifact.extra["parsed_at"] = _utc_now_iso()
            return artifact

        combined_path = self._combine_results(results_dir)
        if not combined_path:
            logger.error(
                "No normalized OpenAI results found in %s for batch %s.",
                results_dir,
                artifact.batch_id,
            )
            raise RuntimeError("No normalized OpenAI results found to parse.")
        predictions_csv = os.path.join(results_dir, "predictions.csv")
        process_results(combined_path, predictions_csv)
        logger.info(
            "Parsed OpenAI batch %s results into %s.",
            artifact.batch_id,
            predictions_csv,
        )
        artifact.extra["predictions_csv"] = predictions_csv
        artifact.extra["parsed_at"] = _utc_now_iso()
        return artifact
