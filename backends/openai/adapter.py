from __future__ import annotations

import glob
import json
import os
from datetime import datetime, timezone
from typing import Any, Optional

from backends.openai.build_inputs import build_batch_requests
from backends.openai.normalize import normalize_jsonl
from backends.openai.poll_and_download import download_and_extract, poll_until_complete
from backends.openai.start_batch_job import start_batch_job
from fireworks.parse_results import process_results


def _utc_now_iso() -> str:
    return datetime.now(tz=timezone.utc).isoformat().replace("+00:00", "Z")


def _format_temp_label(temp: float) -> str:
    value = f"{float(temp):.1f}"
    return "0" if value == "0.0" else value.replace(".", "")


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
        self.endpoint = provider_cfg.get("endpoint", "/v1/responses")
        self.completion_window = provider_cfg.get("completion_window", "24h")
        self.poll_seconds = float(provider_cfg.get("poll_seconds") or 30.0)
        self.batch_params = dict(provider_cfg.get("batch_params") or {})
        self.request_overrides = dict(provider_cfg.get("request_overrides") or {})
        self.request_metadata = dict(provider_cfg.get("request_metadata") or {})
        self.job_metadata = dict(provider_cfg.get("job_metadata") or {})
        self.provider_cfg = provider_cfg
        self._client = client

    def _ensure_client(self) -> Any:
        if self._client is not None:
            return self._client
        from openai import OpenAI

        client_options = dict(self.provider_cfg.get("client_options") or {})
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
        if artifacts.mode == "reuse":
            artifacts.batch_id = artifacts.batch_id or f"reuse-openai-{trial_slug}-{artifacts.condition}-t{_format_temp_label(artifacts.temp)}"
            artifacts.extra.setdefault("reuse", True)
            artifacts.extra.setdefault("submitted_at", _utc_now_iso())
            return artifacts

        source_path = extra.get("source_jsonl")
        if not source_path or not os.path.isfile(source_path):
            raise FileNotFoundError(f"Missing batch shard for OpenAI submission: {source_path}")

        batch_dir = self._default_batch_dir(extra)
        temp_label = _format_temp_label(artifacts.temp)
        request_path = os.path.join(batch_dir, f"{trial_slug}_{artifacts.condition}_t{temp_label}.jsonl")

        model_id = extra.get("model_id") or self.cfg.get("model_id")
        if not model_id:
            raise ValueError("OpenAI adapter requires a model_id in config or artifact metadata.")
        top_p = extra.get("top_p", self.cfg.get("top_p"))
        top_k = extra.get("top_k", self.cfg.get("top_k"))
        max_tokens = _coerce_max_tokens(extra.get("max_new_tokens") or self.cfg.get("max_new_tokens"))
        endpoint = extra.get("endpoint") or self.endpoint

        request_count = build_batch_requests(
            src_path=source_path,
            dest_path=request_path,
            model=model_id,
            temperature=float(artifacts.temp),
            top_p=top_p,
            top_k=top_k,
            max_new_tokens=max_tokens,
            endpoint=endpoint,
            metadata=self._prepare_request_metadata(trial_slug, artifacts),
            request_overrides=self.request_overrides or None,
        )
        artifacts.extra["request_count"] = request_count
        artifacts.extra["request_path"] = request_path
        artifacts.extra["endpoint"] = endpoint
        artifacts.extra["submitted_at"] = _utc_now_iso()

        if dry_run:
            artifacts.batch_id = f"dry-openai-{trial_slug}-{artifacts.condition}-t{temp_label}"
            return artifacts

        client = self._ensure_client()
        job_metadata = dict(self.job_metadata)
        job_metadata.setdefault("trial", trial_slug)
        job_metadata.setdefault("condition", artifacts.condition)
        job_metadata.setdefault("temperature", f"{float(artifacts.temp):.2f}")
        input_file_id, job = start_batch_job(
            client,
            request_path=request_path,
            endpoint=endpoint,
            completion_window=self.completion_window,
            metadata=job_metadata,
            batch_params=self.batch_params or None,
        )
        artifacts.batch_id = getattr(job, "id", None)
        artifacts.extra["input_file_id"] = input_file_id
        artifacts.extra["job_status"] = getattr(job, "status", None)
        artifacts.extra["job_metadata"] = job_metadata
        artifacts.extra["submitted_at"] = _utc_now_iso()
        return artifacts

    def poll(
        self,
        *,
        results_dir: str,
        artifact: Any,
        dry_run: bool,
    ) -> Any:
        temp_label = _format_temp_label(artifact.temp)
        if artifact.mode == "reuse":
            artifact.extra.setdefault("reuse", True)
            artifact.extra.setdefault("poll_completed_at", _utc_now_iso())
            return artifact

        if dry_run:
            placeholder = os.path.join(results_dir, f"{artifact.condition}_t{temp_label}_dry.jsonl")
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
            return artifact

        if not artifact.batch_id:
            raise ValueError("Cannot poll OpenAI batch without batch_id.")

        client = self._ensure_client()
        job = poll_until_complete(client, artifact.batch_id, poll_seconds=self.poll_seconds)
        artifact.extra["job_status"] = getattr(job, "status", None)
        artifact.extra["request_counts"] = _maybe_model_dump(getattr(job, "request_counts", None))
        artifact.extra["usage"] = _maybe_model_dump(getattr(job, "usage", None))
        output_file_id = getattr(job, "output_file_id", None)
        if not output_file_id:
            raise RuntimeError(f"OpenAI batch {artifact.batch_id} completed without output_file_id.")

        endpoint = artifact.extra.get("endpoint") or self.endpoint
        extracted = download_and_extract(client, output_file_id=output_file_id, out_dir=results_dir)
        normalized_path = os.path.join(results_dir, f"{artifact.condition}_t{temp_label}_results.jsonl")
        normalize_jsonl(extracted, normalized_path, endpoint=endpoint)

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
        if artifact.mode == "reuse":
            artifact.extra.setdefault("parsed_at", _utc_now_iso())
            return artifact

        if dry_run:
            artifact.output_file_id = artifact.output_file_id or f"dry-output-openai-{artifact.condition}-t{_format_temp_label(artifact.temp)}"
            artifact.extra["parsed_at"] = _utc_now_iso()
            return artifact

        combined_path = self._combine_results(results_dir)
        if not combined_path:
            raise RuntimeError("No normalized OpenAI results found to parse.")
        predictions_csv = os.path.join(results_dir, "predictions.csv")
        process_results(combined_path, predictions_csv)
        artifact.extra["predictions_csv"] = predictions_csv
        artifact.extra["parsed_at"] = _utc_now_iso()
        return artifact
