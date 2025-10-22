from __future__ import annotations

import json
import os
from datetime import datetime, timezone
from typing import Any, Optional

from backends.anthropic.build_requests import build_message_requests, write_requests_preview
from backends.anthropic.normalize_to_openai import normalize_jsonl
from backends.anthropic.poll_and_stream import poll_until_complete, stream_results_to_jsonl
from backends.anthropic.start_message_batch import start_message_batch
from fireworks.parse_results import process_results


def _utc_now_iso() -> str:
    return datetime.now(tz=timezone.utc).isoformat().replace("+00:00", "Z")


def _format_temp_label(temp: float) -> str:
    value = f"{float(temp):.1f}"
    return "0" if value == "0.0" else value.replace(".", "")


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
        self._client = client

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
        return self._client

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
        if artifacts.mode == "reuse":
            artifacts.batch_id = artifacts.batch_id or f"reuse-{self.backend}-{trial_slug}-{artifacts.condition}-t{_format_temp_label(artifacts.temp)}"
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
            f"{trial_slug}_{artifacts.condition}_t{temp_label}_requests.jsonl",
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

        artifacts.extra["request_count"] = len(requests)
        artifacts.extra["request_preview"] = request_preview_path
        artifacts.extra["submitted_at"] = _utc_now_iso()

        if dry_run:
            artifacts.batch_id = f"dry-{self.backend}-{trial_slug}-{artifacts.condition}-t{temp_label}"
            return artifacts

        client = self._ensure_client()
        batch_metadata = self._prepare_batch_metadata(trial_slug, artifacts)
        batch = start_message_batch(
            client,
            requests,
            metadata=batch_metadata,
            batch_params=self.batch_params or None,
        )
        batch_id = getattr(batch, "id", None) or getattr(batch, "batch_id", None) or (batch.get("id") if isinstance(batch, dict) else None)
        if not batch_id:
            raise RuntimeError("Anthropic batch submission did not return a batch id.")
        artifacts.batch_id = batch_id
        artifacts.extra["batch_metadata"] = batch_metadata
        artifacts.extra["batch_status"] = getattr(batch, "processing_status", None) or (
            batch.get("processing_status") if isinstance(batch, dict) else None
        )
        artifacts.extra["request_counts"] = _maybe_model_dump(getattr(batch, "request_counts", None))
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
            placeholder = os.path.join(results_dir, f"{artifact.condition}_t{temp_label}_anthropic_dry.jsonl")
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
        batch = poll_until_complete(client, artifact.batch_id, poll_seconds=self.poll_seconds)
        status = getattr(batch, "processing_status", None) or (
            batch.get("processing_status") if isinstance(batch, dict) else None
        )

        raw_path = os.path.join(results_dir, f"{artifact.condition}_t{temp_label}_anthropic_raw.jsonl")
        normalized_path = os.path.join(results_dir, f"{artifact.condition}_t{temp_label}_results.jsonl")

        streamed = stream_results_to_jsonl(client, artifact.batch_id, raw_path)
        normalized = normalize_jsonl(raw_path, normalized_path)

        artifact.results_uri = normalized_path
        artifact.extra["batch_status"] = status
        artifact.extra["raw_results_path"] = raw_path
        artifact.extra["normalized_path"] = normalized_path
        artifact.extra["streamed_rows"] = streamed
        artifact.extra["normalized_rows"] = normalized
        artifact.extra["request_counts"] = _maybe_model_dump(getattr(batch, "request_counts", None))
        artifact.extra["poll_completed_at"] = _utc_now_iso()
        return artifact

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

        if not artifact.results_uri or not os.path.isfile(artifact.results_uri):
            raise RuntimeError("No Anthropic normalized results available to parse.")
        predictions_csv = os.path.join(results_dir, "predictions.csv")
        process_results(artifact.results_uri, predictions_csv)
        artifact.extra["predictions_csv"] = predictions_csv
        artifact.extra["parsed_at"] = _utc_now_iso()
        return artifact
