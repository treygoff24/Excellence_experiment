from __future__ import annotations

import argparse
import glob
import os
import shutil
import sys
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Iterable, Optional, Protocol

import json
import yaml

from dotenv import load_dotenv

from fireworks.parse_results import process_results

from backends.anthropic import AnthropicBatchAdapter
from backends.openai import OpenAIBatchAdapter

from config.schema import load_config
from scripts import manifest_v2 as mf
from scripts.run_all import (  # type: ignore[attr-defined]
    _compute_control_key,
    _expand_trials,
    _format_temp_label,
    _load_build_manifest,
    _lookup_shard_meta,
    _split_list_arg,
    _trial_slug,
    _prompt_suffix,
    ensure_dirs,
    run_cmd,
    token_len,
    write_manifest,
)
from scripts.shared_controls import (
    refresh_registry,
    shared_rel_default,
    write_control_registry,
)
from scripts.state_utils import (
    RunStateLock,
    StopToken,
    compute_config_hash,
    init_run_state,
    load_run_state,
    run_state_path,
    update_phase,
    write_json_atomic,
)


def _utc_now_iso() -> str:
    return datetime.utcnow().isoformat() + "Z"


def _sha256_file(path: str) -> str:
    import hashlib

    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def _condition_shards(
    build_manifest: dict[str, Any] | None,
    *,
    prompt_set: str,
    temp: float,
    condition: str,
) -> list[tuple[str, dict[str, Any]]]:
    if not build_manifest:
        return []
    shards_obj = build_manifest.get("shards")
    if not isinstance(shards_obj, dict):
        return []
    shards: dict[str, dict[str, Any]] = {}
    for name, meta in shards_obj.items():
        if isinstance(meta, dict):
            shards[str(name)] = meta
    matches: list[tuple[int, str, dict[str, Any]]] = []
    for name, meta in shards.items():
        if meta.get("prompt_set") != prompt_set:
            continue
        if meta.get("condition") != condition:
            continue
        temp_val = meta.get("temp")
        if temp_val is None:
            continue
        try:
            meta_temp = float(temp_val)
        except (TypeError, ValueError):
            continue
        if abs(meta_temp - float(temp)) > 1e-6:
            continue
        try:
            part_index = int(meta.get("part_index") or 0)
        except (TypeError, ValueError):
            part_index = 0
        matches.append((part_index, name, meta))
    matches.sort(key=lambda item: (item[0], item[1]))
    return [(name, meta) for _, name, meta in matches]


@dataclass
class ProviderArtifacts:
    condition: str
    temp: float
    mode: str = "producer"
    batch_id: Optional[str] = None
    results_uri: Optional[str] = None
    output_file_id: Optional[str] = None
    extra: dict[str, Any] = field(default_factory=dict)


class BaseAdapter(Protocol):
    backend: str

    def submit(
        self,
        *,
        trial_slug: str,
        artifacts: "ProviderArtifacts",
        dry_run: bool,
    ) -> "ProviderArtifacts":
        ...

    def poll(
        self,
        *,
        results_dir: str,
        artifact: "ProviderArtifacts",
        dry_run: bool,
    ) -> "ProviderArtifacts":
        ...

    def parse(
        self,
        *,
        results_dir: str,
        artifact: "ProviderArtifacts",
        dry_run: bool,
    ) -> "ProviderArtifacts":
        ...


class DryRunAdapter(BaseAdapter):
    def __init__(self, backend: str):
        self.backend = backend

    def submit(
        self,
        *,
        trial_slug: str,
        artifacts: ProviderArtifacts,
        dry_run: bool,
    ) -> ProviderArtifacts:
        label = _format_temp_label(artifacts.temp)
        if not dry_run:
            raise NotImplementedError(
                f"{self.backend} adapter submit is not implemented yet. "
                "Use --dry_run or complete provider adapter integration."
            )
        if artifacts.mode == "reuse":
            artifacts.batch_id = f"reuse-{self.backend}-{trial_slug}-{artifacts.condition}-t{label}"
        else:
            artifacts.batch_id = f"dry-{self.backend}-{trial_slug}-{artifacts.condition}-t{label}"
        artifacts.extra["submitted_at"] = _utc_now_iso()
        return artifacts

    def poll(
        self,
        *,
        results_dir: str,
        artifact: ProviderArtifacts,
        dry_run: bool,
    ) -> ProviderArtifacts:
        label = _format_temp_label(artifact.temp)
        if not dry_run:
            raise NotImplementedError(
                f"{self.backend} adapter poll is not implemented yet. "
                "Use --dry_run or complete provider adapter integration."
            )
        placeholder = os.path.join(results_dir, f"dry_{artifact.condition}_t{label}.jsonl")
        os.makedirs(os.path.dirname(placeholder), exist_ok=True)
        if not os.path.isfile(placeholder):
            with open(placeholder, "w", encoding="utf-8") as f:
                f.write(
                    json.dumps(
                        {
                            "custom_id": f"{artifact.condition}|dry|{label}|0",
                            "response": {
                                "body": {
                                    "choices": [
                                        {
                                            "message": {"content": "dry-run placeholder"},
                                            "finish_reason": "stop",
                                        }
                                    ]
                                },
                                "usage": {"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2},
                            },
                        }
                    )
                    + "\n"
                )
        artifact.results_uri = f"file://{placeholder}"
        artifact.extra["poll_completed_at"] = _utc_now_iso()
        return artifact

    def parse(
        self,
        *,
        results_dir: str,
        artifact: ProviderArtifacts,
        dry_run: bool,
    ) -> ProviderArtifacts:
        if not dry_run:
            raise NotImplementedError(
                f"{self.backend} adapter parse is not implemented yet. "
                "Use --dry_run or complete provider adapter integration."
            )
        artifact.output_file_id = f"dry-output-{self.backend}-{artifact.condition}-{_format_temp_label(artifact.temp)}"
        artifact.extra["parsed_at"] = _utc_now_iso()
        return artifact


class ReplayAdapter(BaseAdapter):
    def __init__(self, backend: str, *, cfg: dict[str, Any], replay_dir: str) -> None:
        self.backend = backend
        self.cfg = cfg
        self.replay_dir = os.path.abspath(replay_dir)
        self.results_dir = os.path.join(self.replay_dir, "results")
        self.metadata = self._load_metadata()
        self.failure_mode = (self.metadata.get("failure_mode") or "none").lower()
        self.expected_records = self.metadata.get("expected_records")
        self.expected_predictions = self._resolve_metadata_path(self.metadata.get("expected_predictions"))
        self.expected_results = self._resolve_metadata_path(self.metadata.get("expected_results"))

    def _resolve_metadata_path(self, rel: Any) -> str | None:
        if not rel:
            return None
        path = os.path.join(self.replay_dir, str(rel))
        return path if os.path.exists(path) else None

    def _load_metadata(self) -> dict[str, Any]:
        meta_path = os.path.join(self.replay_dir, "metadata.json")
        if not os.path.isfile(meta_path):
            return {}
        try:
            with open(meta_path, "r", encoding="utf-8") as f:
                data = json.load(f)
            if isinstance(data, dict):
                return data
        except Exception:
            pass
        return {}

    def _condition_filename(self, *, condition: str, temp: float) -> str:
        label = _format_temp_label(temp)
        return os.path.join(self.results_dir, f"{condition}_t{label}.jsonl")

    def _combine_results(self, results_dir: str) -> str:
        dest = os.path.join(results_dir, "results.jsonl")
        if self.expected_results and os.path.isfile(self.expected_results):
            shutil.copyfile(self.expected_results, dest)
            return dest
        pattern = os.path.join(results_dir, "*_results.jsonl")
        matches = sorted(glob.glob(pattern))
        if not matches:
            raise RuntimeError("Replay simulation: no per-condition result files found to combine.")
        with open(dest, "w", encoding="utf-8") as fout:
            for path in matches:
                with open(path, "r", encoding="utf-8") as fin:
                    for line in fin:
                        s = line.rstrip()
                        if s:
                            fout.write(s + "\n")
        return dest

    @staticmethod
    def _count_records(path: str) -> int:
        count = 0
        try:
            with open(path, "r", encoding="utf-8") as f:
                for line in f:
                    if line.strip():
                        count += 1
        except FileNotFoundError:
            return 0
        return count

    def submit(
        self,
        *,
        trial_slug: str,
        artifacts: ProviderArtifacts,
        dry_run: bool,
    ) -> ProviderArtifacts:
        label = _format_temp_label(artifacts.temp)
        artifacts.batch_id = artifacts.batch_id or f"replay-{self.backend}-{trial_slug}-{artifacts.condition}-t{label}"
        artifacts.extra.setdefault("replay", True)
        artifacts.extra.setdefault("submitted_at", _utc_now_iso())
        return artifacts

    def poll(
        self,
        *,
        results_dir: str,
        artifact: ProviderArtifacts,
        dry_run: bool,
    ) -> ProviderArtifacts:
        if self.failure_mode == "expired":
            raise RuntimeError(
                f"Replay simulation: batch for {artifact.condition} temp {artifact.temp} marked as expired."
            )

        source = self._condition_filename(condition=artifact.condition, temp=artifact.temp)
        if not os.path.isfile(source):
            raise FileNotFoundError(
                f"Replay fixture missing result JSONL for {artifact.condition} at temp {artifact.temp}: {source}"
            )
        os.makedirs(results_dir, exist_ok=True)
        dest = os.path.join(results_dir, f"{artifact.condition}_t{_format_temp_label(artifact.temp)}_results.jsonl")
        shutil.copyfile(source, dest)
        artifact.results_uri = dest
        artifact.extra["normalized_path"] = dest
        artifact.extra["poll_completed_at"] = _utc_now_iso()
        return artifact

    def parse(
        self,
        *,
        results_dir: str,
        artifact: ProviderArtifacts,
        dry_run: bool,
    ) -> ProviderArtifacts:
        if self.failure_mode == "validation":
            raise ValueError(
                f"Replay simulation: validation error for {artifact.condition} temp {artifact.temp}."
            )

        os.makedirs(results_dir, exist_ok=True)
        combined_path = self._combine_results(results_dir)
        actual_records = self._count_records(combined_path)
        if self.failure_mode == "partial":
            expected = self.expected_records
            if expected is None:
                raise RuntimeError("Replay simulation: partial failure requested but expected_records missing in metadata.json")
            raise RuntimeError(
                f"Replay simulation: expected {expected} records but found {actual_records} in combined results."
            )
        if self.expected_records is not None and actual_records != self.expected_records:
            raise RuntimeError(
                f"Replay simulation: record count mismatch (expected {self.expected_records}, got {actual_records})."
            )

        predictions_path = os.path.join(results_dir, "predictions.csv")
        process_results(combined_path, predictions_path)
        artifact.extra["predictions_csv"] = predictions_path
        artifact.extra["parsed_at"] = _utc_now_iso()
        return artifact


def resolve_adapter(name: str, *, cfg: dict[str, Any], replay_dir: str | None = None) -> BaseAdapter:
    normalized = (name or "").strip().lower()
    if replay_dir:
        if normalized not in {"openai", "anthropic"}:
            raise SystemExit("Replay mode only supports 'openai' or 'anthropic' backends.")
        return ReplayAdapter(normalized, cfg=cfg, replay_dir=replay_dir)
    if normalized == "openai":
        return OpenAIBatchAdapter(cfg)
    if normalized == "anthropic":
        return AnthropicBatchAdapter(cfg)
    raise SystemExit(f"Unsupported backend '{name}'. Choose from 'openai' or 'anthropic'.")


def _collect_conditions(selected: str) -> list[str]:
    if selected in {"control", "treatment"}:
        return [selected]
    return ["control", "treatment"]


def _serialize_artifacts(entries: Iterable[ProviderArtifacts]) -> list[dict[str, Any]]:
    serialized: list[dict[str, Any]] = []
    for art in entries:
        record = {
            "condition": art.condition,
            "temp": art.temp,
            "mode": art.mode,
            "batch_id": art.batch_id,
            "results_uri": art.results_uri,
            "output_file_id": art.output_file_id,
        }
        if art.extra:
            record["extra"] = dict(art.extra)
        serialized.append(record)
    return serialized


def _update_stage(
    *,
    manifest: dict[str, Any],
    manifest_path: str,
    stage: str,
    status: str,
    provider_entries: Iterable[ProviderArtifacts],
    backend: str,
) -> None:
    stage_map = manifest.setdefault("stage_status", {})
    stage_map[stage] = {
        "status": status,
        "updated_at": _utc_now_iso(),
        "artifacts": {
            "backend": backend,
            "provider_runs": _serialize_artifacts(provider_entries),
        },
    }
    mf.write_manifest(manifest_path, manifest)


def _load_or_create_manifest(
    *,
    manifest_path: str,
    run_id: str,
    trial: dict[str, Any],
    prompt_paths: dict[str, str],
) -> dict[str, Any]:
    if os.path.isfile(manifest_path):
        try:
            data, upgraded = mf.load_manifest(manifest_path)
            if upgraded:
                print(f"Upgraded manifest at {manifest_path} to schema v{mf.SCHEMA_VERSION}")
            return data
        except Exception as exc:  # pragma: no cover - fallback path
            print(f"WARNING: failed to load existing manifest {manifest_path}: {exc}")
    ctrl_prompt = open(prompt_paths["control"], "r", encoding="utf-8").read()
    trt_prompt = open(prompt_paths["treatment"], "r", encoding="utf-8").read()
    manifest = {
        "schema_version": mf.SCHEMA_VERSION,
        "timestamps": {"created_at": _utc_now_iso(), "updated_at": _utc_now_iso()},
        "run_id": run_id,
        "trial": {
            "model_id": trial["model_id"],
            "prompt_set": trial["prompt_set"],
            "top_p": trial.get("top_p"),
            "top_k": trial.get("top_k"),
            "max_new_tokens": trial.get("max_new_tokens"),
            "id": trial.get("id"),
        },
        "temps": [float(t) for t in (trial.get("temps") or [])],
        "samples_per_item": trial.get("samples_per_item"),
        "prompts": {
            "control": {"sha256": _sha256_file(prompt_paths["control"]), "tokens": token_len(ctrl_prompt)},
            "treatment": {"sha256": _sha256_file(prompt_paths["treatment"]), "tokens": token_len(trt_prompt)},
        },
        "datasets": {},
        "jobs": {},
        "job_status": {},
        "control_registry": {},
        "stage_status": {},
    }
    mf.write_manifest(manifest_path, manifest)
    return manifest


def _ensure_control_entry(
    *,
    run_root: Optional[str],
    control_registry: Optional[dict[str, Any]],
    manifest: dict[str, Any],
    label: str,
    ctrl_key: str,
    slug: str,
    prompt_sha: str,
    input_sha: str,
    backend_tag: str,
    model_id: str,
    temp: float,
) -> tuple[dict[str, Any], bool]:
    mutated = False
    entry = manifest.setdefault("control_registry", {}).setdefault(label, {})
    shared_rel = entry.get("shared_rel") or shared_rel_default(ctrl_key)
    entry.update({
        "key": ctrl_key,
        "temp": float(temp),
        "shared_rel": shared_rel,
    })
    reg_entry = None
    if control_registry is not None:
        reg_controls = control_registry.setdefault("controls", {})
        candidate = reg_controls.get(ctrl_key)
        if candidate and candidate.get("status") == "completed":
            from scripts.run_all import _control_entry_files_exist  # local import to avoid cycle

            if _control_entry_files_exist(run_root or "", candidate):
                reg_entry = candidate
    if reg_entry:
        entry.update({
            "mode": "reuse",
            "status": reg_entry.get("status", "completed"),
            "producer_trial": reg_entry.get("producer_trial"),
            "files": reg_entry.get("files", {}),
            "counts": reg_entry.get("counts", {}),
        })
    else:
        entry.update({
            "mode": "producer",
            "status": "pending",
            "producer_trial": slug,
        })
        if control_registry is not None:
            reg_controls = control_registry.setdefault("controls", {})
            existing = reg_controls.get(ctrl_key)
            previous = json.dumps(existing, sort_keys=True) if existing is not None else None
            info = dict(existing or {})
            info["status"] = info.get("status") or "pending"
            info["shared_rel"] = info.get("shared_rel") or shared_rel
            info["created_at"] = info.get("created_at") or _utc_now_iso()
            if not info.get("producer_trial"):
                info["producer_trial"] = slug
            if prompt_sha and not info.get("prompt_sha"):
                info["prompt_sha"] = prompt_sha
            if input_sha and not info.get("input_sha"):
                info["input_sha"] = input_sha
            info["backend"] = info.get("backend") or backend_tag
            info["model_id"] = info.get("model_id") or model_id
            info["temp"] = info.get("temp") if info.get("temp") is not None else float(temp)
            reg_controls[ctrl_key] = info
            if json.dumps(info, sort_keys=True) != previous:
                mutated = True
    return entry, mutated


@dataclass
class TrialRun:
    slug: str
    trial: dict[str, Any]
    results_dir: str
    reports_dir: str
    manifest_path: str
    manifest: dict[str, Any]
    artifacts: list[ProviderArtifacts] = field(default_factory=list)


def _write_multi_manifest(run_root: str, run_id: str, trials: list[TrialRun]) -> None:
    summary = {
        "schema_version": 2,
        "created_utc": _utc_now_iso(),
        "updated_utc": _utc_now_iso(),
        "run_id": run_id,
        "num_trials": len(trials),
        "trials": [],
    }
    for tr in trials:
        rel_results = os.path.relpath(tr.results_dir, run_root) if run_root else tr.results_dir
        rel_reports = os.path.relpath(tr.reports_dir, run_root) if run_root else tr.reports_dir
        summary["trials"].append({
            "slug": tr.slug,
            "model_id": tr.trial["model_id"],
            "prompt_set": tr.trial.get("prompt_set"),
            "temps": tr.trial.get("temps"),
            "results_dir": rel_results,
            "reports_dir": rel_reports,
        })
    manifest_path = os.path.join(run_root, "multi_trial_manifest.json")
    write_manifest(manifest_path, summary)


def main() -> None:
    load_dotenv()
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="config/eval_config.yaml")
    ap.add_argument("--backend", choices=["openai", "anthropic"])
    ap.add_argument("--account_id", default=None)
    ap.add_argument("--condition", choices=["control", "treatment", "both"], default="both")
    ap.add_argument("--skip_prepare", action="store_true")
    ap.add_argument("--skip_build", action="store_true")
    ap.add_argument("--skip_score", action="store_true")
    ap.add_argument("--skip_stats", action="store_true")
    ap.add_argument("--skip_costs", action="store_true")
    ap.add_argument("--skip_report", action="store_true")
    ap.add_argument("--run_id", help="Custom run ID (auto-generated if not provided)")
    ap.add_argument("--models", nargs="+")
    ap.add_argument("--prompt_sets", nargs="+")
    ap.add_argument("--temps", nargs="+")
    ap.add_argument("--no_sweep", action="store_true")
    ap.add_argument("--plan_only", action="store_true")
    ap.add_argument(
        "--only_step",
        choices=["prepare", "build", "submit", "poll", "parse", "score", "stats", "costs", "report"],
    )
    ap.add_argument(
        "--from_step",
        choices=["prepare", "build", "submit", "poll", "parse", "score", "stats", "costs", "report"],
    )
    ap.add_argument(
        "--to_step",
        choices=["prepare", "build", "submit", "poll", "parse", "score", "stats", "costs", "report"],
    )
    ap.add_argument("--archive", action="store_true")
    ap.add_argument("--ignore_stop", action="store_true")
    ap.add_argument("--stop_stale_minutes", type=int, default=60)
    ap.add_argument("--max_concurrent_jobs", type=int, default=4)
    ap.add_argument("--parts_per_dataset", type=int, default=None)
    ap.add_argument("--lines_per_part", type=int, default=None)
    ap.add_argument("--limit_items", type=int, default=None)
    ap.add_argument("--resume", action="store_true")
    ap.add_argument("--dry_run", action="store_true")
    ap.add_argument("--replay_dir", default=None)
    args = ap.parse_args()

    cfg = load_config(args.config)
    backend_name = args.backend or (cfg.get("provider", {}) or {}).get("name") or "openai"
    if args.replay_dir and args.dry_run:
        print("NOTE: --replay_dir specified; ignoring --dry_run to allow parse and downstream checks.")
        args.dry_run = False
    adapter = resolve_adapter(backend_name, cfg=cfg, replay_dir=args.replay_dir)

    if args.models:
        cfg["models"] = _split_list_arg(args.models)
    if args.prompt_sets:
        cfg["prompt_sets_run"] = _split_list_arg(args.prompt_sets)
    if args.temps:
        cfg["temps_override"] = [float(t) for t in _split_list_arg(args.temps)]
    if args.no_sweep:
        cfg["sweep"] = None

    ensure_dirs(cfg)
    run_id = args.run_id or datetime.utcnow().strftime("r%Y%m%d%H%M%S")
    trials = _expand_trials(cfg, args)
    experiments_dir = (cfg.get("paths", {}) or {}).get("experiments_dir") or ""
    use_exp_root = bool(experiments_dir)
    if use_exp_root:
        os.makedirs(experiments_dir, exist_ok=True)
    run_root = os.path.join(experiments_dir, f"run_{run_id}") if use_exp_root else None
    if run_root:
        os.makedirs(run_root, exist_ok=True)
    control_registry = refresh_registry(run_root) if run_root else None
    control_registry_dirty = False

    effective_config_path = args.config
    if run_root:
        effective_config_path = os.path.join(run_root, "effective_config.yaml")
        with open(effective_config_path, "w", encoding="utf-8") as f:
            yaml.safe_dump(cfg, f, sort_keys=False)

    stop_token = StopToken(
        run_root or os.getcwd(),
        ignore_file=bool(args.ignore_stop),
        stale_minutes=args.stop_stale_minutes,
    )
    state = None
    if run_root:
        state = load_run_state(run_root)
        if args.resume and state is None:
            raise SystemExit(f"--resume requested but run_state.json missing at {run_state_path(run_root)}")
        if state is None:
            state = init_run_state(run_root, run_id, cfg)
            with RunStateLock(run_root):
                write_json_atomic(run_state_path(run_root), state)
        else:
            try:
                if compute_config_hash(cfg) != state.get("config_hash"):
                    print("WARNING: Effective config differs from stored config hash. Resume may be inconsistent.")
            except Exception:
                pass

    PHASES = ["prepare", "build", "submit", "poll", "parse", "score", "stats", "costs", "report"]

    def _select_phases() -> list[str]:
        if args.only_step:
            return [args.only_step]
        if args.from_step and args.to_step:
            si = PHASES.index(args.from_step)
            ei = PHASES.index(args.to_step)
            if si > ei:
                raise SystemExit("--from_step must be <= --to_step")
            return PHASES[si:ei + 1]
        if args.from_step:
            return PHASES[PHASES.index(args.from_step):]
        if args.to_step:
            return PHASES[: PHASES.index(args.to_step) + 1]
        return PHASES[:]

    selected_phases = _select_phases()
    if args.skip_prepare:
        selected_phases = [p for p in selected_phases if p != "prepare"]
    if args.skip_build:
        selected_phases = [p for p in selected_phases if p != "build"]
    if args.skip_score:
        selected_phases = [p for p in selected_phases if p != "score"]
    if args.skip_stats:
        selected_phases = [p for p in selected_phases if p != "stats"]
    if args.skip_costs:
        selected_phases = [p for p in selected_phases if p != "costs"]
    if args.skip_report:
        selected_phases = [p for p in selected_phases if p != "report"]
    if args.dry_run:
        selected_phases = [p for p in selected_phases if p not in {"parse", "score", "stats", "costs", "report"}]

    if args.plan_only:
        print(f"Run ID: {run_id}")
        print(f"Backend: {adapter.backend}")
        print(f"Selected phases: {', '.join(selected_phases) or 'none'}")
        print(f"Trials: {len(trials)} configured")
        return

    prompt_sets_cfg = cfg.get("prompt_sets") or {}
    default_ps = cfg.get("default_prompt_set") or (
        sorted(prompt_sets_cfg.keys())[0] if prompt_sets_cfg else "default"
    )
    conditions = _collect_conditions(args.condition)

    def mark_state(phase: str, status: str) -> None:
        if state is None or run_root is None:
            return
        with RunStateLock(run_root):
            update_phase(state, phase, status=status)
            write_json_atomic(run_state_path(run_root), state)

    if "prepare" in selected_phases:
        mark_state("prepare", "in_progress")
        stop_token.check()
        cmd = [sys.executable, "-m", "scripts.prepare_data", "--config", effective_config_path]
        if args.resume:
            cmd.append("--resume")
        run_cmd(cmd)
        mark_state("prepare", "completed")
    else:
        print("Gating: skipping prepare")

    if "build" in selected_phases:
        mark_state("build", "in_progress")
        stop_token.check()
        temps_per_ps: dict[str, set[float]] = {}
        for trial in trials:
            ps_name = trial.get("prompt_set") or default_ps
            temps_per_ps.setdefault(ps_name, set()).update(float(t) for t in (trial.get("temps") or []))
        for ps_name, temps in temps_per_ps.items():
            temp_arg = ",".join(str(t) for t in sorted(temps))
            cmd = [
                sys.executable,
                "-m",
                "scripts.build_batches",
                "--config",
                effective_config_path,
                "--prompt_set",
                ps_name,
            ]
            if temp_arg:
                cmd.extend(["--temps", temp_arg])
            if args.resume:
                cmd.append("--resume")
            if args.limit_items:
                cmd.extend(["--limit_items", str(args.limit_items)])
            run_cmd(cmd)
        mark_state("build", "completed")
    else:
        print("Gating: skipping build")

    trial_runs: list[TrialRun] = []
    build_manifest = _load_build_manifest(cfg["paths"]["batch_inputs_dir"])
    backend_tag = adapter.backend

    for trial in trials:
        model_id = trial["model_id"]
        ps_name = trial.get("prompt_set") or default_ps
        temps = trial.get("temps") or (cfg.get("temps") or [0.0])
        top_p = trial.get("top_p", cfg.get("top_p"))
        top_k = trial.get("top_k", cfg.get("top_k"))
        mx = trial.get("max_new_tokens")
        if not mx:
            mx = cfg.get("max_new_tokens") or {"closed_book": 1024, "open_book": 1024}
        slug = _trial_slug(model_id, ps_name, top_p, top_k, mx, aliases=cfg.get("model_aliases") or {})
        if run_root:
            trial_root = os.path.join(run_root, slug)
            results_dir = os.path.join(trial_root, "results")
            reports_dir = os.path.join(trial_root, "reports")
        else:
            results_dir = cfg["paths"]["results_dir"]
            reports_dir = cfg["paths"]["reports_dir"]
        os.makedirs(results_dir, exist_ok=True)
        os.makedirs(reports_dir, exist_ok=True)

        prompt_meta = prompt_sets_cfg.get(ps_name) or {}
        ctrl_path = os.path.expanduser(prompt_meta.get("control", "config/prompts/control_system.txt"))
        trt_path = os.path.expanduser(prompt_meta.get("treatment", "config/prompts/treatment_system.txt"))
        manifest_path = os.path.join(results_dir, "trial_manifest.json")
        manifest = _load_or_create_manifest(
            manifest_path=manifest_path,
            run_id=run_id,
            trial={
                "model_id": model_id,
                "prompt_set": ps_name,
                "top_p": top_p,
                "top_k": top_k,
                "max_new_tokens": mx,
                "temps": temps,
                "samples_per_item": cfg.get("samples_per_item"),
                "id": slug,
            },
            prompt_paths={"control": ctrl_path, "treatment": trt_path},
        )
        trial_runs.append(
            TrialRun(
                slug=slug,
                trial={
                    "model_id": model_id,
                    "prompt_set": ps_name,
                    "temps": temps,
                    "top_p": top_p,
                    "top_k": top_k,
                    "max_new_tokens": mx,
                },
                results_dir=results_dir,
                reports_dir=reports_dir,
                manifest_path=manifest_path,
                manifest=manifest,
            )
        )

    if "submit" in selected_phases:
        mark_state("submit", "in_progress")
        for tr in trial_runs:
            stop_token.check()
            manifest = tr.manifest
            artifacts: list[ProviderArtifacts] = []
            if run_root:
                batch_inputs_root = os.path.join(run_root, "batch_inputs")
            else:
                batch_inputs_root = cfg["paths"]["batch_inputs_dir"]
            provider_batch_dir = os.path.join(batch_inputs_root, adapter.backend)
            os.makedirs(provider_batch_dir, exist_ok=True)
            for temp in tr.trial.get("temps") or []:
                t_label = _format_temp_label(float(temp))
                suffix = _prompt_suffix(tr.trial["prompt_set"], prompt_sets_cfg)
                for cond in conditions:
                    shard_entries = _condition_shards(
                        build_manifest,
                        prompt_set=tr.trial["prompt_set"],
                        temp=float(temp),
                        condition=cond,
                    )
                    if not shard_entries:
                        cond_fname = f"t{t_label}{suffix}_{cond}.jsonl"
                        shard_entries = [
                            (
                                os.path.basename(cond_fname),
                                _lookup_shard_meta(build_manifest, os.path.basename(cond_fname)) or {},
                            )
                        ]
                    for shard_idx, (cond_fname, shard_meta) in enumerate(shard_entries):
                        cond_source = os.path.join(cfg["paths"]["batch_inputs_dir"], cond_fname)
                        try:
                            part_index = int((shard_meta or {}).get("part_index") or 0)
                        except (TypeError, ValueError):
                            part_index = shard_idx if len(shard_entries) > 1 else 0
                        part_suffix = "" if part_index <= 0 else f"_p{part_index + 1:02d}"

                        entry = None
                        mode = "producer"
                        if cond == "control":
                            shard_sha = (shard_meta or {}).get("sha256") or (
                                _sha256_file(cond_source) if os.path.isfile(cond_source) else ""
                            )
                            ctrl_key = _compute_control_key(
                                run_id=run_id,
                                backend=backend_tag,
                                model_id=tr.trial["model_id"],
                                temp=float(temp),
                                top_p=tr.trial.get("top_p"),
                                top_k=tr.trial.get("top_k"),
                                max_new_tokens=tr.trial.get("max_new_tokens"),
                                samples_per_item=cfg.get("samples_per_item") or {},
                                control_prompt_sha=manifest.get("prompts", {}).get("control", {}).get("sha256"),
                                input_sha=shard_sha,
                            )
                            label = f"t{t_label}{part_suffix}"
                            entry, mutated = _ensure_control_entry(
                                run_root=run_root,
                                control_registry=control_registry,
                                manifest=manifest,
                                label=label,
                                ctrl_key=ctrl_key,
                                slug=tr.slug,
                                prompt_sha=manifest.get("prompts", {}).get("control", {}).get("sha256"),
                                input_sha=shard_sha,
                                backend_tag=backend_tag,
                                model_id=tr.trial["model_id"],
                                temp=float(temp),
                            )
                            if mutated:
                                control_registry_dirty = True
                            manifest["control_registry"][label] = entry
                            mf.write_manifest(tr.manifest_path, manifest)
                            if entry.get("mode") == "reuse":
                                mode = "reuse"

                        art = ProviderArtifacts(condition=cond, temp=float(temp), mode=mode)
                        top_p = tr.trial.get("top_p")
                        if top_p is None:
                            top_p = cfg.get("top_p")
                        top_k = tr.trial.get("top_k")
                        if top_k is None:
                            top_k = cfg.get("top_k")
                        art.extra.update({
                            "model_id": tr.trial["model_id"],
                            "top_p": top_p,
                            "top_k": top_k,
                            "max_new_tokens": tr.trial.get("max_new_tokens") or cfg.get("max_new_tokens"),
                            "source_jsonl": cond_source,
                            "provider_batch_dir": provider_batch_dir,
                            "prompt_set": tr.trial.get("prompt_set"),
                            "run_root": run_root,
                            "shard_name": cond_fname,
                            "part_index": part_index,
                            "part_suffix": part_suffix,
                        })
                        submitted_artifact = adapter.submit(trial_slug=tr.slug, artifacts=art, dry_run=args.dry_run)
                        artifacts.append(submitted_artifact)
                        cache_metrics = getattr(submitted_artifact, "extra", {}).get("cache_metrics") if submitted_artifact else None
                        if cache_metrics and cache_metrics.get("cache_read_input_tokens") is not None:
                            read_tokens = cache_metrics.get("cache_read_input_tokens")
                            created_tokens = cache_metrics.get("cache_creation_input_tokens")
                            batch_reference = submitted_artifact.batch_id or "pending"
                            print(
                                f"[cache] {adapter.backend} batch {batch_reference}: cache_read_input_tokens={read_tokens}, "
                                f"cache_creation_input_tokens={created_tokens}"
                            )
            tr.artifacts = artifacts
            _update_stage(
                manifest=manifest,
                manifest_path=tr.manifest_path,
                stage="submitted",
                status="completed",
                provider_entries=artifacts,
                backend=adapter.backend,
            )
        mark_state("submit", "completed")
    else:
        print("Gating: skipping submit")

    if control_registry_dirty and run_root and control_registry is not None:
        write_control_registry(run_root, control_registry)

    if "poll" in selected_phases:
        mark_state("poll", "in_progress")
        for tr in trial_runs:
            stop_token.check()
            updated: list[ProviderArtifacts] = []
            for art in tr.artifacts:
                updated.append(adapter.poll(results_dir=tr.results_dir, artifact=art, dry_run=args.dry_run))
            tr.artifacts = updated
            _update_stage(
                manifest=tr.manifest,
                manifest_path=tr.manifest_path,
                stage="downloaded",
                status="completed",
                provider_entries=tr.artifacts,
                backend=adapter.backend,
            )
        mark_state("poll", "completed")
    else:
        print("Gating: skipping poll")

    if "parse" in selected_phases:
        if not args.dry_run and isinstance(adapter, DryRunAdapter):
            raise NotImplementedError(
                "Parse phase for alternative backends requires provider adapters (tickets 202/203). "
                "Run with --dry_run for now."
            )
        for tr in trial_runs:
            stop_token.check()
            parsed = [
                adapter.parse(results_dir=tr.results_dir, artifact=art, dry_run=args.dry_run) for art in tr.artifacts
            ]
            tr.artifacts = parsed
            _update_stage(
                manifest=tr.manifest,
                manifest_path=tr.manifest_path,
                stage="parsed",
                status="completed",
                provider_entries=parsed,
                backend=adapter.backend,
            )
    else:
        print("Gating: skipping parse")

    def _results_file(tr: TrialRun, name: str) -> str:
        return os.path.join(tr.results_dir, name)

    def _reports_file(tr: TrialRun, name: str) -> str:
        return os.path.join(tr.reports_dir, name)

    def _needs_refresh(output_path: str, inputs: Iterable[str]) -> bool:
        if not output_path or not os.path.isfile(output_path):
            return True
        try:
            out_mtime = os.path.getmtime(output_path)
        except OSError:
            return True
        for dep in inputs:
            if not dep:
                continue
            if not os.path.isfile(dep):
                continue
            try:
                if os.path.getmtime(dep) > out_mtime:
                    return True
            except OSError:
                return True
        return False

    # Score phase
    if "score" in selected_phases:
        prepared_dir = (cfg.get("paths") or {}).get("prepared_dir")
        if not prepared_dir or not os.path.isdir(prepared_dir):
            print(f"NOTE: skipping score phase because prepared_dir is unavailable ({prepared_dir!r}).")
        else:
            score_targets: list[TrialRun] = []
            for tr in trial_runs:
                preds_path = _results_file(tr, "predictions.csv")
                if not os.path.isfile(preds_path):
                    raise FileNotFoundError(f"Expected predictions.csv before scoring: {preds_path}")
                per_item_path = _results_file(tr, "per_item_scores.csv")
                if _needs_refresh(per_item_path, [preds_path, effective_config_path]):
                    score_targets.append(tr)
            if score_targets:
                mark_state("score", "in_progress")
                for tr in score_targets:
                    preds_path = _results_file(tr, "predictions.csv")
                    per_item_path = _results_file(tr, "per_item_scores.csv")
                    try:
                        run_cmd(
                            [
                                sys.executable,
                                "-m",
                                "scoring.score_predictions",
                                "--pred_csv",
                                preds_path,
                                "--prepared_dir",
                                prepared_dir,
                                "--out_dir",
                                tr.results_dir,
                                "--config",
                                effective_config_path,
                            ]
                        )
                    except Exception:
                        print(
                            f"WARNING: scoring failed for {tr.slug}; downstream stats/costs/report will be skipped for this trial."
                        )
                        continue
                    if os.path.isfile(per_item_path):
                        tr.manifest.setdefault("stage_status", {})["score"] = {
                            "status": "completed",
                            "updated_at": _utc_now_iso(),
                            "artifacts": {"per_item_scores_csv": os.path.relpath(per_item_path, tr.results_dir)},
                        }
                        mf.write_manifest(tr.manifest_path, tr.manifest)
                if all(os.path.isfile(_results_file(tr, "per_item_scores.csv")) for tr in trial_runs):
                    mark_state("score", "completed")
            else:
                if all(os.path.isfile(_results_file(tr, "per_item_scores.csv")) for tr in trial_runs):
                    mark_state("score", "completed")
                print("Gating: skipping score (up-to-date)")
    else:
        print("Gating: skipping score")

    # Stats phase
    if "stats" in selected_phases:
        missing_per_item = [tr.slug for tr in trial_runs if not os.path.isfile(_results_file(tr, "per_item_scores.csv"))]
        if missing_per_item:
            print("NOTE: skipping stats phase because per_item_scores.csv is missing for:", ", ".join(missing_per_item))
        else:
            stats_targets: list[TrialRun] = []
            for tr in trial_runs:
                sig_path = _results_file(tr, "significance.json")
                per_item_path = _results_file(tr, "per_item_scores.csv")
                if _needs_refresh(sig_path, [per_item_path, effective_config_path]):
                    stats_targets.append(tr)
            if stats_targets:
                mark_state("stats", "in_progress")
                for tr in stats_targets:
                    per_item_path = _results_file(tr, "per_item_scores.csv")
                    sig_path = _results_file(tr, "significance.json")
                    try:
                        run_cmd(
                            [
                                sys.executable,
                                "-m",
                                "scoring.stats",
                                "--per_item_csv",
                                per_item_path,
                                "--config",
                                effective_config_path,
                                "--out_path",
                                sig_path,
                            ]
                        )
                    except Exception:
                        with open(sig_path, "w", encoding="utf-8") as fh:
                            json.dump({}, fh)
                    tr.manifest.setdefault("stage_status", {})["stats"] = {
                        "status": "completed",
                        "updated_at": _utc_now_iso(),
                        "artifacts": {"significance_json": os.path.relpath(sig_path, tr.results_dir)},
                    }
                    mf.write_manifest(tr.manifest_path, tr.manifest)
                if all(os.path.isfile(_results_file(tr, "significance.json")) for tr in trial_runs):
                    mark_state("stats", "completed")
            else:
                if all(os.path.isfile(_results_file(tr, "significance.json")) for tr in trial_runs):
                    mark_state("stats", "completed")
                print("Gating: skipping stats (up-to-date)")
    else:
        print("Gating: skipping stats")

    # Costs phase
    if "costs" in selected_phases:
        missing_preds = [tr.slug for tr in trial_runs if not os.path.isfile(_results_file(tr, "predictions.csv"))]
        if missing_preds:
            print("NOTE: skipping costs phase because predictions.csv is missing for:", ", ".join(missing_preds))
        else:
            costs_targets: list[TrialRun] = []
            for tr in trial_runs:
                costs_path = _results_file(tr, "costs.json")
                preds_path = _results_file(tr, "predictions.csv")
                if _needs_refresh(costs_path, [preds_path, effective_config_path]):
                    costs_targets.append(tr)
            if costs_targets:
                mark_state("costs", "in_progress")
                for tr in costs_targets:
                    costs_path = _results_file(tr, "costs.json")
                    try:
                        run_cmd(
                            [
                                sys.executable,
                                "-m",
                                "scripts.summarize_costs",
                                "--pred_csv",
                                _results_file(tr, "predictions.csv"),
                                "--config",
                                effective_config_path,
                                "--out_path",
                                costs_path,
                            ]
                        )
                    except Exception:
                        print(f"WARNING: cost summarization failed for {tr.slug}; skipping costs output for this trial.")
                        continue
                    tr.manifest.setdefault("stage_status", {})["costs"] = {
                        "status": "completed",
                        "updated_at": _utc_now_iso(),
                        "artifacts": {"costs_json": os.path.relpath(costs_path, tr.results_dir)},
                    }
                    mf.write_manifest(tr.manifest_path, tr.manifest)
                if all(os.path.isfile(_results_file(tr, "costs.json")) for tr in trial_runs):
                    mark_state("costs", "completed")
            else:
                if all(os.path.isfile(_results_file(tr, "costs.json")) for tr in trial_runs):
                    mark_state("costs", "completed")
                print("Gating: skipping costs (up-to-date)")
    else:
        print("Gating: skipping costs")

    # Report phase
    if "report" in selected_phases:
        missing_report_inputs: list[str] = []
        for tr in trial_runs:
            required = [
                _results_file(tr, "per_item_scores.csv"),
                _results_file(tr, "significance.json"),
                _results_file(tr, "costs.json"),
            ]
            if not all(os.path.isfile(path) for path in required):
                missing_report_inputs.append(tr.slug)
        if missing_report_inputs:
            print("NOTE: skipping report phase because prerequisites are missing for:", ", ".join(missing_report_inputs))
        else:
            report_targets: list[TrialRun] = []
            for tr in trial_runs:
                report_path = _reports_file(tr, "report.md")
                deps = [
                    _results_file(tr, "per_item_scores.csv"),
                    _results_file(tr, "significance.json"),
                    _results_file(tr, "costs.json"),
                    effective_config_path,
                ]
                if _needs_refresh(report_path, deps):
                    report_targets.append(tr)
            if report_targets:
                mark_state("report", "in_progress")
                for tr in report_targets:
                    run_cmd(
                        [
                            sys.executable,
                            "-m",
                            "scripts.generate_report",
                            "--config",
                            effective_config_path,
                            "--results_dir",
                            tr.results_dir,
                            "--reports_dir",
                            tr.reports_dir,
                        ]
                    )
                    tr.manifest, _ = mf.load_manifest(tr.manifest_path)
                if all(os.path.isfile(_reports_file(tr, "report.md")) for tr in trial_runs):
                    mark_state("report", "completed")
            else:
                if all(os.path.isfile(_reports_file(tr, "report.md")) for tr in trial_runs):
                    mark_state("report", "completed")
                print("Gating: skipping report (up-to-date)")
    else:
        print("Gating: skipping report")

    if run_root:
        _write_multi_manifest(run_root, run_id, trial_runs)

    if control_registry_dirty and run_root and control_registry is not None:
        write_control_registry(run_root, control_registry)


if __name__ == "__main__":
    main()
