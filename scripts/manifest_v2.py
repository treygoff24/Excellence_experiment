from __future__ import annotations

import json
import os
from dataclasses import dataclass
from datetime import datetime
from typing import Any, Dict, Optional, Tuple

from scripts.state_utils import write_json_atomic


SCHEMA_VERSION = 2


def utc_now() -> str:
    return datetime.utcnow().isoformat() + "Z"


def _rel(path: str, base_dir: str) -> str:
    try:
        return os.path.relpath(path, start=base_dir)
    except Exception:
        return path


def _file_size(path: str) -> int:
    try:
        return int(os.path.getsize(path))
    except Exception:
        return 0


def _validate_manifest_v2(data: Dict[str, Any]) -> bool:
    try:
        if not isinstance(data, dict):
            return False
        if int(data.get("schema_version", 0)) != SCHEMA_VERSION:
            return False
        ts = data.get("timestamps", {}) or {}
        if not isinstance(ts, dict):
            return False
        if not ts.get("created_at"):
            return False
        # stage_status is required but may be empty initially
        if not isinstance(data.get("stage_status", {}), dict):
            return False
        # jobs/job_status may be omitted early
        return True
    except Exception:
        return False


def load_manifest(manifest_path: str) -> Tuple[Dict[str, Any], bool]:
    """Load per-trial manifest.

    Returns (manifest_dict, upgraded_flag). If the file is missing or corrupt,
    attempts to repair by inferring stage_status from artifacts.
    """
    results_dir = os.path.dirname(manifest_path)
    try:
        with open(manifest_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        # Auto-upgrade v1/legacy structure
        if int(data.get("schema_version", 0)) == SCHEMA_VERSION and _validate_manifest_v2(data):
            return data, False
        upgraded = False
        # Backup original before upgrade
        try:
            backup_path = f"{manifest_path}.backup.{datetime.utcnow().strftime('%Y%m%d%H%M%S')}.json"
            with open(backup_path, "w", encoding="utf-8") as bf:
                json.dump(data, bf, indent=2)
        except Exception:
            pass
        data_v2 = _upgrade_to_v2(data, results_dir)
        write_manifest(manifest_path, data_v2)
        upgraded = True
        return data_v2, upgraded
    except Exception:
        # Corrupt or unreadable → copy aside and synthesize minimal v2
        try:
            corrupt_path = f"{manifest_path}.corrupt.{datetime.utcnow().strftime('%Y%m%d%H%M%S')}"
            if os.path.isfile(manifest_path):
                try:
                    with open(manifest_path, "rb") as rf, open(corrupt_path, "wb") as wf:
                        wf.write(rf.read())
                except Exception:
                    pass
        except Exception:
            pass
        data_v2 = _synthesize_from_artifacts(results_dir)
        write_manifest(manifest_path, data_v2)
        return data_v2, True


def write_manifest(manifest_path: str, data: Dict[str, Any]) -> None:
    # Ensure timestamps
    ts = data.setdefault("timestamps", {})
    if not ts.get("created_at"):
        ts["created_at"] = utc_now()
    ts["updated_at"] = utc_now()
    data["schema_version"] = SCHEMA_VERSION
    write_json_atomic(manifest_path, data)


def _upgrade_to_v2(data_v1: Dict[str, Any], results_dir: str) -> Dict[str, Any]:
    # Map legacy keys where available
    created = data_v1.get("created_utc") or utc_now()
    out: Dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "timestamps": {"created_at": created, "updated_at": utc_now()},
        "run_id": data_v1.get("run_id"),
        "trial": data_v1.get("trial", {}),
        "temps": data_v1.get("temps", []),
        "samples_per_item": data_v1.get("samples_per_item", {}),
        "prompts": data_v1.get("prompts", {}),
        "jobs": data_v1.get("jobs", {}),
        "job_status": data_v1.get("job_status", {}),
        "stage_status": {},
    }
    # Infer stage statuses from existing artifacts
    st = compute_stage_statuses(results_dir)
    out["stage_status"] = st
    return out


def _synthesize_from_artifacts(results_dir: str) -> Dict[str, Any]:
    # Minimal v2 manifest built from what exists on disk
    st = compute_stage_statuses(results_dir)
    data: Dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "timestamps": {"created_at": utc_now(), "updated_at": utc_now()},
        "run_id": None,
        "trial": {},
        "temps": [],
        "samples_per_item": {},
        "prompts": {},
        "jobs": {},
        "job_status": {},
        "stage_status": st,
    }
    return data


def _read_unique_custom_ids(jsonl_path: str) -> int:
    seen: set[str] = set()
    try:
        with open(jsonl_path, "r", encoding="utf-8") as f:
            for line in f:
                s = line.strip()
                if not s:
                    continue
                try:
                    obj = json.loads(s)
                except Exception:
                    continue
                cid = obj.get("custom_id") or obj.get("customId")
                if cid:
                    seen.add(str(cid))
    except Exception:
        return 0
    return len(seen)


def compute_stage_statuses(results_dir: str) -> Dict[str, Any]:
    """Return stage_status mapping by inspecting files on disk."""
    st: Dict[str, Any] = {}
    now = utc_now()

    def set_stage(name: str, status: str, artifacts: Optional[Dict[str, Any]] = None):
        st[name] = {"status": status, "updated_at": now, "artifacts": artifacts or {}}

    # downloaded
    parts: list[Dict[str, Any]] = []
    part_dirs = []
    try:
        for nm in os.listdir(results_dir):
            p = os.path.join(results_dir, nm)
            if os.path.isdir(p) and nm.endswith(tuple(["_p%02d" % i for i in range(1, 100)])):
                part_dirs.append((nm, p))
    except Exception:
        part_dirs = []
    all_ok = True if part_dirs else False
    for nm, p in sorted(part_dirs):
        rp = os.path.join(p, "results.jsonl")
        sz = _file_size(rp)
        parts.append({"job_key": nm, "path": _rel(rp, results_dir), "size": sz})
        if sz <= 0:
            all_ok = False
    combined_path = os.path.join(results_dir, "results_combined.jsonl")
    combined_rel = _rel(combined_path, results_dir)
    combined_n = _read_unique_custom_ids(combined_path) if os.path.isfile(combined_path) else 0
    set_stage(
        "downloaded",
        "completed" if all_ok and (not part_dirs or combined_n >= 0) else ("in_progress" if parts else "pending"),
        {
            "parts": parts,
            "combined_path": combined_rel if os.path.exists(combined_path) else None,
            "n_results": combined_n,
        },
    )

    # parsed
    pred_path = os.path.join(results_dir, "predictions.csv")
    pred_rel = _rel(pred_path, results_dir)
    pred_rows = 0
    if os.path.isfile(pred_path):
        try:
            import csv
            with open(pred_path, "r", encoding="utf-8") as f:
                r = csv.reader(f)
                pred_rows = max(0, sum(1 for _ in r) - 1)
        except Exception:
            pred_rows = 0
    ok_parsed = bool(os.path.isfile(pred_path) and pred_rows >= 0)
    set_stage(
        "parsed",
        "completed" if ok_parsed else "pending",
        {"predictions_csv": pred_rel if ok_parsed else None, "row_count": pred_rows},
    )

    # scored
    per_item = os.path.join(results_dir, "per_item_scores.csv")
    per_item_rel = _rel(per_item, results_dir)
    scored_rows = 0
    if os.path.isfile(per_item):
        try:
            import csv
            with open(per_item, "r", encoding="utf-8") as f:
                r = csv.reader(f)
                scored_rows = max(0, sum(1 for _ in r) - 1)
        except Exception:
            scored_rows = 0
    set_stage(
        "scored",
        "completed" if os.path.isfile(per_item) and scored_rows >= 0 else "pending",
        {"per_item_scores_csv": per_item_rel if os.path.isfile(per_item) else None, "row_count": scored_rows},
    )

    # stats
    sig_path = os.path.join(results_dir, "significance.json")
    sig_rel = _rel(sig_path, results_dir)
    sig_ok = False
    if os.path.isfile(sig_path):
        try:
            with open(sig_path, "r", encoding="utf-8") as f:
                sj = json.load(f)
            sig_ok = int(sj.get("schema_version", 0)) >= 2
        except Exception:
            sig_ok = False
    set_stage("stats", "completed" if sig_ok else "pending", {"significance_json": sig_rel if sig_ok else None})

    # costs
    c_path = os.path.join(results_dir, "costs.json")
    c_rel = _rel(c_path, results_dir)
    c_ok = False
    if os.path.isfile(c_path):
        try:
            with open(c_path, "r", encoding="utf-8") as f:
                cj = json.load(f)
            required = {"prompt_tokens", "completion_tokens", "total_tokens", "usd"}
            c_ok = required.issubset(set(cj.keys()))
        except Exception:
            c_ok = False
    set_stage("costs", "completed" if c_ok else "pending", {"costs_json": c_rel if c_ok else None})

    # report
    # Accept either local results_dir/../reports or results_dir/reports when collocated
    reports_dir = os.path.join(os.path.dirname(results_dir), "reports")
    if not os.path.isdir(reports_dir):
        reports_dir = os.path.join(results_dir, "..", "reports")
        reports_dir = os.path.normpath(reports_dir)
    rep_path = os.path.join(reports_dir, "report.md")
    rep_ok = os.path.isfile(rep_path)
    set_stage("report", "completed" if rep_ok else "pending", {"report_md": _rel(rep_path, os.path.dirname(results_dir)) if rep_ok else None})

    # archived tracked externally; default pending
    set_stage("archived", "pending", {})

    return st


def update_stage_status(manifest_path: str, stage: str, status: str, artifacts: Optional[Dict[str, Any]] = None) -> None:
    data, _ = load_manifest(manifest_path)
    st = data.setdefault("stage_status", {})
    st[stage] = {"status": status, "updated_at": utc_now(), "artifacts": artifacts or {}}
    write_manifest(manifest_path, data)


