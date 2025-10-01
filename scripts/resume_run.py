#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import sys
import subprocess
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple

import yaml

from scripts.state_utils import (
    RunStateLock,
    write_json_atomic,
    load_run_state,
    init_run_state,
    update_phase,
    run_state_path,
    compute_config_hash,
)


def _now() -> str:
    return datetime.utcnow().isoformat() + "Z"


def _discover_run_root(run_id: str, experiments_dir: str = "experiments") -> str:
    root = os.path.join(experiments_dir, f"run_{run_id}")
    return root


def _load_yaml(path: str) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def _backup_corrupt(path: str, err: Exception) -> None:
    try:
        suffix = _now().replace(":", "_")
        newp = f"{path}.corrupt.{suffix}"
        os.replace(path, newp)
        print(f"WARNING: Backed up corrupt manifest to {newp}: {err}")
    except Exception as e:
        print(f"WARNING: Could not back up corrupt file {path}: {e}")


def _find_trial_result_dirs(run_root: str) -> List[str]:
    found: List[str] = []
    try:
        for name in os.listdir(run_root):
            tdir = os.path.join(run_root, name)
            if not os.path.isdir(tdir):
                continue
            res = os.path.join(tdir, "results")
            if os.path.isdir(res) and os.path.isfile(os.path.join(res, "trial_manifest.json")):
                found.append(res)
    except FileNotFoundError:
        pass
    return sorted(found)


def _load_trial_manifest(path: str) -> Optional[Dict[str, Any]]:
    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception as e:
        _backup_corrupt(path, e)
        return None


def _infer_phase_from_trials(trial_results_dirs: List[str]) -> Dict[str, str]:
    """Best-effort inference of phase completion from per-trial artifacts.

    Returns a map phase -> status (completed|in_progress|not_started).
    Conservative: any missing artifact across trials yields in_progress/not_started.
    """
    if not trial_results_dirs:
        return {p: "not_started" for p in [
            "prepare", "build", "submit", "poll", "parse", "score", "stats", "costs", "report"
        ]}

    # Helper to check all trials
    def all_trials_have(filename: str, *, nonempty: bool = False) -> Tuple[bool, bool]:
        any_present = False
        for res_dir in trial_results_dirs:
            fpath = os.path.join(res_dir, filename)
            if os.path.isfile(fpath):
                any_present = True
                if nonempty and os.path.getsize(fpath) <= 0:
                    return False, True
            else:
                return False, any_present
        return True, any_present

    # Load manifests and check job_status/results presence
    submits_done = True
    polls_done = True
    saw_any_submit = False
    for res_dir in trial_results_dirs:
        mp = os.path.join(res_dir, "trial_manifest.json")
        man = _load_trial_manifest(mp)
        if not man:
            submits_done = False
            polls_done = False
            continue
        job_status = (man.get("job_status", {}) or {})
        datasets = (man.get("datasets", {}) or {})
        # For submit: require job_status contains an entry per expected part
        expected_keys: List[str] = []
        for k, dsids in datasets.items():
            try:
                n = len(dsids or [])
            except Exception:
                n = 0
            expected_keys.extend([f"{k}_p{i:02d}" for i in range(1, n + 1)])
        if expected_keys:
            saw_any_submit = True
        for ek in expected_keys:
            if ek not in job_status:
                submits_done = False
                break
        # For poll: require either per-part results.jsonl or a combined results file
        # Accept either complete per-part results or presence of results_combined.jsonl
        combined = os.path.join(res_dir, "results_combined.jsonl")
        if os.path.isfile(combined) and os.path.getsize(combined) > 0:
            continue
        for ek in expected_keys:
            part_dir = os.path.join(res_dir, ek)
            part_res = os.path.join(part_dir, "results.jsonl")
            if not (os.path.isfile(part_res) and os.path.getsize(part_res) > 0):
                polls_done = False
                break

    # Prepare/build predicates are approximate but consistent with run_all
    phases: Dict[str, str] = {}
    phases["prepare"] = "completed"  # we only migrate existing runs post-prepare
    phases["build"] = "completed"     # conservative assumption once trials exist
    if not saw_any_submit:
        phases["submit"] = "not_started"
        phases["poll"] = "not_started"
    else:
        phases["submit"] = "completed" if submits_done else "in_progress"
        phases["poll"] = "completed" if polls_done else "in_progress"

    # Parse/score/stats/costs/report based on file presence across trials
    preds_all, preds_any = all_trials_have("predictions.csv", nonempty=True)
    phases["parse"] = "completed" if preds_all else ("in_progress" if preds_any else "not_started")

    score_all, score_any = all_trials_have("per_item_scores.csv", nonempty=True)
    phases["score"] = "completed" if score_all else ("in_progress" if score_any else "not_started")

    sig_all, sig_any = all_trials_have("significance.json", nonempty=True)
    phases["stats"] = "completed" if sig_all else ("in_progress" if sig_any else "not_started")

    costs_all, costs_any = all_trials_have("costs.json", nonempty=True)
    phases["costs"] = "completed" if costs_all else ("in_progress" if costs_any else "not_started")

    # Report can be per-trial or aggregated; consider per-trial reports
    rpt_all = True
    rpt_any = False
    for res_dir in trial_results_dirs:
        rpt_dir = os.path.dirname(res_dir).replace("/results", "/reports")
        rpt = os.path.join(rpt_dir, "report.md")
        if os.path.isfile(rpt) and os.path.getsize(rpt) > 0:
            rpt_any = True
        else:
            rpt_all = False
    phases["report"] = "completed" if rpt_all else ("in_progress" if rpt_any else "not_started")
    return phases


def _migrate_state_if_missing(run_root: str, run_id: str, cfg: Dict[str, Any]) -> Dict[str, Any]:
    os.makedirs(run_root, exist_ok=True)
    st = load_run_state(run_root)
    if st is not None:
        return st
    # Find trial dirs for this run
    trials_res = _find_trial_result_dirs(run_root)
    st = init_run_state(run_root, run_id, cfg)
    # Mark as migrated and add a note
    st["migrated"] = True
    st.setdefault("notes", []).append({
        "created_by": "scripts.resume_run",
        "created_at": _now(),
        "message": "State synthesized from on-disk artifacts (best effort)."
    })
    # Infer phase statuses
    inferred = _infer_phase_from_trials(trials_res)
    for phase, status in inferred.items():
        update_phase(st, phase, status=status)
    # Persist
    with RunStateLock(run_root):
        write_json_atomic(run_state_path(run_root), st)
    print(f"Migration: wrote synthesized run_state.json at {run_state_path(run_root)}")
    return st


def main() -> None:
    ap = argparse.ArgumentParser(description="Resume an experiment run with config drift checks and state migration")
    ap.add_argument("--run_id", required=True, help="Run ID, e.g. r20250101123456")
    ap.add_argument("--config", help="Path to config (used if effective_config.yaml is missing)")
    ap.add_argument("--only_step", choices=[
        "prepare","build","submit","poll","parse","score","stats","costs","report"
    ])
    ap.add_argument("--from_step", choices=[
        "prepare","build","submit","poll","parse","score","stats","costs","report"
    ])
    ap.add_argument("--to_step", choices=[
        "prepare","build","submit","poll","parse","score","stats","costs","report"
    ])
    ap.add_argument("--force", action="store_true", help="Proceed on config drift; record drift in state")
    ap.add_argument("--experiments_dir", default="experiments", help="Experiments base directory")
    args = ap.parse_args()

    run_id = args.run_id
    run_root = _discover_run_root(run_id, args.experiments_dir)
    eff_cfg = os.path.join(run_root, "effective_config.yaml")

    cfg_path: Optional[str] = None
    if os.path.isfile(eff_cfg):
        cfg_path = eff_cfg
    elif args.config and os.path.isfile(args.config):
        cfg_path = args.config
    else:
        print("ERROR: No effective config found and no --config provided.")
        print(f"Looked for: {eff_cfg}")
        sys.exit(2)

    try:
        cfg = _load_yaml(cfg_path)
    except Exception as e:
        print(f"ERROR: Failed to load config at {cfg_path}: {e}")
        sys.exit(2)

    # Load or synthesize run state
    state = load_run_state(run_root)
    if state is None:
        # Attempt migration from artifacts
        state = _migrate_state_if_missing(run_root, run_id, cfg)

    # Config drift check
    actual_hash = compute_config_hash(cfg)
    recorded_hash = state.get("config_hash")
    if recorded_hash and (recorded_hash != actual_hash) and not args.force:
        print("CONFIG DRIFT DETECTED: effective config does not match recorded config_hash.")
        print(f"  recorded: {recorded_hash}")
        print(f"  current : {actual_hash}")
        print("Refuse to resume. Pass --force to proceed and record drift.")
        sys.exit(3)

    if recorded_hash != actual_hash:
        # Record drift
        drift_rec = {
            "detected_at": _now(),
            "expected": recorded_hash,
            "actual": actual_hash,
        }
        state.setdefault("config_drift", []).append(drift_rec)
        state["config_hash"] = actual_hash
        with RunStateLock(run_root):
            write_json_atomic(run_state_path(run_root), state)
        print("WARNING: Proceeding despite config drift; recorded in run_state.json")

    # Dispatch to run_all with --resume and gating flags
    cmd: List[str] = [sys.executable, "-m", "scripts.run_all", "--config", cfg_path, "--run_id", run_id, "--resume"]
    if args.only_step:
        cmd += ["--only_step", args.only_step]
    if args.from_step:
        cmd += ["--from_step", args.from_step]
    if args.to_step:
        cmd += ["--to_step", args.to_step]

    print("+", " ".join(cmd))
    rc = subprocess.call(cmd)
    if rc != 0:
        print(f"ERROR: run_all exited with status {rc}")
        sys.exit(rc)


if __name__ == "__main__":
    import multiprocessing as mp

    mp.freeze_support()
    main()
