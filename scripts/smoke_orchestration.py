from __future__ import annotations
import os
import sys
import json
import time
import argparse
import subprocess
import shutil
from typing import Dict, Any, Optional, Tuple, List
from config.schema import load_config
from scripts.state_utils import STOP_FILENAME


def _copy_first_n(src: str, dst: str, n: int) -> None:
    os.makedirs(os.path.dirname(dst), exist_ok=True)
    cnt = 0
    with open(src, "r", encoding="utf-8") as fin, open(dst, "w", encoding="utf-8") as fout:
        for line in fin:
            s = line.strip()
            if not s:
                continue
            fout.write(s + "\n")
            cnt += 1
            if cnt >= max(1, int(n)):
                break


def _ensure_small_prepared(base_out: str, prepared_dir: str | None, n: int) -> str:
    """Create a tiny prepared dataset for orchestration smoke tests."""
    dst_prepared = os.path.join(base_out, "prepared")
    os.makedirs(dst_prepared, exist_ok=True)
    dst_ob = os.path.join(dst_prepared, "open_book.jsonl")
    dst_cb = os.path.join(dst_prepared, "closed_book.jsonl")

    src_ob = os.path.join(prepared_dir or "data/prepared", "open_book.jsonl")
    src_cb = os.path.join(prepared_dir or "data/prepared", "closed_book.jsonl")
    if os.path.isfile(src_ob) and os.path.isfile(src_cb):
        _copy_first_n(src_ob, dst_ob, n)
        _copy_first_n(src_cb, dst_cb, n)
        return dst_prepared

    # Synthesize a minimal set
    ob_rows = [
        {"dataset": "squad_v2", "id": "ob1", "context": "Paris is in France.", "question": "Which country is Paris in?", "answers": ["France"], "is_unanswerable": False},
        {"dataset": "squad_v2", "id": "ob2", "context": "No one knows this.", "question": "What is the answer?", "answers": [], "is_unanswerable": True},
    ]
    cb_rows = [
        {"dataset": "triviaqa", "id": "cb1", "question": "How many valves does a trumpet have?", "answers": ["3", "Three"]},
        {"dataset": "nq_open", "id": "cb2", "question": "Who wrote The Hobbit?", "answers": ["J. R. R. Tolkien", "JRR Tolkien", "Tolkien"]},
    ]
    with open(dst_ob, "w", encoding="utf-8") as f:
        for r in ob_rows[: max(1, int(n))]: f.write(json.dumps(r) + "\n")
    with open(dst_cb, "w", encoding="utf-8") as f:
        for r in cb_rows[: max(1, int(n))]: f.write(json.dumps(r) + "\n")
    return dst_prepared


def _wait_for(predicate, timeout_s: float, poll_s: float = 0.05) -> bool:
    start = time.time()
    while (time.time() - start) < timeout_s:
        try:
            if predicate():
                return True
        except Exception:
            pass
        time.sleep(poll_s)
    return False


def _build_done(cfg2: dict, trials: List[dict], experiments_dir: str, run_id: str) -> bool:
    # Mimic run_all._build_done logic minimally
    prompt_sets_cfg = cfg2.get("prompt_sets") or {}
    default_ps = cfg2.get("default_prompt_set") or (sorted(list(prompt_sets_cfg.keys()))[0] if prompt_sets_cfg else "default")
    backend = str((cfg2.get("backend") or "fireworks")).strip().lower()
    temps_per_ps: dict[str, set[float]] = {}
    for tr in trials:
        psn = tr.get("prompt_set") or default_ps
        temps_per_ps.setdefault(psn, set()).update(float(t) for t in (tr.get("temps") or cfg2.get("temps") or [0.0]))
    # batch_inputs live under experiments/run_<id>/batch_inputs when using experiments dir
    batch_dir = os.path.join(experiments_dir, f"run_{run_id}", "batch_inputs")
    configured_batch_dir = os.path.expanduser(
        os.path.expandvars((cfg2.get("paths") or {}).get("batch_inputs_dir", ""))
    )
    search_roots = []
    if os.path.isdir(batch_dir):
        search_roots.append(batch_dir)
    if configured_batch_dir and os.path.isdir(configured_batch_dir):
        norm_configured = os.path.abspath(configured_batch_dir)
        if all(os.path.abspath(p) != norm_configured for p in search_roots):
            search_roots.append(configured_batch_dir)
    if backend == "alt":
        expected_names: set[str] = set()
        for psn, tset in temps_per_ps.items():
            suffix = f"_{psn}" if (len(prompt_sets_cfg) > 1 or psn not in ("default", None)) else ""
            for t in sorted(tset):
                t_label = "0" if f"{float(t):.1f}" == "0.0" else f"{float(t):.1f}".replace(".", "")
                for cond in ("control", "treatment"):
                    expected_names.add(f"t{t_label}{suffix}_{cond}.jsonl")
        if not expected_names:
            return True
        if not search_roots:
            return False
        discovered: set[str] = set()
        for root in search_roots:
            for dirpath, _dirs, files in os.walk(root):
                for fname in files:
                    if fname.endswith(".jsonl"):
                        discovered.add(fname)
        return expected_names.issubset(discovered)
    for psn, tset in temps_per_ps.items():
        suffix = f"_{psn}" if (len(prompt_sets_cfg) > 1 or psn not in ("default", None)) else ""
        for t in sorted(tset):
            t_label = "0" if f"{float(t):.1f}" == "0.0" else f"{float(t):.1f}".replace(".", "")
            for cond in ("control", "treatment"):
                fpath = os.path.join(batch_dir, f"t{t_label}{suffix}_{cond}.jsonl")
                if not (os.path.isfile(fpath) and os.path.getsize(fpath) > 0):
                    return False
    return True


def _manifest_has_jobs(manifest_path: str) -> bool:
    try:
        with open(manifest_path, "r", encoding="utf-8") as f:
            m = json.load(f)
        jobs = m.get("jobs") or {}
        job_status = m.get("job_status") or {}
        return bool(jobs) or bool(job_status)
    except Exception:
        return False


def _results_combined_exists(results_dir: str) -> bool:
    p = os.path.join(results_dir, "results_combined.jsonl")
    return os.path.isfile(p) and os.path.getsize(p) > 0


def _write_stop(run_root: str) -> None:
    os.makedirs(run_root, exist_ok=True)
    p = os.path.join(run_root, STOP_FILENAME)
    with open(p, "w", encoding="utf-8") as f:
        f.write(time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()))


def main() -> None:
    ap = argparse.ArgumentParser(description="Orchestration smoke test with optional STOP injection")
    ap.add_argument("--config", default="config/eval_config.yaml")
    ap.add_argument("--n", type=int, default=10, help="Items per split for the tiny prepared set")
    ap.add_argument("--prompt_set", default="operational_only")
    ap.add_argument("--keep", action="store_true", help="Keep temporary smoke artifacts (default: delete)")
    ap.add_argument("--dry_run", action="store_true", help="Run orchestrator in --dry_run mode (recommended)")
    ap.add_argument("--stop_point", choices=[
        "none", "post_build", "pre_submit", "pre_poll", "pre_parse"
    ], default="pre_parse", help="When to inject a STOP request during the first pass")
    args = ap.parse_args()

    cfg = load_config(args.config)
    ts = time.strftime("%Y%m%d_%H%M%S", time.localtime())
    run_id = f"smoke_orch_{ts}"
    base_out = os.path.join("results", run_id)
    os.makedirs(base_out, exist_ok=True)

    # Build a small prepared dir
    prepared_small = _ensure_small_prepared(base_out, cfg["paths"].get("prepared_dir"), args.n)

    # Write a temporary config pointing to the small prepared dir
    cfg2 = dict(cfg)
    cfg2["temps"] = [float((cfg.get("temps") or [0.0])[0] or 0.0)]
    cfg2["samples_per_item"] = {f"{cfg2['temps'][0]:.1f}": 1}
    cfg2.setdefault("paths", {}).update({
        "prepared_dir": prepared_small,
        # Allow run_all to place batch_inputs/results under experiments/run_* automatically
    })
    # Constrain to a single trial for determinism: model_id + selected prompt_set only
    ps_cfg = (cfg.get("prompt_sets") or {})
    if args.prompt_set not in ps_cfg:
        # Best-effort: if missing, fall back to default
        pass
    cfg2["default_prompt_set"] = args.prompt_set
    cfg2["prompt_sets"] = {args.prompt_set: ps_cfg.get(args.prompt_set, {
        "control": "config/prompts/control_system.txt",
        "treatment": "config/prompts/treatment_system.txt",
    })}
    # Disable config sweeps to avoid expanding extra prompt sets/models during smoke
    cfg2["sweep"] = {}
    tmp_cfg = os.path.join(base_out, "config.smoke_orch.yaml")
    with open(tmp_cfg, "w", encoding="utf-8") as f:
        import yaml
        yaml.safe_dump(cfg2, f, sort_keys=False)

    # Run the orchestrator (pass 1), optionally injecting STOP at a checkpoint
    backend = str((cfg.get("backend") or "fireworks")).strip().lower()
    runner = "scripts.run_all"
    if backend == "alt":
        runner = "scripts.alt_run_all"
    temp_label = f"{cfg2['temps'][0]:.1f}"
    common = [
        sys.executable, "-m", runner,
        "--config", tmp_cfg,
        "--run_id", run_id,
        "--prompt_sets", args.prompt_set,
        "--temps", temp_label,
        "--skip_prepare",
        "--parts_per_dataset", "3",
        "--max_concurrent_jobs", "2",
    ]
    if args.dry_run:
        common.append("--dry_run")

    # Compute experiments/run_<id>/ path
    exp_root = cfg2.get("paths", {}).get("experiments_dir", "experiments")
    run_root = os.path.join(exp_root, f"run_{run_id}")

    # Launch pass-1
    p1 = subprocess.Popen(common, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)

    # Derive trial directories (only one trial in this smoke)
    # We cannot know the slug ahead of time without duplicating logic; search later.

    try:
        # Wait for checkpoint then inject STOP
        if args.stop_point != "none":
            # Ensure run root exists first
            _wait_for(lambda: os.path.isdir(run_root), timeout_s=30.0)

            if args.stop_point == "post_build":
                # Build completes when batch_inputs files exist
                # Construct a single-trial placeholder to reuse helper
                trial_temps = [float(t) for t in (cfg2.get("temps") or [0.0])]
                trials = [{
                    "model_id": cfg2.get("model_id"),
                    "prompt_set": args.prompt_set,
                    "temps": trial_temps,
                }]
                ok = _wait_for(lambda: _build_done(cfg2, trials, exp_root, run_id), timeout_s=60.0)
                if not ok:
                    raise SystemExit("Timeout waiting for build to complete in smoke orchestrator")
                _write_stop(run_root)
            elif args.stop_point in ("pre_submit", "pre_poll"):
                # pre_submit: wait until the per-trial manifest exists with jobs planned
                # pre_poll: same condition works in dry_run because submit writes manifest before polling
                # Find the trial results dir by scanning the run tree
                def _trial_manifest_path() -> Optional[str]:
                    if not os.path.isdir(run_root):
                        return None
                    for root, _d, files in os.walk(run_root):
                        if "trial_manifest.json" in files and os.path.basename(root) == "results":
                            return os.path.join(root, "trial_manifest.json")
                    return None
                ok = _wait_for(lambda: (_path := _trial_manifest_path()) is not None and _manifest_has_jobs(_path), timeout_s=120.0)
                if not ok:
                    raise SystemExit("Timeout waiting for trial_manifest with jobs")
                _write_stop(run_root)
            elif args.stop_point == "pre_parse":
                # Wait for combined results to exist, then stop before parse
                def _results_dir() -> Optional[str]:
                    if not os.path.isdir(run_root):
                        return None
                    for root, _d, files in os.walk(run_root):
                        if os.path.basename(root) == "results" and any(fn.lower().endswith('.jsonl') for fn in files):
                            return root
                    return None
                def _has_combined() -> bool:
                    rd = _results_dir()
                    return bool(rd and _results_combined_exists(rd))
                ok = _wait_for(_has_combined, timeout_s=180.0)
                if not ok:
                    raise SystemExit("Timeout waiting for combined results before parse")
                _write_stop(run_root)

        # Wait for process to exit (expect non-zero due to STOP), but tolerate success
        try:
            rc = p1.wait(timeout=240)
        except subprocess.TimeoutExpired:
            p1.kill()
            rc = -9
        if args.stop_point != "none" and rc == 0:
            print("Warning: STOP was injected but process exited cleanly; continuing with resume")
    finally:
        if p1 and p1.stdout:
            try:
                p1.stdout.close()
            except Exception:
                pass

    # Remove STOP and resume to completion
    stop_path = os.path.join(run_root, STOP_FILENAME)
    if os.path.exists(stop_path):
        try:
            os.remove(stop_path)
        except Exception:
            pass
    p2_cmd = list(common)
    p2_cmd.append("--resume")
    subprocess.check_call(p2_cmd)

    # Best-effort cleanup unless --keep
    if not args.keep:
        try:
            if os.path.isdir(base_out):
                shutil.rmtree(base_out, ignore_errors=True)
        except Exception:
            pass

    # Verify outputs exist inside experiments dir
    exp_root = cfg2.get("paths", {}).get("experiments_dir", "experiments")
    if not os.path.isdir(exp_root):
        raise SystemExit("experiments root not created; run_all did not execute as expected")

    # Find newest run matching smoke_orch
    candidates = [d for d in os.listdir(exp_root) if d.startswith("run_") and run_id in d]
    if not candidates:
        # fallback: pick newest
        candidates = sorted(os.listdir(exp_root))
    chosen = sorted(candidates)[-1]
    chosen_dir = os.path.join(exp_root, chosen)

    # Check for any predictions.csv in trial results and basic uniqueness of ids
    pred_paths: list[str] = []
    for root, _dirs, files in os.walk(chosen_dir):
        if "predictions.csv" in files:
            pred_paths.append(os.path.join(root, "predictions.csv"))
    if args.dry_run:
        print("NOTE: Dry run completed; skipping predictions.csv presence/duplication checks.")
    else:
        if not pred_paths:
            raise SystemExit("orchestration smoke: predictions.csv not found; post-processing failed")
        # Basic duplicate check: ensure custom_id column has no duplicates
        import csv
        for pp in pred_paths:
            try:
                ids: list[str] = []
                with open(pp, "r", encoding="utf-8") as f:
                    r = csv.DictReader(f)
                    for row in r:
                        if row.get("custom_id"):
                            ids.append(row["custom_id"])
                if ids and (len(set(ids)) != len(ids)):
                    raise SystemExit(f"Duplicate custom_id detected in {pp}; resume should avoid duplicates")
            except Exception as e:
                print(f"Warning: duplicate check skipped for {pp}: {e}")
    print(f"OK: Orchestration smoke completed. See {chosen_dir}")
    if not args.keep:
        try:
            shutil.rmtree(chosen_dir, ignore_errors=True)
            print("Cleaned up smoke artifacts.")
        except Exception:
            pass


if __name__ == "__main__":
    import multiprocessing as mp

    mp.freeze_support()
    main()
