from __future__ import annotations
import os
import sys
import json
import time
import glob
import argparse
import subprocess
import shutil
from typing import Dict, Any, List

from config.schema import load_config


def _copy_first_n(src: str, dst: str, n: int) -> None:
    os.makedirs(os.path.dirname(dst), exist_ok=True)
    cnt = 0
    with open(src, "r", encoding="utf-8") as fin, open(dst, "w", encoding="utf-8") as fout:
        for line in fin:
            s = line.strip()
            if not s:
                continue
            fout.write(s + "\n"); cnt += 1
            if cnt >= max(1, int(n)):
                break


def _ensure_small_prepared(base_out: str, prepared_dir: str | None, n: int) -> str:
    """Create a tiny prepared dataset for dry-run tests."""
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
        for r in ob_rows[: max(1, int(n))]:
            f.write(json.dumps(r) + "\n")
    with open(dst_cb, "w", encoding="utf-8") as f:
        for r in cb_rows[: max(1, int(n))]:
            f.write(json.dumps(r) + "\n")
    return dst_prepared


def _slug_from_path(p: str) -> str:
    base = os.path.basename(p)
    name, _ext = os.path.splitext(base)
    return name


def main() -> None:
    ap = argparse.ArgumentParser(description="Dry-run: run many treatment prompts with one control, with concurrency=4")
    ap.add_argument("--config", default="config/eval_config.yaml")
    ap.add_argument("--prompts_dir", default="config/prompts", help="Directory containing treatment prompt markdown files")
    ap.add_argument("--pattern", default="*.md", help="Glob for treatment prompt files")
    ap.add_argument("--control", default="config/prompts/control_system.txt", help="Path to control system prompt")
    ap.add_argument("--temps", default="0.0", help="Temperatures to test (comma or space separated)")
    ap.add_argument("--n", type=int, default=5, help="Items per split for tiny prepared set")
    ap.add_argument("--parts_per_dataset", type=int, default=6)
    ap.add_argument("--max_concurrent_jobs", type=int, default=4)
    ap.add_argument("--keep", action="store_true", help="Keep experiment artifacts")
    args = ap.parse_args()

    cfg = load_config(args.config)

    # Discover treatment files
    treat_files = sorted(glob.glob(os.path.join(args.prompts_dir, args.pattern)))
    if not treat_files:
        raise SystemExit(f"No treatment files matched in {args.prompts_dir} with pattern {args.pattern}")

    ts = time.strftime("%Y%m%d_%H%M%S", time.localtime())
    run_id = f"dry_multi_treat_{ts}"

    # Build small prepared dir
    base_out = os.path.join("results", run_id)
    os.makedirs(base_out, exist_ok=True)
    prepared_small = _ensure_small_prepared(base_out, cfg["paths"].get("prepared_dir"), args.n)

    # Build a temporary config with prompt_sets for each treatment file
    cfg2 = dict(cfg)
    cfg2["temps"] = [float(s) for s in str(args.temps).replace(",", " ").split()]
    # Make per-temp samples small
    cfg2["samples_per_item"] = {f"{float(t):.1f}": 1 for t in cfg2["temps"]}
    cfg2.setdefault("paths", {}).update({
        "prepared_dir": prepared_small,
    })
    prompt_sets: Dict[str, Dict[str, str]] = {}
    for tf in treat_files:
        key = _slug_from_path(tf)
        prompt_sets[key] = {"control": os.path.abspath(args.control), "treatment": os.path.abspath(tf)}
    cfg2["prompt_sets"] = prompt_sets
    # Disable any existing sweep so CLI prompt_sets are respected
    cfg2["sweep"] = None
    # Make the first discovered set the default
    cfg2["default_prompt_set"] = sorted(prompt_sets.keys())[0]

    # Write temporary config
    tmp_cfg = os.path.join(base_out, "config.multi_treat.yaml")
    with open(tmp_cfg, "w", encoding="utf-8") as f:
        import yaml
        yaml.safe_dump(cfg2, f, sort_keys=False)

    # Compose CLI args: use all prompt set keys
    keys: List[str] = sorted(prompt_sets.keys())
    cmd = [
        sys.executable, "-m", "scripts.run_all",
        "--config", tmp_cfg,
        "--run_id", run_id,
        "--prompt_sets", *keys,
        "--temps", *(str(t) for t in cfg2["temps"]),
        "--skip_prepare",
        "--dry_run",
        "--parts_per_dataset", str(int(args.parts_per_dataset)),
        "--max_concurrent_jobs", str(int(args.max_concurrent_jobs)),
    ]

    print("+", " ".join(cmd)); sys.stdout.flush()
    try:
        subprocess.check_call(cmd)
    finally:
        if not args.keep:
            # Remove the results base and experiments run directory
            try:
                if os.path.isdir(base_out):
                    shutil.rmtree(base_out, ignore_errors=True)
            except Exception:
                pass

    # After run_all, verify the experiments directory contains the run
    exp_root = cfg2.get("paths", {}).get("experiments_dir", "experiments")
    if not os.path.isdir(exp_root):
        raise SystemExit("experiments root not created; run_all did not execute as expected")
    candidates = [d for d in os.listdir(exp_root) if d.startswith("run_") and run_id in d]
    if not candidates:
        # fallback: newest
        candidates = sorted(os.listdir(exp_root))
    chosen = sorted(candidates)[-1]
    chosen_dir = os.path.join(exp_root, chosen)

    # Check for multi-trial manifest and number of trials
    mtm = os.path.join(chosen_dir, "multi_trial_manifest.json")
    if not os.path.isfile(mtm):
        raise SystemExit("Missing multi_trial_manifest.json; dry-run orchestration failed")
    try:
        data = json.load(open(mtm, "r", encoding="utf-8"))
        num_trials = int(data.get("num_trials") or 0)
        if num_trials < len(keys):
            print(f"Warning: Expected >= {len(keys)} trials, found {num_trials}")
    except Exception:
        pass

    print(f"OK: Multi-treatment dry-run completed. See {chosen_dir}")
    if not args.keep:
        try:
            shutil.rmtree(chosen_dir, ignore_errors=True)
            print("Cleaned up dry-run artifacts.")
        except Exception:
            pass


if __name__ == "__main__":
    main()
