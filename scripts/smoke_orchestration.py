from __future__ import annotations
import os
import sys
import json
import time
import argparse
import subprocess
import shutil
from typing import Dict, Any
from config.schema import load_config


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


def main() -> None:
    ap = argparse.ArgumentParser(description="Orchestration smoke test (dry-run)")
    ap.add_argument("--config", default="config/eval_config.yaml")
    ap.add_argument("--n", type=int, default=10, help="Items per split for the tiny prepared set")
    ap.add_argument("--prompt_set", default="operational_only")
    ap.add_argument("--keep", action="store_true", help="Keep temporary smoke artifacts (default: delete)")
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
    cfg2["temps"] = [0.0]
    cfg2["samples_per_item"] = {"0.0": 1}
    cfg2.setdefault("paths", {}).update({
        "prepared_dir": prepared_small,
        # Allow run_all to place batch_inputs/results under experiments/run_* automatically
    })
    tmp_cfg = os.path.join(base_out, "config.smoke_orch.yaml")
    with open(tmp_cfg, "w", encoding="utf-8") as f:
        import yaml
        yaml.safe_dump(cfg2, f, sort_keys=False)

    # Run the full orchestrator in dry-run with tiny splits
    cmd = [
        sys.executable, "-m", "scripts.run_all",
        "--config", tmp_cfg,
        "--run_id", run_id,
        "--prompt_sets", args.prompt_set,
        "--temps", "0.0",
        "--skip_prepare",
        "--dry_run",
        "--parts_per_dataset", "3",
        "--max_concurrent_jobs", "2",
    ]
    try:
        subprocess.check_call(cmd)
    finally:
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

    # Check for any predictions.csv in trial results
    found_csv = False
    for root, _dirs, files in os.walk(chosen_dir):
        if "predictions.csv" in files:
            found_csv = True
            break
    if not found_csv:
        raise SystemExit("orchestration smoke: predictions.csv not found; dry-run post-processing failed")
    print(f"OK: Orchestration smoke dry-run completed. See {chosen_dir}")
    if not args.keep:
        try:
            shutil.rmtree(chosen_dir, ignore_errors=True)
            print("Cleaned up smoke artifacts.")
        except Exception:
            pass


if __name__ == "__main__":
    main()
