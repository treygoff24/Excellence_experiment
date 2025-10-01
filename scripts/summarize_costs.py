from __future__ import annotations
import os
import csv
import argparse
import json
import yaml
from scripts import manifest_v2 as mf


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pred_csv", default="results/predictions.csv")
    ap.add_argument("--config", default="config/eval_config.yaml")
    ap.add_argument("--out_path", default="results/costs.json")
    args = ap.parse_args()
    with open(args.config, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    pricing = cfg.get("pricing", {})
    batch = cfg.get("use_batch_api", True)
    disc = pricing.get("batch_discount", 0.5) if batch else 1.0
    pt = ct = 0
    with open(args.pred_csv, "r", encoding="utf-8") as f:
        r = csv.DictReader(f)
        for row in r:
            pt += int(row["prompt_tokens"] or 0)
            ct += int(row["completion_tokens"] or 0)
    input_per_m = pricing.get("input_per_million", 0.15)
    output_per_m = pricing.get("output_per_million", 0.60)
    cost = (pt / 1_000_000) * input_per_m + (ct / 1_000_000) * output_per_m
    cost *= disc
    data = {"prompt_tokens": pt, "completion_tokens": ct, "total_tokens": pt + ct, "usd": cost, "batch_discount_applied": batch}
    # Idempotency
    preds_mtime = os.path.getmtime(args.pred_csv) if os.path.exists(args.pred_csv) else 0.0
    if os.path.isfile(args.out_path) and os.path.getmtime(args.out_path) >= preds_mtime:
        print("Idempotent skip: costs.json up-to-date")
        _update_manifest_costs(args.out_path)
        return
    with open(args.out_path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)
    print("Wrote", args.out_path)
    _update_manifest_costs(args.out_path)


def _update_manifest_costs(costs_path: str) -> None:
    results_dir = os.path.dirname(costs_path)
    manifest_path = os.path.join(results_dir, "trial_manifest.json")
    if os.path.isfile(manifest_path):
        try:
            mf.update_stage_status(
                manifest_path,
                "costs",
                "completed",
                {"costs_json": os.path.relpath(costs_path, results_dir)},
            )
        except Exception:
            pass


if __name__ == "__main__":
    import multiprocessing as mp

    mp.freeze_support()
    main()
