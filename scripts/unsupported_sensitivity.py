from __future__ import annotations

import os
import json
import csv
import argparse
from collections import defaultdict
from typing import Dict, List, Tuple

import numpy as np

from config.schema import load_config
from scoring.normalize import normalize_answer
from scoring.unsupported import is_unsupported


def read_preds_csv(path: str):
    with open(path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for r in reader:
            yield r


def load_open_book(prepared_dir: str) -> Dict[str, dict]:
    path = os.path.join(prepared_dir, "open_book.jsonl")
    data: Dict[str, dict] = {}
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            s = line.strip()
            if not s:
                continue
            row = json.loads(s)
            data[f"{row['dataset']}|{row['id']}"] = row
    return data


def bootstrap_ci(deltas: List[float], B: int, seed: int) -> Tuple[float, float]:
    if not deltas:
        return 0.0, 0.0
    arr = np.array([float(x) for x in deltas], dtype=float)
    if arr.size == 1:
        return float(arr[0]), float(arr[0])
    rng = np.random.default_rng(seed)
    n = arr.size
    idx = np.arange(n)
    boots = np.empty(B, dtype=float)
    for i in range(B):
        sample = rng.choice(idx, size=n, replace=True)
        boots[i] = float(np.mean(arr[sample]))
    lo = float(np.percentile(boots, 2.5))
    hi = float(np.percentile(boots, 97.5))
    return lo, hi


def main():
    ap = argparse.ArgumentParser(description="Unsupported sensitivity sweep")
    ap.add_argument("--pred_csv", default="results/predictions.csv")
    ap.add_argument("--prepared_dir", default="data/prepared")
    ap.add_argument("--config", default="config/eval_config.yaml")
    ap.add_argument("--out_path", default="results/unsupported_sensitivity.json")
    ap.add_argument("--thresholds", help="Comma-separated thresholds override (0..1)")
    args = ap.parse_args()

    cfg = load_config(args.config)
    uns_cfg = cfg.get("unsupported", {})
    strategy = (uns_cfg.get("strategy") or "baseline").lower()
    params = {"min_token_overlap": float(uns_cfg.get("min_token_overlap", 0.6))}
    thresholds = []
    if args.thresholds:
        thresholds = [float(x.strip()) for x in args.thresholds.split(",") if x.strip()]
    else:
        thresholds = [round(x, 2) for x in np.linspace(0.0, 1.0, 11)]
    ob = load_open_book(cfg["paths"]["prepared_dir"])
    # Collect per (item, temp, condition) unsupported flags per threshold
    by_key = defaultdict(lambda: defaultdict(list))  # (item_key,temp,condition) -> thr -> list[flag]
    for r in read_preds_csv(args.pred_csv):
        if (r.get("type") or "") != "open":
            continue
        item_key = f"{r['dataset']}|{r['item_id']}"
        temp = float(r.get("temp") or 0.0)
        condition = r.get("condition") or "control"
        pred = r.get("response_text") or ""
        ex = ob.get(item_key) or {}
        abstained = 1.0 if normalize_answer(pred) in ("",) else 0.0  # fallback; true abstention already reflected in score file, but re-computing is out of scope
        for thr in thresholds:
            flag = is_unsupported(pred, ex.get("context", ""), abstained=abstained, strategy=strategy, threshold=float(thr), params=params)
            by_key[(item_key, temp, condition)][thr].append(int(flag))

    # Aggregate per item (avg over replicates), then compute deltas trt-ctrl per temp
    results: Dict[str, dict] = {}
    B = int(cfg.get("stats", {}).get("bootstrap_samples", 5000))
    seed = int(cfg.get("stats", {}).get("random_seed", 1337))
    by_temp: Dict[float, Dict[float, List[float]]] = defaultdict(lambda: defaultdict(list))
    # Map temp -> thr -> list of deltas per item
    by_item_temp: Dict[Tuple[str, float], Dict[str, Dict[float, float]]] = defaultdict(lambda: {"control": {}, "treatment": {}})
    # Aggregate per item/cond/thr
    for (item_key, temp, condition), thr_map in by_key.items():
        for thr, flags in thr_map.items():
            rate = float(sum(flags)) / float(len(flags)) if flags else 0.0
            by_item_temp[(item_key, temp)][condition][thr] = rate
    # Deltas
    for (item_key, temp), sides in by_item_temp.items():
        ctrl = sides.get("control", {})
        trt = sides.get("treatment", {})
        for thr in thresholds:
            if thr in ctrl and thr in trt:
                by_temp[temp][thr].append(float(trt[thr]) - float(ctrl[thr]))

    for temp, thr_map in by_temp.items():
        tkey = f"{float(temp):.1f}"
        results.setdefault(tkey, {})
        for thr, deltas in sorted(thr_map.items()):
            mean_delta = float(np.mean(deltas)) if deltas else 0.0
            lo, hi = bootstrap_ci(deltas, B, seed)
            results[tkey][f"{thr:.2f}"] = {"delta_mean": mean_delta, "ci_95": [lo, hi], "n_items": len(deltas)}

    os.makedirs(os.path.dirname(args.out_path), exist_ok=True)
    with open(args.out_path, "w", encoding="utf-8") as f:
        json.dump({
            "strategy": strategy,
            "thresholds": [float(t) for t in thresholds],
            "results": results,
        }, f, indent=2)
    print("Wrote", args.out_path)


if __name__ == "__main__":
    import multiprocessing as mp

    mp.freeze_support()
    main()
