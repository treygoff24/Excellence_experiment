from __future__ import annotations

import os
import csv
import json
import argparse
from collections import defaultdict
from typing import Dict, List, Tuple

import numpy as np
from scipy.stats import norm

from config.schema import load_config


def load_unanswerable_map(prepared_dir: str) -> Dict[str, bool]:
    path = os.path.join(prepared_dir, "open_book.jsonl")
    mapping: Dict[str, bool] = {}
    try:
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                s = line.strip()
                if not s:
                    continue
                try:
                    row = json.loads(s)
                except Exception:
                    continue
                key = f"{row.get('dataset')}|{row.get('id')}"
                mapping[key] = bool(row.get("is_unanswerable", False))
    except FileNotFoundError:
        pass
    return mapping


def read_per_item_scores(path: str):
    with open(path, "r", encoding="utf-8") as f:
        r = csv.DictReader(f)
        for row in r:
            yield row


def collect_deltas_false_answer(per_item_csv: str, ua_map: Dict[str, bool]) -> Dict[float, List[float]]:
    buckets: Dict[Tuple[str, float], Dict[str, float]] = defaultdict(dict)
    for row in read_per_item_scores(per_item_csv):
        try:
            if (row.get("type") or "") != "open":
                continue
            item_key = str(row.get("item_key"))
            if ua_map and not ua_map.get(item_key, False):
                continue
            temp = float(row.get("temp") or 0.0)
            cond = row.get("condition") or "control"
            far = row.get("false_answer_rate")
            if far in (None, ""):
                continue
            buckets[(item_key, temp)][cond] = float(far)
        except Exception:
            continue
    deltas_by_temp: Dict[float, List[float]] = defaultdict(list)
    for (item_key, temp), sides in buckets.items():
        if "control" in sides and "treatment" in sides:
            deltas_by_temp[temp].append(float(sides["treatment"]) - float(sides["control"]))
    return deltas_by_temp


def compute_power_stats(deltas_by_temp: Dict[float, List[float]], alpha: float, power: float, targets: List[float]) -> Dict[str, dict]:
    z_alpha = float(norm.ppf(1 - alpha / 2.0))
    z_power = float(norm.ppf(power))
    out: Dict[str, dict] = {}
    for temp, deltas in sorted(deltas_by_temp.items()):
        arr = np.array([float(x) for x in deltas], dtype=float)
        n = arr.size
        mean = float(np.mean(arr)) if n else 0.0
        sd = float(np.std(arr, ddof=1)) if n > 1 else 0.0
        if n <= 0:
            mde = None
        else:
            mde = float((z_alpha + z_power) * (sd / np.sqrt(n)) if sd > 0 else 0.0)
        reqN: Dict[str, int | None] = {}
        for t in targets:
            t = abs(float(t))
            if t <= 0 or sd == 0:
                reqN[f"{t:.3f}"] = None
            else:
                n_need = ((z_alpha + z_power) * sd / t) ** 2
                reqN[f"{t:.3f}"] = int(np.ceil(n_need))
        out[f"{float(temp):.1f}"] = {
            "n_items": int(n),
            "mean_delta": mean,
            "sd_delta": sd,
            "mde": mde,
            "required_n_for_target": reqN,
        }
    return out


def main():
    ap = argparse.ArgumentParser(description="Power/MDE analysis for primary endpoint (false_answer_rate on unanswerables)")
    ap.add_argument("--per_item_csv", default="results/per_item_scores.csv")
    ap.add_argument("--prepared_dir", default="data/prepared")
    ap.add_argument("--config", default="config/eval_config.yaml")
    ap.add_argument("--out_path", default="results/power_analysis.json")
    ap.add_argument("--alpha", type=float, default=0.05)
    ap.add_argument("--power", type=float, default=0.80)
    ap.add_argument("--targets", help="Comma-separated absolute target improvements (e.g., 0.02,0.05,0.10)")
    args = ap.parse_args()

    cfg = load_config(args.config)
    ua_map = load_unanswerable_map(cfg["paths"]["prepared_dir"])
    deltas_by_temp = collect_deltas_false_answer(args.per_item_csv, ua_map)
    targets = [0.02, 0.05, 0.10]
    if args.targets:
        try:
            targets = [float(x.strip()) for x in args.targets.split(",") if x.strip()]
        except Exception:
            pass
    results = compute_power_stats(deltas_by_temp, float(args.alpha), float(args.power), targets)
    payload = {
        "alpha": float(args.alpha),
        "power": float(args.power),
        "metric": "false_answer_rate",
        "targets": targets,
        "results": results,
    }
    os.makedirs(os.path.dirname(args.out_path), exist_ok=True)
    with open(args.out_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)
    print("Wrote", args.out_path)


if __name__ == "__main__":
    main()

