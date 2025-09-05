from __future__ import annotations

import os
import csv
import json
import argparse
from collections import defaultdict
from typing import Dict, Tuple

import yaml


def load_cfg(path: str) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def read_preds(path: str):
    with open(path, "r", encoding="utf-8") as f:
        r = csv.DictReader(f)
        for row in r:
            yield row


def read_per_item(path: str):
    with open(path, "r", encoding="utf-8") as f:
        r = csv.DictReader(f)
        for row in r:
            yield row


def pricing_from_cfg(cfg: dict) -> Tuple[float, float, float]:
    pr = cfg.get("pricing", {}) or {}
    input_per_m = float(pr.get("input_per_million", 0.15))
    output_per_m = float(pr.get("output_per_million", 0.60))
    disc = float(pr.get("batch_discount", 0.5)) if bool(cfg.get("use_batch_api", True)) else 1.0
    return input_per_m, output_per_m, disc


def compute_costs_by_temp_condition(pred_csv: str, input_per_m: float, output_per_m: float, discount: float) -> Dict[str, Dict[str, float]]:
    # returns {temp_key: {"control": usd, "treatment": usd, "tokens_control": int, ...}}
    acc: Dict[Tuple[str, str], Dict[str, float]] = defaultdict(lambda: {"prompt_tokens": 0.0, "completion_tokens": 0.0})
    for row in read_preds(pred_csv):
        try:
            tkey = f"{float(row.get('temp') or 0.0):.1f}"
            cond = (row.get("condition") or "control").lower()
            acc[(tkey, cond)]["prompt_tokens"] += float(row.get("prompt_tokens") or 0.0)
            acc[(tkey, cond)]["completion_tokens"] += float(row.get("completion_tokens") or 0.0)
        except Exception:
            continue
    out: Dict[str, Dict[str, float]] = defaultdict(dict)
    for (tkey, cond), tk in acc.items():
        usd = (tk["prompt_tokens"] / 1_000_000.0) * input_per_m + (tk["completion_tokens"] / 1_000_000.0) * output_per_m
        usd *= discount
        out.setdefault(tkey, {})[f"usd_{cond}"] = float(usd)
        out.setdefault(tkey, {})[f"tokens_{cond}"] = int(tk["prompt_tokens"] + tk["completion_tokens"])
    return out


def compute_metric_means(per_item_csv: str) -> Dict[str, Dict[str, Dict[str, float]]]:
    # returns {tkey: {condition: {metric: mean}}}
    sums: Dict[Tuple[str, str], Dict[str, float]] = defaultdict(lambda: defaultdict(float))
    counts: Dict[Tuple[str, str], int] = defaultdict(int)
    metrics = ["em", "f1", "abstain_rate", "false_answer_rate", "unsupported_rate"]
    for row in read_per_item(per_item_csv):
        try:
            tkey = f"{float(row.get('temp') or 0.0):.1f}"
            cond = (row.get("condition") or "control").lower()
        except Exception:
            continue
        for m in metrics:
            v = row.get(m)
            if v not in (None, ""):
                try:
                    sums[(tkey, cond)][m] += float(v)
                except Exception:
                    pass
        counts[(tkey, cond)] += 1
    means: Dict[str, Dict[str, Dict[str, float]]] = defaultdict(lambda: defaultdict(dict))
    for (tkey, cond), sd in sums.items():
        n = max(1, counts[(tkey, cond)])
        for m, s in sd.items():
            means[tkey][cond][m] = float(s) / float(n)
    return means


def per_point_cost(delta: float, usd_diff: float) -> float | None:
    try:
        dpp = abs(float(delta)) * 100.0  # convert to percentage points
    except Exception:
        return None
    if dpp <= 0:
        return None
    return float(usd_diff) / dpp


def main():
    ap = argparse.ArgumentParser(description="Cost-effectiveness summary by temperature")
    ap.add_argument("--pred_csv", default="results/predictions.csv")
    ap.add_argument("--per_item_csv", default="results/per_item_scores.csv")
    ap.add_argument("--config", default="config/eval_config.yaml")
    ap.add_argument("--out_path", default="results/cost_effectiveness.json")
    args = ap.parse_args()

    cfg = load_cfg(args.config)
    ipm, opm, disc = pricing_from_cfg(cfg)
    cost_by_temp = compute_costs_by_temp_condition(args.pred_csv, ipm, opm, disc)
    means = compute_metric_means(args.per_item_csv)

    out: Dict[str, dict] = {}
    for tkey, cdict in cost_by_temp.items():
        usd_ctrl = float(cdict.get("usd_control", 0.0))
        usd_trt = float(cdict.get("usd_treatment", 0.0))
        usd_diff = usd_trt - usd_ctrl
        m_ctrl = means.get(tkey, {}).get("control", {})
        m_trt = means.get(tkey, {}).get("treatment", {})
        deltas = {k: float(m_trt.get(k, 0.0)) - float(m_ctrl.get(k, 0.0)) for k in set(list(m_ctrl.keys()) + list(m_trt.keys()))}
        out[tkey] = {
            "usd_control": usd_ctrl,
            "usd_treatment": usd_trt,
            "usd_diff": usd_diff,
            "delta_metrics": deltas,
            "usd_per_1pp_gain_em": per_point_cost(deltas.get("em", 0.0), usd_diff),
            "usd_per_1pp_gain_f1": per_point_cost(deltas.get("f1", 0.0), usd_diff),
            "usd_per_1pp_reduction_false_answer": per_point_cost(-deltas.get("false_answer_rate", 0.0), usd_diff),
        }

    os.makedirs(os.path.dirname(args.out_path), exist_ok=True)
    with open(args.out_path, "w", encoding="utf-8") as f:
        json.dump({"pricing": {"input_per_m": ipm, "output_per_m": opm, "batch_discount": disc}, "results": out}, f, indent=2)
    print("Wrote", args.out_path)


if __name__ == "__main__":
    main()
