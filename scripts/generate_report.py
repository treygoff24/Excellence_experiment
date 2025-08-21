from __future__ import annotations

import os
import csv
import json
import argparse
from typing import Dict, Tuple
from math import sqrt
from scipy.stats import t as student_t

from config.schema import load_config


def read_manifest(path: str) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def compute_metric_means(per_item_csv_path: str) -> tuple[dict, dict, dict]:
    # returns (means_by_key, counts_by_key, series_by_key)
    aggregates: Dict[Tuple[float, str, str], Dict[str, float]] = {}
    counts: Dict[Tuple[float, str, str], int] = {}
    series: Dict[Tuple[float, str, str], Dict[str, list]] = {}
    with open(per_item_csv_path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            try:
                temp = float(row.get("temp") or 0.0)
            except Exception:
                temp = 0.0
            condition = row.get("condition") or ""
            typ = row.get("type") or ""
            key = (temp, condition, typ)
            if key not in aggregates:
                aggregates[key] = {
                    "em": 0.0,
                    "f1": 0.0,
                    "abstain_rate": 0.0,
                    "false_answer_rate": 0.0,
                    "unsupported_rate": 0.0,
                }
                counts[key] = 0
                series[key] = {"em": [], "f1": [], "abstain_rate": [], "false_answer_rate": [], "unsupported_rate": []}

            def _to_float(x):
                try:
                    return float(x)
                except Exception:
                    return 0.0

            aggregates[key]["em"] += _to_float(row.get("em"))
            series[key]["em"].append(_to_float(row.get("em")))
            aggregates[key]["f1"] += _to_float(row.get("f1"))
            if row.get("f1") not in (None, ""):
                series[key]["f1"].append(_to_float(row.get("f1")))
            aggregates[key]["abstain_rate"] += _to_float(row.get("abstain_rate"))
            series[key]["abstain_rate"].append(_to_float(row.get("abstain_rate")))
            aggregates[key]["false_answer_rate"] += _to_float(row.get("false_answer_rate"))
            if row.get("false_answer_rate") not in (None, ""):
                series[key]["false_answer_rate"].append(_to_float(row.get("false_answer_rate")))
            aggregates[key]["unsupported_rate"] += _to_float(row.get("unsupported_rate"))
            if row.get("unsupported_rate") not in (None, ""):
                series[key]["unsupported_rate"].append(_to_float(row.get("unsupported_rate")))
            counts[key] += 1

    means: Dict[Tuple[float, str, str], Dict[str, float]] = {}
    for key, sums in aggregates.items():
        n = max(1, counts.get(key, 1))
        means[key] = {k: (v / n) for k, v in sums.items()}
    return means, counts, series


def load_json(path: str) -> dict:
    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return {}


def format_percent_or_number(x: float) -> str:
    try:
        val = float(x)
    except Exception:
        val = 0.0
    if 0.0 <= val <= 1.0:
        return f"{val*100:.1f}%"
    return f"{val:.3f}"


def write_report(cfg: dict, manifest: dict, means: dict, significance: dict, costs: dict, out_path: str, series: dict | None = None, counts: dict | None = None) -> None:
    lines: list[str] = []
    lines.append("# Evaluation Report")
    lines.append("")
    lines.append(f"Model: {cfg['model_id']}")
    try:
        rid = manifest.get("run_id")
        if rid:
            lines.append(f"Run ID: {rid}")
    except Exception:
        pass
    lines.append(f"Temperatures: {', '.join(str(t) for t in cfg['temps'])}")
    # Report per-temp replicate counts, not hard-coded 0.7
    for t in cfg["temps"]:
        k = (
            cfg["samples_per_item"].get(str(float(t)))
            or cfg["samples_per_item"].get(f"{float(t):.1f}")
            or cfg["samples_per_item"].get(float(t))
            or "?"
        )
        lines.append(f"Samples per item @T={float(t):.1f}: {k}")
    lines.append("")
    lines.append("## Prompts")
    lines.append("")
    try:
        lines.append(f"- Control prompt tokens: {manifest['prompts']['control']['tokens']}")
        lines.append(f"- Treatment prompt tokens: {manifest['prompts']['treatment']['tokens']}")
    except Exception:
        pass
    lines.append("")

    for temp in cfg["temps"]:
        lines.append(f"## Results @ T={float(temp):.1f}")
        for typ in ("closed", "open"):
            lines.append("")
            lines.append(f"### {typ.capitalize()}-book")
            lines.append("")
            lines.append("| Condition | EM | F1 | Abstain | False-Ans | Unsupported |")
            lines.append("|---|---:|---:|---:|---:|---:|")
            row_ctrl = means.get((float(temp), "control", typ), {})
            row_trt = means.get((float(temp), "treatment", typ), {})
            def val(d, k):
                return d.get(k, 0.0)
            lines.append(
                f"| Control | {format_percent_or_number(val(row_ctrl,'em'))} | {format_percent_or_number(val(row_ctrl,'f1'))} | {format_percent_or_number(val(row_ctrl,'abstain_rate'))} | {format_percent_or_number(val(row_ctrl,'false_answer_rate'))} | {format_percent_or_number(val(row_ctrl,'unsupported_rate'))} |"
            )
            lines.append(
                f"| Treatment | {format_percent_or_number(val(row_trt,'em'))} | {format_percent_or_number(val(row_trt,'f1'))} | {format_percent_or_number(val(row_trt,'abstain_rate'))} | {format_percent_or_number(val(row_trt,'false_answer_rate'))} | {format_percent_or_number(val(row_trt,'unsupported_rate'))} |"
            )
            # 95% CIs over items (t-based) for EM (and F1 for open)
            if series is not None and counts is not None:
                def ci_for(key_tuple: tuple, metric: str) -> tuple[float, float] | None:
                    lst = (series.get(key_tuple, {}) or {}).get(metric) or []
                    n = len(lst)
                    if n <= 1:
                        return None
                    mean_val = sum(lst) / n
                    sd = (sum((x - mean_val) ** 2 for x in lst) / (n - 1)) ** 0.5
                    se = sd / sqrt(n)
                    try:
                        tcrit = float(student_t.ppf(0.975, df=n-1))
                    except Exception:
                        tcrit = 1.96
                    return mean_val - tcrit * se, mean_val + tcrit * se
                k_ctrl = (float(temp), "control", typ)
                k_trt = (float(temp), "treatment", typ)
                ci_em_ctrl = ci_for(k_ctrl, "em")
                ci_em_trt = ci_for(k_trt, "em")
                if ci_em_ctrl or ci_em_trt:
                    def fmt_ci(ci):
                        if not ci:
                            return "N/A"
                        lo, hi = ci
                        return f"[{format_percent_or_number(lo)}, {format_percent_or_number(hi)}]"
                    lines.append("")
                    lines.append(f"- EM 95% CI — Control: {fmt_ci(ci_em_ctrl)}; Treatment: {fmt_ci(ci_em_trt)}")
                if typ == "open":
                    ci_f1_ctrl = ci_for(k_ctrl, "f1")
                    ci_f1_trt = ci_for(k_trt, "f1")
                    if ci_f1_ctrl or ci_f1_trt:
                        lines.append(f"- F1 95% CI — Control: {fmt_ci(ci_f1_ctrl)}; Treatment: {fmt_ci(ci_f1_trt)}")
        # Significance per temp
        s = significance.get(str(float(temp))) or significance.get(float(temp)) or {}
        if s:
            lines.append("")
            lines.append("### Significance")
            lines.append("")
            m = s.get("mcnemar", {})
            w = s.get("wilcoxon", {})
            lines.append(f"- McNemar: b={m.get('b')}, c={m.get('c')}, p={m.get('p_value')}")
            lines.append(f"- Wilcoxon: W={w.get('W')}, p={w.get('p_value')}")
        lines.append("")

    if costs:
        lines.append("## Cost")
        lines.append("")
        lines.append(f"- Prompt tokens: {costs.get('prompt_tokens')}")
        lines.append(f"- Completion tokens: {costs.get('completion_tokens')}")
        lines.append(f"- Total tokens: {costs.get('total_tokens')}")
        try:
            usd_val = float(costs.get('usd') or 0.0)
        except Exception:
            usd_val = 0.0
        lines.append(f"- Estimated USD: ${usd_val:.4f}")
        if costs.get("batch_discount_applied"):
            lines.append("- Batch discount applied")
        lines.append("")

    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="config/eval_config.yaml")
    ap.add_argument("--results_dir", default="results")
    ap.add_argument("--reports_dir", default="reports")
    args = ap.parse_args()

    cfg = load_config(args.config)
    per_item_csv = os.path.join(args.results_dir, "per_item_scores.csv")
    sig_path = os.path.join(args.results_dir, "significance.json")
    costs_path = os.path.join(args.results_dir, "costs.json")
    manifest_path = cfg["paths"].get("run_manifest", os.path.join(args.results_dir, "run_manifest.json"))

    means, counts, series = compute_metric_means(per_item_csv)
    significance = load_json(sig_path)
    costs = load_json(costs_path)
    try:
        manifest = read_manifest(manifest_path)
    except Exception:
        manifest = {}

    out_path = os.path.join(args.reports_dir, "report.md")
    write_report(cfg, manifest, means, significance, costs, out_path, series=series, counts=counts)
    print("Wrote", out_path)


if __name__ == "__main__":
    main()


