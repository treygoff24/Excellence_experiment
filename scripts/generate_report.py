from __future__ import annotations

import os
import csv
import json
import argparse
from typing import Dict, Tuple
from math import sqrt
from scipy.stats import t as student_t

from config.schema import load_config
from scripts import manifest_v2 as mf


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

    # Add data balance information
    lines.append("")
    lines.append("## Data Balance")
    lines.append("")

    if counts:
        # Calculate condition totals
        control_total = sum(count for (temp, cond, typ), count in counts.items() if cond == "control")
        treatment_total = sum(count for (temp, cond, typ), count in counts.items() if cond == "treatment")

        lines.append(f"- Control items: {control_total:,}")
        lines.append(f"- Treatment items: {treatment_total:,}")

        if control_total > 0 and treatment_total > 0:
            ratio = max(control_total, treatment_total) / min(control_total, treatment_total)
            lines.append(f"- Ratio: {ratio:.1f}:1")
            if ratio > 5.0:
                lines.append("- ⚠️  **Warning**: Severe data imbalance detected")
            elif ratio > 2.0:
                lines.append("- ℹ️  **Note**: Moderate data imbalance")
        elif control_total == 0:
            lines.append("- ❌ **Error**: No control data found")
        elif treatment_total == 0:
            lines.append("- ❌ **Error**: No treatment data found")

        # Calculate pairing rate from significance results
        if significance and "results" in significance and counts:
            # Determine the number of unique items evaluated per type
            items_by_type: dict[str, list[int]] = {}
            for (_, _, typ), count in counts.items():
                items_by_type.setdefault(typ, []).append(count)

            # Extract paired counts by type from significance outputs
            paired_by_type: dict[str, int] = {}
            for temp_results in significance["results"].values():
                for typ_key, typ_results in temp_results.items():
                    if isinstance(typ_results, dict) and "n_items" in typ_results:
                        paired_by_type[typ_key] = max(paired_by_type.get(typ_key, 0), int(typ_results["n_items"]))

            total_items = sum(max(vals) for vals in items_by_type.values() if vals)
            total_paired = sum(paired_by_type.get(typ, 0) for typ in items_by_type.keys())
            pairing_rate = (total_paired / total_items) if total_items else 0.0

            lines.append("")
            lines.append("**Pairing Analysis:**")
            lines.append(f"- Total items: {total_items:,}")
            lines.append(f"- Paired items: {total_paired:,} ({pairing_rate:.1%})")
            if pairing_rate < 0.9:
                lines.append("- ⚠️  **Warning**: Low pairing rate suggests many items lack matches")
            lines.append("- Statistical analysis uses only paired items for valid comparisons")
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
                        tcrit = float(student_t.ppf(0.975, df=n - 1))
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
        sig = significance
        # Support schema v2 {results: {"1.0": {open:{...}, closed:{...}}}} and legacy keyed by temp
        results_v2 = sig.get("results") if isinstance(sig, dict) else None
        if results_v2:
            tkey = f"{float(temp):.1f}"
            s_obj = results_v2.get(tkey, {}) if isinstance(results_v2, dict) else {}
            if s_obj:
                lines.append("")
                lines.append("### Significance")
                lines.append("")
                for typ in ("closed", "open"):
                    if typ not in s_obj:
                        continue
                    lines.append(f"- {typ.capitalize()}-book:")
                    mcn = s_obj[typ].get("mcnemar", {})
                    or_part = ""
                    try:
                        or_est = mcn.get("odds_ratio")
                        or_ci = mcn.get("or_ci_95") or [None, None]
                        if or_est is not None and or_ci:
                            or_part = f", OR={or_est:.3f} CI95=[{or_ci[0]}, {or_ci[1]}]"
                    except Exception:
                        pass
                    qv = mcn.get("q_value")
                    q_part = f", q={qv:.4f}" if isinstance(qv, (int, float)) else ""
                    lines.append(f"  - McNemar: b={mcn.get('b')}, c={mcn.get('c')}, p_exact={mcn.get('p_exact')}{or_part}{q_part}")
                    # Deltas with CI
                    met = s_obj[typ].get("metrics", {})
                    if met:
                        lines.append("  - Deltas (Treatment − Control) with 95% CI:")
                        order = ["em", "f1", "abstain_rate", "false_answer_rate", "unsupported_rate"]
                        for k in order:
                            if k not in met:
                                continue
                            try:
                                dm = float(met[k].get("delta_mean", 0.0))
                            except Exception:
                                dm = 0.0
                            ci = met[k].get("ci_95") or [None, None]
                            qv = (met[k].get("wilcoxon", {}) or {}).get("q_value")
                            q_part = f", q={qv:.4f}" if isinstance(qv, (int, float)) else ""
                            lines.append(f"    - {k}: Δ={dm:+.4f} CI95=[{ci[0]}, {ci[1]}]{q_part}")
                    # Selective risk summary
                    sel = s_obj[typ].get("selective_risk", {})
                    if sel:
                        aurc = sel.get("aurc", {})
                        if aurc:
                            lines.append(
                                f"  - Risk–Coverage AURC: Control={aurc.get('control')}, Treatment={aurc.get('treatment')}"
                            )
                        pts = sel.get("points", {})
                        if pts:
                            ctrl_pts = pts.get("control") or []
                            trt_pts = pts.get("treatment") or []
                            if ctrl_pts and trt_pts:
                                lines.append("  - Risk–Coverage points (threshold, coverage, risk):")

                                def _fmt(pt):
                                    return f"(τ={pt.get('threshold')}, cov={pt.get('coverage')}, risk={pt.get('risk')})"
                                lines.append(
                                    "    - Control: " + ", ".join(_fmt(p) for p in ctrl_pts)
                                )
                                lines.append(
                                    "    - Treatment: " + ", ".join(_fmt(p) for p in trt_pts)
                                )
                    # TOST non-inferiority
                    tost = s_obj[typ].get("tost", {})
                    if tost:
                        lines.append("  - Non-inferiority (TOST):")
                        for key in ("em", "f1"):
                            if key in tost:
                                ti = tost[key]
                                lines.append(
                                    f"    - {key.upper()}: margin={ti.get('margin')}, p={ti.get('p_value')}, non_inferior={ti.get('non_inferior')} (Δ={ti.get('mean_delta'):+.4f}, CI95=[{ti.get('ci_95',[None,None])[0]}, {ti.get('ci_95',[None,None])[1]}])"
                                )
                    # Meta-analysis across datasets (fixed/random effects)
                    meta = s_obj[typ].get("meta", {})
                    if meta:
                        lines.append("  - Meta-analysis (across datasets):")
                        for key in ("em", "f1"):
                            if key in meta:
                                m = meta[key]
                                fe = m.get("fixed", {}) or {}
                                re = m.get("random", {}) or {}
                                het = m.get("heterogeneity", {}) or {}
                                lines.append(
                                    f"    - {key.upper()} (fixed): Δ={fe.get('delta_mean', 0.0):+.4f} CI95=[{fe.get('ci_95', [None,None])[0]}, {fe.get('ci_95', [None,None])[1]}]"
                                )
                                lines.append(
                                    f"      {key.upper()} (random): Δ={re.get('delta_mean', 0.0):+.4f} CI95=[{re.get('ci_95', [None,None])[0]}, {re.get('ci_95', [None,None])[1]}] (τ²={re.get('tau2')})"
                                )
                                if het:
                                    lines.append(
                                        f"      Heterogeneity: Q={het.get('Q')}, df={het.get('df')}, p={het.get('p_value')}, I²={het.get('I2')}"
                                    )
                # Subgroups by dataset
                for typ in ("closed", "open"):
                    if typ not in s_obj:
                        continue
                    sub = (s_obj.get(typ, {}) or {}).get("subgroups", {})
                    ds = (sub.get("dataset") or {}) if isinstance(sub, dict) else {}
                if ds:
                    lines.append(f"  - Subgroups ({typ}):")
                    for ds_name, ds_block in ds.items():
                        met = ds_block.get("metrics", {})
                        em = met.get("em", {})
                        ci = em.get("ci_95") or [None, None]
                        dm = em.get("delta_mean", 0.0)
                        qv = (em.get("wilcoxon", {}) or {}).get("q_value")
                        q_part = f", q={qv:.4f}" if isinstance(qv, (int, float)) else ""
                        lines.append(f"    - {ds_name}: EM Δ={dm:+.4f} CI95=[{ci[0]}, {ci[1]}]{q_part}")
        # Unsupported sensitivity (optional)
        sens_path = os.path.join(os.path.dirname(out_path).replace("reports", "results"), "unsupported_sensitivity.json")
        try:
            with open(sens_path, "r", encoding="utf-8") as f:
                sens = json.load(f)
        except Exception:
            sens = None
        if sens and isinstance(sens, dict):
            lines.append("")
            lines.append("### Unsupported Sensitivity")
            lines.append("")
            res = sens.get("results", {}) or {}
            tkey = f"{float(temp):.1f}"
            if tkey in res:
                lines.append("- Δ unsupported_rate across thresholds (Treatment − Control):")
                for thr_s, obj in sorted(res.get(tkey, {}).items(), key=lambda kv: float(kv[0])):
                    dm = obj.get("delta_mean", 0.0)
                    ci = obj.get("ci_95") or [None, None]
                    lines.append(f"  - τ={thr_s}: Δ={dm:+.4f} CI95=[{ci[0]}, {ci[1]}]")
        # Power/MDE (optional)
        power_path = os.path.join(os.path.dirname(out_path).replace("reports", "results"), "power_analysis.json")
        try:
            with open(power_path, "r", encoding="utf-8") as f:
                pwr = json.load(f)
        except Exception:
            pwr = None
        if pwr and isinstance(pwr, dict):
            lines.append("")
            lines.append("### Power / MDE (Primary Endpoint)")
            lines.append("")
            lines.append(f"- alpha={pwr.get('alpha')}, power={pwr.get('power')}, metric={pwr.get('metric')}")
            tkey = f"{float(temp):.1f}"
            res = (pwr.get("results") or {}).get(tkey)
            if res:
                lines.append(f"- @T={tkey}: n_items={res.get('n_items')}, sd_delta={res.get('sd_delta')}")
                lines.append(f"  - MDE (two-sided): {res.get('mde')}")
                rn = res.get("required_n_for_target", {}) or {}
                if rn:
                    pretty = ", ".join(f"Δ={k} → N={v}" for k, v in sorted(rn.items(), key=lambda kv: float(kv[0])))
                    lines.append(f"  - Required N for targets: {pretty}")
        # Mixed-effects / GEE robustness (optional)
        mixed_path = os.path.join(os.path.dirname(out_path).replace("reports", "results"), "mixed_models.json")
        try:
            with open(mixed_path, "r", encoding="utf-8") as f:
                mixed = json.load(f)
        except Exception:
            mixed = None
        if mixed and isinstance(mixed, dict):
            lines.append("")
            lines.append("### Mixed-Effects (Robustness)")
            lines.append("")
            if mixed.get("status") == "unavailable":
                reason = mixed.get('reason') or ((mixed.get('models') or {}).get('reason') if isinstance(mixed.get('models'), dict) else None)
                lines.append(f"- Skipped: {reason}")
            else:
                m = mixed.get("models", {}) or mixed
                logit = m.get("logistic_binary_correct", {})
                if logit:
                    lines.append("- Logistic (binary_correct via GEE):")
                    or_map = logit.get("odds_ratio", {}) or {}
                    or_ci = logit.get("or_ci", {}) or {}
                    key = "C(condition)[T.treatment]"
                    if key in or_map:
                        ci = or_ci.get(key) or [None, None]
                        lines.append(f"  - Treatment OR={or_map.get(key):.3f} CI95=[{ci[0]}, {ci[1]}]")
                lin = m.get("linear_f1_open", {})
                if lin:
                    lines.append("- Linear (F1, open-book):")
                    params = lin.get("params", {}) or {}
                    ci_all = lin.get("conf_int", {}) or {}
                    key = "C(condition)[T.treatment]"
                    if key in params:
                        ci = ci_all.get(key) or [None, None]
                        lines.append(f"  - Treatment Δ={params.get(key):+.4f} CI95=[{ci[0]}, {ci[1]}]")
        # Cost-effectiveness (optional)
        ce_path = os.path.join(os.path.dirname(out_path).replace("reports", "results"), "cost_effectiveness.json")
        try:
            with open(ce_path, "r", encoding="utf-8") as f:
                ce = json.load(f)
        except Exception:
            ce = None
        if ce and isinstance(ce, dict):
            lines.append("")
            lines.append("### Cost-Effectiveness")
            lines.append("")
            tkey = f"{float(temp):.1f}"
            row = (ce.get("results") or {}).get(tkey)
            if row:
                lines.append(f"- Δ Cost (Treatment − Control): ${row.get('usd_diff', 0.0):.4f}")
                lines.append(f"  - $ per 1pp EM gain: {row.get('usd_per_1pp_gain_em')}")
                lines.append(f"  - $ per 1pp F1 gain: {row.get('usd_per_1pp_gain_f1')}")
                lines.append(f"  - $ per 1pp false-answer reduction: {row.get('usd_per_1pp_reduction_false_answer')}")
        else:
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
    ap.add_argument("--force", action="store_true", help="Force regenerate report even if up-to-date")
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
    # Idempotency: if report exists and newer than inputs and not --force, skip
    newest_input = 0.0
    for p in [per_item_csv, sig_path, costs_path]:
        if os.path.exists(p):
            newest_input = max(newest_input, os.path.getmtime(p))
    if (not args.force) and os.path.isfile(out_path) and os.path.getmtime(out_path) >= newest_input:
        print("Idempotent skip: report.md up-to-date (use --force to regenerate)")
        _update_manifest_report(args.results_dir, out_path)
        return
    write_report(cfg, manifest, means, significance, costs, out_path, series=series, counts=counts)
    print("Wrote", out_path)
    _update_manifest_report(args.results_dir, out_path)


def _update_manifest_report(results_dir: str, report_path: str) -> None:
    manifest_path = os.path.join(results_dir, "trial_manifest.json")
    if os.path.isfile(manifest_path):
        try:
            mf.update_stage_status(
                manifest_path,
                "report",
                "completed",
                {"report_md": os.path.relpath(report_path, os.path.dirname(results_dir))},
            )
        except Exception:
            pass


if __name__ == "__main__":
    import multiprocessing as mp

    mp.freeze_support()
    main()
