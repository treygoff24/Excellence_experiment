from __future__ import annotations
import os, csv, json, argparse, math
from collections import defaultdict
from typing import Dict, List, Tuple, Any

import numpy as np
from scipy import stats

from config.schema import load_config


def load_per_item(path: str) -> list[dict]:
    rows: list[dict] = []
    with open(path, "r", encoding="utf-8") as f:
        r = csv.DictReader(f)
        for row in r:
            try:
                row["temp"] = float(row["temp"])  # type: ignore[index]
            except Exception:
                continue
            for k in ["em", "f1", "abstain_rate", "false_answer_rate", "unsupported_rate"]:
                if k in row and row[k] not in (None, "", "nan"):
                    try:
                        row[k] = float(row[k])
                    except Exception:
                        row[k] = 0.0
            rows.append(row)
    return rows


def _binom_two_sided_p(b: int, c: int) -> float:
    n = b + c
    if n == 0:
        return 1.0
    k = min(b, c)
    try:
        res = stats.binomtest(k, n, 0.5, alternative="two-sided")
        return float(res.pvalue)
    except Exception:
        # Fallback: sum of tail probabilities
        # Compute p = sum_{i<=k} C(n,i) 0.5^n; double and cap at 1
        from math import comb
        tail = sum(comb(n, i) for i in range(0, k + 1)) / (2 ** n)
        p = min(1.0, 2.0 * tail)
        return float(p)


def _baptista_pike_or_ci(b: int, c: int, alpha: float = 0.05) -> tuple[float, float | None, float | None]:
    # Odds ratio estimate with 0.5 continuity
    or_est = (b + 0.5) / (c + 0.5)
    n = b + c
    if n == 0:
        return float("nan"), None, None
    # Exact binomial (Clopper–Pearson) CI for p=b/n then transform to OR=p/(1-p)
    # Handle boundary cases gracefully
    try:
        lo_p = stats.beta.ppf(alpha / 2.0, max(b, 0), c + 1) if b > 0 else 0.0
        hi_p = stats.beta.ppf(1 - alpha / 2.0, b + 1, max(c, 0)) if c > 0 else 1.0
    except Exception:
        lo_p, hi_p = 0.0, 1.0
    def to_or(p: float) -> float:
        if p <= 0.0:
            return 0.0
        if p >= 1.0:
            return float("inf")
        return p / (1.0 - p)
    lo_or = to_or(lo_p)
    hi_or = to_or(hi_p)
    return float(or_est), float(lo_or), float(hi_or)


def _majority(x: float, thr: float = 0.5) -> int:
    return 1 if float(x) >= thr else 0


def _paired_lists(rows: list[dict]) -> Dict[Tuple[float, str], list[Tuple[str, dict, dict]]]:
    """Return mapping keyed by (temp, type) -> list of (item_key, ctrl_row, trt_row)."""
    by_key = defaultdict(list)
    for r in rows:
        by_key[(r.get("item_key"), r.get("type"), float(r.get("temp")))].append(r)
    paired: Dict[Tuple[float, str], list[Tuple[str, dict, dict]]] = defaultdict(list)
    for (item_key, typ, temp), lst in by_key.items():
        if not item_key or not typ:
            continue
        r_ctrl = next((x for x in lst if x.get("condition") == "control"), None)
        r_trt = next((x for x in lst if x.get("condition") == "treatment"), None)
        if r_ctrl is None or r_trt is None:
            continue
        paired[(float(temp), str(typ))].append((str(item_key), r_ctrl, r_trt))
    return paired


def _bootstrap_ci(deltas: np.ndarray, B: int, rng: np.random.Generator, alpha: float = 0.05) -> tuple[float, float, list[float]]:
    n = len(deltas)
    if n == 0:
        return 0.0, 0.0, []
    if n == 1:
        return float(deltas[0]), float(deltas[0]), [float(deltas[0])]
    idx = np.arange(n)
    boots = np.empty(B, dtype=float)
    for i in range(B):
        sample_idx = rng.choice(idx, size=n, replace=True)
        boots[i] = float(np.mean(deltas[sample_idx]))
    lo, hi = float(np.percentile(boots, 100 * (alpha / 2.0))), float(np.percentile(boots, 100 * (1 - alpha / 2.0)))
    return lo, hi, boots.tolist()


def _wilcoxon_p(deltas: np.ndarray) -> tuple[float, float]:
    non_zero = deltas[np.abs(deltas) > 1e-12]
    if non_zero.size == 0:
        return 0.0, 1.0
    if non_zero.size == 1:
        return 1.0, 1.0
    try:
        W, p = stats.wilcoxon(non_zero, zero_method="wilcox", alternative="two-sided")
        return float(W), float(p)
    except Exception:
        return float("nan"), float("nan")


def _hodges_lehmann(deltas: np.ndarray) -> float:
    n = deltas.size
    if n == 0:
        return 0.0
    if n > 2000:
        # Avoid O(n^2) for very large n; use median as approximation
        return float(np.median(deltas))
    vals = []
    for i in range(n):
        for j in range(i, n):
            vals.append(0.5 * (float(deltas[i]) + float(deltas[j])))
    return float(np.median(np.array(vals))) if vals else 0.0


def _cohens_d(deltas: np.ndarray) -> float:
    if deltas.size == 0:
        return 0.0
    mu = float(np.mean(deltas))
    sd = float(np.std(deltas, ddof=1)) if deltas.size > 1 else 0.0
    if sd == 0.0:
        return 0.0
    return mu / sd


def _cliffs_delta(deltas: np.ndarray) -> float:
    if deltas.size == 0:
        return 0.0
    pos = int(np.sum(deltas > 0))
    neg = int(np.sum(deltas < 0))
    n = pos + neg
    if n == 0:
        return 0.0
    return float((pos - neg) / n)


def _perm_p_value(deltas: np.ndarray, R: int, rng: np.random.Generator) -> float:
    if deltas.size == 0:
        return 1.0
    obs = abs(float(np.mean(deltas)))
    if obs == 0.0 or R <= 0:
        return 1.0
    n = deltas.size
    ge = 0
    for _ in range(R):
        signs = rng.choice(np.array([-1.0, 1.0]), size=n, replace=True)
        val = abs(float(np.mean(signs * deltas)))
        if val >= obs - 1e-15:
            ge += 1
    return float(ge / R)


def compute_stats(per_item_csv: str, cfg: dict) -> dict:
    rows = load_per_item(per_item_csv)
    paired = _paired_lists(rows)
    rng = np.random.default_rng(int(cfg.get("stats", {}).get("random_seed", 1337)))
    B = int(cfg.get("stats", {}).get("bootstrap_samples", 5000))
    Rperm = int(cfg.get("stats", {}).get("permutation_samples", 5000))
    enable_perm = bool(cfg.get("stats", {}).get("enable_permutation", True))

    def compute_block(items: list[Tuple[str, dict, dict]], typ: str) -> dict:
        # Build metric deltas per item
        metrics = ["em", "f1", "abstain_rate", "false_answer_rate", "unsupported_rate"]
        if typ == "closed":
            metrics = ["em", "abstain_rate"]
        deltas_by_metric: dict[str, list[float]] = {m: [] for m in metrics}
        b = 0
        c = 0
        # For selective risk curves we also collect per-condition aggregates
        ctrl_em_vals: list[float] = []
        ctrl_abst_vals: list[float] = []
        trt_em_vals: list[float] = []
        trt_abst_vals: list[float] = []

        for item_key, r_ctrl, r_trt in items:
            for m in metrics:
                if m in r_ctrl and m in r_trt and (r_ctrl[m] not in (None, "") and r_trt[m] not in (None, "")):
                    try:
                        deltas_by_metric[m].append(float(r_trt[m]) - float(r_ctrl[m]))
                    except Exception:
                        pass
            # Collect condition-specific values for risk/coverage
            try:
                ctrl_em_vals.append(float(r_ctrl.get("em", 0.0)))
                ctrl_abst_vals.append(float(r_ctrl.get("abstain_rate", 0.0)))
                trt_em_vals.append(float(r_trt.get("em", 0.0)))
                trt_abst_vals.append(float(r_trt.get("abstain_rate", 0.0)))
            except Exception:
                pass
            ctrl_bin = _majority(float(r_ctrl.get("em", 0.0)))
            trt_bin = _majority(float(r_trt.get("em", 0.0)))
            if ctrl_bin == 0 and trt_bin == 1:
                b += 1
            elif ctrl_bin == 1 and trt_bin == 0:
                c += 1
        p_exact = _binom_two_sided_p(b, c)
        or_est, or_lo, or_hi = _baptista_pike_or_ci(b, c)
        metrics_out: dict = {}
        for m, dl in deltas_by_metric.items():
            arr = np.array(dl, dtype=float)
            mean_delta = float(np.mean(arr)) if arr.size else 0.0
            lo_ci, hi_ci, _ = _bootstrap_ci(arr, B, rng)
            W, p_wil = _wilcoxon_p(arr)
            hl = _hodges_lehmann(arr)
            d = _cohens_d(arr)
            cd = _cliffs_delta(arr)
            p_perm = _perm_p_value(arr, Rperm, rng) if enable_perm else None
            metrics_out[m] = {
                "n_items": int(len(arr)),
                "delta_mean": mean_delta,
                "ci_95": [lo_ci, hi_ci],
                "wilcoxon": {"W": W, "p_value": p_wil},
                "hodges_lehmann": hl,
                "cohens_d": d,
                "cliffs_delta": cd,
                "perm_p_value": p_perm,
            }
        # Selective risk curves
        thresholds: list[float] = list(cfg.get("stats", {}).get("risk_thresholds", [0.0, 0.5, 1.0]))
        def _risk_points(em_list: list[float], abst_list: list[float]) -> list[dict]:
            pts: list[dict] = []
            # assume same K across items; compute pooled metrics via sums
            K = 1.0  # cancels out since we use means
            for thr in thresholds:
                mask = [ab <= thr for ab in abst_list]
                if not any(mask):
                    pts.append({"threshold": float(thr), "coverage": 0.0, "selective_accuracy": None, "risk": None})
                    continue
                sum_em = sum(em for em, keep in zip(em_list, mask) if keep) * K
                sum_nonabst = sum((1.0 - ab) for ab, keep in zip(abst_list, mask) if keep) * K
                coverage = sum_nonabst / max(1e-12, len([1 for k in mask if k]))  # relative per included item
                if sum_nonabst <= 0:
                    sel_acc = None
                    risk = None
                else:
                    sel_acc = float(sum_em / sum_nonabst)
                    risk = float(1.0 - sel_acc)
                pts.append({"threshold": float(thr), "coverage": float(coverage), "selective_accuracy": sel_acc, "risk": risk})
            return pts

        ctrl_pts = _risk_points(ctrl_em_vals, ctrl_abst_vals)
        trt_pts = _risk_points(trt_em_vals, trt_abst_vals)
        def _aurc(points: list[dict]) -> float | None:
            # Trapezoid area over coverage vs risk; skip None
            xs = [p["coverage"] for p in points if p["coverage"] is not None and p["risk"] is not None]
            ys = [p["risk"] for p in points if p["coverage"] is not None and p["risk"] is not None]
            if len(xs) < 2:
                return None
            # ensure sorted by coverage
            pairs = sorted(zip(xs, ys), key=lambda z: z[0])
            area = 0.0
            for (x0, y0), (x1, y1) in zip(pairs[:-1], pairs[1:]):
                area += (x1 - x0) * (y0 + y1) / 2.0
            return float(area)

        selective_risk = {
            "thresholds": thresholds,
            "points": {
                "control": ctrl_pts,
                "treatment": trt_pts,
            },
            "aurc": {
                "control": _aurc(ctrl_pts),
                "treatment": _aurc(trt_pts),
            },
        }

        return {
            "n_items": int(len(items)),
            "mcnemar": {
                "b": int(b),
                "c": int(c),
                "p_exact": float(p_exact),
                "odds_ratio": float(or_est) if not math.isnan(or_est) else None,
                "or_ci_95": [or_lo, or_hi],
            },
            "metrics": metrics_out,
            "selective_risk": selective_risk,
        }

    # Build overall and subgroup (by dataset) results
    out: dict[str, Any] = {}
    tests: list[Tuple[float, str, str, str, dict, str]] = []
    # tuple: (temp, typ, scope, metric, dict_ref, p_key)

    for (temp, typ), items in sorted(paired.items(), key=lambda x: (x[0][0], x[0][1])):
        tkey = f"{float(temp):.1f}"
        block = compute_block(items, typ)
        out.setdefault(tkey, {})[typ] = block
        # Register tests for FDR
        # McNemar
        tests.append((float(temp), typ, "overall", "mcnemar", block["mcnemar"], "p_exact"))
        # Metrics (use Wilcoxon p)
        for m, info in block.get("metrics", {}).items():
            tests.append((float(temp), typ, "overall", m, info["wilcoxon"], "p_value"))

        # Subgroups by dataset
        by_ds: Dict[str, list[Tuple[str, dict, dict]]] = defaultdict(list)
        for item_key, rc, rt in items:
            ds = str(item_key).split("|", 1)[0]
            by_ds[ds].append((item_key, rc, rt))
        if by_ds:
            out[tkey][typ]["subgroups"] = {"dataset": {}}
            for ds, sub_items in sorted(by_ds.items()):
                sub_block = compute_block(sub_items, typ)
                out[tkey][typ]["subgroups"]["dataset"][ds] = sub_block
                # Register subgroup tests for FDR
                tests.append((float(temp), typ, f"dataset:{ds}", "mcnemar", sub_block["mcnemar"], "p_exact"))
                for m, info in sub_block.get("metrics", {}).items():
                    tests.append((float(temp), typ, f"dataset:{ds}", m, info["wilcoxon"], "p_value"))

    # Apply Benjamini–Hochberg FDR if enabled
    if bool(cfg.get("stats", {}).get("enable_fdr", True)) and tests:
        # Collect p-values
        vals = []
        for i, (_t, _typ, _scope, _metric, ref, pkey) in enumerate(tests):
            try:
                p = float(ref.get(pkey))
            except Exception:
                p = float("nan")
            if not (p is None or math.isnan(p)):
                vals.append((i, p))
        m = len(vals)
        if m > 0:
            vals_sorted = sorted(vals, key=lambda x: x[1])
            # Compute BH q-values
            q_tmp = [0.0] * m
            min_q = 1.0
            for rank, (idx, p) in enumerate(vals_sorted, start=1):
                q = (m / rank) * p
                if q < min_q:
                    min_q = q
                q_tmp[rank - 1] = min_q
            # Assign back to each test
            for (rank, (idx, _p)) in enumerate(vals_sorted, start=1):
                qval = min(1.0, q_tmp[rank - 1])
                ref = tests[idx][4]
                # Attach q_value alongside the p-value container
                ref["q_value"] = float(qval)

    # TOST non-inferiority for EM/F1
    def _tost_noninferiority(deltas: List[float], margin: float, alpha: float) -> dict:
        arr = np.array([float(x) for x in deltas], dtype=float)
        n = arr.size
        if n == 0:
            return {"margin": margin, "alpha": alpha, "p_value": None, "non_inferior": None, "mean_delta": 0.0, "ci_95": [0.0, 0.0]}
        mean = float(np.mean(arr))
        sd = float(np.std(arr, ddof=1)) if n > 1 else 0.0
        if sd == 0.0:
            # If all deltas equal d, then if d > -margin we declare non-inferiority with p=0, else p=1
            pval = 0.0 if mean > -margin else 1.0
        else:
            se = sd / math.sqrt(n)
            t_stat = (mean + margin) / se  # H0: mean <= -margin vs H1: mean > -margin
            pval = 1.0 - float(stats.t.cdf(t_stat, df=n - 1))
        # 95% CI for mean difference
        try:
            tcrit = float(stats.t.ppf(1 - 0.5 * 0.05, df=n - 1))
        except Exception:
            tcrit = 1.96
        lo = mean - tcrit * (sd / math.sqrt(n) if n > 1 else 0.0)
        hi = mean + tcrit * (sd / math.sqrt(n) if n > 1 else 0.0)
        return {"margin": margin, "alpha": alpha, "p_value": float(pval), "non_inferior": bool(pval < alpha), "mean_delta": mean, "ci_95": [float(lo), float(hi)]}

    # Build deltas for TOST per temp/type for em and f1
    margins = cfg.get("stats", {}).get("tost_margins", {"em": 0.01, "f1": 0.01})
    alpha = float(cfg.get("stats", {}).get("tost_alpha", 0.05))
    # Recompute paired lists to capture deltas by temp/type
    for (temp, typ), items in sorted(paired.items(), key=lambda x: (x[0][0], x[0][1])):
        tkey = f"{float(temp):.1f}"
        em_d = []
        f1_d = []
        for _item_key, rc, rt in items:
            try:
                em_d.append(float(rt.get("em", 0.0)) - float(rc.get("em", 0.0)))
                if typ == "open":
                    f1_d.append(float(rt.get("f1", 0.0)) - float(rc.get("f1", 0.0)))
            except Exception:
                pass
        out[tkey][typ]["tost"] = {}
        out[tkey][typ]["tost"]["em"] = _tost_noninferiority(em_d, float(margins.get("em", 0.01)), alpha)
        if typ == "open":
            out[tkey][typ]["tost"]["f1"] = _tost_noninferiority(f1_d, float(margins.get("f1", 0.01)), alpha)

    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--per_item_csv", default="results/per_item_scores.csv")
    ap.add_argument("--config", default="config/eval_config.yaml")
    ap.add_argument("--out_path", default="results/significance.json")
    args = ap.parse_args()
    cfg = load_config(args.config)
    results = compute_stats(args.per_item_csv, cfg)
    os.makedirs(os.path.dirname(args.out_path), exist_ok=True)
    with open(args.out_path, "w", encoding="utf-8") as f:
        json.dump({
            "schema_version": 2,
            "temps": [float(t) for t in cfg.get("temps", [])],
            "stats_config": cfg.get("stats", {}),
            "results": results,
        }, f, indent=2)
    print("Wrote", args.out_path)
    
if __name__ == "__main__":
    main()
