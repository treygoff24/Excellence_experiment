from __future__ import annotations
import os, csv, argparse
from collections import defaultdict
from math import sqrt
from scipy import stats
def load_per_item(path: str):
    rows = []
    with open(path, "r", encoding="utf-8") as f:
        r = csv.DictReader(f)
        for row in r:
            row["temp"] = float(row["temp"])
            for k in ["em","f1","abstain_rate","false_answer_rate","unsupported_rate"]:
                if k in row and row[k] not in (None, "", "nan"):
                    row[k] = float(row[k])
            rows.append(row)
    return rows
def mcnemar_p_value(b: int, c: int):
    # If there are no discordant pairs, there is no evidence against H0; return p=1.0
    if (b + c) == 0:
        return 1.0
    num = (abs(b - c) - 1)**2
    den = (b + c)
    chi2 = num / den
    return 1 - stats.chi2.cdf(chi2, df=1)
def majority(x: float, thr: float=0.5) -> int:
    return 1 if x >= thr else 0
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--per_item_csv", default="results/per_item_scores.csv")
    ap.add_argument("--metric", default="em", choices=["em","f1"])
    ap.add_argument("--out_path", default="results/significance.json")
    args = ap.parse_args()
    rows = load_per_item(args.per_item_csv)
    groups = defaultdict(list)
    for r in rows: groups[(r["item_key"], r["type"], r["temp"])].append(r)
    summaries = []
    for (item_key, typ, temp), lst in groups.items():
        if len(lst) != 2: continue
        r_ctrl = next((x for x in lst if x["condition"]=="control"), None)
        r_trt  = next((x for x in lst if x["condition"]=="treatment"), None)
        if not r_ctrl or not r_trt: continue
        metric = args.metric if args.metric in r_ctrl and args.metric in r_trt else "em"
        summaries.append({"key": (item_key, typ, temp), "ctrl": float(r_ctrl[metric]), "trt": float(r_trt[metric])})
    per_temp = defaultdict(list)
    for s in summaries: per_temp[s["key"][2]].append(s)
    results = {}
    for temp, lst in per_temp.items():
        b = c = 0; deltas = []
        for s in lst:
            ctrl_bin = majority(s["ctrl"]); trt_bin = majority(s["trt"])
            if ctrl_bin == 0 and trt_bin == 1: b += 1
            if ctrl_bin == 1 and trt_bin == 0: c += 1
            deltas.append(s["trt"] - s["ctrl"])
        p_mcn = mcnemar_p_value(b, c)
        # Guard against degenerate cases that trigger SciPy warnings (e.g., all-zero deltas)
        non_zero_deltas = [d for d in deltas if abs(d) > 1e-12]
        if len(non_zero_deltas) == 0:
            w, p_wil = 0.0, 1.0
        elif len(non_zero_deltas) == 1:
            # With a single non-zero difference, Wilcoxon is not informative; treat as no evidence
            w, p_wil = 1.0, 1.0
        else:
            try:
                w, p_wil = stats.wilcoxon(non_zero_deltas, zero_method="wilcox", alternative="two-sided")
            except ValueError:
                w, p_wil = float("nan"), float("nan")
        results[temp] = {"mcnemar": {"b": b, "c": c, "p_value": p_mcn}, "wilcoxon": {"W": w, "p_value": p_wil}, "mean_delta": sum(deltas)/len(deltas) if deltas else 0.0}
    with open(args.out_path, "w", encoding="utf-8") as f:
        import json; json.dump(results, f, indent=2)
    print("Wrote", args.out_path)
if __name__ == "__main__":
    main()
