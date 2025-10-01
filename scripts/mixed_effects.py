from __future__ import annotations

import os
import json
import csv
import argparse
from typing import Dict, Any, List

import numpy as np
import pandas as pd

from config.schema import load_config
from scoring import squad_v2, triviaqa, nq_open
from scoring.unsupported import is_unsupported


def load_jsonl(path: str):
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            s = line.strip()
            if s:
                yield json.loads(s)


def load_canonical(prepared_dir: str) -> dict:
    data = {"open": {}, "closed": {}}
    with open(os.path.join(prepared_dir, "open_book.jsonl"), "r", encoding="utf-8") as f:
        for line in f:
            row = json.loads(line)
            data["open"][f"{row['dataset']}|{row['id']}"] = row
    with open(os.path.join(prepared_dir, "closed_book.jsonl"), "r", encoding="utf-8") as f:
        for line in f:
            row = json.loads(line)
            data["closed"][f"{row['dataset']}|{row['id']}"] = row
    return data


def read_preds(pred_csv: str):
    with open(pred_csv, "r", encoding="utf-8") as f:
        r = csv.DictReader(f)
        for row in r:
            yield row


def build_long_table(pred_csv: str, prepared_dir: str, cfg: dict) -> pd.DataFrame:
    canon = load_canonical(prepared_dir)
    uns_cfg = cfg.get("unsupported", {})
    strategy = (uns_cfg.get("strategy") or "baseline").lower()
    threshold = float(uns_cfg.get("threshold", 0.5))
    params = {"min_token_overlap": float(uns_cfg.get("min_token_overlap", 0.6))}

    rows: List[Dict[str, Any]] = []
    for r in read_preds(pred_csv):
        item_key = f"{r['dataset']}|{r['item_id']}"
        typ = r.get("type") or "closed"
        temp = float(r.get("temp") or 0.0)
        condition = r.get("condition") or "control"
        sample_index = int(r.get("sample_index") or 0)
        pred = r.get("response_text") or ""
        abstained = 1.0 if squad_v2.is_abstention(pred) else 0.0
        if typ == "open":
            ex = canon["open"].get(item_key) or {}
            sc = squad_v2.score_item(pred, ex.get("answers") or [], bool(ex.get("is_unanswerable", False)))
            f1 = sc.get("f1", 0.0)
            em = sc.get("em", 0.0)
            unsupported = float(is_unsupported(pred, ex.get("context") or "", abstained=sc.get("abstained", abstained), strategy=strategy, threshold=threshold, params=params))
            false_ans = float(sc.get("false_answer", 0.0))
            unans = bool(ex.get("is_unanswerable", False))
            dataset = (ex.get("dataset") or item_key.split("|", 1)[0])
        else:
            ex = canon["closed"].get(item_key) or {}
            scorer = triviaqa if (ex.get("dataset") or "triviaqa") == "triviaqa" else nq_open
            sc = scorer.score_item(pred, ex.get("answers") or [])
            f1 = None
            em = sc.get("em", 0.0)
            unsupported = 0.0
            false_ans = 0.0
            unans = False
            dataset = (ex.get("dataset") or item_key.split("|", 1)[0])
        rows.append({
            "item_key": item_key,
            "dataset": dataset,
            "type": typ,
            "temp": temp,
            "condition": condition,
            "sample_index": sample_index,
            "binary_correct": 1 if float(em) >= 0.5 else 0,
            "em": float(em),
            "f1": float(f1) if f1 is not None else None,
            "abstained": float(abstained),
            "unsupported": float(unsupported),
            "false_answer": float(false_ans),
            "unanswerable": bool(unans),
        })
    return pd.DataFrame(rows)


def fit_models(df: pd.DataFrame) -> dict:
    out: dict = {"status": "ok"}
    try:
        import statsmodels.api as sm  # type: ignore
        import statsmodels.formula.api as smf  # type: ignore
    except Exception as e:  # pragma: no cover
        return {"status": "unavailable", "reason": f"{type(e).__name__}: {e}"}

    # Ensure proper dtypes
    df = df.copy()
    df["condition"] = df["condition"].astype("category")
    df["type"] = df["type"].astype("category")
    # Center temp to improve interpretability
    df["temp_c"] = df["temp"] - df["temp"].mean()

    models = {}
    # Logistic via GEE with clustering by item_key (robust SE)
    try:
        fam = sm.families.Binomial()
        cov = sm.cov_struct.Exchangeable()
        gee = smf.gee("binary_correct ~ C(condition) * temp_c", groups="item_key", data=df, family=fam, cov_struct=cov)
        res = gee.fit()
        params = res.params.to_dict()
        bse = res.bse.to_dict()
        conf = res.conf_int()
        conf_dict = {k: [float(conf.loc[k, 0]), float(conf.loc[k, 1])] for k in conf.index}
        or_params = {k: float(np.exp(v)) for k, v in params.items()}
        or_ci = {k: [float(np.exp(v[0])), float(np.exp(v[1]))] for k, v in conf_dict.items()}
        models["logistic_binary_correct"] = {
            "params": params,
            "bse": bse,
            "conf_int": conf_dict,
            "odds_ratio": or_params,
            "or_ci": or_ci,
            "n_obs": int(res.nobs),
        }
    except Exception as e:
        models["logistic_binary_correct"] = {"error": f"{type(e).__name__}: {e}"}

    # Linear model for F1 (open-book only) with cluster-robust SE by item_key
    try:
        dfo = df[df["type"] == "open"].dropna(subset=["f1"])
        if len(dfo) >= 5:
            ols = smf.ols("f1 ~ C(condition) * temp_c", data=dfo)
            res = ols.fit(cov_type="cluster", cov_kwds={"groups": dfo["item_key"]})
            params = res.params.to_dict()
            bse = res.bse.to_dict()
            conf = res.conf_int()
            conf_dict = {k: [float(conf.loc[k, 0]), float(conf.loc[k, 1])] for k in conf.index}
            models["linear_f1_open"] = {
                "params": params,
                "bse": bse,
                "conf_int": conf_dict,
                "n_obs": int(res.nobs),
            }
        else:
            models["linear_f1_open"] = {"skipped": "insufficient data"}
    except Exception as e:
        models["linear_f1_open"] = {"error": f"{type(e).__name__}: {e}"}

    out["models"] = models
    return out


def main():
    ap = argparse.ArgumentParser(description="Mixed-effects style robustness models (GEE/OLS)")
    ap.add_argument("--pred_csv", default="results/predictions.csv")
    ap.add_argument("--prepared_dir", default="data/prepared")
    ap.add_argument("--config", default="config/eval_config.yaml")
    ap.add_argument("--out_path", default="results/mixed_models.json")
    args = ap.parse_args()

    cfg = load_config(args.config)
    df = build_long_table(args.pred_csv, cfg["paths"]["prepared_dir"], cfg)
    models = fit_models(df)
    payload = {
        "status": models.get("status", "ok"),
        "n_rows": int(df.shape[0]),
        "n_items": int(df["item_key"].nunique()),
        "models": models.get("models") if "models" in models else models,
    }
    os.makedirs(os.path.dirname(args.out_path), exist_ok=True)
    with open(args.out_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)
    print("Wrote", args.out_path)


if __name__ == "__main__":
    import multiprocessing as mp

    mp.freeze_support()
    main()
