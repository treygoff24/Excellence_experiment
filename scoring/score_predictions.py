from __future__ import annotations
import os
import json
import csv
import argparse
from collections import defaultdict
from . import squad_v2, triviaqa, nq_open
from .unsupported import is_unsupported
from config.schema import load_config
from scripts import manifest_v2 as mf
from scripts.shared_controls import gather_control_entries, iter_control_results, count_control_results


def load_jsonl(path: str):
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line: yield json.loads(line)


def load_canonical(prepared_dir: str):
    data = {"open": {}, "closed": {}}
    for row in load_jsonl(os.path.join(prepared_dir, "open_book.jsonl")):
        data["open"][f"{row['dataset']}|{row['id']}"] = row
    for row in load_jsonl(os.path.join(prepared_dir, "closed_book.jsonl")):
        data["closed"][f"{row['dataset']}|{row['id']}"] = row
    return data


def read_preds_csv(path: str):
    with open(path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for r in reader:
            yield r


def aggregate_replicates(values):
    return sum(values) / len(values) if values else 0.0


def stdev(values):
    n = len(values)
    if n <= 1:
        return 0.0
    mean_val = sum(values) / n
    var = sum((x - mean_val) ** 2 for x in values) / (n - 1)
    return var ** 0.5


def _write_rows(rows, out_csv):
    import csv
    fieldnames = sorted({k for r in rows for k in r.keys()})
    with open(out_csv, "w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for r in rows: w.writerow(r)


def _load_prediction_rows(pred_csv: str, results_dir: str) -> list[dict]:
    rows = [dict(r) for r in read_preds_csv(pred_csv)]
    run_root, ctrl_entries, _manifest = gather_control_entries(results_dir)
    reuse_entries = [info for info in (ctrl_entries or {}).values() if isinstance(info, dict) and info.get("mode") == "reuse"]
    if not reuse_entries:
        return rows

    control_ids = {str(r.get("custom_id")) for r in rows if str(r.get("condition")) == "control" and r.get("custom_id")}
    expected_controls = count_control_results(run_root, reuse_entries)
    if expected_controls <= len(control_ids):
        return rows

    added = 0
    for entry in reuse_entries:
        for obj in iter_control_results(run_root, [entry]):
            cid = obj.get("custom_id") or obj.get("customId")
            if not cid or cid in control_ids:
                continue
            try:
                dataset, item_id, condition, temp_str, sample_idx_str, typ = tuple(str(cid).split("|"))
            except ValueError:
                continue
            try:
                temp_val = float(temp_str)
            except Exception:
                temp_val = 0.0
            try:
                sample_idx = int(sample_idx_str)
            except Exception:
                sample_idx = 0
            resp = obj.get("response") or {}
            body = resp.get("body") or resp or {}
            response_text = ""
            finish_reason = None
            try:
                choices = body.get("choices") or []
                if choices and isinstance(choices[0], dict):
                    finish_reason = choices[0].get("finish_reason")
                    msg = choices[0].get("message", {}) or {}
                    response_text = (msg.get("content") or "").strip()
            except Exception:
                pass
            usage = resp.get("usage") or body.get("usage") or {}
            rows.append({
                "custom_id": cid,
                "dataset": dataset,
                "item_id": item_id,
                "condition": condition,
                "temp": temp_val,
                "sample_index": sample_idx,
                "type": typ,
                "request_id": resp.get("request_id") or resp.get("id") or body.get("id"),
                "finish_reason": finish_reason,
                "response_text": response_text,
                "prompt_tokens": usage.get("prompt_tokens"),
                "completion_tokens": usage.get("completion_tokens"),
                "total_tokens": usage.get("total_tokens"),
            })
            control_ids.add(cid)
            added += 1
    if added:
        print(f"Rehydrated {added} shared control row(s) into per-item scoring inputs")
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pred_csv", default="results/predictions.csv")
    ap.add_argument("--prepared_dir", default="data/prepared")
    ap.add_argument("--out_dir", default="results")
    ap.add_argument("--config", default="config/eval_config.yaml")
    args = ap.parse_args()
    os.makedirs(args.out_dir, exist_ok=True)
    cfg = load_config(args.config)
    # Unsupported detection settings
    uns_cfg = (cfg.get("unsupported") or {})
    unsupported_threshold = float(uns_cfg.get("threshold", cfg.get("unsupported_threshold", 0.5)))
    unsupported_strategy = (uns_cfg.get("strategy") or "baseline").lower()
    unsupported_params = {"min_token_overlap": float(uns_cfg.get("min_token_overlap", 0.6))}
    canon = load_canonical(args.prepared_dir)
    bucket = defaultdict(lambda: defaultdict(list))
    abst_bucket = defaultdict(list)
    false_ans_bucket = defaultdict(list)
    unsupported_bucket = defaultdict(list)
    pred_rows = _load_prediction_rows(args.pred_csv, args.out_dir)

    for r in pred_rows:
        key_item = f"{r['dataset']}|{r['item_id']}"
        typ = r["type"]
        condition = r["condition"]
        temp = float(r["temp"])
        pred = r["response_text"] or ""
        key = (key_item, typ, condition, temp)
        if typ == "open":
            ex = canon["open"][key_item]
            score = squad_v2.score_item(pred, ex["answers"], ex["is_unanswerable"])
            # Improved unsupported detection
            unsupported = float(
                is_unsupported(
                    pred,
                    ex["context"],
                    abstained=score.get("abstained", 0.0),
                    strategy=unsupported_strategy,
                    threshold=unsupported_threshold,
                    params=unsupported_params,
                )
            )
            bucket[key]["em"].append(score["em"])
            bucket[key]["f1"].append(score["f1"])
            abst_bucket[key].append(score["abstained"])
            false_ans_bucket[key].append(score["false_answer"])
            unsupported_bucket[key].append(unsupported)
        else:
            ex = canon["closed"][key_item]
            scorer = triviaqa if ex["dataset"] == "triviaqa" else nq_open
            score = scorer.score_item(pred, ex["answers"])
            bucket[key]["em"].append(score["em"])
            abst_bucket[key].append(score["abstained"])
    rows = []
    for key, metrics in bucket.items():
        item_key, typ, condition, temp = key
        row = {"item_key": item_key, "type": typ, "condition": condition, "temp": temp, "em": aggregate_replicates(metrics.get("em", []))}
        row["em_std"] = stdev(metrics.get("em", []))
        if typ == "open":
            row["f1"] = aggregate_replicates(metrics.get("f1", []))
            row["f1_std"] = stdev(metrics.get("f1", []))
            row["false_answer_rate"] = aggregate_replicates(false_ans_bucket.get(key, []))
            row["false_answer_rate_std"] = stdev(false_ans_bucket.get(key, []))
            row["unsupported_rate"] = aggregate_replicates(unsupported_bucket.get(key, []))
            row["unsupported_rate_std"] = stdev(unsupported_bucket.get(key, []))
        row["abstain_rate"] = aggregate_replicates(abst_bucket.get(key, []))
        row["abstain_rate_std"] = stdev(abst_bucket.get(key, []))
        rows.append(row)
    out_csv = os.path.join(args.out_dir, "per_item_scores.csv")
    import csv
    fieldnames = sorted({k for r in rows for k in r.keys()})
    # Idempotency: if per_item_scores.csv exists and is newer than preds, skip
    preds_mtime = os.path.getmtime(args.pred_csv) if os.path.exists(args.pred_csv) else 0.0
    if os.path.isfile(out_csv) and os.path.getmtime(out_csv) >= preds_mtime:
        print(f"Idempotent skip: per_item_scores.csv up-to-date")
        _update_manifest_scored(out_csv)
        return
    with open(out_csv, "w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for r in rows: w.writerow(r)
    print("Wrote", out_csv)
    _update_manifest_scored(out_csv)


def _update_manifest_scored(per_item_csv: str) -> None:
    results_dir = os.path.dirname(per_item_csv)
    manifest_path = os.path.join(results_dir, "trial_manifest.json")
    if os.path.isfile(manifest_path):
        try:
            import csv as _csv
            n_rows = 0
            with open(per_item_csv, "r", encoding="utf-8") as f:
                r = _csv.reader(f)
                n_rows = max(0, sum(1 for _ in r) - 1)
            mf.update_stage_status(
                manifest_path,
                "scored",
                "completed",
                {"per_item_scores_csv": os.path.relpath(per_item_csv, results_dir), "row_count": int(n_rows)},
            )
        except Exception:
            pass


if __name__ == "__main__":
    main()
