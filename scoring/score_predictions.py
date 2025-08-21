from __future__ import annotations
import os, json, csv, argparse
from collections import defaultdict
from . import squad_v2, triviaqa, nq_open
from config.schema import load_config
def load_jsonl(path: str):
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line=line.strip()
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
        for r in reader: yield r
def aggregate_replicates(values):
    return sum(values)/len(values) if values else 0.0

def stdev(values):
    n = len(values)
    if n <= 1:
        return 0.0
    mean_val = sum(values) / n
    var = sum((x - mean_val) ** 2 for x in values) / (n - 1)
    return var ** 0.5
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pred_csv", default="results/predictions.csv")
    ap.add_argument("--prepared_dir", default="data/prepared")
    ap.add_argument("--out_dir", default="results")
    ap.add_argument("--config", default="config/eval_config.yaml")
    args = ap.parse_args()
    os.makedirs(args.out_dir, exist_ok=True)
    cfg = load_config(args.config)
    unsupported_threshold = float(cfg.get("unsupported_threshold", 0.5))
    canon = load_canonical(args.prepared_dir)
    bucket = defaultdict(lambda: defaultdict(list))
    abst_bucket = defaultdict(list)
    false_ans_bucket = defaultdict(list)
    unsupported_bucket = defaultdict(list)
    for r in read_preds_csv(args.pred_csv):
        key_item = f"{r['dataset']}|{r['item_id']}"
        typ = r["type"]; condition = r["condition"]; temp = float(r["temp"]); pred = r["response_text"] or ""
        key = (key_item, typ, condition, temp)
        if typ == "open":
            ex = canon["open"][key_item]
            score = squad_v2.score_item(pred, ex["answers"], ex["is_unanswerable"])
            from .normalize import normalize_answer
            ctx = normalize_answer(ex["context"]); pt = normalize_answer(pred)
            # Unsupported claim heuristic: penalize when confident non-abstention and
            # predicted content is not a substring of context by normalized tokens.
            unsupported = 0.0
            if (score["abstained"] <= (1.0 - unsupported_threshold)) and pt and (pt not in ctx):
                unsupported = 1.0
            bucket[key]["em"].append(score["em"]); bucket[key]["f1"].append(score["f1"])
            abst_bucket[key].append(score["abstained"]); false_ans_bucket[key].append(score["false_answer"]); unsupported_bucket[key].append(unsupported)
        else:
            ex = canon["closed"][key_item]
            scorer = triviaqa if ex["dataset"] == "triviaqa" else nq_open
            score = scorer.score_item(pred, ex["answers"])
            bucket[key]["em"].append(score["em"]); abst_bucket[key].append(score["abstained"])
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
    with open(out_csv, "w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames); w.writeheader()
        for r in rows: w.writerow(r)
    print("Wrote", out_csv)
if __name__ == "__main__":
    main()
