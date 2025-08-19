from __future__ import annotations
import os, csv, argparse, json, yaml
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pred_csv", default="results/predictions.csv")
    ap.add_argument("--config", default="config/eval_config.yaml")
    ap.add_argument("--out_path", default="results/costs.json")
    args = ap.parse_args()
    with open(args.config, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    pricing = cfg.get("pricing", {})
    batch = cfg.get("use_batch_api", True)
    disc = pricing.get("batch_discount", 0.5) if batch else 1.0
    pt = ct = 0
    with open(args.pred_csv, "r", encoding="utf-8") as f:
        r = csv.DictReader(f)
        for row in r:
            pt += int(row["prompt_tokens"] or 0)
            ct += int(row["completion_tokens"] or 0)
    input_per_m = pricing.get("input_per_million", 0.15)
    output_per_m = pricing.get("output_per_million", 0.60)
    cost = (pt/1_000_000)*input_per_m + (ct/1_000_000)*output_per_m
    cost *= disc
    data = {"prompt_tokens": pt, "completion_tokens": ct, "total_tokens": pt+ct, "usd": cost, "batch_discount_applied": batch}
    with open(args.out_path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)
    print("Wrote", args.out_path)
if __name__ == "__main__":
    main()
