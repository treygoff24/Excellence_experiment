from __future__ import annotations
import os
import json
import argparse
import csv


def parse_custom_id(cid: str):
    parts = cid.split("|")
    if len(parts) != 6: raise ValueError(f"Bad custom_id: {cid}")
    dataset, item_id, condition, temp_str, sample_str, typ = parts
    return dataset, item_id, condition, float(temp_str), int(sample_str), typ


def iter_jsonl(path: str):
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line: yield json.loads(line)


def extract_text_from_body(body: dict) -> str:
    # Support both shapes:
    # - OpenAI-compatible: response has choices/message directly
    # - Wrapped shape: response.body.choices[0].message
    try:
        choices = body.get("choices") or []
        if choices and isinstance(choices[0], dict):
            msg = choices[0].get("message", {}) or {}
            return (msg.get("content") or "").strip()
    except Exception:
        pass
    if "message" in body and isinstance(body["message"], dict):
        return (body["message"].get("content") or "").strip()
    if isinstance(body.get("content"), str):
        return (body.get("content") or "").strip()
    return ""


def process_results(results_jsonl: str, out_csv: str):
    fieldnames = ["custom_id", "dataset", "item_id", "condition", "temp", "sample_index", "type", "request_id", "finish_reason", "response_text", "prompt_tokens", "completion_tokens", "total_tokens"]
    os.makedirs(os.path.dirname(out_csv), exist_ok=True)
    with open(out_csv, "w", encoding="utf-8", newline="") as fcsv:
        w = csv.DictWriter(fcsv, fieldnames=fieldnames, quoting=csv.QUOTE_MINIMAL, escapechar='\\')
        w.writeheader()
        for row in iter_jsonl(results_jsonl):
            cid = row.get("custom_id") or row.get("customId")
            resp = row.get("response") or {}
            # Fireworks responses may put payload directly under `response`, or under `response.body`
            body = resp.get("body") or resp or {}
            # Usage may appear either at resp-level or inside body
            usage = resp.get("usage") or body.get("usage") or {}
            finish = None
            try:
                choices = body.get("choices") or []
                if choices and isinstance(choices[0], dict):
                    finish = choices[0].get("finish_reason")
            except Exception:
                pass
            # Prefer explicit request_id, else fall back to generic id fields
            request_id = resp.get("request_id") or resp.get("id") or body.get("id")
            text = extract_text_from_body(body)
            dataset, item_id, condition, temp, idx, typ = parse_custom_id(cid)
            w.writerow({
                "custom_id": cid,
                "dataset": dataset,
                "item_id": item_id,
                "condition": condition,
                "temp": temp,
                "sample_index": idx,
                "type": typ,
                "request_id": request_id,
                "finish_reason": finish,
                "response_text": text,
                "prompt_tokens": usage.get("prompt_tokens"),
                "completion_tokens": usage.get("completion_tokens"),
                "total_tokens": usage.get("total_tokens"),
            })


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results_jsonl", required=True)
    ap.add_argument("--out_csv", default="results/predictions.csv")
    args = ap.parse_args()
    process_results(args.results_jsonl, args.out_csv)
    print("Wrote", args.out_csv)


if __name__ == "__main__":
    main()
