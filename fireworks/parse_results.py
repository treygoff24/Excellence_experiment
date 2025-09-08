from __future__ import annotations
import os
import json
import argparse
import csv
from typing import Tuple

from scripts import manifest_v2 as mf


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
    # Idempotency: if up-to-date and counts match, skip
    results_mtime = os.path.getmtime(results_jsonl) if os.path.exists(results_jsonl) else 0.0
    if os.path.isfile(out_csv) and os.path.getmtime(out_csv) >= results_mtime:
        # Validate row count vs unique IDs in results
        expected = _count_unique_custom_ids(results_jsonl)
        actual = _count_rows(out_csv)
        if expected == actual and actual > 0:
            print(f"Idempotent skip: predictions.csv is up-to-date ({actual} rows)")
            _update_manifest_parsed(out_csv, actual)
            return
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
    # Validate outcome and update manifest
    expected = _count_unique_custom_ids(results_jsonl)
    actual = _count_rows(out_csv)
    if expected != actual:
        raise SystemExit(f"Parsed row count mismatch: expected {expected} (unique custom_id in combined results) but wrote {actual}")
    _update_manifest_parsed(out_csv, actual)


def _count_unique_custom_ids(path: str) -> int:
    seen: set[str] = set()
    try:
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                s = line.strip()
                if not s:
                    continue
                try:
                    obj = json.loads(s)
                except Exception:
                    continue
                cid = obj.get("custom_id") or obj.get("customId")
                if cid:
                    seen.add(str(cid))
    except Exception:
        return 0
    return len(seen)


def _count_rows(csv_path: str) -> int:
    try:
        with open(csv_path, "r", encoding="utf-8") as f:
            r = csv.reader(f)
            return max(0, sum(1 for _ in r) - 1)
    except Exception:
        return 0


def _update_manifest_parsed(out_csv: str, n_rows: int) -> None:
    results_dir = os.path.dirname(out_csv)
    manifest_path = os.path.join(results_dir, "trial_manifest.json")
    if os.path.isfile(manifest_path):
        try:
            mf.update_stage_status(
                manifest_path,
                "parsed",
                "completed",
                {"predictions_csv": os.path.relpath(out_csv, results_dir), "row_count": int(n_rows)},
            )
        except Exception:
            pass


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results_jsonl", required=True)
    ap.add_argument("--out_csv", default="results/predictions.csv")
    args = ap.parse_args()
    process_results(args.results_jsonl, args.out_csv)
    print("Wrote", args.out_csv)


if __name__ == "__main__":
    main()
