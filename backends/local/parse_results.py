from __future__ import annotations

import argparse
import csv
import json
import os
from typing import Iterable, Tuple


FIELDNAMES = [
    "custom_id",
    "dataset",
    "item_id",
    "condition",
    "temp",
    "sample_index",
    "type",
    "request_id",
    "finish_reason",
    "response_text",
    "prompt_tokens",
    "completion_tokens",
    "total_tokens",
]


def iter_jsonl(path: str) -> Iterable[dict]:
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            s = line.strip()
            if not s:
                continue
            try:
                yield json.loads(s)
            except Exception:
                # Skip malformed lines rather than aborting
                continue


def parse_custom_id(cid: str) -> Tuple[str, str, str, float, int, str]:
    parts = cid.split("|")
    if len(parts) != 6:
        raise ValueError(f"Bad custom_id: {cid}")
    dataset, item_id, condition, temp_str, sample_str, typ = parts
    return dataset, item_id, condition, float(temp_str), int(sample_str), typ


def extract_text_from_response_obj(resp: dict) -> tuple[str, str | None, dict, str | None]:
    """Return (text, finish_reason, usage_dict, request_id) from a local result object.

    Supports the local schema emitted by backends.local.local_batch and the shape
    also handled by fireworks.parse_results for parity.
    """
    body = (resp.get("body") or resp) if isinstance(resp, dict) else {}
    usage = body.get("usage") or resp.get("usage") or {}
    request_id = resp.get("request_id") or resp.get("id") or body.get("id")
    text = ""
    finish = None
    try:
        choices = body.get("choices") or []
        if choices and isinstance(choices[0], dict):
            msg = choices[0].get("message") or {}
            text = (msg.get("content") or "").strip()
            finish = choices[0].get("finish_reason")
    except Exception:
        pass
    if not text and isinstance(body.get("content"), str):
        text = (body.get("content") or "").strip()
    return text, finish, usage, request_id


def validate_unique_custom_ids(results_jsonl: str) -> int:
    seen: set[str] = set()
    total = 0
    for row in iter_jsonl(results_jsonl):
        total += 1
        cid = row.get("custom_id") or row.get("customId")
        if not cid:
            raise SystemExit(f"Result row missing custom_id at line {total}")
        if cid in seen:
            raise SystemExit(f"Duplicate custom_id detected: {cid}")
        seen.add(str(cid))
    return len(seen)


def process_results(results_jsonl: str, out_csv: str) -> None:
    # Validate custom_ids first; ensures 1:1 mapping and explicit failures on duplicates
    expected = validate_unique_custom_ids(results_jsonl)

    os.makedirs(os.path.dirname(out_csv), exist_ok=True)
    with open(out_csv, "w", encoding="utf-8", newline="") as fcsv:
        w = csv.DictWriter(fcsv, fieldnames=FIELDNAMES, quoting=csv.QUOTE_MINIMAL, escapechar='\\')
        w.writeheader()
        actual = 0
        for row in iter_jsonl(results_jsonl):
            cid = row.get("custom_id") or row.get("customId")
            resp = row.get("response") or {}
            text, finish, usage, request_id = extract_text_from_response_obj(resp)
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
                "prompt_tokens": (usage or {}).get("prompt_tokens"),
                "completion_tokens": (usage or {}).get("completion_tokens"),
                "total_tokens": (usage or {}).get("total_tokens"),
            })
            actual += 1

    if actual != expected:
        raise SystemExit(f"Parsed row count mismatch: expected {expected} (unique custom_id) but wrote {actual}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--results_jsonl", required=True)
    ap.add_argument("--out_csv", default="results/predictions.csv")
    args = ap.parse_args()
    process_results(args.results_jsonl, args.out_csv)
    print("Wrote", args.out_csv)


if __name__ == "__main__":
    main()

