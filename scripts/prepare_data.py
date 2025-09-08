from __future__ import annotations
import os
import json
import random
import hashlib
import argparse
from dataclasses import dataclass, asdict
from datetime import datetime
from typing import List, Optional, Dict, Any
from datasets import load_dataset
from tqdm import tqdm
import yaml
from config.schema import load_config
from scripts.state_utils import write_json_atomic
RANDOM_SEED = 2025


@dataclass
class OpenBookItem:
    dataset: str
    id: str
    context: str
    question: str
    answers: list
    is_unanswerable: bool


@dataclass
class ClosedBookItem:
    dataset: str
    id: str
    question: str
    answers: list


def write_jsonl(path: str, rows: list) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


def limit(items: list, n: Optional[int]) -> list:
    return items if n is None else items[:n]


def clean_text(s: str) -> str:
    return " ".join(s.strip().split())


def _sha256_file(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def _count_lines(path: str) -> int:
    n = 0
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                n += 1
    return n


def _prepared_manifest_path(prepared_dir: str) -> str:
    return os.path.join(prepared_dir, "prepared_manifest.json")


def _load_manifest(path: str) -> Dict[str, Any] | None:
    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return None


def _manifest_valid(prepared_dir: str, m: Dict[str, Any]) -> bool:
    try:
        ds = m.get("datasets", {}) or {}
        ob = ds.get("open_book", {}) or {}
        cb = ds.get("closed_book", {}) or {}
        for entry, fname in ((ob, "open_book.jsonl"), (cb, "closed_book.jsonl")):
            path = os.path.join(prepared_dir, entry.get("path") or fname)
            if not os.path.isfile(path):
                return False
            # Validate checksum
            want = str(entry.get("sha256") or "").strip()
            if not want:
                return False
            have = _sha256_file(path)
            if want != have:
                return False
            # Validate item count
            try:
                if int(entry.get("items", -1)) <= 0:
                    return False
            except Exception:
                return False
        return True
    except Exception:
        return False


def load_squad_v2(max_items: Optional[int] = None) -> List[OpenBookItem]:
    ds = load_dataset("squad_v2", split="validation")
    rows = []
    for ex in tqdm(ds, desc="SQuAD v2"):
        answers = ex.get("answers", {}).get("text", []) or []
        is_unans = ex.get("is_impossible", False) or len(answers) == 0
        rows.append(OpenBookItem(
            dataset="squad_v2",
            id=str(ex["id"]),
            context=clean_text(ex["context"]),
            question=clean_text(ex["question"]),
            answers=[clean_text(a) for a in answers],
            is_unanswerable=bool(is_unans),
        ))
    random.Random(RANDOM_SEED).shuffle(rows)
    return limit(rows, max_items)


def load_triviaqa_rc_nocontext(max_items: Optional[int] = None) -> List[ClosedBookItem]:
    ds = load_dataset("trivia_qa", "rc.nocontext", split="validation")
    rows = []
    for ex in tqdm(ds, desc="TriviaQA rc.nocontext"):
        ans = ex.get("answer", {}) or {}
        vals = set()
        if ans.get("value"): vals.add(clean_text(ans["value"]))
        for a in ans.get("aliases", []) or []:
            if a: vals.add(clean_text(a))
        qid = ex.get("question_id") or ex.get("id") or hashlib.md5(ex["question"].encode("utf-8")).hexdigest()
        rows.append(ClosedBookItem(
            dataset="triviaqa",
            id=str(qid),
            question=clean_text(ex["question"]),
            answers=sorted(vals),
        ))
    random.Random(RANDOM_SEED).shuffle(rows)
    return limit(rows, max_items)


def load_nq_open(max_items: Optional[int] = None) -> List[ClosedBookItem]:
    ds = load_dataset("nq_open", split="validation")
    rows = []
    for ex in tqdm(ds, desc="NQ-Open"):
        answers = ex.get("answer") or ex.get("answers") or []
        if isinstance(answers, str): answers = [answers]
        rows.append(ClosedBookItem(
            dataset="nq_open",
            id=str(ex.get("id") or hashlib.md5(ex["question"].encode("utf-8")).hexdigest()),
            question=clean_text(ex["question"]),
            answers=[clean_text(a) for a in answers if a],
        ))
    random.Random(RANDOM_SEED).shuffle(rows)
    return limit(rows, max_items)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="config/eval_config.yaml")
    ap.add_argument("--resume", action="store_true", help="Skip when prepared_manifest.json validates existing outputs")
    args = ap.parse_args()
    cfg = load_config(args.config)
    sizes = cfg.get("sizes", {})
    prepared_dir = cfg["paths"]["prepared_dir"]
    os.makedirs(prepared_dir, exist_ok=True)
    # Resume fast-path when manifest is present and valid
    man_path = _prepared_manifest_path(prepared_dir)
    if args.resume and os.path.isfile(man_path):
        m = _load_manifest(man_path) or {}
        if _manifest_valid(prepared_dir, m):
            print(f"Resume: prepared outputs already valid per manifest → skipping (path={man_path})")
            return
    ob_max = sizes.get("open_book_max_items") or sizes.get("squad_v2_max_items")
    tqa_max = sizes.get("triviaqa_max_items")
    nq_max = sizes.get("nq_open_max_items")
    open_book = load_squad_v2(max_items=ob_max)
    closed_book = []
    closed_book.extend(load_triviaqa_rc_nocontext(max_items=tqa_max))
    closed_book.extend(load_nq_open(max_items=nq_max))
    ob_path = os.path.join(prepared_dir, "open_book.jsonl")
    cb_path = os.path.join(prepared_dir, "closed_book.jsonl")
    write_jsonl(ob_path, [asdict(x) for x in open_book])
    write_jsonl(cb_path, [asdict(x) for x in closed_book])
    # Compute manifest metadata and write atomically
    ob_items = _count_lines(ob_path)
    cb_items = _count_lines(cb_path)
    manifest = {
        "schema_version": 1,
        "created_at": datetime.utcnow().isoformat() + "Z",
        "updated_at": datetime.utcnow().isoformat() + "Z",
        "datasets": {
            "open_book": {"path": os.path.basename(ob_path), "items": ob_items, "sha256": _sha256_file(ob_path)},
            "closed_book": {"path": os.path.basename(cb_path), "items": cb_items, "sha256": _sha256_file(cb_path)},
        },
    }
    write_json_atomic(man_path, manifest)
    print("Prepared", ob_items, "open-book;", cb_items, "closed-book")


if __name__ == "__main__":
    main()
