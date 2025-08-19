from __future__ import annotations
import os, json, random, hashlib, argparse
from dataclasses import dataclass, asdict
from typing import List, Optional
from datasets import load_dataset
from tqdm import tqdm
import yaml
from config.schema import load_config
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
def load_squad_v2(max_items: Optional[int]=None) -> List[OpenBookItem]:
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
def load_triviaqa_rc_nocontext(max_items: Optional[int]=None) -> List[ClosedBookItem]:
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
def load_nq_open(max_items: Optional[int]=None) -> List[ClosedBookItem]:
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
    args = ap.parse_args()
    cfg = load_config(args.config)
    sizes = cfg.get("sizes", {})
    prepared_dir = cfg["paths"]["prepared_dir"]
    os.makedirs(prepared_dir, exist_ok=True)
    ob_max = sizes.get("open_book_max_items") or sizes.get("squad_v2_max_items")
    tqa_max = sizes.get("triviaqa_max_items")
    nq_max = sizes.get("nq_open_max_items")
    open_book = load_squad_v2(max_items=ob_max)
    closed_book = []
    closed_book.extend(load_triviaqa_rc_nocontext(max_items=tqa_max))
    closed_book.extend(load_nq_open(max_items=nq_max))
    write_jsonl(os.path.join(prepared_dir, "open_book.jsonl"), [asdict(x) for x in open_book])
    write_jsonl(os.path.join(prepared_dir, "closed_book.jsonl"), [asdict(x) for x in closed_book])
    print("Prepared", len(open_book), "open-book;", len(closed_book), "closed-book")
if __name__ == "__main__":
    main()
