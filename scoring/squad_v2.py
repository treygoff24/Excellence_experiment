from __future__ import annotations
from .normalize import best_over_ground_truths, normalize_answer
ABSTAIN_MARKERS = ["i don't know","i do not know","unknown","not sure","cannot answer","can't answer","insufficient information","not enough information","not provided in the passage"]
def is_abstention(text: str) -> bool:
    t = normalize_answer(text)
    return any(m in t for m in ABSTAIN_MARKERS)
def score_item(pred: str, truths: list, is_unanswerable: bool) -> dict:
    if is_unanswerable:
        if is_abstention(pred):
            return {"em": 1.0, "f1": 1.0, "abstained": 1.0, "false_answer": 0.0}
        else:
            return {"em": 0.0, "f1": 0.0, "abstained": 0.0, "false_answer": 1.0}
    em = 1.0 if best_over_ground_truths(pred, truths, metric="em") else 0.0
    from .normalize import f1_score
    f1 = max(f1_score(pred, g) for g in truths or [""])
    abst = 1.0 if is_abstention(pred) else 0.0
    return {"em": em, "f1": f1, "abstained": abst, "false_answer": 0.0}
