from __future__ import annotations
from .normalize import best_over_ground_truths
from .squad_v2 import is_abstention
def score_item(pred: str, truths: list) -> dict:
    em = 1.0 if best_over_ground_truths(pred, truths, metric="em") else 0.0
    abst = 1.0 if is_abstention(pred) else 0.0
    return {"em": em, "abstained": abst}
