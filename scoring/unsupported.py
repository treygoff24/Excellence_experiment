from __future__ import annotations

import re
from typing import Dict, Iterable, List, Tuple

from .normalize import normalize_answer


def _tokens(text: str) -> List[str]:
    t = normalize_answer(text)
    return [tok for tok in re.split(r"\W+", t) if tok]


def support_score_baseline(pred: str, context: str) -> float:
    """Baseline support score: 1.0 if normalized prediction is a substring of context, else 0.0."""
    p = normalize_answer(pred)
    c = normalize_answer(context)
    if not p:
        return 0.0
    return 1.0 if p in c else 0.0


def support_score_overlap(pred: str, context: str) -> float:
    """Token overlap support score: fraction of prediction tokens found in context tokens.

    Returns a score in [0,1] equal to containment overlap: |tok(pred) ∩ tok(ctx)| / |tok(pred)|.
    """
    pt = _tokens(pred)
    if not pt:
        return 0.0
    ct = set(_tokens(context))
    inter = sum(1 for t in pt if t in ct)
    return float(inter) / float(len(pt))


def is_supported(pred: str, context: str, *, strategy: str = "baseline", params: Dict | None = None) -> bool:
    """Return True if pred is supported by context under given strategy.

    - baseline: normalized substring check
    - overlap: token containment >= params.get('min_token_overlap', 0.6)
    - nli: optional future extension (currently falls back to overlap+baseline combo)
    """
    strategy = (strategy or "baseline").lower()
    params = params or {}
    if strategy == "baseline":
        return support_score_baseline(pred, context) >= 1.0
    if strategy == "overlap":
        thr = float(params.get("min_token_overlap", 0.6))
        return support_score_overlap(pred, context) >= thr
    if strategy == "nli":
        # Placeholder fallback: require both baseline OR overlap>=0.7
        return (support_score_baseline(pred, context) >= 1.0) or (support_score_overlap(pred, context) >= float(params.get("min_token_overlap", 0.7)))
    # Default to conservative False if unknown strategy
    return False


def is_unsupported(pred: str, context: str, *, abstained: float | int = 0, strategy: str = "baseline", threshold: float = 0.5, params: Dict | None = None) -> int:
    """Return 1 if pred is marked unsupported, else 0.

    We treat abstentions as not-unsupported.
    Threshold is used as a gating confidence for non-abstentions only: if abstained>=threshold → not unsupported.
    """
    try:
        abst = float(abstained)
    except Exception:
        abst = 0.0
    # If abstained (>= threshold), do not mark unsupported
    if abst >= threshold:
        return 0
    return 0 if is_supported(pred, context, strategy=strategy, params=params) else 1


def sweep_thresholds(pred: str, context: str, *, abstained: float | int = 0, strategy: str = "baseline", thresholds: Iterable[float] = (0.0, 0.25, 0.5, 0.75, 1.0), params: Dict | None = None) -> List[Tuple[float, int]]:
    """Return list of (threshold, unsupported_flag) across thresholds.

    Useful for sensitivity analyses where abstention gating is varied.
    """
    out: List[Tuple[float, int]] = []
    for thr in thresholds:
        out.append((float(thr), is_unsupported(pred, context, abstained=abstained, strategy=strategy, threshold=float(thr), params=params)))
    return out
