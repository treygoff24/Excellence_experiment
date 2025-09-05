import string
_ARTICLES = {"a", "an", "the"}
_PUNC_TABLE = str.maketrans("", "", string.punctuation)


def normalize_answer(s: str) -> str:
    s = s.lower().strip()
    s = s.translate(_PUNC_TABLE)
    s = " ".join(s.split())
    tokens = [t for t in s.split() if t not in _ARTICLES]
    return " ".join(tokens)


def f1_score(prediction: str, ground_truth: str) -> float:
    pred_tokens = normalize_answer(prediction).split()
    truth_tokens = normalize_answer(ground_truth).split()
    if len(pred_tokens) == 0 and len(truth_tokens) == 0: return 1.0
    if len(pred_tokens) == 0 or len(truth_tokens) == 0: return 0.0
    common = 0
    used = [False] * len(truth_tokens)
    for t in pred_tokens:
        for i, g in enumerate(truth_tokens):
            if (not used[i]) and t == g:
                used[i] = True
                common += 1
                break
    if common == 0: return 0.0
    precision = common / len(pred_tokens)
    recall = common / len(truth_tokens)
    return 2 * precision * recall / (precision + recall)


def exact_match(prediction: str, ground_truth: str) -> bool:
    return normalize_answer(prediction) == normalize_answer(ground_truth)


def best_over_ground_truths(pred: str, truths: list, metric="f1") -> float:
    truths = truths or [""]
    if metric == "f1":
        return max(f1_score(pred, g) for g in truths)
    elif metric == "em":
        return 1.0 if any(exact_match(pred, g) for g in truths) else 0.0
    else:
        raise ValueError("metric must be 'f1' or 'em'")
