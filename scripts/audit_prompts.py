from __future__ import annotations

import os
import argparse
import hashlib
import json
from typing import Dict, Tuple

from config.schema import load_config


def _read(path: str) -> str:
    with open(path, "r", encoding="utf-8") as f:
        return f.read()


def _token_len(text: str) -> int:
    try:
        import tiktoken

        enc = tiktoken.get_encoding("cl100k_base")
        return len(enc.encode(text))
    except Exception:
        # Fallback: rough word count
        return max(1, len((text or "").split()))


def _count_lines(path: str) -> int:
    try:
        with open(path, "r", encoding="utf-8") as f:
            return sum(1 for ln in f if ln.strip())
    except FileNotFoundError:
        return 0


def _fmt_usd(x: float) -> str:
    return f"${x:,.4f}"


def audit_prompts(cfg: dict) -> Dict:
    paths = cfg.get("paths", {})
    prepared_dir = paths.get("prepared_dir", "data/prepared")

    # Prefer real prepared counts if available; else infer from config.sizes
    n_open = _count_lines(os.path.join(prepared_dir, "open_book.jsonl"))
    n_closed = _count_lines(os.path.join(prepared_dir, "closed_book.jsonl"))
    if not (n_open or n_closed):
        sizes = cfg.get("sizes", {}) or {}
        n_open = int(sizes.get("open_book_max_items") or 0) or 0
        # closed_book_max_items may refer to aggregate closed book size;
        # else fall back to per-dataset caps if present
        n_closed = int(
            sizes.get("closed_book_max_items")
            or sizes.get("triviaqa_max_items")
            or 0
        ) + int(sizes.get("nq_open_max_items") or 0)

    temps = [float(t) for t in (cfg.get("temps") or [0.0])]
    samples_per_item = {str(float(k)): int(v) for k, v in (cfg.get("samples_per_item") or {"0.0": 1}).items()}
    K_total = sum(samples_per_item.get(str(float(t)), 1) for t in temps)

    pricing = cfg.get("pricing", {})
    input_per_m = float(pricing.get("input_per_million", 0.0) or 0.0)
    batch_discount = float(pricing.get("batch_discount", 0.0) or 0.0)
    use_batch = bool(cfg.get("use_batch_api", True))

    prompt_sets = cfg.get("prompt_sets") or {
        "default": {
            "control": "config/prompts/control_system.txt",
            "treatment": "config/prompts/treatment_system.txt",
        }
    }

    out: Dict = {
        "schema_version": 1,
        "meta": {
            "prepared_counts": {"open": n_open, "closed": n_closed},
            "temps": temps,
            "samples_per_item": samples_per_item,
            "requests_per_condition": (n_open + n_closed) * K_total,
            "pricing": {
                "input_per_million": input_per_m,
                "batch_discount": batch_discount,
                "use_batch_api": use_batch,
            },
        },
        "prompt_sets": {},
    }

    for name, spec in prompt_sets.items():
        ctrl_path = spec.get("control")
        trt_path = spec.get("treatment")
        if not (ctrl_path and trt_path) or not (os.path.isfile(ctrl_path) and os.path.isfile(trt_path)):
            out["prompt_sets"][name] = {
                "error": "Missing control or treatment prompt file",
                "control": ctrl_path,
                "treatment": trt_path,
            }
            continue

        ctrl_text = _read(ctrl_path)
        trt_text = _read(trt_path)

        ctrl_tok = _token_len(ctrl_text)
        trt_tok = _token_len(trt_text)
        delta_tok = trt_tok - ctrl_tok

        # Lower-bound prompt token costs: system message only (excludes per-item instructions/context)
        req_per_cond = out["meta"]["requests_per_condition"]
        est_ctrl_tokens = ctrl_tok * req_per_cond
        est_trt_tokens = trt_tok * req_per_cond

        # Cost is only for input side here. Output costs excluded intentionally.
        def _cost_for_tokens(tok: int) -> float:
            usd = (tok / 1_000_000.0) * input_per_m
            if use_batch:
                usd *= (1.0 - batch_discount)
            return usd

        out["prompt_sets"][name] = {
            "control": {
                "path": ctrl_path,
                "sha256": hashlib.sha256(ctrl_text.encode()).hexdigest(),
                "tokens": ctrl_tok,
                "est_input_tokens_total": est_ctrl_tokens,
                "est_input_usd": _cost_for_tokens(est_ctrl_tokens),
            },
            "treatment": {
                "path": trt_path,
                "sha256": hashlib.sha256(trt_text.encode()).hexdigest(),
                "tokens": trt_tok,
                "est_input_tokens_total": est_trt_tokens,
                "est_input_usd": _cost_for_tokens(est_trt_tokens),
            },
            "delta": {
                "tokens": delta_tok,
                "est_input_tokens_total": est_trt_tokens - est_ctrl_tokens,
                "est_input_usd": _cost_for_tokens(est_trt_tokens) - _cost_for_tokens(est_ctrl_tokens),
            },
            "checks": {
                "empty_control": (ctrl_tok == 0),
                "empty_treatment": (trt_tok == 0),
                "treatment_5x_longer": (trt_tok >= 5 * max(1, ctrl_tok)),
            },
        }

    return out


def main() -> None:
    ap = argparse.ArgumentParser(description="Audit prompt token lengths and estimated input cost deltas")
    ap.add_argument("--config", default="config/eval_config.yaml")
    ap.add_argument("--out_json", default=None, help="Optional path to write JSON summary")
    args = ap.parse_args()

    cfg = load_config(args.config)
    audit = audit_prompts(cfg)

    # Pretty print summary
    print("Prompt Audit Summary")
    print("---------------------")
    meta = audit["meta"]
    print(f"Prepared counts: open={meta['prepared_counts']['open']} closed={meta['prepared_counts']['closed']}")
    print(f"Temps: {', '.join(str(t) for t in meta['temps'])}  |  Samples per item: {meta['samples_per_item']}")
    print(f"Requests per condition (approx): {meta['requests_per_condition']}")
    print(f"Pricing: input_per_million={meta['pricing']['input_per_million']}  batch_discount={meta['pricing']['batch_discount']}  use_batch_api={meta['pricing']['use_batch_api']}")
    print()

    header = f"{'Prompt Set':<28} {'CtrlTok':>8} {'TrtTok':>8} {'ΔTok':>8} {'CtrlCost':>12} {'TrtCost':>12} {'ΔCost':>12}"
    print(header)
    print("-" * len(header))
    for name, row in audit["prompt_sets"].items():
        if row.get("error"):
            print(f"{name:<28} [ERROR] {row['error']}")
            continue
        c = row["control"]
        t = row["treatment"]
        d = row["delta"]
        print(
            f"{name:<28} {c['tokens']:>8} {t['tokens']:>8} {d['tokens']:>8} "
            f"{_fmt_usd(c['est_input_usd']):>12} {_fmt_usd(t['est_input_usd']):>12} {_fmt_usd(d['est_input_usd']):>12}"
        )
    print()

    problems: list[Tuple[str, str]] = []
    for name, row in audit["prompt_sets"].items():
        if row.get("error"):
            problems.append((name, row["error"]))
            continue
        chk = row.get("checks", {})
        if chk.get("empty_control"):
            problems.append((name, "Control prompt is empty"))
        if chk.get("empty_treatment"):
            problems.append((name, "Treatment prompt is empty"))
        if chk.get("treatment_5x_longer"):
            problems.append((name, "Treatment prompt is >= 5x control length"))

    if problems:
        print("Potential Issues:")
        for ps, msg in problems:
            print(f"- {ps}: {msg}")
        print()

    if args.out_json:
        os.makedirs(os.path.dirname(args.out_json) or ".", exist_ok=True)
        with open(args.out_json, "w", encoding="utf-8") as f:
            json.dump(audit, f, indent=2)
        print("Wrote:", args.out_json)


if __name__ == "__main__":
    main()
