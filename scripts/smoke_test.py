from __future__ import annotations
import os
import argparse
import json
import time
import csv
import hashlib
import traceback
import tempfile
from typing import List, Dict, Any, Tuple
from dotenv import load_dotenv
import yaml
from config.schema import load_config

try:
    # openai>=1.0
    from openai import OpenAI
except Exception as e:  # pragma: no cover
    OpenAI = None  # type: ignore


def read_jsonl(path: str) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            s = line.strip()
            if not s:
                continue
            try:
                rows.append(json.loads(s))
            except Exception:
                continue
    return rows


def assemble_user_open(instructions: str, context: str, question: str) -> str:
    return f"{instructions}\n\nCONTEXT:\n{context}\n\nQUESTION:\n{question}"


def assemble_user_closed(instructions: str, question: str) -> str:
    return f"{instructions}\n\nQUESTION:\n{question}"


def get_base_url() -> str:
    base = os.environ.get("FIREWORKS_API_BASE", "https://api.fireworks.ai")
    # Ensure we point to the OpenAI-compatible path
    if base.endswith("/v1") or base.endswith("/inference/v1"):
        return base
    # Prefer explicit inference/v1 path
    return f"{base.rstrip('/')}/inference/v1"


def _flow_mode(args) -> None:
    """Offline, end-to-end smoke covering build → parse → score → stats → cost.

    Assumptions:
    - Uses existing prepared data if present; otherwise creates a tiny synthetic set.
    - Limits to args.n items per split, temps=[0.0], samples_per_item={"0.0":1}.
    - Does not hit network or external APIs.
    """
    cfg = load_config(args.config)
    # Resolve paths and ensure isolated smoke run directories
    prepared_dir = cfg["paths"]["prepared_dir"]
    run_ts = time.strftime("%Y%m%d_%H%M%S", time.localtime())
    run_id = f"smoke_{run_ts}"
    base_out = os.path.join(args.out_dir, run_id)
    os.makedirs(base_out, exist_ok=True)
    batch_dir = os.path.join(base_out, "batch_inputs")
    results_dir = os.path.join(base_out, "results")
    os.makedirs(batch_dir, exist_ok=True)
    os.makedirs(results_dir, exist_ok=True)

    # Prepare a minimal config override written to a temp file
    cfg_override = dict(cfg)
    cfg_override["temps"] = [0.0]
    cfg_override["samples_per_item"] = {"0.0": 1}
    cfg_override.setdefault("paths", {}).update({
        "batch_inputs_dir": batch_dir,
        "results_dir": results_dir,
    })

    # If prepared data is missing, synthesize a tiny set
    def _ensure_prepared_minimal() -> str:
        nonlocal prepared_dir
        src_ob = os.path.join(prepared_dir, "open_book.jsonl")
        src_cb = os.path.join(prepared_dir, "closed_book.jsonl")
        # Always build a reduced prepared set for smoke under base_out
        dst_prepared = os.path.join(base_out, "prepared")
        os.makedirs(dst_prepared, exist_ok=True)
        dst_ob = os.path.join(dst_prepared, "open_book.jsonl")
        dst_cb = os.path.join(dst_prepared, "closed_book.jsonl")
        if os.path.isfile(src_ob) and os.path.isfile(src_cb):
            # Copy only the first N items from each split
            def _copy_first_n(src, dst, n):
                cnt = 0
                with open(src, "r", encoding="utf-8") as fin, open(dst, "w", encoding="utf-8") as fout:
                    for line in fin:
                        s = line.strip()
                        if not s: continue
                        fout.write(s + "\n")
                        cnt += 1
                        if cnt >= max(1, int(args.n)):
                            break
            _copy_first_n(src_ob, dst_ob, args.n)
            _copy_first_n(src_cb, dst_cb, args.n)
            prepared_dir = dst_prepared
            return prepared_dir
        # No source prepared data; synthesize a tiny set
        ob_rows = [
            {"dataset": "squad_v2", "id": "ob1", "context": "Paris is in France.", "question": "Which country is Paris in?", "answers": ["France"], "is_unanswerable": False},
            {"dataset": "squad_v2", "id": "ob2", "context": "No one knows this.", "question": "What is the answer?", "answers": [], "is_unanswerable": True},
        ]
        cb_rows = [
            {"dataset": "triviaqa", "id": "cb1", "question": "How many valves does a trumpet have?", "answers": ["3", "Three"]},
            {"dataset": "nq_open", "id": "cb2", "question": "Who wrote The Hobbit?", "answers": ["J. R. R. Tolkien", "JRR Tolkien", "Tolkien"]},
        ]
        with open(dst_ob, "w", encoding="utf-8") as f:
            for r in ob_rows[: max(1, int(args.n))]: f.write(json.dumps(r) + "\n")
        with open(dst_cb, "w", encoding="utf-8") as f:
            for r in cb_rows[: max(1, int(args.n))]: f.write(json.dumps(r) + "\n")
        prepared_dir = dst_prepared
        return prepared_dir

    prepared_dir = _ensure_prepared_minimal()
    cfg_override["paths"]["prepared_dir"] = prepared_dir

    # Write override config
    tmp_cfg_path = os.path.join(base_out, "config.smoke.yaml")
    with open(tmp_cfg_path, "w", encoding="utf-8") as f:
        yaml.safe_dump(cfg_override, f)

    # Build minimal batches (temps fixed to 0.0; use default prompt set)
    import subprocess
    import sys
    subprocess.check_call([sys.executable, "-m", "scripts.build_batches", "--config", tmp_cfg_path, "--temps", "0.0"])  # validation: raises on failure
    print("build_batches: ok (t=0.0, K=1)")

    # Identify generated batch files (without prompt set suffix for default)
    t0_ctrl = os.path.join(batch_dir, "t0_control.jsonl")
    t0_trt = os.path.join(batch_dir, "t0_treatment.jsonl")
    # Fallback if prompt set suffix present
    if not os.path.isfile(t0_ctrl):
        # Find any control/treatment files for t0
        for name in os.listdir(batch_dir):
            if name.startswith("t0_") and name.endswith("_control.jsonl"): t0_ctrl = os.path.join(batch_dir, name)
            if name.startswith("t0_") and name.endswith("_treatment.jsonl"): t0_trt = os.path.join(batch_dir, name)

    def _load_canon(prep_dir: str) -> Tuple[dict, dict]:
        ob = {}
        cb = {}
        for path, target in [(os.path.join(prep_dir, "open_book.jsonl"), ob), (os.path.join(prep_dir, "closed_book.jsonl"), cb)]:
            with open(path, "r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if not line: continue
                    row = json.loads(line)
                    target[f"{row.get('dataset')}|{row.get('id')}"] = row
        return ob, cb

    open_canon, closed_canon = _load_canon(prepared_dir)

    # Helper to choose a plausible prediction from canon
    def _predict_from_custom_id(cid: str) -> str:
        dataset, item_id, condition, temp_str, sample_idx, typ = cid.split("|")
        key = f"{dataset}|{item_id}"
        if typ == "open":
            ex = open_canon.get(key) or {}
            if ex.get("is_unanswerable"):
                return "I don't know."
            ans = (ex.get("answers") or [])
            return (ans[0] if ans else "I don't know.")
        else:
            ex = closed_canon.get(key) or {}
            ans = (ex.get("answers") or [])
            return (ans[0] if ans else "I don't know.")

    # Create a combined results JSONL simulating Fireworks batch output
    results_jsonl = os.path.join(results_dir, "results_combined.jsonl")
    total = 0
    with open(results_jsonl, "w", encoding="utf-8") as fout:
        for part in [t0_ctrl, t0_trt]:
            with open(part, "r", encoding="utf-8") as f:
                for line in f:
                    row = json.loads(line)
                    cid = row.get("custom_id")
                    pred = _predict_from_custom_id(cid)
                    payload = {
                        "custom_id": cid,
                        "response": {
                            "id": f"local-sim-{hashlib.sha1(cid.encode()).hexdigest()[:12]}",
                            "body": {
                                "choices": [
                                    {"index": 0, "finish_reason": "stop", "message": {"role": "assistant", "content": pred}}
                                ],
                                "usage": {"prompt_tokens": 32, "completion_tokens": max(1, len(pred.split()))}
                            }
                        }
                    }
                    fout.write(json.dumps(payload) + "\n")
                    total += 1
    print(f"simulate_results: ok ({total} responses)")

    # Parse → Score → Stats → Costs
    import subprocess
    import sys
    subprocess.check_call([sys.executable, "-m", "fireworks.parse_results", "--results_jsonl", results_jsonl, "--out_csv", os.path.join(results_dir, "predictions.csv")])
    print("parse_results: ok")
    subprocess.check_call([sys.executable, "-m", "scoring.score_predictions", "--pred_csv", os.path.join(results_dir, "predictions.csv"), "--prepared_dir", prepared_dir, "--out_dir", results_dir, "--config", tmp_cfg_path])
    print("score_predictions: ok")
    try:
        subprocess.check_call([sys.executable, "-m", "scoring.stats", "--per_item_csv", os.path.join(results_dir, "per_item_scores.csv"), "--config", args.config, "--out_path", os.path.join(results_dir, "significance.json")])
        print("stats: ok")
    except Exception as e:
        # Provide a graceful fallback when optional deps (e.g., scipy) are unavailable
        sig_path = os.path.join(results_dir, "significance.json")
        with open(sig_path, "w", encoding="utf-8") as f:
            json.dump({}, f)
        print(f"stats: skipped ({type(e).__name__}) → wrote empty significance.json")
    # Unsupported sensitivity (optional)
    try:
        subprocess.check_call([sys.executable, "-m", "scripts.unsupported_sensitivity", "--pred_csv", os.path.join(results_dir, "predictions.csv"), "--prepared_dir", prepared_dir, "--config", tmp_cfg_path, "--out_path", os.path.join(results_dir, "unsupported_sensitivity.json")])
        print("unsupported_sensitivity: ok")
    except Exception as e:
        print(f"unsupported_sensitivity: skipped ({type(e).__name__})")
    # Mixed-effects (optional)
    try:
        subprocess.check_call([sys.executable, "-m", "scripts.mixed_effects", "--pred_csv", os.path.join(results_dir, "predictions.csv"), "--prepared_dir", prepared_dir, "--config", tmp_cfg_path, "--out_path", os.path.join(results_dir, "mixed_models.json")])
        print("mixed_effects: ok")
    except Exception as e:
        print(f"mixed_effects: skipped ({type(e).__name__})")
    # Power/MDE (optional)
    try:
        subprocess.check_call([sys.executable, "-m", "scripts.power_analysis", "--per_item_csv", os.path.join(results_dir, "per_item_scores.csv"), "--prepared_dir", prepared_dir, "--config", tmp_cfg_path, "--out_path", os.path.join(results_dir, "power_analysis.json")])
        print("power_analysis: ok")
    except Exception as e:
        print(f"power_analysis: skipped ({type(e).__name__})")
    # Cost-effectiveness (optional)
    try:
        subprocess.check_call([sys.executable, "-m", "scripts.cost_effectiveness", "--pred_csv", os.path.join(results_dir, "predictions.csv"), "--per_item_csv", os.path.join(results_dir, "per_item_scores.csv"), "--config", tmp_cfg_path, "--out_path", os.path.join(results_dir, "cost_effectiveness.json")])
        print("cost_effectiveness: ok")
    except Exception as e:
        print(f"cost_effectiveness: skipped ({type(e).__name__})")
    subprocess.check_call([sys.executable, "-m", "scripts.summarize_costs", "--pred_csv", os.path.join(results_dir, "predictions.csv"), "--config", tmp_cfg_path, "--out_path", os.path.join(results_dir, "costs.json")])
    print("summarize_costs: ok")

    print(f"Smoke flow complete. Outputs in {results_dir}")


def main():
    load_dotenv()

    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="config/eval_config.yaml")
    ap.add_argument("--mode", choices=["flow", "api"], default="flow", help="'flow' for offline full pipeline; 'api' to hit chat API")
    # Flow mode options
    ap.add_argument("--n", type=int, default=5, help="Number of items per split (flow mode)")
    # API mode options
    ap.add_argument("--temp", type=float, default=0.0)
    ap.add_argument("--type", choices=["closed", "open", "both"], default="closed")
    ap.add_argument("--condition", choices=["control", "treatment", "both"], default="both")
    ap.add_argument("--out_dir", default="results/smoke")
    args = ap.parse_args()

    if args.mode == "flow":
        _flow_mode(args)
        return

    # API mode (previous behavior)
    if OpenAI is None:
        raise SystemExit("The 'openai' package is required. Please install requirements.txt")

    cfg = load_config(args.config)

    prepared_dir = cfg["paths"]["prepared_dir"]
    results_dir = args.out_dir
    os.makedirs(results_dir, exist_ok=True)

    # Structured JSONL logging setup (one line per event)
    run_ts = time.strftime("%Y%m%d_%H%M%S", time.localtime())
    run_id = f"smoke_{run_ts}"
    log_path = os.path.join(results_dir, f"{run_id}_log.jsonl")

    def _truncate(text: str, max_len: int = 400) -> str:
        if text is None:
            return ""
        text = str(text)
        return text if len(text) <= max_len else text[: max_len] + " …[truncated]"

    def _log_event(event: Dict[str, Any]) -> None:
        try:
            base: Dict[str, Any] = {
                "ts": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                "run_id": run_id,
            }
            with open(log_path, "a", encoding="utf-8") as lf:
                lf.write(json.dumps({**base, **event}, ensure_ascii=False) + "\n")
        except Exception:
            # Last-resort: print to stdout if file logging fails
            print("[log-fallback]", json.dumps(event)[:500])

    # Prompts and instructions
    ctrl_path = "config/prompts/control_system.txt"
    trt_path = "config/prompts/treatment_system.txt"
    if not os.path.isfile(ctrl_path) or not os.path.isfile(trt_path):
        raise SystemExit("Missing prompts. Please add control and treatment prompts under config/prompts/.")
    control_system = open(ctrl_path, "r", encoding="utf-8").read().strip()
    treatment_system = open(trt_path, "r", encoding="utf-8").read().strip()
    closed_instr = open("config/task_instructions/closed_book.txt", "r", encoding="utf-8").read().strip()
    open_instr = open("config/task_instructions/open_book.txt", "r", encoding="utf-8").read().strip()

    # Sampling
    do_closed = args.type in ("closed", "both")
    do_open = args.type in ("open", "both")

    closed_items: List[Dict[str, Any]] = []
    open_items: List[Dict[str, Any]] = []
    if do_closed:
        cb_path = os.path.join(prepared_dir, "closed_book.jsonl")
        closed_items = read_jsonl(cb_path)[: max(0, args.n)]
    if do_open:
        ob_path = os.path.join(prepared_dir, "open_book.jsonl")
        open_items = read_jsonl(ob_path)[: max(0, args.n)]

    # Client
    api_key = os.environ.get("FIREWORKS_API_KEY")
    if not api_key:
        raise SystemExit("FIREWORKS_API_KEY is not set")
    base_url = get_base_url()
    client = OpenAI(api_key=api_key, base_url=base_url)

    _log_event({
        "event": "run_start",
        "results_dir": results_dir,
        "log_path": log_path,
        "base_url": base_url,
        "config_path": args.config,
        "args": {"n": args.n, "temp": args.temp, "type": args.type, "condition": args.condition},
    })

    conditions: List[str] = [args.condition] if args.condition in ("control", "treatment") else ["control", "treatment"]

    # Params
    model = cfg["model_id"]
    top_p = cfg.get("top_p", 1.0)
    top_k = cfg.get("top_k")
    stop_seqs = cfg.get("stop") or None

    def _run_once(meta: Dict[str, Any], messages: List[Dict[str, str]], max_tokens: int) -> str:
        # Simple retry for 429/5xx with detailed logging
        delay = 1.0
        attempt = 0
        while attempt < 5:
            attempt += 1
            _log_event({
                "event": "request",
                **meta,
                "attempt": attempt,
                "model": model,
                "parameters": {
                    "temperature": float(args.temp),
                    "top_p": float(top_p) if top_p is not None else None,
                    "top_k": int(top_k) if top_k is not None else None,
                    "max_tokens": int(max_tokens),
                    "stop": stop_seqs,
                },
                "messages_preview": [
                    {"role": m.get("role"), "len": len(m.get("content") or ""), "preview": _truncate(m.get("content") or "", 300)}
                    for m in messages
                ],
            })
            try:
                t0 = time.time()
                resp = client.chat.completions.create(
                    model=model,
                    messages=messages,
                    temperature=float(args.temp),
                    top_p=float(top_p) if top_p is not None else None,
                    max_tokens=int(max_tokens),
                    stop=stop_seqs,
                    extra_body={"top_k": int(top_k)} if top_k is not None else None,
                )
                dt = time.time() - t0
                # Extract fields defensively
                finish_reason = None
                content = ""
                request_id = None
                prompt_tokens = completion_tokens = total_tokens = None
                try:
                    request_id = getattr(resp, "id", None)
                    if getattr(resp, "choices", None):
                        first_choice = resp.choices[0]
                        finish_reason = getattr(first_choice, "finish_reason", None)
                        if getattr(first_choice, "message", None):
                            content = (first_choice.message.content or "").strip()
                    usage_obj = getattr(resp, "usage", None)
                    if usage_obj is not None:
                        prompt_tokens = getattr(usage_obj, "prompt_tokens", None)
                        completion_tokens = getattr(usage_obj, "completion_tokens", None)
                        total_tokens = getattr(usage_obj, "total_tokens", None)
                except Exception:
                    pass

                _log_event({
                    "event": "response",
                    **meta,
                    "attempt": attempt,
                    "latency_ms": int(dt * 1000),
                    "request_id": request_id,
                    "finish_reason": finish_reason,
                    "content_len": len(content or ""),
                    "content_preview": _truncate(content or "", 300),
                    "usage": {
                        "prompt_tokens": prompt_tokens,
                        "completion_tokens": completion_tokens,
                        "total_tokens": total_tokens,
                    },
                })

                return content
            except Exception as e:
                err_info: Dict[str, Any] = {
                    "event": "exception",
                    **meta,
                    "attempt": attempt,
                    "error_type": type(e).__name__,
                    "error_message": str(e),
                }
                # Try to pull HTTP status / response if present
                try:
                    status_code = getattr(e, "status_code", None) or getattr(e, "http_status", None)
                    if status_code is not None:
                        err_info["status_code"] = status_code
                    resp_obj = getattr(e, "response", None)
                    if resp_obj is not None:
                        # Some clients have .text or .json()
                        body_text = None
                        try:
                            body_text = resp_obj.text
                        except Exception:
                            pass
                        if body_text:
                            err_info["error_response_text"] = _truncate(body_text, 500)
                except Exception:
                    pass
                # Attach last traceback line for context
                try:
                    err_info["traceback_last"] = _truncate(traceback.format_exc().splitlines()[-1] if traceback.format_exc() else "", 300)
                except Exception:
                    pass
                _log_event(err_info)
                time.sleep(delay)
                delay = min(delay * 2.0, 8.0)
        _log_event({"event": "give_up", **meta})
        return ""

    out_csv = os.path.join(results_dir, "predictions.csv")
    # Optional debug CSV with meta for quick inspection
    debug_csv = os.path.join(results_dir, f"{run_id}_predictions_debug.csv")
    blanks_count = 0
    total_count = 0
    with open(out_csv, "w", encoding="utf-8", newline="") as fout:
        writer = csv.writer(fout)
        writer.writerow([
            "dataset",
            "id",
            "condition",
            "temp",
            "type",
            "question",
            "prediction",
        ])
        with open(debug_csv, "w", encoding="utf-8", newline="") as fdbg:
            dbg_writer = csv.writer(fdbg)
            dbg_writer.writerow([
                "dataset", "id", "condition", "temp", "type", "request_id", "finish_reason", "pred_len", "prompt_len", "system_sha1"
            ])

            if do_closed:
                max_tok_closed = int(cfg.get("max_new_tokens", {}).get("closed_book", 1024))
                for row in closed_items:
                    dataset = row.get("dataset") or "closed_book"
                    rid = row.get("id") or row.get("qid") or ""
                    question = row.get("question") or row.get("query") or ""
                    user = assemble_user_closed(closed_instr, question)
                    for cond in conditions:
                        system_text = control_system if cond == "control" else treatment_system
                        system_sha1 = hashlib.sha1(system_text.encode("utf-8")).hexdigest()
                        messages = [
                            {"role": "system", "content": system_text},
                            {"role": "user", "content": user},
                        ]
                        meta = {
                            "phase": "closed",
                            "dataset": dataset,
                            "item_id": rid,
                            "condition": cond,
                            "temp": float(args.temp),
                        }
                        pred = _run_once(meta, messages, max_tok_closed)
                        if not pred:
                            blanks_count += 1
                            _log_event({"event": "blank_prediction", **meta})
                        total_count += 1
                        writer.writerow([dataset, rid, cond, f"{float(args.temp):.1f}", "closed", question, pred])
                        # Try to surface last response meta from logs via minimal fields we still know
                        dbg_writer.writerow([
                            dataset, rid, cond, f"{float(args.temp):.1f}", "closed", "", "", len(pred or ""), len(user), system_sha1
                        ])

            if do_open:
                max_tok_open = int(cfg.get("max_new_tokens", {}).get("open_book", 1024))
                for row in open_items:
                    dataset = row.get("dataset") or "squad_v2"
                    rid = row.get("id") or row.get("qid") or ""
                    question = row.get("question") or row.get("query") or ""
                    context = row.get("context", "")
                    user = assemble_user_open(open_instr, context, question)
                    for cond in conditions:
                        system_text = control_system if cond == "control" else treatment_system
                        system_sha1 = hashlib.sha1(system_text.encode("utf-8")).hexdigest()
                        messages = [
                            {"role": "system", "content": system_text},
                            {"role": "user", "content": user},
                        ]
                        meta = {
                            "phase": "open",
                            "dataset": dataset,
                            "item_id": rid,
                            "condition": cond,
                            "temp": float(args.temp),
                        }
                        pred = _run_once(meta, messages, max_tok_open)
                        if not pred:
                            blanks_count += 1
                            _log_event({"event": "blank_prediction", **meta})
                        total_count += 1
                        writer.writerow([dataset, rid, cond, f"{float(args.temp):.1f}", "open", question, pred])
                        dbg_writer.writerow([
                            dataset, rid, cond, f"{float(args.temp):.1f}", "open", "", "", len(pred or ""), len(user), system_sha1
                        ])

    _log_event({
        "event": "run_end",
        "out_csv": out_csv,
        "debug_csv": debug_csv,
        "blanks": blanks_count,
        "total": total_count,
    })
    print(f"Wrote smoke test predictions to {out_csv}")
    print(f"Debug CSV: {debug_csv}")
    print(f"JSONL log: {log_path}")


if __name__ == "__main__":
    import multiprocessing as mp

    mp.freeze_support()
    main()
