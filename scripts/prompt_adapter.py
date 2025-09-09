from __future__ import annotations

import argparse
import hashlib
import json
import os
from typing import Dict, Literal, Optional, Tuple

from config.schema import load_config


Cond = Literal["control", "treatment"]
TaskType = Literal["open", "closed", "open_book", "closed_book"]
OutFormat = Literal["messages", "prompt"]


def _read_text(path: str) -> str:
    with open(path, "r", encoding="utf-8") as f:
        return f.read().strip()


def _task_type_norm(t: str) -> TaskType:
    t2 = (t or "").strip().lower()
    if t2 in {"open", "open_book"}:
        return "open_book"
    if t2 in {"closed", "closed_book"}:
        return "closed_book"
    raise SystemExit("task type must be one of: open, open_book, closed, closed_book")


def _render_user_message(task_type: TaskType, question: str, context: Optional[str]) -> str:
    base_dir = "config/task_instructions"
    if task_type == "open_book":
        instr = _read_text(os.path.join(base_dir, "open_book.txt"))
        ctx = context or ""
        return f"{instr}\n\nCONTEXT:\n{ctx}\n\nQUESTION:\n{question}"
    else:
        instr = _read_text(os.path.join(base_dir, "closed_book.txt"))
        return f"{instr}\n\nQUESTION:\n{question}"


def _system_prompt_from_cfg(cfg: Dict, prompt_set: str, condition: Cond) -> str:
    ps = (cfg.get("prompt_sets") or {}).get(prompt_set)
    if not ps:
        raise SystemExit(f"Unknown prompt set '{prompt_set}'. Available: {', '.join(sorted((cfg.get('prompt_sets') or {}).keys()))}")
    path = ps.get("control" if condition == "control" else "treatment")
    if not path or not os.path.isfile(path):
        raise SystemExit("Missing system prompt file for the selected prompt set and condition")
    return _read_text(path)


def render_payload(
    cfg: Dict,
    *,
    condition: Cond,
    prompt_set: str,
    task_type: TaskType,
    temp: float,
    question: str,
    context: Optional[str] = None,
    out_format: OutFormat = "messages",
) -> Dict:
    system_text = _system_prompt_from_cfg(cfg, prompt_set, condition)
    user_text = _render_user_message(task_type, question, context)

    # max_new_tokens selection by task type
    mnt = cfg.get("max_new_tokens", {})
    mnt_val: Optional[int]
    if task_type == "open_book":
        mnt_val = int(mnt.get("open_book") or 0) or None
    else:
        mnt_val = int(mnt.get("closed_book") or 0) or None

    stop = cfg.get("stop") or []

    body: Dict
    if out_format == "messages":
        body = {
            "messages": [
                {"role": "system", "content": system_text},
                {"role": "user", "content": user_text},
            ]
        }
    else:
        # Combine system + user for engines that only accept a single prompt string
        body = {
            "prompt": f"[SYSTEM]\n{system_text}\n\n[USER]\n{user_text}",
        }

    if mnt_val is not None:
        body["max_new_tokens"] = mnt_val
    if stop:
        body["stop"] = list(stop)

    meta = {
        "condition": condition,
        "prompt_set": prompt_set,
        "temp": float(temp),
        "type": task_type,
    }
    return {"meta": meta, "body": body}


def _sha1(text: str) -> str:
    return hashlib.sha1(text.encode("utf-8")).hexdigest()


def check_render_equivalence(cfg: Dict, *, prompt_set: str, question: str, context: Optional[str], task_type: TaskType) -> Dict[str, Dict[str, str]]:
    # Render both conditions with messages
    a = render_payload(
        cfg,
        condition="control",
        prompt_set=prompt_set,
        task_type=task_type,
        temp=float((cfg.get("temps") or [0.0])[0]),
        question=question,
        context=context,
        out_format="messages",
    )
    b = render_payload(
        cfg,
        condition="treatment",
        prompt_set=prompt_set,
        task_type=task_type,
        temp=float((cfg.get("temps") or [0.0])[0]),
        question=question,
        context=context,
        out_format="messages",
    )

    sys_a = a["body"]["messages"][0]["content"]
    usr_a = a["body"]["messages"][1]["content"]
    sys_b = b["body"]["messages"][0]["content"]
    usr_b = b["body"]["messages"][1]["content"]

    return {
        "control": {"system_sha1": _sha1(sys_a), "user_sha1": _sha1(usr_a)},
        "treatment": {"system_sha1": _sha1(sys_b), "user_sha1": _sha1(usr_b)},
        "checks": {
            "system_differs": _sha1(sys_a) != _sha1(sys_b),
            "user_equal": _sha1(usr_a) == _sha1(usr_b),
        },
    }


def main() -> None:
    ap = argparse.ArgumentParser(description="Render prompts/messages for A/B experiments from config")
    ap.add_argument("--config", default="config/eval_config.yaml")
    ap.add_argument("--condition", choices=["control", "treatment"], default="control")
    ap.add_argument("--prompt_set", default=None, help="Prompt set name; defaults to config.default_prompt_set")
    ap.add_argument("--type", dest="task_type", default="closed", help="Task type: open|open_book|closed|closed_book")
    ap.add_argument("--temp", type=float, default=None, help="Temperature label to embed in meta")
    ap.add_argument("--format", choices=["messages", "prompt"], default="messages")
    ap.add_argument("--question", required=True)
    ap.add_argument("--context", default=None)
    ap.add_argument("--check", action="store_true", help="Print hashes to verify system differs and user is equal across conditions")
    args = ap.parse_args()

    cfg = load_config(args.config)
    ps_name = args.prompt_set or (cfg.get("default_prompt_set") or "default")
    t = float(args.temp if args.temp is not None else (cfg.get("temps") or [0.0])[0])
    task_type = _task_type_norm(args.task_type)

    payload = render_payload(
        cfg,
        condition=args.condition,  # type: ignore[arg-type]
        prompt_set=ps_name,
        task_type=task_type,  # type: ignore[arg-type]
        temp=t,
        question=args.question,
        context=args.context,
        out_format=args.format,  # type: ignore[arg-type]
    )

    print(json.dumps(payload, ensure_ascii=False, indent=2))

    if args.check:
        res = check_render_equivalence(
            cfg,
            prompt_set=ps_name,
            question=args.question,
            context=args.context,
            task_type=task_type,
        )
        print("\n--- Equivalence Check ---")
        print(json.dumps(res, indent=2))


if __name__ == "__main__":
    main()

