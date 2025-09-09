from __future__ import annotations
import os
import json
import argparse
import yaml
from tqdm import tqdm
from config.schema import load_config
import hashlib
from datetime import datetime
from typing import Dict, Any
from scripts.state_utils import write_json_atomic


def read_lines(path: str):
    rows = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                try:
                    rows.append(json.loads(line))
                except json.JSONDecodeError:
                    # Skip malformed lines rather than crashing batch build
                    continue
    return rows


def assemble_user_open(instructions: str, context: str, question: str) -> str:
    return f"{instructions}\n\nCONTEXT:\n{context}\n\nQUESTION:\n{question}"


def assemble_user_closed(instructions: str, question: str) -> str:
    return f"{instructions}\n\nQUESTION:\n{question}"


def build_line(custom_id: str, system_text: str, user_text: str, stop: list[str] | None = None):
    body = {
        "messages": [
            {"role": "system", "content": system_text},
            {"role": "user", "content": user_text},
        ]
    }
    # Keep row bodies minimal; only include stop if explicitly configured
    if stop:
        body["stop"] = list(stop)
    return {"custom_id": custom_id, "body": body}


def format_temp_label(t: float) -> str:
    s = f"{float(t):.1f}"
    return "0" if s == "0.0" else s.replace(".", "")


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


def _manifest_path(batch_dir: str) -> str:
    return os.path.join(batch_dir, "build_manifest.json")


def _load_manifest(path: str) -> Dict[str, Any] | None:
    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return None


def _manifest_has_valid_entry(m: Dict[str, Any], shard_path: str) -> bool:
    try:
        shards = m.get("shards", {}) or {}
        meta = shards.get(os.path.basename(shard_path)) or shards.get(shard_path)
        if not meta:
            return False
        if not os.path.isfile(shard_path):
            return False
        want = str(meta.get("sha256") or "").strip()
        if not want:
            return False
        have = _sha256_file(shard_path)
        return want == have and int(meta.get("lines", 0)) > 0
    except Exception:
        return False


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="config/eval_config.yaml")
    ap.add_argument("--prompt_set", default=None, help="Name of prompt set in config.prompt_sets")
    ap.add_argument("--temps", default=None, help="Comma-separated temperatures to build (override config)")
    ap.add_argument("--resume", action="store_true", help="Skip shards when present and valid per manifest")
    args = ap.parse_args()

    cfg = load_config(args.config)

    prepared_dir = cfg["paths"]["prepared_dir"]
    batch_dir = cfg["paths"]["batch_inputs_dir"]
    os.makedirs(batch_dir, exist_ok=True)

    # Determine prompt set
    prompt_sets = cfg.get("prompt_sets") or {}
    default_ps = cfg.get("default_prompt_set") or "default"
    ps_name = (args.prompt_set or default_ps)
    ps = prompt_sets.get(ps_name)
    if not ps:
        raise SystemExit(f"Unknown prompt set '{ps_name}'. Available: {', '.join(sorted(prompt_sets.keys()))}")
    ctrl_path = ps.get("control")
    trt_path = ps.get("treatment")
    if not os.path.isfile(ctrl_path) or not os.path.isfile(trt_path):
        raise SystemExit("Missing prompts. Please add control and treatment prompts and reference them via config.prompt_sets.")
    control_system = open(ctrl_path, "r", encoding="utf-8").read().strip()
    treatment_system = open(trt_path, "r", encoding="utf-8").read().strip()
    closed_instr = open("config/task_instructions/closed_book.txt", "r", encoding="utf-8").read().strip()
    open_instr = open("config/task_instructions/open_book.txt", "r", encoding="utf-8").read().strip()

    # Determine temperatures and validate against samples_per_item
    if args.temps:
        try:
            temps = [float(t.strip()) for t in args.temps.split(",") if t.strip()]
        except Exception:
            raise SystemExit("--temps must be a comma-separated list of numbers, e.g., '0.2,0.7,1.0'")
    else:
        temps = cfg["temps"]
    samples_per_item = cfg["samples_per_item"]

    ob_rows = read_lines(os.path.join(prepared_dir, "open_book.jsonl"))
    cb_rows = read_lines(os.path.join(prepared_dir, "closed_book.jsonl"))

    stop_cfg = cfg.get("stop") or []
    include_row_stop = not bool(cfg.get("use_batch_api", True))

    manifest_path = _manifest_path(batch_dir)
    manifest = _load_manifest(manifest_path) or {"schema_version": 1, "created_at": datetime.utcnow().isoformat() + "Z", "updated_at": datetime.utcnow().isoformat() + "Z", "shards": {}, "prompt_sets": {}}

    for t in temps:
        t_str = format_temp_label(t)
        # Include prompt set in filenames only when multiple sets are defined to preserve backward compatibility
        include_ps = len(prompt_sets) > 1 or (ps_name not in ("default", None))
        suffix = f"_{ps_name}" if include_ps else ""
        out_control = os.path.join(batch_dir, f"t{t_str}{suffix}_control.jsonl")
        out_treat = os.path.join(batch_dir, f"t{t_str}{suffix}_treatment.jsonl")
        # Resume check: if both shards are valid per manifest, skip building
        if args.resume:
            m_ok_ctrl = _manifest_has_valid_entry(manifest, out_control)
            m_ok_trt = _manifest_has_valid_entry(manifest, out_treat)
            if m_ok_ctrl and m_ok_trt:
                print(f"Resume: skipping build for t={float(t):.1f}, ps={ps_name} (valid shards)")
                continue
        with open(out_control, "w", encoding="utf-8") as fc, open(out_treat, "w", encoding="utf-8") as ft:
            # Ensure per-file uniqueness of custom_id to satisfy Fireworks dataset validation
            seen_control: set[str] = set()
            seen_treat: set[str] = set()
            skipped_control = 0
            skipped_treat = 0
            K = int(samples_per_item[str(float(t))])
            for rows, open_flag in [(cb_rows, False), (ob_rows, True)]:
                # enumerate to create a stable fallback id when missing
                for idx, row in enumerate(tqdm(rows, desc=f"t={t} {'open' if open_flag else 'closed'}")):
                    # derive dataset and id with safe fallbacks
                    dataset = (
                        row.get("dataset")
                        or row.get("source")
                        or row.get("collection")
                        or ("open_book" if open_flag else "closed_book")
                    )
                    rid = (
                        row.get("id")
                        or row.get("example_id")
                        or row.get("qid")
                        or row.get("uuid")
                        or str(idx)
                    )
                    custom_prefix = f"{dataset}|{rid}"

                    # Prepare user content with fallbacks; skip if no question
                    question = row.get("question") or row.get("query") or row.get("prompt")
                    if not question:
                        # cannot build a prompt without a question; skip this row
                        continue
                    if open_flag:
                        context = row.get("context", "")
                        user = assemble_user_open(open_instr, context, question)
                    else:
                        user = assemble_user_closed(closed_instr, question)

                    for k in range(K):
                        typ = "open" if open_flag else "closed"
                        # CONTROL
                        cid = f"{custom_prefix}|control|{float(t):.1f}|{k}|{typ}"
                        if cid in seen_control:
                            skipped_control += 1
                        else:
                            seen_control.add(cid)
                            payload = build_line(cid, control_system, user, stop=(stop_cfg if include_row_stop else None))
                            fc.write(json.dumps(payload, ensure_ascii=False) + "\n")
                        # TREATMENT
                        cid = f"{custom_prefix}|treatment|{float(t):.1f}|{k}|{typ}"
                        if cid in seen_treat:
                            skipped_treat += 1
                        else:
                            seen_treat.add(cid)
                            payload = build_line(cid, treatment_system, user, stop=(stop_cfg if include_row_stop else None))
                            ft.write(json.dumps(payload, ensure_ascii=False) + "\n")
        print("Wrote", out_control, "and", out_treat)
        if skipped_control or skipped_treat:
            print(
                f"De-dup summary @ T={float(t):.1f}: skipped {skipped_control} duplicate custom_id(s) in control, {skipped_treat} in treatment."
            )
        # Update manifest entries for these shards
        ctrl_meta = {
            "path": os.path.basename(out_control),
            "prompt_set": ps_name,
            "condition": "control",
            "temp": float(t),
            "lines": _count_lines(out_control),
            "size": int(os.path.getsize(out_control)),
            "sha256": _sha256_file(out_control),
        }
        trt_meta = {
            "path": os.path.basename(out_treat),
            "prompt_set": ps_name,
            "condition": "treatment",
            "temp": float(t),
            "lines": _count_lines(out_treat),
            "size": int(os.path.getsize(out_treat)),
            "sha256": _sha256_file(out_treat),
        }
        # Maintain temps list as sorted unique values
        manifest.setdefault("temps", [])
        manifest["temps"] = sorted(list(set([float(x) for x in (manifest.get("temps") or [])] + [float(t)])))
        shards = manifest.setdefault("shards", {})
        shards[os.path.basename(out_control)] = ctrl_meta
        shards[os.path.basename(out_treat)] = trt_meta
        manifest["updated_at"] = datetime.utcnow().isoformat() + "Z"
        write_json_atomic(manifest_path, manifest)


if __name__ == "__main__":
    main()
