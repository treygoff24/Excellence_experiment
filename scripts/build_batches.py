from __future__ import annotations
import os
import json
import argparse
from tqdm import tqdm
from config.schema import load_config
import hashlib
from datetime import datetime
from typing import Dict, Any, Iterable
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


MAX_LINES_PER_SHARD = 9_999


class _ShardWriter:
    """Rotate JSONL shard files once a max line threshold is hit."""

    def __init__(self, base_path: str, *, max_lines: int = MAX_LINES_PER_SHARD) -> None:
        self.base_path = base_path
        self.max_lines = max_lines
        self._parts: list[dict[str, Any]] = []
        self._current: dict[str, Any] | None = None

    def _part_path(self, index: int) -> str:
        if index == 0:
            return self.base_path
        stem, ext = os.path.splitext(self.base_path)
        return f"{stem}_part{index + 1:02d}{ext}"

    def _open_new_part(self) -> None:
        part_index = len(self._parts)
        path = self._part_path(part_index)
        os.makedirs(os.path.dirname(path), exist_ok=True)
        fout = open(path, "w", encoding="utf-8")
        part = {
            "path": path,
            "file": fout,
            "lines": 0,
            "bytes": 0,
            "sha256": hashlib.sha256(),
            "part_index": part_index,
        }
        self._parts.append(part)
        self._current = part

    def write_json(self, payload: dict[str, Any]) -> None:
        if self._current is None or self._current["lines"] >= self.max_lines:
            self._close_current()
            self._open_new_part()
        line = json.dumps(payload, ensure_ascii=False)
        data = (line + "\n").encode("utf-8")
        assert self._current is not None  # for mypy
        fout = self._current["file"]
        fout.write(line + "\n")
        self._current["lines"] += 1
        self._current["bytes"] += len(data)
        self._current["sha256"].update(data)

    def _close_current(self) -> None:
        if self._current is not None:
            self._current["file"].close()
            self._current = None

    def finalize(self) -> list[dict[str, Any]]:
        self._close_current()
        finalized: list[dict[str, Any]] = []
        for part in self._parts:
            if part["lines"] == 0:
                try:
                    os.remove(part["path"])
                except FileNotFoundError:
                    pass
                continue
            finalized.append(
                {
                    "path": part["path"],
                    "lines": part["lines"],
                    "size": part["bytes"],
                    "sha256": part["sha256"].hexdigest(),
                    "part_index": part["part_index"],
                }
            )
        return finalized


def _remove_stale_shards(
    manifest: dict[str, Any],
    *,
    prompt_set: str,
    temp: float,
    condition: str,
    keep_files: Iterable[str],
) -> None:
    shards = manifest.setdefault("shards", {})
    keep = {os.path.basename(path) for path in keep_files}
    for name, meta in list(shards.items()):
        if meta.get("prompt_set") != prompt_set:
            continue
        try:
            meta_temp = float(meta.get("temp"))
        except (TypeError, ValueError):
            continue
        if abs(meta_temp - float(temp)) > 1e-6:
            continue
        if meta.get("condition") != condition:
            continue
        if name not in keep:
            shards.pop(name, None)


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
    ap.add_argument("--limit_items", type=int, default=None, help="Take only the first N items from each dataset split")
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
    if args.limit_items is not None and args.limit_items > 0:
        limit = int(args.limit_items)
        ob_rows = ob_rows[:limit]
        cb_rows = cb_rows[:limit]

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
        control_writer = _ShardWriter(out_control, max_lines=MAX_LINES_PER_SHARD)
        treatment_writer = _ShardWriter(out_treat, max_lines=MAX_LINES_PER_SHARD)
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
                        control_writer.write_json(payload)
                    # TREATMENT
                    cid = f"{custom_prefix}|treatment|{float(t):.1f}|{k}|{typ}"
                    if cid in seen_treat:
                        skipped_treat += 1
                    else:
                        seen_treat.add(cid)
                        payload = build_line(cid, treatment_system, user, stop=(stop_cfg if include_row_stop else None))
                        treatment_writer.write_json(payload)
        control_entries = control_writer.finalize()
        treatment_entries = treatment_writer.finalize()
        all_paths = [entry["path"] for entry in control_entries + treatment_entries]
        if all_paths:
            print("Wrote shards:", ", ".join(all_paths))
        if skipped_control or skipped_treat:
            print(
                f"De-dup summary @ T={float(t):.1f}: skipped {skipped_control} duplicate custom_id(s) in control, {skipped_treat} in treatment."
            )
        # Remove stale manifest entries for this prompt/temp/condition combo
        _remove_stale_shards(manifest, prompt_set=ps_name, temp=t, condition="control", keep_files=[e["path"] for e in control_entries])
        _remove_stale_shards(manifest, prompt_set=ps_name, temp=t, condition="treatment", keep_files=[e["path"] for e in treatment_entries])

        # Update manifest entries for these shards
        ctrl_meta = {
            "prompt_set": ps_name,
            "condition": "control",
            "temp": float(t),
        }
        trt_meta = {
            "prompt_set": ps_name,
            "condition": "treatment",
            "temp": float(t),
        }
        shards = manifest.setdefault("shards", {})
        for entry in control_entries:
            meta = dict(ctrl_meta)
            meta.update(
                {
                    "path": os.path.basename(entry["path"]),
                    "lines": entry["lines"],
                    "size": entry["size"],
                    "sha256": entry["sha256"],
                    "part_index": entry["part_index"],
                }
            )
            shards[os.path.basename(entry["path"])] = meta
        for entry in treatment_entries:
            meta = dict(trt_meta)
            meta.update(
                {
                    "path": os.path.basename(entry["path"]),
                    "lines": entry["lines"],
                    "size": entry["size"],
                    "sha256": entry["sha256"],
                    "part_index": entry["part_index"],
                }
            )
            shards[os.path.basename(entry["path"])] = meta
        # Maintain temps list as sorted unique values
        manifest.setdefault("temps", [])
        manifest["temps"] = sorted(list(set([float(x) for x in (manifest.get("temps") or [])] + [float(t)])))
        manifest["updated_at"] = datetime.utcnow().isoformat() + "Z"
        write_json_atomic(manifest_path, manifest)


if __name__ == "__main__":
    import multiprocessing as mp

    mp.freeze_support()
    main()
