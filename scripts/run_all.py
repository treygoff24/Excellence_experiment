from __future__ import annotations
import os
import sys
import json
import hashlib
import argparse
import subprocess
import re
import shutil
import yaml
import csv
import tarfile
import zipfile
import gzip
import shutil
import fcntl
import tempfile
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
from dotenv import load_dotenv
from fireworks.upload_dataset import create_dataset, upload_dataset_file
from fireworks.poll_and_download import poll_until_done, get_dataset, try_download_external_url, _try_extract_jsonls
from scripts import manifest_v2 as mf
from fireworks.batch_queue_manager import QueueManager
from scripts.state_utils import StopToken
from config.schema import load_config
from scripts.state_utils import (
    RunStateLock,
    write_json_atomic,
    init_run_state,
    load_run_state,
    update_phase,
    run_state_path,
    StopToken,
    compute_config_hash,
)


def _split_list_arg(val) -> list[str]:
    """Normalize an argparse value into a flat list.

    Accepts None, a comma-separated string, or a list of strings (space-separated usage).
    """
    if val is None:
        return []
    if isinstance(val, (list, tuple)):
        parts: list[str] = []
        for x in val:
            parts.extend(p.strip() for p in str(x).split(",") if p and p.strip())
        return parts
    return [p.strip() for p in str(val).split(",") if p and p.strip()]


def _format_temp_label(t: float) -> str:
    s = f"{float(t):.1f}"
    return "0" if s == "0.0" else s.replace(".", "")


def token_len(text: str) -> int:
    try:
        import tiktoken
        enc = tiktoken.get_encoding("cl100k_base")
        return len(enc.encode(text))
    except Exception:
        return len(text.split())


def _format_float_for_slug(x: float) -> str:
    try:
        s = f"{float(x):.2f}"
    except Exception:
        return str(x)
    return s.replace(".", "").rstrip("0") or "0"


def _resolve_model_id(raw_model: str, aliases: dict) -> str:
    return (aliases or {}).get(raw_model, raw_model)


def _model_short_name(model_id: str, aliases: dict) -> str:
    for k, v in (aliases or {}).items():
        if v == model_id:
            return k
    base = model_id.split("/")[-1]
    return re.sub(r"[^a-zA-Z0-9_-]+", "-", base)


def _trial_slug(model_id: str, prompt_set: str, top_p: float | None, top_k: int | None, mx: dict, *, aliases: dict | None = None) -> str:
    parts = [
        _model_short_name(model_id, aliases or {}),
        prompt_set,
    ]
    if top_p is not None:
        parts.append(f"tp{_format_float_for_slug(top_p)}")
    if top_k is not None:
        parts.append(f"tk{int(top_k)}")
    try:
        ob = int(mx.get("open_book")) if isinstance(mx, dict) else None
        cb = int(mx.get("closed_book")) if isinstance(mx, dict) else None
        if ob is not None and cb is not None:
            parts.append(f"mx{cb}-{ob}")
    except Exception:
        pass
    return "-".join(str(p) for p in parts if p)


def _abbr_prompt_set(name: str) -> str:
    """Create a compact, stable abbreviation for long prompt set names.

    Example: "structure_without_content" -> "swc"; "length_matched_best_practices" -> "lmbp".
    Falls back to a short hash if the abbreviation is too short.
    """
    if not name:
        return "def"
    tokens = re.split(r"[^a-zA-Z0-9]+", str(name))
    letters = "".join(t[0] for t in tokens if t)
    if len(letters) >= 3:
        return letters.lower()
    # Fallback: first 6 chars of sha1
    h = hashlib.sha1(str(name).encode()).hexdigest()[:6]
    return (letters + h).lower()


def _make_dataset_display_name(ps_name: str, t_label: str, cond: str, run_id: str, part_number: int) -> str:
    """Construct a dataset display name guaranteed to be <64 chars.

    Strategy: use abbreviated prompt set + short run id; aggressively trim if needed.
    """
    ps_abbr = _abbr_prompt_set(ps_name)
    rid_short = (run_id or "").strip()
    if len(rid_short) > 8:
        rid_short = rid_short[-8:]
    base = f"ex-{ps_abbr}-t{t_label}-{cond}-{rid_short}-p{int(part_number):02d}"
    if len(base) < 64:
        return base
    # Further trim: shorten cond and ps_abbr if still too long
    cond_short = cond[0] if cond else "c"
    ps_short = ps_abbr[:6]
    rid_short = rid_short[-6:] if len(rid_short) > 6 else rid_short
    base2 = f"x-{ps_short}-t{t_label}-{cond_short}-{rid_short}-p{int(part_number):02d}"
    if len(base2) < 64:
        return base2
    # Last resort: use hash suffix
    suffix = hashlib.sha1(base.encode()).hexdigest()[:6]
    base3 = f"x-{ps_short}-t{t_label}-{cond_short}-{suffix}-p{int(part_number):02d}"
    return base3[:63]


def _expand_trials(cfg: dict, args) -> list[dict]:
    aliases = cfg.get("model_aliases", {}) or {}
    cli_models = _split_list_arg(getattr(args, "models", None))
    cfg_models = cfg.get("models") or []
    base_models = cli_models or cfg_models or [cfg["model_id"]]
    models_full = [_resolve_model_id(m, aliases) for m in base_models]

    cli_ps = _split_list_arg(getattr(args, "prompt_sets", None))
    ps_cfg = cfg.get("prompt_sets") or {}
    default_ps = cfg.get("default_prompt_set") or (sorted(list(ps_cfg.keys()))[0] if ps_cfg else "default")
    base_ps = cli_ps or [default_ps]

    def _temps_from_arg_or_cfg():
        if getattr(args, "temps", None):
            vals: list[float] = []
            for tok in _split_list_arg(args.temps):
                # Tolerate stray '-' or mistakenly concatenated tokens like 'parts_per_dataset=24'
                if tok in ("-", "--"):
                    print("Warning: ignoring stray '-' token in --temps")
                    continue
                try:
                    vals.append(float(tok))
                except Exception:
                    # Ignore clearly non-numeric tokens that likely belong to other flags
                    if any(ch.isalpha() for ch in tok) or ("=" in tok):
                        print(f"Warning: ignoring non-numeric token in --temps: {tok}")
                        continue
                    # Last resort: re-raise
                    raise
            if not vals:
                return cfg.get("temps") or [0.0]
            return vals
        return cfg.get("temps") or [0.0]

    base_temps = _temps_from_arg_or_cfg()
    base_top_p = cfg.get("top_p")
    base_top_k = cfg.get("top_k")
    base_mx = cfg.get("max_new_tokens", {"closed_book": 1024, "open_book": 1024})

    trials: list[dict] = []
    if cfg.get("trials"):
        for tr in cfg["trials"]:
            md = tr.get("model_id") or tr.get("model") or cfg["model_id"]
            md_full = _resolve_model_id(md, aliases)
            ps = tr.get("prompt_set") or default_ps
            temps = tr.get("temps") or base_temps
            trial = {
                "id": tr.get("id"),
                "model_id": md_full,
                "prompt_set": ps,
                "temps": [float(t) for t in temps],
                "top_p": tr.get("top_p", base_top_p),
                "top_k": tr.get("top_k", base_top_k),
                "max_new_tokens": (tr.get("max_new_tokens") or base_mx),
            }
            trials.append(trial)
        return trials

    sw = cfg.get("sweep") or {}
    sw_models = sw.get("models") or base_models
    sw_ps = sw.get("prompt_sets") or base_ps
    sw_temps = sw.get("temps") or base_temps
    sw_top_p = sw.get("top_p") or [base_top_p]
    sw_top_k = sw.get("top_k") or [base_top_k]
    mx_sw = sw.get("max_new_tokens")
    if mx_sw:
        ob_list = mx_sw.get("open_book") or [base_mx.get("open_book", 1024)]
        cb_list = mx_sw.get("closed_book") or [base_mx.get("closed_book", 1024)]
        mx_list = [{"open_book": int(ob), "closed_book": int(cb)} for ob in ob_list for cb in cb_list]
    else:
        mx_list = [base_mx]

    for m in sw_models:
        m_full = _resolve_model_id(m, aliases)
        for ps in sw_ps:
            for tp in sw_top_p:
                for tk in sw_top_k:
                    for mx in mx_list:
                        trials.append({
                            "id": None,
                            "model_id": m_full,
                            "prompt_set": ps,
                            "temps": [float(t) for t in sw_temps],
                            "top_p": (None if tp is None else float(tp)),
                            "top_k": (None if tk is None else int(tk)),
                            "max_new_tokens": mx,
                        })
    if not trials:
        trials.append({
            "id": None,
            "model_id": models_full[0],
            "prompt_set": base_ps[0],
            "temps": [float(t) for t in base_temps],
            "top_p": base_top_p,
            "top_k": base_top_k,
            "max_new_tokens": base_mx,
        })
    return trials


def ensure_dirs(cfg: dict):
    for k in ["prepared_dir", "batch_inputs_dir", "results_dir", "reports_dir"]:
        os.makedirs(cfg["paths"][k], exist_ok=True)


def run_cmd(args: list[str]):
    print("+", " ".join(args))
    sys.stdout.flush()
    subprocess.run(args, check=True)


def _validate_manifest_schema(data: dict) -> bool:
    """Validate manifest structure and required fields.

    Supports:
    - Per-trial manifest v1 (legacy) and v2 (schema_version=2)
    - Multi-trial summary (includes trials list and num_trials)
    """
    try:
        # Multi-trial summary manifest
        if isinstance(data, dict) and "trials" in data:
            required_fields = ["created_utc", "run_id", "num_trials", "trials"]
            for field in required_fields:
                if field not in data:
                    print(f"Manifest validation failed: missing field '{field}'")
                    return False
            if not isinstance(data.get("trials"), list):
                print("Manifest validation failed: 'trials' must be a list")
                return False
            # Light validation of entries
            for entry in data.get("trials", []):
                tr = (entry or {}).get("trial", {})
                if tr and ("model_id" not in tr or "prompt_set" not in tr):
                    print("Manifest validation failed: multi-trial entry missing trial metadata")
                    return False
            return True

        # Per-trial manifest v2
        if int(data.get("schema_version", 0)) == 2:
            ts = data.get("timestamps", {}) or {}
            if not isinstance(ts, dict) or not ts.get("created_at"):
                print("Manifest validation failed: v2 requires timestamps.created_at")
                return False
            if not isinstance(data.get("stage_status", {}), dict):
                print("Manifest validation failed: v2 requires stage_status dict")
                return False
            trial = data.get("trial", {})
            if not isinstance(trial, dict) or ("model_id" not in trial or "prompt_set" not in trial):
                print("Manifest validation failed: v2 missing trial metadata")
                return False
            if not isinstance(data.get("temps", []), list):
                print("Manifest validation failed: v2 requires temps list")
                return False
            return True

        # Per-trial manifest v1 (legacy)
        else:
            required_fields = ["created_utc", "run_id", "temps", "samples_per_item"]
            for field in required_fields:
                if field not in data:
                    print(f"Manifest validation failed: missing field '{field}'")
                    return False

            # Validate trial structure
            trial = data.get("trial", {})
            trial_fields = ["model_id", "prompt_set"]
            for field in trial_fields:
                if field not in trial:
                    print(f"Manifest validation failed: missing trial field '{field}'")
                    return False

            # Validate temps is a list
            if not isinstance(data.get("temps"), list):
                print(f"Manifest validation failed: 'temps' must be a list")
                return False

            # Validate job_status structure (if exists)
            job_status = data.get("job_status", {})
            if job_status and not isinstance(job_status, dict):
                print(f"Manifest validation failed: 'job_status' must be a dict")
                return False

            return True
    except Exception as e:
        print(f"Manifest validation error: {e}")
        return False


def write_manifest(path: str, data: dict):
    """Thread-safe atomic manifest writing with file locking and validation."""
    os.makedirs(os.path.dirname(path), exist_ok=True)

    # Validate schema before writing
    if not _validate_manifest_schema(data):
        raise ValueError(f"Manifest validation failed for {path}")

    # Write to temporary file first, then atomic move
    temp_path = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            dir=os.path.dirname(path),
            delete=False,
            prefix=".tmp_manifest_",
            suffix=".json"
        ) as temp_file:
            temp_path = temp_file.name

            # Acquire exclusive lock on temp file
            fcntl.flock(temp_file.fileno(), fcntl.LOCK_EX)

            # Write data to temp file
            json.dump(data, temp_file, indent=2)
            temp_file.flush()
            os.fsync(temp_file.fileno())

        # Atomic move to final location
        if os.name == 'nt':  # Windows
            # Windows doesn't support atomic replace, so remove first
            if os.path.exists(path):
                os.remove(path)
        os.rename(temp_path, path)
        temp_path = None  # Successfully moved

    except Exception as e:
        # Clean up temp file on error
        if temp_path and os.path.exists(temp_path):
            try:
                os.unlink(temp_path)
            except Exception:
                pass
        raise e


def _split_jsonl_file(
    src_path: str,
    out_dir: str,
    base_prefix: str,
    parts: int | None = None,
    *,
    lines_per_part: int | None = None,
    limit_items: int | None = None,
) -> list[tuple[int, str]]:
    """Split a JSONL file into N approximately equal parts with validation.

    Returns a list of (part_number, part_path). If parts <= 1, returns [(1, src_path)].
    If part files already exist and are valid, reuse them.
    """
    # Validate input file exists
    if not os.path.exists(src_path):
        raise FileNotFoundError(f"Source JSONL file not found: {src_path}")

    # Determine splitting mode: by lines per part or fixed number of parts
    parts = int(parts) if parts and int(parts) > 1 else None
    lines_per_part = int(lines_per_part) if lines_per_part and int(lines_per_part) > 0 else None
    if parts is None and lines_per_part is None:
        # nothing to do - validate source file has content
        if os.path.getsize(src_path) == 0:
            print(f"Warning: Source file {src_path} is empty")
        return [(1, src_path)]
    os.makedirs(out_dir, exist_ok=True)
    # If splits already exist, validate and reuse them
    existing: list[tuple[int, str]] = []
    if parts is not None:
        for i in range(1, parts + 1):
            p = os.path.join(out_dir, f"{base_prefix}.p{i:02d}.jsonl")
            if os.path.exists(p) and os.path.getsize(p) >= 0:  # Allow empty parts
                existing.append((i, p))
        if len(existing) == parts:
            # Validate that split files contain expected total lines
            total_split_lines = 0
            try:
                for _, split_path in existing:
                    with open(split_path, 'r', encoding='utf-8') as f:
                        total_split_lines += sum(1 for line in f if line.strip())

                # Count original lines for comparison
                with open(src_path, 'r', encoding='utf-8') as f:
                    original_lines = sum(1 for line in f if line.strip())
                    if limit_items:
                        original_lines = min(original_lines, int(limit_items))

                if total_split_lines == original_lines:
                    print(f"Reusing validated splits: {len(existing)} parts, {total_split_lines} total lines")
                    return existing
                else:
                    print(f"Split validation failed: {total_split_lines} != {original_lines} lines, regenerating")
            except Exception as e:
                print(f"Split validation error: {e}, regenerating")
    # Read all lines once and split (optionally limit first N lines)
    with open(src_path, "r", encoding="utf-8") as f:
        if limit_items is not None and int(limit_items) > 0:
            lines: list[str] = []
            for ln in f:
                s = ln.strip()
                if not s:
                    continue
                lines.append(ln)
                if len(lines) >= int(limit_items):
                    break
        else:
            lines = [ln for ln in f.readlines() if ln.strip()]
    n = len(lines)
    if n == 0:
        # create empty part files
        out_files: list[tuple[int, str]] = []
        for i in range(1, parts + 1):
            p = os.path.join(out_dir, f"{base_prefix}.p{i:02d}.jsonl")
            open(p, "w", encoding="utf-8").close()
            out_files.append((i, p))
        return out_files
    # Decide split sizes
    if lines_per_part is not None:
        chunk = max(1, int(lines_per_part))
        parts = max(1, (n + chunk - 1) // chunk)
    else:
        parts = max(1, int(parts) or 1)
        chunk = max(1, (n + parts - 1) // parts)
    out_files: list[tuple[int, str]] = []
    for i in range(parts):
        start = i * chunk
        end = min(n, (i + 1) * chunk)
        if start >= n:
            # still create an empty file for consistency
            part_lines: list[str] = []
        else:
            part_lines = lines[start:end]
        p = os.path.join(out_dir, f"{base_prefix}.p{i+1:02d}.jsonl")
        with open(p, "w", encoding="utf-8") as out:
            out.writelines(part_lines)
        out_files.append((i + 1, p))

    # Post-split validation
    try:
        total_written = sum(len([line for line in open(path, 'r', encoding='utf-8') if line.strip()])
                            for _, path in out_files)
        if total_written != n:
            print(f"Warning: Split validation failed - wrote {total_written} lines, expected {n}")
        else:
            print(f"Split validation passed: {parts} parts, {total_written} total lines")
    except Exception as e:
        print(f"Warning: Could not validate split: {e}")

    return out_files


def main():
    load_dotenv()
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="config/eval_config.yaml")
    ap.add_argument("--account_id", default=os.environ.get("FIREWORKS_ACCOUNT_ID"))
    ap.add_argument("--condition", choices=["control", "treatment", "both"], default="both")
    ap.add_argument("--skip_prepare", action="store_true")
    ap.add_argument("--skip_build", action="store_true")
    ap.add_argument("--skip_batch", action="store_true")
    ap.add_argument("--run_id", help="Custom run ID (auto-generated if not provided)")
    # Overrides and planning
    ap.add_argument("--models", nargs="+", help="Models or aliases (space or comma separated)")
    ap.add_argument("--prompt_sets", nargs="+", help="Prompt set names (space or comma separated)")
    ap.add_argument("--temps", nargs="+", help="Temperatures override (space or comma separated)")
    ap.add_argument("--plan_only", action="store_true", help="Show phase plan and exit")
    ap.add_argument("--only_step", choices=[
        "prepare","build","submit","poll","parse","score","stats","costs","report"
    ], help="Run only the specified phase")
    ap.add_argument("--from_step", choices=[
        "prepare","build","submit","poll","parse","score","stats","costs","report"
    ], help="Start from this phase (inclusive)")
    ap.add_argument("--to_step", choices=[
        "prepare","build","submit","poll","parse","score","stats","costs","report"
    ], help="Stop after this phase (inclusive)")
    ap.add_argument("--archive", action="store_true", help="Archive results after completion")
    # Stop/Resume behavior controls
    ap.add_argument("--ignore_stop", action="store_true", help="Ignore STOP_REQUESTED sentinel (still honors Ctrl-C for this process)")
    ap.add_argument("--stop_stale_minutes", type=int, default=60, help="Treat STOP_REQUESTED older than N minutes as stale and ignore (default: 60)")
    ap.add_argument("--max_concurrent_jobs", type=int, default=4, help="Max concurrent Fireworks batch jobs (default: 4)")
    ap.add_argument("--parts_per_dataset", type=int, default=None, help="How many parts to split each input into (decoupled from concurrency)")
    ap.add_argument("--lines_per_part", type=int, default=None, help="Alternatively, target number of lines per part (overrides parts_per_dataset)")
    ap.add_argument("--limit_items", type=int, default=None, help="Take only the first N items from each input when splitting (useful for smoke/dry runs)")
    ap.add_argument("--resume", action="store_true", help="Resume from existing manifests; skip uploaded datasets and completed jobs")
    ap.add_argument("--dry_run", action="store_true", help="Offline mode: do not hit Fireworks; synthesize completed jobs and results for testing")
    args = ap.parse_args()
    cfg = load_config(args.config)

    # Generate or use provided run_id
    run_id = args.run_id or datetime.utcnow().strftime("r%Y%m%d%H%M%S")

    # Expand trials and determine run root
    experiments_dir = cfg.get("paths", {}).get("experiments_dir", "experiments")
    trials = _expand_trials(cfg, args)
    use_exp_root = os.path.exists(experiments_dir)
    if use_exp_root:
        run_root = os.path.join(experiments_dir, f"run_{run_id}")
        os.makedirs(run_root, exist_ok=True)
        cfg["paths"]["batch_inputs_dir"] = os.path.join(run_root, "batch_inputs")
        multi_manifest_path = os.path.join(run_root, "multi_trial_manifest.json")
    else:
        run_root = None
        multi_manifest_path = cfg["paths"]["run_manifest"]

    # When using experiments dir, persist an effective config so subprocesses see updated paths
    effective_config_path = args.config
    if use_exp_root:
        try:
            effective_config_path = os.path.join(run_root, "effective_config.yaml")
            with open(effective_config_path, "w", encoding="utf-8") as f:
                yaml.safe_dump(cfg, f, sort_keys=False)
        except Exception as e:
            print(f"WARNING: Failed to write effective config: {e}. Using original config.")

    # Compute the ordered phases and gating selection
    PHASES = ["prepare","build","submit","poll","parse","score","stats","costs","report"]
    def _compute_phase_selection() -> list[str]:
        if args.only_step:
            return [args.only_step]
        if args.from_step and args.to_step:
            si = PHASES.index(args.from_step)
            ei = PHASES.index(args.to_step)
            if si > ei:
                raise SystemExit("Invalid gating: --from_step must be <= --to_step")
            return PHASES[si:ei+1]
        if args.from_step:
            return PHASES[PHASES.index(args.from_step):]
        if args.to_step:
            return PHASES[:PHASES.index(args.to_step)+1]
        return PHASES[:]

    selected_phases = _compute_phase_selection()

    # Initialize run-level state and stop token (honors STOP file semantics)
    stop_token = StopToken(
        run_root or os.getcwd(),
        ignore_file=bool(args.ignore_stop),
        stale_minutes=int(args.stop_stale_minutes) if args.stop_stale_minutes is not None else None,
    )
    state = None
    if use_exp_root:
        # Ensure run root exists
        os.makedirs(run_root, exist_ok=True)
        state = load_run_state(run_root)
        if args.resume and state is None:
            print(f"ERROR: --resume requested but missing run_state.json at {run_state_path(run_root)}")
            print("Hint: Try --plan_only to inspect, or run without --resume for a new run.")
            raise SystemExit(2)
        if state is None:
            state = init_run_state(run_root, run_id, cfg)
            with RunStateLock(run_root):
                write_json_atomic(run_state_path(run_root), state)
        else:
            # Best-effort config hash check
            try:
                cfg_hash = compute_config_hash(cfg)
                if cfg_hash != state.get("config_hash"):
                    print("WARNING: Effective config differs from recorded config_hash; proceed with caution for resume.")
            except Exception:
                pass

    # Helpers to compute simple done-predicates from on-disk artifacts
    def _prepared_done() -> bool:
        pdir = cfg["paths"]["prepared_dir"]
        open_b = os.path.join(pdir, "open_book.jsonl")
        closed_b = os.path.join(pdir, "closed_book.jsonl")
        return os.path.isfile(open_b) and os.path.getsize(open_b) > 0 and os.path.isfile(closed_b) and os.path.getsize(closed_b) > 0

    def _build_done() -> bool:
        prompt_sets_cfg = cfg.get("prompt_sets") or {}
        default_ps = cfg.get("default_prompt_set") or (sorted(list(prompt_sets_cfg.keys()))[0] if prompt_sets_cfg else "default")
        temps_per_ps: dict[str, set[float]] = {}
        for tr in trials:
            psn = tr.get("prompt_set") or default_ps
            temps_per_ps.setdefault(psn, set()).update(float(t) for t in (tr.get("temps") or cfg.get("temps") or [0.0]))
        for psn, tset in temps_per_ps.items():
            suffix = f"_{psn}" if (len(prompt_sets_cfg) > 1 or psn not in ("default", None)) else ""
            for t in sorted(tset):
                t_str = _format_temp_label(float(t))
                for cond in ("control","treatment"):
                    fpath = os.path.join(cfg["paths"]["batch_inputs_dir"], f"t{t_str}{suffix}_{cond}.jsonl")
                    if not (os.path.isfile(fpath) and os.path.getsize(fpath) > 0):
                        return False
        return True

    def _trial_dirs() -> list[tuple[dict, str, str]]:
        prompt_sets_cfg = cfg.get("prompt_sets") or {}
        default_ps = cfg.get("default_prompt_set") or (sorted(list(prompt_sets_cfg.keys()))[0] if prompt_sets_cfg else "default")
        out: list[tuple[dict, str, str]] = []
        for trial in trials:
            model_id = trial["model_id"]
            ps_name = trial.get("prompt_set") or default_ps
            top_p = trial.get("top_p", cfg.get("top_p"))
            top_k = trial.get("top_k", cfg.get("top_k"))
            mx = trial.get("max_new_tokens") or cfg.get("max_new_tokens", {"closed_book": 1024, "open_book": 1024})
            if os.path.exists(experiments_dir):
                slug = _trial_slug(model_id, ps_name, top_p, top_k, mx, aliases=cfg.get("model_aliases") or {})
                trial_root = os.path.join(experiments_dir, f"run_{run_id}", slug)
                results_dir = os.path.join(trial_root, "results")
                reports_dir = os.path.join(trial_root, "reports")
            else:
                results_dir = cfg["paths"]["results_dir"]
                reports_dir = cfg["paths"]["reports_dir"]
            out.append((trial, results_dir, reports_dir))
        return out

    def _submit_done() -> bool:
        # Minimal check: manifests exist and have jobs entries
        for _trial, results_dir, _reports_dir in _trial_dirs():
            mp = os.path.join(results_dir, "trial_manifest.json")
            try:
                with open(mp, "r", encoding="utf-8") as f:
                    m = json.load(f)
                jobs = m.get("jobs") or {}
                if not jobs:
                    return False
            except Exception:
                return False
        return True

    def _poll_done() -> bool:
        # Done if any results JSONL detected or combined exists and has content
        for _trial, results_dir, _reports_dir in _trial_dirs():
            combined = os.path.join(results_dir, "results_combined.jsonl")
            if os.path.isfile(combined) and os.path.getsize(combined) > 0:
                continue
            any_jsonl = False
            for root, _d, files in os.walk(results_dir):
                if any(nm.lower().endswith('.jsonl') for nm in files):
                    any_jsonl = True
                    break
            if not any_jsonl:
                return False
        return True

    def _parse_done() -> bool:
        for _trial, results_dir, _reports_dir in _trial_dirs():
            if not os.path.isfile(os.path.join(results_dir, "predictions.csv")):
                return False
        return True

    def _score_done() -> bool:
        for _trial, results_dir, _reports_dir in _trial_dirs():
            if not os.path.isfile(os.path.join(results_dir, "per_item_scores.csv")):
                return False
        return True

    def _stats_done() -> bool:
        for _trial, results_dir, _reports_dir in _trial_dirs():
            if not os.path.isfile(os.path.join(results_dir, "significance.json")):
                return False
        return True

    def _costs_done() -> bool:
        for _trial, results_dir, _reports_dir in _trial_dirs():
            if not os.path.isfile(os.path.join(results_dir, "costs.json")):
                return False
        return True

    def _report_done() -> bool:
        for _trial, _results_dir, reports_dir in _trial_dirs():
            if not os.path.isfile(os.path.join(reports_dir, "report.md")):
                return False
        return True

    phase_done_fn = {
        "prepare": _prepared_done,
        "build": _build_done,
        "submit": _submit_done,
        "poll": _poll_done,
        "parse": _parse_done,
        "score": _score_done,
        "stats": _stats_done,
        "costs": _costs_done,
        "report": _report_done,
    }

    # Plan-only mode prints the phase plan and exits
    if args.plan_only:
        plan_lines: list[str] = []
        plan_lines.append(f"Run ID: {run_id}")
        plan_lines.append(f"Config hash: {compute_config_hash(cfg)}")
        plan_lines.append("Phases:")
        for ph in PHASES:
            selected = (ph in selected_phases)
            try:
                done = phase_done_fn[ph]()
            except Exception:
                done = False
            action = "skip" if (not selected or done) else "execute"
            reason = "not selected" if not selected else ("already done" if done else "pending")
            plan_lines.append(f"- {ph}: {action} ({reason})")
        print("\n".join(plan_lines))
        return

    ensure_dirs(cfg)

    # Phase: prepare
    if "prepare" in selected_phases and not _prepared_done():
        if state is not None:
            with RunStateLock(run_root):
                update_phase(state, "prepare", status="in_progress")
                write_json_atomic(run_state_path(run_root), state)
        stop_token.check()
        run_cmd([sys.executable, "-m", "scripts.prepare_data", "--config", effective_config_path])
        if state is not None:
            with RunStateLock(run_root):
                update_phase(state, "prepare", status="completed")
                write_json_atomic(run_state_path(run_root), state)
    else:
        print("Gating: skipping prepare (already done or not selected)")

    # Phase: build
    if "build" in selected_phases and not _build_done():
        if state is not None:
            with RunStateLock(run_root):
                update_phase(state, "build", status="in_progress")
                write_json_atomic(run_state_path(run_root), state)
        stop_token.check()
        # Build once per prompt set with union of temps across trials
        temps_per_ps: dict[str, set[float]] = {}
        prompt_sets_cfg = cfg.get("prompt_sets") or {}
        default_ps = cfg.get("default_prompt_set") or (sorted(list(prompt_sets_cfg.keys()))[0] if prompt_sets_cfg else "default")
        for tr in trials:
            psn = tr.get("prompt_set") or default_ps
            temps_per_ps.setdefault(psn, set()).update(float(t) for t in (tr.get("temps") or cfg.get("temps") or [0.0]))
        for psn, tset in temps_per_ps.items():
            temps_arg = ",".join(str(t) for t in sorted(tset))
            run_cmd([sys.executable, "-m", "scripts.build_batches", "--config", effective_config_path, "--prompt_set", psn, "--temps", temps_arg])
        if state is not None:
            with RunStateLock(run_root):
                update_phase(state, "build", status="completed")
                write_json_atomic(run_state_path(run_root), state)
    else:
        print("Gating: skipping build (already done or not selected)")
    if args.skip_batch:
        # Write a minimal multi-trial plan and exit
        plan = {"created_utc": datetime.utcnow().isoformat() + "Z", "run_id": run_id, "num_trials": len(trials), "trials": trials}
        write_manifest(multi_manifest_path, plan)
        print("Prepared inputs only (skip_batch). Plan written:", multi_manifest_path)
        return

    def _derive_account_id(raw: str | None) -> str:
        """Return a Fireworks account slug suitable for /v1/accounts/{slug}/... paths.

        Accepts any of:
        - Bare slug (e.g., "my-team") → returns as-is
        - Full resource (e.g., "accounts/my-team" or "accounts/my-team/models/...") → extracts "my-team"
        - Email or invalid string → raises with guidance
        """
        # Helper: extract first segment after "accounts/" from a resource string
        def _extract_from_resource(s: str) -> str | None:
            m = re.search(r"accounts/([^/]+)", s)
            return m.group(1) if m else None

        s = (raw or "").strip()
        # If the provided value already looks like a usable slug
        if s and "/" not in s and "@" not in s:
            return s
        # If an email was provided, try a best-effort derivation from domain root
        if s and "@" in s:
            try:
                domain = s.split("@", 1)[1]
                slug = domain.split(".", 1)[0]
                if slug:
                    print(f"Note: Deriving FIREWORKS_ACCOUNT_ID='{slug}' from email domain '{domain}'.")
                    return slug
            except Exception:
                pass
        # If a full resource was provided, e.g., accounts/foo or accounts/foo/models/bar
        if s:
            from_res = _extract_from_resource(s)
            if from_res:
                return from_res
        raise SystemExit(
            "Unable to determine Fireworks account id. Set FIREWORKS_ACCOUNT_ID to your account slug (e.g., 'my-team'), not an email."
        )

    account_id = _derive_account_id(args.account_id) if not args.dry_run else "dryrun"
    conditions: list[str] = [args.condition] if args.condition in ("control", "treatment") else ["control", "treatment"]
    # Submit datasets and jobs per trial and write per-trial manifests
    prompt_sets_cfg = cfg.get("prompt_sets") or {}
    default_ps = cfg.get("default_prompt_set") or (sorted(list(prompt_sets_cfg.keys()))[0] if prompt_sets_cfg else "default")
    if "submit" in selected_phases and not _submit_done():
        if state is not None:
            with RunStateLock(run_root):
                update_phase(state, "submit", status="in_progress")
                write_json_atomic(run_state_path(run_root), state)
        for trial in trials:
            stop_token.check()
        model_id = trial["model_id"]
        ps_name = trial.get("prompt_set") or default_ps
        temps = trial.get("temps") or (cfg.get("temps") or [0.0])
        top_p = trial.get("top_p", cfg.get("top_p"))
        top_k = trial.get("top_k", cfg.get("top_k"))
        mx = trial.get("max_new_tokens") or cfg.get("max_new_tokens", {"closed_book": 1024, "open_book": 1024})

        # Trial directories
        if os.path.exists(experiments_dir):
            slug = _trial_slug(model_id, ps_name, top_p, top_k, mx, aliases=cfg.get("model_aliases") or {})
            trial_root = os.path.join(experiments_dir, f"run_{run_id}", slug)
            results_dir = os.path.join(trial_root, "results")
            reports_dir = os.path.join(trial_root, "reports")
        else:
            results_dir = cfg["paths"]["results_dir"]
            reports_dir = cfg["paths"]["reports_dir"]
        os.makedirs(results_dir, exist_ok=True)
        os.makedirs(reports_dir, exist_ok=True)

        # Prompts
        ps_meta = prompt_sets_cfg.get(ps_name) or {}
        ctrl_path = ps_meta.get("control", "config/prompts/control_system.txt")
        trt_path = ps_meta.get("treatment", "config/prompts/treatment_system.txt")
        control_prompt = open(ctrl_path, "r", encoding="utf-8").read()
        treatment_prompt = open(trt_path, "r", encoding="utf-8").read()

        # If resuming and a manifest exists, load and validate it; else start fresh
        existing_manifest: dict | None = None
        manifest_path_pre = os.path.join(results_dir, "trial_manifest.json")
        if args.resume and os.path.isfile(manifest_path_pre):
            try:
                existing_manifest, upgraded = mf.load_manifest(manifest_path_pre)
                if upgraded:
                    print(f"Upgraded manifest to v2 at {manifest_path_pre}")
            except Exception as e:
                print(f"Warning: Could not load/upgrade existing manifest: {e}")
                existing_manifest = None

        trial_manifest = existing_manifest or {
            "schema_version": 2,
            "timestamps": {"created_at": datetime.utcnow().isoformat() + "Z", "updated_at": datetime.utcnow().isoformat() + "Z"},
            "run_id": run_id,
            "trial": {
                "model_id": model_id,
                "prompt_set": ps_name,
                "top_p": top_p,
                "top_k": top_k,
                "max_new_tokens": mx,
            },
            "temps": [float(t) for t in temps],
            "samples_per_item": cfg["samples_per_item"],
            "prompts": {
                "control": {"sha256": hashlib.sha256(control_prompt.encode()).hexdigest(), "tokens": token_len(control_prompt)},
                "treatment": {"sha256": hashlib.sha256(treatment_prompt.encode()).hexdigest(), "tokens": token_len(treatment_prompt)},
            },
            "datasets": {},
            "jobs": {},
            "job_status": {},
            "stage_status": {},
        }

        for temp in temps:
            t_str = _format_temp_label(float(temp))
            for cond in conditions:
                suffix = f"_{ps_name}" if (len(prompt_sets_cfg) > 1 or ps_name not in ("default", None)) else ""
                jsonl_path = os.path.join(cfg["paths"]["batch_inputs_dir"], f"t{t_str}{suffix}_{cond}.jsonl")
                # Keep the human-readable trial display separate from dataset names (which must be short)
                display_name = f"excellence-{ps_name}-t{t_str}-{cond}-{run_id}"
                if not os.path.isfile(jsonl_path):
                    raise SystemExit(f"Missing input file: {jsonl_path}")

                # Split the dataset into up to max_concurrent parts for parallel jobs
                base_prefix = f"t{t_str}{suffix}_{cond}"
                part_files = _split_jsonl_file(
                    jsonl_path,
                    cfg["paths"]["batch_inputs_dir"],
                    base_prefix,
                    parts=(int(args.parts_per_dataset) if (args.lines_per_part is None and args.parts_per_dataset is not None) else None),
                    lines_per_part=(int(args.lines_per_part) if args.lines_per_part is not None else None),
                    limit_items=(int(args.limit_items) if args.limit_items is not None else None),
                )

                # Enforce Fireworks batch concurrency via QueueManager
                queue = QueueManager(
                    account_id=account_id,
                    model_id=model_id,
                    config=cfg,
                    max_concurrent=int(args.max_concurrent_jobs) if args.max_concurrent_jobs else 4,
                    temp_label=t_str,
                    temperature=float(temp),
                    condition=cond,
                    run_id=run_id,
                    stop_event=stop_token,
                )
                # If resuming and this temp/cond appears complete in manifest, skip queueing
                resume_key = f"t{t_str}_{cond}"
                if args.resume and existing_manifest:
                    jobs_map = (existing_manifest or {}).get("jobs", {}) or {}
                    job_status = (existing_manifest or {}).get("job_status", {}) or {}
                    names = jobs_map.get(resume_key) or []
                    if names and len(names) == len(part_files):
                        # Check that all parts appear completed AND have downloaded results
                        def _is_completed_and_downloaded(jkey: str) -> bool:
                            status = (str(job_status.get(jkey, "")).lower())
                            if not (("complete" in status) or (status == "completed") or (status == "downloaded")):
                                return False

                            # Check if results file exists locally
                            results_path = os.path.join(results_dir, jkey, "results.jsonl")
                            if os.path.exists(results_path) and os.path.getsize(results_path) > 0:
                                return True

                            # Check for combined results file
                            combined_path = os.path.join(results_dir, "results_combined.jsonl")
                            if os.path.exists(combined_path):
                                # Verify this job's results are in the combined file
                                try:
                                    with open(combined_path, 'r', encoding='utf-8') as f:
                                        for line in f:
                                            if line.strip() and jkey in line:
                                                return True
                                except Exception:
                                    pass

                            return False

                        all_done = True
                        for i in range(1, len(part_files) + 1):
                            jkey = f"{resume_key}_p{i:02d}"
                            if not _is_completed_and_downloaded(jkey):
                                all_done = False
                                print(f"Resume: {jkey} incomplete or missing results, will retry")
                                break
                        if all_done:
                            print(f"Resume: {resume_key} already completed; skipping queue.")
                            # ensure fields exist in manifest
                            trial_manifest.setdefault("datasets", {}).setdefault(resume_key, existing_manifest.get("datasets", {}).get(resume_key, []))
                            trial_manifest.setdefault("jobs", {}).setdefault(resume_key, names)
                            continue

                # Upload each part as a separate dataset and enqueue a job
                dsids_for_cond: list[str] = []
                planned_jobs: list[tuple[int, str]] = []
                for part_number, part_path in part_files:
                    # Per-part resume: skip parts already completed AND downloaded
                    jkey_resume = f"{resume_key}_p{part_number:02d}"
                    if args.resume and existing_manifest:
                        prev_status = (existing_manifest.get("job_status", {}) or {}).get(jkey_resume)
                        status_lower = str(prev_status or "").lower()
                        if ("complete" in status_lower) or (status_lower == "downloaded"):
                            # Also verify results file exists
                            results_path = os.path.join(results_dir, jkey_resume, "results.jsonl")
                            if os.path.exists(results_path) and os.path.getsize(results_path) > 0:
                                print(f"Resume: skipping already completed part {jkey_resume} (results verified)")
                                continue
                            else:
                                # Attempt repair via OUTPUT_DATASET_ID or job.json instead of requeueing
                                print(f"Resume: part {jkey_resume} marked completed but missing results; attempting repair download")
                                try:
                                    ds_id_txt = os.path.join(results_dir, f"{jkey_resume}_OUTPUT_DATASET_ID.txt")
                                    job_json = os.path.join(results_dir, f"{jkey_resume}_job.json")
                                    dsid = None
                                    if os.path.isfile(ds_id_txt):
                                        with open(ds_id_txt, "r", encoding="utf-8") as f:
                                            dsid = (f.read() or "").strip()
                                    if (not dsid) and os.path.isfile(job_json):
                                        try:
                                            with open(job_json, "r", encoding="utf-8") as f:
                                                jj = json.load(f)
                                            dsid = jj.get("outputDatasetId") or jj.get("output_dataset_id")
                                        except Exception:
                                            pass
                                    if dsid:
                                        try:
                                            ds = get_dataset(account_id, str(dsid).split("/")[-1])
                                            ext = ds.get("externalUrl") or ds.get("external_url")
                                        except Exception:
                                            ext = None
                                        if ext:
                                            job_dir = os.path.join(results_dir, jkey_resume)
                                            os.makedirs(job_dir, exist_ok=True)
                                            bundle = try_download_external_url(ext, job_dir)
                                            if bundle:
                                                extracted = _try_extract_jsonls(bundle, job_dir)
                                                if extracted:
                                                    print(f"Repair successful: downloaded results for {jkey_resume}")
                                                    continue
                                except Exception as _e:
                                    print(f"WARNING: repair attempt failed for {jkey_resume}: {_e}")
                    # Create a safe, short dataset display name (<64 chars) to satisfy API constraints
                    ds_name = _make_dataset_display_name(ps_name, t_str, cond, run_id, part_number)
                    if args.dry_run:
                        dsid = f"dryrun-{ds_name}"
                    else:
                        dsid = create_dataset(ds_name, account_id)
                        remote_fname = os.path.basename(part_path)
                        upload_dataset_file(account_id, dsid, part_path, filename=remote_fname)
                    dsids_for_cond.append(dsid)
                    queue.add_job(part_number, "", dsid)
                    planned_jobs.append((part_number, dsid))
                trial_manifest["datasets"][f"t{t_str}_{cond}"] = dsids_for_cond
                # Initialize manifest job_status entries before running the queue
                trial_manifest.setdefault("job_status", {})
                for part_number, _ in planned_jobs:
                    jkey = f"t{t_str}_{cond}_p{part_number:02d}"
                    trial_manifest["job_status"][jkey] = "pending"
                # Persist the manifest immediately (early state)
                mf.write_manifest(os.path.join(results_dir, "trial_manifest.json"), trial_manifest)

                # Dry-run path: simulate queue concurrency lifecycle (no API calls)
                if args.dry_run:
                    try:
                        # Exercise queue scheduling beyond concurrency limit without submitting real jobs
                        queue.run_queue_simulated(results_dir)
                    except Exception:
                        # Best-effort; continue to synthesize artifacts
                        pass

                    # Then synthesize completion and results for each part
                    import json as _json
                    os.makedirs(results_dir, exist_ok=True)
                    for part_number, dsid in planned_jobs:
                        jkey = f"t{t_str}_{cond}_p{part_number:02d}"
                        trial_manifest["job_status"][jkey] = "completed"
                        # Create a dummy job.json and results
                        with open(os.path.join(results_dir, f"{jkey}_job.json"), "w", encoding="utf-8") as jf:
                            _json.dump({"state": "COMPLETED", "name": f"dryrun/{jkey}", "outputDatasetId": f"dry-{jkey}"}, jf, indent=2)
                        # Place a small results file under a part dir
                        job_dir = os.path.join(results_dir, jkey)
                        os.makedirs(job_dir, exist_ok=True)
                        results_path = os.path.join(job_dir, "results.jsonl")
                        # Use the input custom_ids to keep dataset/id alignment
                        part_input = os.path.join(cfg["paths"]["batch_inputs_dir"], f"t{t_str}{suffix}_{cond}.p{part_number:02d}.jsonl")
                        count = 0
                        with open(results_path, "w", encoding="utf-8") as rf:
                            try:
                                with open(part_input, "r", encoding="utf-8") as fin:
                                    for line in fin:
                                        s = line.strip()
                                        if not s:
                                            continue
                                        obj = _json.loads(s)
                                        cid = obj.get("custom_id") or obj.get("customId")
                                        if not cid:
                                            continue
                                        rf.write(_json.dumps({
                                            "custom_id": cid,
                                            "response": {"body": {"choices": [{"message": {"content": "dummy"}, "finish_reason": "stop"}]}, "usage": {"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2}}
                                        }) + "\n")
                                        count += 1
                            except FileNotFoundError:
                                # Fallback: write a single minimal row
                                rf.write(_json.dumps({
                                    "custom_id": f"closed_book|dry-{part_number}|{cond}|{float(temp):.1f}|0|closed",
                                    "response": {"body": {"choices": [{"message": {"content": "dummy"}, "finish_reason": "stop"}]}, "usage": {"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2}}
                                }) + "\n")
                        # Mark output dataset id for reference
                        with open(os.path.join(results_dir, f"{jkey}_OUTPUT_DATASET_ID.txt"), "w", encoding="utf-8") as of:
                            of.write(f"dry-{jkey}")
                    # Persist updated manifest after dry-run synthesis
                    mf.write_manifest(os.path.join(results_dir, "trial_manifest.json"), trial_manifest)
                else:
                    # Real queue run with simple progress callback that updates the manifest
                    def _progress_cb(ev: dict):
                        try:
                            jkey = ev.get("job_key")
                            if jkey:
                                trial_manifest.setdefault("job_status", {})
                                if ev.get("event") == "submitted":
                                    trial_manifest["job_status"][jkey] = "submitted"
                                    # Persist job name immediately for resume safety
                                    job_name = ev.get("job_name")
                                    # Group jobs per temp/cond key
                                    grp = jkey.rsplit("_p", 1)[0]
                                    trial_manifest.setdefault("jobs", {})
                                    arr = trial_manifest["jobs"].setdefault(grp, [])
                                    try:
                                        # Ensure array length covers this index
                                        idx = int(jkey.split("_p")[-1]) - 1
                                        if idx >= len(arr):
                                            arr.extend([None] * (idx + 1 - len(arr)))
                                        arr[idx] = job_name
                                    except Exception:
                                        pass
                                elif ev.get("event") == "state":
                                    st = ev.get("state")
                                    if st:
                                        trial_manifest["job_status"][jkey] = str(st).lower()
                                elif ev.get("event") == "downloaded":
                                    trial_manifest["job_status"][jkey] = "completed"
                                elif ev.get("event") == "download_pending":
                                    trial_manifest["job_status"][jkey] = "completed"
                            mf.write_manifest(os.path.join(results_dir, "trial_manifest.json"), trial_manifest)
                        except Exception:
                            pass
                    queue.progress_cb = _progress_cb  # type: ignore[attr-defined]
                    # Respect cooperative stop before running queue
                    try:
                        stop_token.check()
                    except Exception:
                        print("Stop requested before queue; skipping remaining submissions for this condition")
                        continue
                    queue.run_queue(results_dir)
                # Persist the job name (if available) for bookkeeping
                jnames = [j.job_name for j in queue.jobs if j.job_name]
                if jnames:
                    trial_manifest["jobs"][f"t{t_str}_{cond}"] = jnames

        # Write per-trial manifest
        mf.write_manifest(os.path.join(results_dir, "trial_manifest.json"), trial_manifest)
        print("Trial manifest written:", os.path.join(results_dir, "trial_manifest.json"))
        if state is not None:
            with RunStateLock(run_root):
                update_phase(state, "submit", status="completed")
                write_json_atomic(run_state_path(run_root), state)
    else:
        print("Gating: skipping submit (already done or not selected)")

    # Helpers to extract any JSONL files from downloaded bundles
    def _try_extract_jsonls(bundle_path: str, out_dir: str) -> list[str]:
        extracted: list[str] = []
        try:
            if zipfile.is_zipfile(bundle_path):
                with zipfile.ZipFile(bundle_path) as zf:
                    for info in zf.infolist():
                        if info.filename.lower().endswith(".jsonl"):
                            target_path = os.path.join(out_dir, os.path.basename(info.filename))
                            os.makedirs(os.path.dirname(target_path), exist_ok=True)
                            with zf.open(info, 'r') as src, open(target_path, 'wb') as dst:
                                shutil.copyfileobj(src, dst)
                            extracted.append(target_path)
                return extracted
            if tarfile.is_tarfile(bundle_path):
                with tarfile.open(bundle_path, 'r:*') as tf:
                    for member in tf.getmembers():
                        if member.isfile() and member.name.lower().endswith('.jsonl'):
                            target_path = os.path.join(out_dir, os.path.basename(member.name))
                            os.makedirs(os.path.dirname(target_path), exist_ok=True)
                            with tf.extractfile(member) as src, open(target_path, 'wb') as dst:
                                if src is not None:
                                    shutil.copyfileobj(src, dst)
                                    extracted.append(target_path)
                return extracted
            # Try gzip → plain file
            with open(bundle_path, 'rb') as f:
                head = f.read(2)
            if head == b"\x1f\x8b":
                decomp_path = os.path.join(out_dir, os.path.splitext(os.path.basename(bundle_path))[0])
                try:
                    with gzip.open(bundle_path, 'rb') as src, open(decomp_path, 'wb') as dst:
                        shutil.copyfileobj(src, dst)
                    # If the decompressed file is a tar, recurse; else if it's jsonl, keep it
                    if tarfile.is_tarfile(decomp_path):
                        extracted.extend(_try_extract_jsonls(decomp_path, out_dir))
                    elif decomp_path.lower().endswith('.jsonl'):
                        extracted.append(decomp_path)
                    return extracted
                except Exception:
                    pass
            # As a last resort, if it's actually a JSONL, copy/rename it
            try:
                with open(bundle_path, 'r', encoding='utf-8') as f:
                    for line in f:
                        s = line.strip()
                        if not s:
                            continue
                        json.loads(s)  # validate JSONL
                        break
                candidate = os.path.join(out_dir, 'results.jsonl')
                shutil.copyfile(bundle_path, candidate)
                extracted.append(candidate)
            except Exception:
                # not a text JSONL; ignore
                pass
        except Exception:
            pass
        return extracted

    # Poll and download results for each job (per trial below)
    def _process_job(job_key: str, job_name: str, results_dir: str):
        if args.dry_run:
            return  # Already synthesized in dry run
        print(f"Polling job {job_key}: {job_name}")
        job = poll_until_done(account_id, job_name)
        # Persist job metadata
        with open(os.path.join(results_dir, f"{job_key}_job.json"), "w", encoding="utf-8") as f:
            json.dump(job, f, indent=2)
        # Use normalized state computed by get_batch_job/poll_until_done to handle proto-style enums
        state_norm = job.get("normalizedState") or str(job.get("state") or "").upper()
        if state_norm != "COMPLETED":
            print(f"WARNING: job {job_key} not COMPLETED (state={job.get('state')}). Skipping download.")
            return
        out_ds_id = job.get("outputDatasetId") or job.get("output_dataset_id")
        if not out_ds_id:
            print(f"WARNING: job {job_key} missing outputDatasetId. Skipping download.")
            return
        # Record output dataset id
        with open(os.path.join(results_dir, f"{job_key}_OUTPUT_DATASET_ID.txt"), "w", encoding="utf-8") as f:
            f.write(str(out_ds_id))
        try:
            ds = get_dataset(account_id, out_ds_id.split("/")[-1])
            ext = ds.get("externalUrl") or ds.get("external_url")
        except Exception as e:
            print(f"WARNING: could not fetch dataset metadata for {job_key}: {e}")
            ext = None
        job_dir = os.path.join(results_dir, job_key)
        os.makedirs(job_dir, exist_ok=True)
        if not ext:
            # Fallback: try firectl CLI if available
            try:
                if shutil.which("firectl"):
                    ds_id_short = out_ds_id.split("/")[-1]
                    cmd = ["firectl", "download", "dataset", ds_id_short, "--output-dir", job_dir]
                    print("Attempting CLI download:", " ".join(cmd))
                    subprocess.run(cmd, check=True)
                    # If CLI succeeds, try to find any JSONL
                    extracted_any = False
                    for root, _dirs, files in os.walk(job_dir):
                        for nm in files:
                            if nm.lower().endswith('.jsonl'):
                                extracted_any = True
                                break
                        if extracted_any:
                            break
                    if extracted_any:
                        print(f"Downloaded via firectl for {job_key}")
                    else:
                        print(f"firectl finished but no JSONL found for {job_key}")
                    return
            except Exception as e:
                print(f"WARNING: firectl fallback failed for {job_key}: {e}")
            # Leave OUTPUT_DATASET_ID for later manual or resumed download
            return
        try:
            p = try_download_external_url(ext, job_dir)
            print(f"Downloaded dataset for {job_key} to {p}")
            extracted = _try_extract_jsonls(p, job_dir)
            if extracted:
                print(f"Extracted {len(extracted)} JSONL file(s) for {job_key}")
            else:
                print(f"No JSONL could be extracted for {job_key}; you may need to download manually from the UI.")
        except Exception as e:
            print(f"WARNING: failed to download dataset for {job_key}: {e}")

    # Iterate trials for polling, parsing, scoring, stats, costs, reporting
    prompt_sets_cfg = cfg.get("prompt_sets") or {}
    default_ps = cfg.get("default_prompt_set") or (sorted(list(prompt_sets_cfg.keys()))[0] if prompt_sets_cfg else "default")
    all_trial_summaries: list[dict] = []
    for trial in trials:
        model_id = trial["model_id"]
        ps_name = trial.get("prompt_set") or default_ps
        top_p = trial.get("top_p", cfg.get("top_p"))
        top_k = trial.get("top_k", cfg.get("top_k"))
        mx = trial.get("max_new_tokens") or cfg.get("max_new_tokens", {"closed_book": 1024, "open_book": 1024})
        if os.path.exists(experiments_dir):
            slug = _trial_slug(model_id, ps_name, top_p, top_k, mx, aliases=cfg.get("model_aliases") or {})
            trial_root = os.path.join(experiments_dir, f"run_{run_id}", slug)
            results_dir = os.path.join(trial_root, "results")
            reports_dir = os.path.join(trial_root, "reports")
        else:
            results_dir = cfg["paths"]["results_dir"]
            reports_dir = cfg["paths"]["reports_dir"]
        os.makedirs(results_dir, exist_ok=True)
        os.makedirs(reports_dir, exist_ok=True)

        manifest_path = os.path.join(results_dir, "trial_manifest.json")
        try:
            manifest, _ = mf.load_manifest(manifest_path)
        except Exception as e:
            print(f"WARNING: Missing or invalid trial manifest at {manifest_path}: {e}; skipping this trial")
            continue

        # Polling and download step (gated)
        if "poll" in selected_phases and not _poll_done():
            if state is not None:
                with RunStateLock(run_root):
                    update_phase(state, "poll", status="in_progress")
                    write_json_atomic(run_state_path(run_root), state)
            stop_token.check()
            flat_jobs: list[tuple[str, str]] = []
            for k, v in manifest["jobs"].items():
                if isinstance(v, list):
                    for i, name in enumerate(v, start=1):
                        if not name:
                            continue
                        flat_jobs.append((f"{k}_p{i:02d}", name))
                else:
                    if v:
                        flat_jobs.append((k, v))
            with ThreadPoolExecutor(max_workers=min(4, len(flat_jobs)) or 1) as ex:
                futures = {ex.submit(_process_job, k, v, results_dir): k for k, v in flat_jobs}
                for fut in as_completed(futures):
                    _k = futures[fut]
                    try:
                        fut.result()
                    except Exception as e:
                        print(f"WARNING: polling/downloading failed for {_k}: {e}")
            if state is not None:
                with RunStateLock(run_root):
                    update_phase(state, "poll", status="completed")
                    write_json_atomic(run_state_path(run_root), state)
        else:
            print("Gating: skipping poll for this trial (already done or not selected)")

        # Locate and combine all results JSONL files for parsing (per-trial)
        jsonl_files: list[str] = []
        for root, _dirs, files in os.walk(results_dir):
            for name in files:
                if name.lower().endswith('.jsonl'):
                    jsonl_files.append(os.path.join(root, name))

        print(f"Found {len(jsonl_files)} JSONL files for aggregation:")
        for f in jsonl_files:
            print(f"  {f}")

        combined_path = os.path.join(results_dir, "results_combined.jsonl")
        combined_lines = 0
        if jsonl_files:
            seen_ids: set[str] = set()
            with open(combined_path, 'w', encoding='utf-8') as fout:
                for fp in jsonl_files:
                    try:
                        with open(fp, 'r', encoding='utf-8') as fin:
                            for line in fin:
                                if not line.strip():
                                    continue
                                try:
                                    obj = json.loads(line)
                                except Exception:
                                    continue
                                cid = obj.get('custom_id') or obj.get('customId')
                                if cid and cid not in seen_ids:
                                    seen_ids.add(cid)
                                    fout.write(json.dumps(obj, ensure_ascii=False) + "\n")
                                    combined_lines += 1
                    except Exception:
                        continue

        print(f"Combined {combined_lines} unique items into {combined_path}")
        # Update manifest stage_status.downloaded based on current files
        try:
            st = mf.compute_stage_statuses(results_dir)
            man, _ = mf.load_manifest(os.path.join(results_dir, "trial_manifest.json"))
            man["stage_status"] = st
            mf.write_manifest(os.path.join(results_dir, "trial_manifest.json"), man)
        except Exception as _e:
            print(f"WARNING: failed to update manifest stage_status.downloaded: {_e}")
        if combined_lines == 0:
            print("No parseable results found (.jsonl with custom_id). Skipping parse/score/stats/costs for this trial.")
            continue

        # Parse predictions → CSV (gated)
        if "parse" in selected_phases and not _parse_done():
            if state is not None:
                with RunStateLock(run_root):
                    update_phase(state, "parse", status="in_progress")
                    write_json_atomic(run_state_path(run_root), state)
            stop_token.check()
            run_cmd([sys.executable, "-m", "fireworks.parse_results", "--results_jsonl", combined_path, "--out_csv", os.path.join(results_dir, "predictions.csv")])
            if state is not None:
                with RunStateLock(run_root):
                    update_phase(state, "parse", status="completed")
                    write_json_atomic(run_state_path(run_root), state)
        else:
            print("Gating: skipping parse (already done or not selected)")

        # Score per-item (gated)
        if "score" in selected_phases and not _score_done():
            if state is not None:
                with RunStateLock(run_root):
                    update_phase(state, "score", status="in_progress")
                    write_json_atomic(run_state_path(run_root), state)
            stop_token.check()
            run_cmd([sys.executable, "-m", "scoring.score_predictions", "--pred_csv", os.path.join(results_dir, "predictions.csv"), "--prepared_dir", cfg["paths"]["prepared_dir"], "--out_dir", results_dir, "--config", effective_config_path])
            if state is not None:
                with RunStateLock(run_root):
                    update_phase(state, "score", status="completed")
                    write_json_atomic(run_state_path(run_root), state)
        else:
            print("Gating: skipping score (already done or not selected)")

        # Stats (tolerate optional SciPy not installed in some environments) (gated)
        if "stats" in selected_phases and not _stats_done():
            try:
                if state is not None:
                    with RunStateLock(run_root):
                        update_phase(state, "stats", status="in_progress")
                        write_json_atomic(run_state_path(run_root), state)
                stop_token.check()
                run_cmd([sys.executable, "-m", "scoring.stats", "--per_item_csv", os.path.join(results_dir, "per_item_scores.csv"), "--config", effective_config_path, "--out_path", os.path.join(results_dir, "significance.json")])
            except Exception:
                # Write an empty file so downstream report generation still works
                try:
                    with open(os.path.join(results_dir, "significance.json"), "w", encoding="utf-8") as _sf:
                        json.dump({}, _sf)
                except Exception:
                    pass
            finally:
                if state is not None:
                    with RunStateLock(run_root):
                        update_phase(state, "stats", status="completed")
                        write_json_atomic(run_state_path(run_root), state)
        else:
            print("Gating: skipping stats (already done or not selected)")

        # Optional analyses (always safe, not tracked in run_state)
        try:
            run_cmd([sys.executable, "-m", "scripts.unsupported_sensitivity", "--pred_csv", os.path.join(results_dir, "predictions.csv"), "--prepared_dir", cfg["paths"]["prepared_dir"], "--config", effective_config_path, "--out_path", os.path.join(results_dir, "unsupported_sensitivity.json")])
        except Exception:
            pass
        try:
            run_cmd([sys.executable, "-m", "scripts.mixed_effects", "--pred_csv", os.path.join(results_dir, "predictions.csv"), "--prepared_dir", cfg["paths"]["prepared_dir"], "--config", effective_config_path, "--out_path", os.path.join(results_dir, "mixed_models.json")])
        except Exception:
            pass
        try:
            run_cmd([sys.executable, "-m", "scripts.power_analysis", "--per_item_csv", os.path.join(results_dir, "per_item_scores.csv"), "--prepared_dir", cfg["paths"]["prepared_dir"], "--config", effective_config_path, "--out_path", os.path.join(results_dir, "power_analysis.json")])
        except Exception:
            pass
        try:
            run_cmd([sys.executable, "-m", "scripts.cost_effectiveness", "--pred_csv", os.path.join(results_dir, "predictions.csv"), "--per_item_csv", os.path.join(results_dir, "per_item_scores.csv"), "--config", effective_config_path, "--out_path", os.path.join(results_dir, "cost_effectiveness.json")])
        except Exception:
            pass

        # Costs (gated)
        if "costs" in selected_phases and not _costs_done():
            if state is not None:
                with RunStateLock(run_root):
                    update_phase(state, "costs", status="in_progress")
                    write_json_atomic(run_state_path(run_root), state)
            stop_token.check()
            run_cmd([sys.executable, "-m", "scripts.summarize_costs", "--pred_csv", os.path.join(results_dir, "predictions.csv"), "--config", effective_config_path, "--out_path", os.path.join(results_dir, "costs.json")])
            if state is not None:
                with RunStateLock(run_root):
                    update_phase(state, "costs", status="completed")
                    write_json_atomic(run_state_path(run_root), state)
        else:
            print("Gating: skipping costs (already done or not selected)")
        # Generate a concise Markdown report
        report_path = os.path.join(reports_dir, "report.md")

        # Aggregate metrics (per-trial)
        agg: dict[tuple[float, str, str], dict[str, float]] = {}
        counts: dict[tuple[float, str, str], int] = {}
        per_item_csv = os.path.join(results_dir, "per_item_scores.csv")
        with open(per_item_csv, "r", encoding="utf-8") as f:
            r = csv.DictReader(f)
            for row in r:
                temp = float(row["temp"]) if row.get("temp") not in (None, "") else 0.0
                condition = row["condition"]
                typ = row["type"]
                key = (temp, condition, typ)
                if key not in agg:
                    agg[key] = {"em": 0.0, "f1": 0.0, "abstain_rate": 0.0, "false_answer_rate": 0.0, "unsupported_rate": 0.0}
                    counts[key] = 0

                def _to_float(x):
                    try:
                        return float(x)
                    except Exception:
                        return 0.0
                agg[key]["em"] += _to_float(row.get("em"))
                agg[key]["f1"] += _to_float(row.get("f1"))
                agg[key]["abstain_rate"] += _to_float(row.get("abstain_rate"))
                agg[key]["false_answer_rate"] += _to_float(row.get("false_answer_rate"))
                agg[key]["unsupported_rate"] += _to_float(row.get("unsupported_rate"))
                counts[key] += 1
        # Compute means
        means: dict[tuple[float, str, str], dict[str, float]] = {}
        for key, sums in agg.items():
            n = max(1, counts.get(key, 1))
            means[key] = {k: (v / n) for k, v in sums.items()}

        # Load significance and costs
        sig_path = os.path.join(results_dir, "significance.json")
        costs_path = os.path.join(results_dir, "costs.json")
        significance = {}
        costs = {}
        try:
            with open(sig_path, "r", encoding="utf-8") as f:
                significance = json.load(f)
        except Exception:
            pass
        try:
            with open(costs_path, "r", encoding="utf-8") as f:
                costs = json.load(f)
        except Exception:
            pass

        # Build markdown for this trial
        lines: list[str] = []
        lines.append("# Evaluation Report")
        lines.append("")
        # Identify current trial metadata
        curr_model = model_id
        curr_temps = trial.get('temps') or []
        lines.append(f"Model: {curr_model}")
        lines.append(f"Temperatures: {', '.join(str(t) for t in curr_temps)}")
        # Report per-temp replicate counts
        try:
            for t in curr_temps:
                k = (
                    cfg["samples_per_item"].get(str(float(t)))
                    or cfg["samples_per_item"].get(f"{float(t):.1f}")
                    or cfg["samples_per_item"].get(float(t))
                    or "?"
                )
                lines.append(f"Samples per item @T={float(t):.1f}: {k}")
        except Exception:
            pass
        lines.append("")
        # Prompt tokens
        lines.append("## Prompts")
        lines.append("")
        try:
            with open(os.path.join(results_dir, "trial_manifest.json"), "r", encoding="utf-8") as _mf:
                _m = json.load(_mf)
            lines.append(f"- Control prompt tokens: {_m['prompts']['control']['tokens']}")
            lines.append(f"- Treatment prompt tokens: {_m['prompts']['treatment']['tokens']}")
        except Exception:
            pass
        lines.append("")
        # Metrics table per temp and type
        for temp in curr_temps:
            lines.append(f"## Results @ T={float(temp):.1f}")
            for typ in ("closed", "open"):
                lines.append("")
                lines.append(f"### {typ.capitalize()}-book")
                lines.append("")
                lines.append("| Condition | EM | F1 | Abstain | False-Ans | Unsupported |")
                lines.append("|---|---:|---:|---:|---:|---:|")
                row_ctrl = means.get((float(temp), "control", typ), {})
                row_trt = means.get((float(temp), "treatment", typ), {})

                def fmt(x):
                    return f"{(x or 0.0)*100:.1f}%" if 0.0 <= (x or 0.0) <= 1.0 else f"{x:.3f}"

                def val(d, k):
                    return d.get(k, 0.0)
                lines.append(f"| Control | {fmt(val(row_ctrl,'em'))} | {fmt(val(row_ctrl,'f1'))} | {fmt(val(row_ctrl,'abstain_rate'))} | {fmt(val(row_ctrl,'false_answer_rate'))} | {fmt(val(row_ctrl,'unsupported_rate'))} |")
                lines.append(f"| Treatment | {fmt(val(row_trt,'em'))} | {fmt(val(row_trt,'f1'))} | {fmt(val(row_trt,'abstain_rate'))} | {fmt(val(row_trt,'false_answer_rate'))} | {fmt(val(row_trt,'unsupported_rate'))} |")
            # Stats
            if significance:
                s = significance.get(str(float(temp))) or significance.get(float(temp)) or {}
                if s:
                    lines.append("")
                    lines.append("### Significance")
                    lines.append("")
                    m = s.get("mcnemar", {})
                    w = s.get("wilcoxon", {})
                    lines.append(f"- McNemar: b={m.get('b')}, c={m.get('c')}, p={m.get('p_value')}")
                    lines.append(f"- Wilcoxon: W={w.get('W')}, p={w.get('p_value')}")
            lines.append("")
        # Costs
        if costs:
            lines.append("## Cost")
            lines.append("")
            lines.append(f"- Prompt tokens: {costs.get('prompt_tokens')}")
            lines.append(f"- Completion tokens: {costs.get('completion_tokens')}")
            lines.append(f"- Total tokens: {costs.get('total_tokens')}")
            usd = costs.get('usd')
            try:
                usd_val = float(usd)
            except (TypeError, ValueError):
                usd_val = 0.0
            lines.append(f"- Estimated USD: ${usd_val:.4f}")
            if costs.get("batch_discount_applied"):
                lines.append("- Batch discount applied")
            lines.append("")
        if "report" in selected_phases and not _report_done():
            if state is not None:
                with RunStateLock(run_root):
                    update_phase(state, "report", status="in_progress")
                    write_json_atomic(run_state_path(run_root), state)
            stop_token.check()
            with open(report_path, "w", encoding="utf-8") as f:
                f.write("\n".join(lines) + "\n")
            if state is not None:
                with RunStateLock(run_root):
                    update_phase(state, "report", status="completed")
                    write_json_atomic(run_state_path(run_root), state)
        else:
            print("Gating: skipping report (already done or not selected)")

        print("Trial complete. Parsed predictions, scored, computed stats and costs, and wrote report.")
        all_trial_summaries.append({"trial": trial, "results_dir": results_dir, "reports_dir": reports_dir, "report_path": report_path})

    # Write multi-trial summary
    multi_summary = {"created_utc": datetime.utcnow().isoformat() + "Z", "run_id": run_id, "num_trials": len(trials), "trials": all_trial_summaries}
    write_manifest(multi_manifest_path, multi_summary)
    print("Wrote multi-trial summary:", multi_manifest_path)

    # Optional aggregate report across trials (experiments dir only)
    if use_exp_root:
        try:
            aggregate_lines: list[str] = []
            aggregate_lines.append("# Aggregate Trial Comparison")
            aggregate_lines.append("")
            aggregate_lines.append(f"Run ID: {run_id}")
            aggregate_lines.append("")
            aggregate_lines.append("| Trial | Model | PromptSet | EM(closed) | F1(closed) | EM(open) | F1(open) |")
            aggregate_lines.append("|---|---|---|---:|---:|---:|---:|")

            def _avg_from_csv(csv_path: str, typ: str, metric: str) -> float:
                try:
                    vals = []
                    with open(csv_path, "r", encoding="utf-8") as f:
                        r = csv.DictReader(f)
                        for row in r:
                            if row.get("type") == typ and row.get(metric) not in (None, ""):
                                try:
                                    vals.append(float(row.get(metric)))
                                except Exception:
                                    pass
                    if not vals:
                        return 0.0
                    return sum(vals) / len(vals)
                except Exception:
                    return 0.0

            for entry in all_trial_summaries:
                tr = entry.get("trial", {})
                model_id = tr.get("model_id") or cfg.get("model_id")
                ps_name = tr.get("prompt_set") or (cfg.get("default_prompt_set") or "default")
                results_dir = entry.get("results_dir")
                per_item_csv = os.path.join(results_dir, "per_item_scores.csv")
                em_closed = _avg_from_csv(per_item_csv, "closed", "em") * 100.0
                f1_closed = _avg_from_csv(per_item_csv, "closed", "f1") * 100.0
                em_open = _avg_from_csv(per_item_csv, "open", "em") * 100.0
                f1_open = _avg_from_csv(per_item_csv, "open", "f1") * 100.0
                trial_name = _trial_slug(model_id, ps_name, tr.get("top_p"), tr.get("top_k"), tr.get("max_new_tokens") or {}, aliases=cfg.get("model_aliases") or {})
                aggregate_lines.append(
                    f"| {trial_name} | {model_id} | {ps_name} | {em_closed:.1f}% | {f1_closed:.1f}% | {em_open:.1f}% | {f1_open:.1f}% |"
                )

            agg_path = os.path.join(run_root, "aggregate_report.md")
            with open(agg_path, "w", encoding="utf-8") as f:
                f.write("\n".join(aggregate_lines) + "\n")
            print("Wrote aggregate report:", agg_path)
        except Exception as e:
            print(f"WARNING: Failed to write aggregate report: {e}")


if __name__ == "__main__":
    main()
