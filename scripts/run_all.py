from __future__ import annotations
import os, sys, json, hashlib, argparse, subprocess, re
import yaml
import csv
import tarfile, zipfile, gzip, shutil
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
from dotenv import load_dotenv
from fireworks.upload_dataset import create_dataset, upload_dataset_file
from fireworks.poll_and_download import poll_until_done, get_dataset, try_download_external_url
from fireworks.batch_queue_manager import QueueManager
from config.schema import load_config

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
            return [float(t) for t in _split_list_arg(args.temps)]
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
    for k in ["prepared_dir","batch_inputs_dir","results_dir","reports_dir"]:
        os.makedirs(cfg["paths"][k], exist_ok=True)
def run_cmd(args: list[str]):
    print("+", " ".join(args)); sys.stdout.flush()
    subprocess.run(args, check=True)
def write_manifest(path: str, data: dict):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)

def _split_jsonl_file(
    src_path: str,
    out_dir: str,
    base_prefix: str,
    parts: int | None = None,
    *,
    lines_per_part: int | None = None,
    limit_items: int | None = None,
) -> list[tuple[int, str]]:
    """Split a JSONL file into N approximately equal parts.

    Returns a list of (part_number, part_path). If parts <= 1, returns [(1, src_path)].
    If part files already exist and are non-empty, reuse them.
    """
    # Determine splitting mode: by lines per part or fixed number of parts
    parts = int(parts) if parts and int(parts) > 1 else None
    lines_per_part = int(lines_per_part) if lines_per_part and int(lines_per_part) > 0 else None
    if parts is None and lines_per_part is None:
        # nothing to do
        return [(1, src_path)]
    os.makedirs(out_dir, exist_ok=True)
    # If splits already exist, reuse
    existing: list[tuple[int, str]] = []
    if parts is not None:
        for i in range(1, parts + 1):
            p = os.path.join(out_dir, f"{base_prefix}.p{i:02d}.jsonl")
            if os.path.exists(p) and os.path.getsize(p) > 0:
                existing.append((i, p))
        if len(existing) == parts:
            return existing
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
    ap.add_argument("--plan_only", action="store_true", help="Show expanded trial plan and exit")
    ap.add_argument("--archive", action="store_true", help="Archive results after completion")
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

    # Plan-only mode prints the trial matrix then exits
    if args.plan_only:
        plan = {"created_utc": datetime.utcnow().isoformat() + "Z", "run_id": run_id, "num_trials": len(trials), "trials": trials}
        print(json.dumps(plan, indent=2))
        return

    ensure_dirs(cfg)
    if not args.skip_prepare:
        run_cmd([sys.executable, "-m", "scripts.prepare_data", "--config", effective_config_path])
    if not args.skip_build:
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
    if args.skip_batch:
        # Write a minimal multi-trial plan and exit
        plan = {"created_utc": datetime.utcnow().isoformat() + "Z", "run_id": run_id, "num_trials": len(trials), "trials": trials}
        write_manifest(multi_manifest_path, plan)
        print("Prepared inputs only (skip_batch). Plan written:", multi_manifest_path); return
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
    for trial in trials:
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

        # If resuming and a manifest exists, load and append to it; else start fresh
        existing_manifest: dict | None = None
        manifest_path_pre = os.path.join(results_dir, "trial_manifest.json")
        if args.resume and os.path.isfile(manifest_path_pre):
            try:
                with open(manifest_path_pre, "r", encoding="utf-8") as _mf:
                    existing_manifest = json.load(_mf)
            except Exception:
                existing_manifest = None

        trial_manifest = existing_manifest or {
            "created_utc": datetime.utcnow().isoformat() + "Z",
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
        }

        for temp in temps:
            t_str = _format_temp_label(float(temp))
            for cond in conditions:
                suffix = f"_{ps_name}" if (len(prompt_sets_cfg) > 1 or ps_name not in ("default", None)) else ""
                jsonl_path = os.path.join(cfg["paths"]["batch_inputs_dir"], f"t{t_str}{suffix}_{cond}.jsonl")
                display_name = f"excellence-{ps_name}-t{t_str}-{cond}-{run_id}"
                if not os.path.isfile(jsonl_path):
                    raise SystemExit(f"Missing input file: {jsonl_path}")

                # Split the dataset into up to max_concurrent parts for parallel jobs
                base_prefix = f"t{t_str}{suffix}_{cond}"
                part_files = _split_jsonl_file(
                    jsonl_path,
                    cfg["paths"]["batch_inputs_dir"],
                    base_prefix,
                    parts=int(args.parts_per_dataset) if args.lines_per_part is None else None,
                    lines_per_part=int(args.lines_per_part) if args.lines_per_part else None,
                    limit_items=int(args.limit_items) if args.limit_items else None,
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
                )
                # If resuming and this temp/cond appears complete in manifest, skip queueing
                resume_key = f"t{t_str}_{cond}"
                if args.resume and existing_manifest:
                    jobs_map = (existing_manifest or {}).get("jobs", {}) or {}
                    job_status = (existing_manifest or {}).get("job_status", {}) or {}
                    names = jobs_map.get(resume_key) or []
                    if names and len(names) == len(part_files):
                        # Check that all parts appear completed
                        def _is_completed(val: str | None) -> bool:
                            s = (str(val or "")).lower()
                            return ("complete" in s) or (s == "completed") or (s == "downloaded")
                        all_done = True
                        for i in range(1, len(part_files) + 1):
                            jkey = f"{resume_key}_p{i:02d}"
                            if not _is_completed(job_status.get(jkey)):
                                all_done = False; break
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
                    # Per-part resume: skip parts already marked completed
                    jkey_resume = f"{resume_key}_p{part_number:02d}"
                    if args.resume and existing_manifest:
                        prev_status = (existing_manifest.get("job_status", {}) or {}).get(jkey_resume)
                        if str(prev_status or "").lower().find("complete") != -1 or str(prev_status or "").lower() == "downloaded":
                            print(f"Resume: skipping already completed part {jkey_resume}")
                            continue
                    ds_name = f"{display_name}-p{part_number:02d}"
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
                write_manifest(os.path.join(results_dir, "trial_manifest.json"), trial_manifest)

                # Dry-run path: synthesize completion and results for each part, then skip actual queueing
                if args.dry_run:
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
                    write_manifest(os.path.join(results_dir, "trial_manifest.json"), trial_manifest)
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
                            write_manifest(os.path.join(results_dir, "trial_manifest.json"), trial_manifest)
                        except Exception:
                            pass
                    queue.progress_cb = _progress_cb  # type: ignore[attr-defined]
                    queue.run_queue(results_dir)
                # Persist the job name (if available) for bookkeeping
                jnames = [j.job_name for j in queue.jobs if j.job_name]
                if jnames:
                    trial_manifest["jobs"][f"t{t_str}_{cond}"] = jnames

        # Write per-trial manifest
        write_manifest(os.path.join(results_dir, "trial_manifest.json"), trial_manifest)
        print("Trial manifest written:", os.path.join(results_dir, "trial_manifest.json"))

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
        if job.get("state") != "COMPLETED":
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
        if not ext:
            # Leave OUTPUT_DATASET_ID for later manual or resumed download
            return
        job_dir = os.path.join(results_dir, job_key)
        os.makedirs(job_dir, exist_ok=True)
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

    # Iterate trials for polling and reporting
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
            with open(manifest_path, "r", encoding="utf-8") as f:
                manifest = json.load(f)
        except Exception:
            print(f"WARNING: Missing trial manifest at {manifest_path}; skipping this trial")
            continue

        # Flatten jobs
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

    # Locate and combine all results JSONL files for parsing
    jsonl_files: list[str] = []
    for root, _dirs, files in os.walk(results_dir):
        for name in files:
            if name.lower().endswith('.jsonl'):
                jsonl_files.append(os.path.join(root, name))

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

        if combined_lines == 0:
            print("No parseable results found (.jsonl with custom_id). Skipping parse/score/stats/costs for this trial.")
            continue

        # Parse predictions → CSV
        run_cmd([sys.executable, "-m", "fireworks.parse_results", "--results_jsonl", combined_path, "--out_csv", os.path.join(results_dir, "predictions.csv")])
        # Score per-item
        run_cmd([sys.executable, "-m", "scoring.score_predictions", "--pred_csv", os.path.join(results_dir, "predictions.csv"), "--prepared_dir", cfg["paths"]["prepared_dir"], "--out_dir", results_dir, "--config", effective_config_path])
        # Stats (tolerate optional SciPy not installed in some environments)
        try:
            run_cmd([sys.executable, "-m", "scoring.stats", "--per_item_csv", os.path.join(results_dir, "per_item_scores.csv"), "--config", effective_config_path, "--out_path", os.path.join(results_dir, "significance.json")])
        except Exception:
            # Write an empty file so downstream report generation still works
            try:
                with open(os.path.join(results_dir, "significance.json"), "w", encoding="utf-8") as _sf:
                    json.dump({}, _sf)
            except Exception:
                pass
        # Unsupported sensitivity analysis (Phase 4)
        try:
            run_cmd([sys.executable, "-m", "scripts.unsupported_sensitivity", "--pred_csv", os.path.join(results_dir, "predictions.csv"), "--prepared_dir", cfg["paths"]["prepared_dir"], "--config", effective_config_path, "--out_path", os.path.join(results_dir, "unsupported_sensitivity.json")])
        except Exception:
            pass
        # Mixed-effects robustness models (optional)
        try:
            run_cmd([sys.executable, "-m", "scripts.mixed_effects", "--pred_csv", os.path.join(results_dir, "predictions.csv"), "--prepared_dir", cfg["paths"]["prepared_dir"], "--config", effective_config_path, "--out_path", os.path.join(results_dir, "mixed_models.json")])
        except Exception:
            pass
        # Power/MDE analysis (optional)
        try:
            run_cmd([sys.executable, "-m", "scripts.power_analysis", "--per_item_csv", os.path.join(results_dir, "per_item_scores.csv"), "--prepared_dir", cfg["paths"]["prepared_dir"], "--config", effective_config_path, "--out_path", os.path.join(results_dir, "power_analysis.json")])
        except Exception:
            pass
        # Cost-effectiveness summary (optional)
        try:
            run_cmd([sys.executable, "-m", "scripts.cost_effectiveness", "--pred_csv", os.path.join(results_dir, "predictions.csv"), "--per_item_csv", os.path.join(results_dir, "per_item_scores.csv"), "--config", effective_config_path, "--out_path", os.path.join(results_dir, "cost_effectiveness.json")])
        except Exception:
            pass
        # Costs
        run_cmd([sys.executable, "-m", "scripts.summarize_costs", "--pred_csv", os.path.join(results_dir, "predictions.csv"), "--config", effective_config_path, "--out_path", os.path.join(results_dir, "costs.json")])
        # Generate a concise Markdown report
        report_path = os.path.join(reports_dir, "report.md")

    # Aggregate metrics
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

    # Build markdown for the last processed trial (context: report_path/results_dir refer to current trial loop)
    lines: list[str] = []
    lines.append("# Evaluation Report")
    lines.append("")
    # Identify current trial metadata
    curr_model = model_id if 'model_id' in locals() else cfg.get('model_id')
    curr_temps = trial.get('temps') if 'trial' in locals() else (cfg.get('temps') or [])
    lines.append(f"Model: {curr_model}")
    lines.append(f"Temperatures: {', '.join(str(t) for t in curr_temps)}")
    # Report per-temp replicate counts to avoid stale text when temps != 0.7
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
        with open(report_path, "w", encoding="utf-8") as f:
            f.write("\n".join(lines) + "\n")

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
