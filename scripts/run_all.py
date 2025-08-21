from __future__ import annotations
import os, sys, json, hashlib, argparse, subprocess, re
import csv
import tarfile, zipfile, gzip, shutil
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
from dotenv import load_dotenv
from fireworks.upload_dataset import create_dataset, upload_dataset_file
from fireworks.start_batch_job import create_batch_job
from fireworks.poll_and_download import poll_until_done, get_dataset, try_download_external_url
from config.schema import load_config
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
def split_jsonl_by_size(src_path: str, max_bytes: int) -> list[str]:
    """Split a JSONL file into parts not exceeding max_bytes each (approximate).

    Returns a list of part file paths. If src_path is already <= max_bytes,
    returns [src_path] without creating new files.
    """
    try:
        size = os.stat(src_path).st_size
    except FileNotFoundError:
        return []
    if size <= max_bytes:
        return [src_path]
    parts: list[str] = []
    base_dir = os.path.dirname(src_path)
    base_name = os.path.basename(src_path)
    # Keep .jsonl as the final extension for all parts to ensure downstream
    # systems recognize the file type. Example: foo.p01.jsonl
    name_root, ext = os.path.splitext(base_name)
    if not ext:
        ext = ".jsonl"
    idx = 1
    current_bytes = 0
    current_path = os.path.join(base_dir, f"{name_root}.p{idx:02d}{ext}")
    current_file = open(current_path, "w", encoding="utf-8")
    parts.append(current_path)
    with open(src_path, "r", encoding="utf-8") as fin:
        for line in fin:
            if not line:
                continue
            b = len(line.encode("utf-8"))
            if current_bytes + b > max_bytes and current_bytes > 0:
                current_file.close()
                idx += 1
                current_bytes = 0
                current_path = os.path.join(base_dir, f"{name_root}.p{idx:02d}{ext}")
                current_file = open(current_path, "w", encoding="utf-8")
                parts.append(current_path)
            current_file.write(line)
            current_bytes += b
    current_file.close()
    return parts
def main():
    load_dotenv()
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="config/eval_config.yaml")
    ap.add_argument("--account_id", default=os.environ.get("FIREWORKS_ACCOUNT_ID"))
    ap.add_argument("--condition", choices=["control", "treatment", "both"], default="both")
    ap.add_argument("--skip_prepare", action="store_true")
    ap.add_argument("--skip_build", action="store_true")
    ap.add_argument("--skip_batch", action="store_true")
    args = ap.parse_args()
    cfg = load_config(args.config)
    ensure_dirs(cfg)
    if not args.skip_prepare:
        run_cmd([sys.executable, "-m", "scripts.prepare_data", "--config", args.config])
    if not args.skip_build:
        run_cmd([sys.executable, "-m", "scripts.build_batches", "--config", args.config])
    control_prompt = open("config/prompts/control_system.txt", "r", encoding="utf-8").read()
    treatment_prompt = open("config/prompts/treatment_system.txt", "r", encoding="utf-8").read()
    run_id = datetime.utcnow().strftime("r%Y%m%d%H%M%S")
    manifest = {
        "created_utc": datetime.utcnow().isoformat() + "Z",
        "run_id": run_id,
        "model_id": cfg["model_id"],
        "temps": cfg["temps"],
        "samples_per_item": cfg["samples_per_item"],
        "top_p": cfg.get("top_p"),
        "top_k": cfg.get("top_k"),
        "max_new_tokens": cfg.get("max_new_tokens"),
        "prompts": {
            "control": {"sha256": hashlib.sha256(control_prompt.encode()).hexdigest(), "tokens": token_len(control_prompt)},
            "treatment": {"sha256": hashlib.sha256(treatment_prompt.encode()).hexdigest(), "tokens": token_len(treatment_prompt)},
        },
        "datasets": {},
        "jobs": {},
    }
    if args.skip_batch:
        write_manifest(cfg["paths"]["run_manifest"], manifest)
        print("Prepared inputs only (skip_batch)."); return
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

    account_id = _derive_account_id(args.account_id)
    conditions: list[str] = [args.condition] if args.condition in ("control", "treatment") else ["control", "treatment"]
    for temp in cfg["temps"]:
        t_str = _format_temp_label(temp)
        for cond in conditions:
            jsonl_path = os.path.join(cfg["paths"]["batch_inputs_dir"], f"t{t_str}_{cond}.jsonl")
            display_name = f"excellence-t{t_str}-{cond}-{run_id}"
            # Keep each part <=140MB so uploads always use the multipart JSONL path
            parts = split_jsonl_by_size(jsonl_path, max_bytes=140 * 1024 * 1024)
            if not parts:
                raise SystemExit(f"Missing or empty input file: {jsonl_path}")
            datasets_for_cond: list[str] = []
            for pi, part_path in enumerate(parts, start=1):
                ds_name = display_name if len(parts) == 1 else f"{display_name}-p{pi:02d}"
                dsid = create_dataset(ds_name, account_id)
                # Ensure remote filename clearly ends with .jsonl for correct type detection
                base_no_ext, base_ext = os.path.splitext(os.path.basename(jsonl_path))
                if not base_ext:
                    base_ext = ".jsonl"
                remote_fname = (
                    f"{base_no_ext}{base_ext}" if len(parts) == 1 else f"{base_no_ext}.p{pi:02d}{base_ext}"
                )
                upload_dataset_file(account_id, dsid, part_path, filename=remote_fname)
                datasets_for_cond.append(dsid)
            manifest["datasets"][f"t{t_str}_{cond}"] = datasets_for_cond if len(datasets_for_cond) > 1 else datasets_for_cond[0]
            # Use a single uniform max_tokens per job; choose the maximum across splits
            mn = cfg.get("max_new_tokens", {"closed_book": 1024, "open_book": 1024})
            try:
                job_max_tokens = int(max(int(mn.get("closed_book", 1024)), int(mn.get("open_book", 1024))))
            except Exception:
                job_max_tokens = 1024
            # Create a batch job per dataset part; Fireworks will produce a result per job
            job_names: list[str] = []
            dsid_list = datasets_for_cond
            if isinstance(dsid_list, str):
                dsid_list = [dsid_list]
            for pi, dsid in enumerate(dsid_list, start=1):
                job = create_batch_job(
                    account_id=account_id,
                    model=cfg["model_id"],
                    input_dataset_id=dsid,
                    display_name=(f"{display_name}-job" if len(dsid_list) == 1 else f"{display_name}-job-p{pi:02d}"),
                    temperature=float(temp),
                    max_tokens=job_max_tokens,
                    top_p=cfg.get("top_p"),
                    top_k=cfg.get("top_k"),
                    stop=cfg.get("stop") or None,
                )
                job_name = job.get("name") or job.get("id")
                job_names.append(job_name)
            manifest["jobs"][f"t{t_str}_{cond}"] = job_names if len(job_names) > 1 else job_names[0]
    write_manifest(cfg["paths"]["run_manifest"], manifest)
    print("Run manifest written:", cfg["paths"]["run_manifest"]) 

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

    # Poll and download results for each job
    results_dir = cfg["paths"]["results_dir"]
    os.makedirs(results_dir, exist_ok=True)
    def _process_job(job_key: str, job_name: str):
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

    # Poll in parallel for speed
    # Flatten jobs if any temp/cond has multiple parts
    flat_jobs: list[tuple[str, str]] = []
    for k, v in manifest["jobs"].items():
        if isinstance(v, list):
            for i, name in enumerate(v, start=1):
                flat_jobs.append((f"{k}_p{i:02d}", name))
        else:
            flat_jobs.append((k, v))
    with ThreadPoolExecutor(max_workers=min(4, len(flat_jobs)) or 1) as ex:
        futures = {ex.submit(_process_job, k, v): k for k, v in flat_jobs}
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
        print("No parseable results found (.jsonl with custom_id). Skipping parse/score/stats/costs. See issue #3 (results extraction).")
        return

    # Parse predictions → CSV
    run_cmd([sys.executable, "-m", "fireworks.parse_results", "--results_jsonl", combined_path, "--out_csv", os.path.join(results_dir, "predictions.csv")])
    # Score per-item
    run_cmd([sys.executable, "-m", "scoring.score_predictions", "--pred_csv", os.path.join(results_dir, "predictions.csv"), "--prepared_dir", cfg["paths"]["prepared_dir"], "--out_dir", results_dir, "--config", args.config])
    # Stats
    run_cmd([sys.executable, "-m", "scoring.stats", "--per_item_csv", os.path.join(results_dir, "per_item_scores.csv"), "--metric", "em", "--out_path", os.path.join(results_dir, "significance.json")])
    # Costs
    run_cmd([sys.executable, "-m", "scripts.summarize_costs", "--pred_csv", os.path.join(results_dir, "predictions.csv"), "--config", args.config, "--out_path", os.path.join(results_dir, "costs.json")])
    # Generate a concise Markdown report
    reports_dir = cfg["paths"]["reports_dir"]
    os.makedirs(reports_dir, exist_ok=True)
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

    # Build markdown
    lines: list[str] = []
    lines.append("# Evaluation Report")
    lines.append("")
    lines.append(f"Model: {cfg['model_id']}")
    lines.append(f"Temperatures: {', '.join(str(t) for t in cfg['temps'])}")
    # Report per-temp replicate counts to avoid stale text when temps != 0.7
    try:
        for t in cfg["temps"]:
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
    lines.append(f"- Control prompt tokens: {manifest['prompts']['control']['tokens']}")
    lines.append(f"- Treatment prompt tokens: {manifest['prompts']['treatment']['tokens']}")
    lines.append("")
    # Metrics table per temp and type
    for temp in cfg["temps"]:
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

    print("Done. Parsed predictions, scored, computed stats and costs, and wrote report.")
if __name__ == "__main__":
    main()
