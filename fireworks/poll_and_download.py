from __future__ import annotations
import os
import time
import json
import httpx
import argparse
import random
import zipfile
import tarfile
import gzip
import shutil
from dotenv import load_dotenv
from scripts import manifest_v2 as mf
API_BASE = os.environ.get("FIREWORKS_API_BASE", "https://api.fireworks.ai")
V1 = f"{API_BASE}/v1"


def auth_headers():
    key = os.environ.get("FIREWORKS_API_KEY")
    if not key: raise RuntimeError("FIREWORKS_API_KEY not set")
    return {"Authorization": f"Bearer {key}"}


def _sleep_with_jitter(seconds: float) -> None:
    jitter = min(0.5, seconds * 0.25)
    time.sleep(max(0.0, seconds) + random.random() * jitter)


def _get_with_retries(url: str, headers: dict, max_attempts: int = 8, base_delay: float = 1.0) -> httpx.Response:
    attempt = 0
    delay_seconds = base_delay
    last_exc: Exception | None = None
    while attempt < max_attempts:
        attempt += 1
        try:
            with httpx.Client(timeout=60.0) as client:
                resp = client.get(url, headers=headers)
            # Fast-path success
            if 200 <= resp.status_code < 300:
                return resp
            # Retry on 429/408/5xx and transient 403/404 from object stores
            if resp.status_code in (429, 408, 403, 404) or resp.status_code >= 500:
                # Honor Retry-After if present
                ra = resp.headers.get("retry-after") or resp.headers.get("Retry-After")
                try:
                    ra_s = float(ra) if ra is not None else None
                except Exception:
                    ra_s = None
                wait = max(delay_seconds, ra_s or 0.0)
                _sleep_with_jitter(wait)
                delay_seconds = min(delay_seconds * 2.0, 16.0)
                continue
            # Non-retryable status
            resp.raise_for_status()
            return resp
        except httpx.RequestError as e:
            last_exc = e
            _sleep_with_jitter(delay_seconds)
            delay_seconds = min(delay_seconds * 2.0, 16.0)
        except Exception as e:  # safety net
            last_exc = e
            _sleep_with_jitter(delay_seconds)
            delay_seconds = min(delay_seconds * 2.0, 16.0)
    # Give up
    if last_exc:
        raise last_exc
    raise RuntimeError(f"GET {url} failed after {max_attempts} attempts")


def _normalize_state(state: str | None) -> str:
    """Normalize Fireworks job state strings to canonical tokens.

    Handles both proto-style (e.g., "JOB_STATE_COMPLETED") and plain
    (e.g., "COMPLETED") variants. Returns one of:
    "COMPLETED", "FAILED", "EXPIRED", "RUNNING", "PENDING", or original uppercased fallback.
    """
    if not state:
        return ""
    s = str(state).upper()
    if "COMPLETED" in s:
        return "COMPLETED"
    if "FAILED" in s:
        return "FAILED"
    if "EXPIRED" in s:
        return "EXPIRED"
    if "RUNNING" in s or "PROCESSING" in s:
        return "RUNNING"
    if "PENDING" in s or "QUEUED" in s or "SUBMITTED" in s:
        return "PENDING"
    return s


def get_batch_job(account_id: str, job_name: str):
    url = f"{V1}/accounts/{account_id}/batchInferenceJobs/{job_name.split('/')[-1]}"
    resp = _get_with_retries(url, headers=auth_headers())
    data = resp.json()
    try:
        # Attach a normalizedState helper for consumers
        data["normalizedState"] = _normalize_state(data.get("state"))
    except Exception:
        pass
    return data


def get_dataset(account_id: str, dataset_id: str):
    url = f"{V1}/accounts/{account_id}/datasets/{dataset_id}"
    resp = _get_with_retries(url, headers=auth_headers())
    return resp.json()


def get_dataset_external_url(account_id: str, dataset_id: str) -> str | None:
    ds = get_dataset(account_id, dataset_id)
    return ds.get("externalUrl") or ds.get("external_url")


def try_download_external_url(url: str, out_dir: str):
    if not url:
        return None
    # Support local file URLs to ease testing
    if url.startswith("file://"):
        src_path = url[len("file://"):]
        if os.path.isfile(src_path):
            out_path = os.path.join(out_dir, "dataset_download.bin")
            os.makedirs(os.path.dirname(out_path), exist_ok=True)
            shutil.copyfile(src_path, out_path)
            return out_path
        return None
    out_path = os.path.join(out_dir, "dataset_download.bin")
    attempt = 0
    delay_seconds = 1.0
    last_status = None
    while attempt < 8:
        attempt += 1
        try:
            with httpx.Client(follow_redirects=True, timeout=None) as client:
                with client.stream("GET", url) as resp:
                    status = resp.status_code
                    last_status = status
                    # Success path
                    if 200 <= status < 300:
                        with open(out_path, "wb") as f:
                            for chunk in resp.iter_bytes():
                                f.write(chunk)
                        return out_path
                    # Retry on common transient statuses from blob stores/CDNs
                    if status in (429, 403, 404, 408) or status >= 500:
                        ra = resp.headers.get("retry-after") or resp.headers.get("Retry-After")
                        try:
                            ra_s = float(ra) if ra is not None else None
                        except Exception:
                            ra_s = None
                        wait = max(delay_seconds, ra_s or 0.0)
                        _sleep_with_jitter(wait)
                        delay_seconds = min(delay_seconds * 2.0, 16.0)
                        continue
                    # Otherwise, treat as fatal
                    resp.raise_for_status()
                    return None
        except httpx.RequestError:
            _sleep_with_jitter(delay_seconds)
            delay_seconds = min(delay_seconds * 2.0, 16.0)
        except Exception:
            _sleep_with_jitter(delay_seconds)
            delay_seconds = min(delay_seconds * 2.0, 16.0)
    # Give up gracefully; caller logs a warning
    return None


def _try_extract_jsonls(bundle_path: str, out_dir: str) -> list[str]:
    extracted: list[str] = []
    try:
        if zipfile.is_zipfile(bundle_path):
            with zipfile.ZipFile(bundle_path) as zf:
                for info in zf.infolist():
                    if info.filename.lower().endswith('.jsonl'):
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


def _combine_jsonls(root_dir: str, out_path: str) -> int:
    jsonl_files: list[str] = []
    for r, _dirs, files in os.walk(root_dir):
        for name in files:
            if name.lower().endswith('.jsonl'):
                jsonl_files.append(os.path.join(r, name))
    lines = 0
    if jsonl_files:
        with open(out_path, 'w', encoding='utf-8') as fout:
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
                            if 'custom_id' in obj or 'customId' in obj:
                                fout.write(json.dumps(obj, ensure_ascii=False) + "\n")
                                lines += 1
                except Exception:
                    continue
    return lines


def poll_until_done(account_id: str, job_name: str, poll_seconds: int = 15):
    """Poll a batch job until terminal state.

    - Prints only on state change to reduce log spam.
    - Sleeps `poll_seconds` between checks (default: 15s).
    """
    last_state = None
    while True:
        job = get_batch_job(account_id, job_name)
        raw_state = job.get("state")
        state = job.get("normalizedState") or _normalize_state(raw_state)
        if state != last_state:
            # Show raw state if available for transparency
            disp = raw_state if raw_state is not None else state
            print(f"[poll] job {job_name} -> {disp}")
            last_state = state
        if state in ("COMPLETED", "FAILED", "EXPIRED"):
            return job
        time.sleep(max(1, int(poll_seconds)))


def main():
    load_dotenv()
    ap = argparse.ArgumentParser()
    ap.add_argument("--account", required=True)
    ap.add_argument("--job_name", required=True)
    ap.add_argument("--out_dir", required=True)
    args = ap.parse_args()
    os.makedirs(args.out_dir, exist_ok=True)
    job = poll_until_done(args.account, args.job_name)
    print(json.dumps(job, indent=2))
    # Accept either raw or proto-style completed states
    norm_state = job.get("normalizedState") or _normalize_state(job.get("state"))
    if norm_state != "COMPLETED":
        print(f"WARNING: job not COMPLETED (state={job.get('state')}); exiting without download")
        return
    out_ds_id = job.get("outputDatasetId") or job.get("output_dataset_id")
    if not out_ds_id:
        print("No outputDatasetId present on job.")
        return
    ds = get_dataset(args.account, out_ds_id.split("/")[-1])
    ext = ds.get("externalUrl") or ds.get("external_url")
    if ext:
        p = try_download_external_url(ext, args.out_dir)
        if p:
            print(f"Downloaded dataset bundle to {p}")
            extracted = _try_extract_jsonls(p, args.out_dir)
            if extracted:
                print(f"Extracted {len(extracted)} JSONL file(s)")
                combined = os.path.join(args.out_dir, "results.jsonl")
                n = _combine_jsonls(args.out_dir, combined)
                if n > 0:
                    print(f"Combined {n} records into {combined}")
                    # Update manifest stage if a trial_manifest.json is nearby
                    manifest_path = os.path.join(args.out_dir, "trial_manifest.json")
                    if os.path.isfile(manifest_path):
                        try:
                            st = mf.compute_stage_statuses(args.out_dir)
                            man, _ = mf.load_manifest(manifest_path)
                            man["stage_status"] = st
                            mf.write_manifest(manifest_path, man)
                        except Exception:
                            pass
                return
    with open(os.path.join(args.out_dir, "OUTPUT_DATASET_ID.txt"), "w", encoding="utf-8") as f:
        f.write(out_ds_id)
    print("Wrote OUTPUT_DATASET_ID.txt for manual download via UI.")


if __name__ == "__main__":
    main()
