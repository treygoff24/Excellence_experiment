from __future__ import annotations
import os
import sys
import json
import httpx
import argparse
import random
import time
from urllib.parse import urlparse, parse_qs
from dotenv import load_dotenv
API_BASE = os.environ.get("FIREWORKS_API_BASE", "https://api.fireworks.ai")
DATASETS_BASE = f"{API_BASE}/v1"
MB_150 = 150 * 1024 * 1024
_newly_created_datasets: set[str] = set()


def normalize_dataset_id(display_name: str) -> str:
    """Derive a stable dataset id from a display name.

    - Lowercase, keep [a-z0-9-], replace others with '-'
    - Collapse consecutive '-'
    - Trim leading/trailing '-'
    - Ensure first char is a letter
    - Limit length to 63 chars
    """
    base_id = "".join(
        ch if ("a" <= ch <= "z") or ("0" <= ch <= "9") or ch == "-" else "-"
        for ch in display_name.lower()
    )
    while "--" in base_id:
        base_id = base_id.replace("--", "-")
    base_id = base_id.strip("-")
    if not base_id:
        base_id = "dataset"
    if not base_id[0].isalpha():
        base_id = f"ds-{base_id}"
    if len(base_id) > 63:
        base_id = base_id[:63].rstrip("-")
    return base_id


def dataset_exists(account_id: str, dataset_id: str) -> bool:
    """Return True if the dataset resource already exists (HTTP 200), False if 404.

    Raises for other non-2xx, non-404 statuses.
    """
    url = f"{DATASETS_BASE}/accounts/{account_id}/datasets/{dataset_id}"
    try:
        with httpx.Client(timeout=30.0) as client:
            resp = client.get(url, headers=auth_headers())
        if resp.status_code == 200:
            return True
        if resp.status_code == 404:
            return False
        # Non-retryable error; attach helpful context if account id looks invalid
        if 400 <= resp.status_code < 500:
            hint = ""
            if "/accounts/" in url and "@" in url:
                hint = "\nHint: FIREWORKS_ACCOUNT_ID appears to be an email. Set it to your account slug (e.g., 'my-team'), not an email."
            raise httpx.HTTPStatusError(f"{resp.status_code} error for {url}: {resp.text}{hint}", request=resp.request, response=resp)
        resp.raise_for_status()
        return True
    except httpx.RequestError:
        # Be conservative: if we cannot confirm due to a transient network error,
        # assume it does not exist so the subsequent create/upload path can proceed
        return False


def auth_headers():
    key = os.environ.get("FIREWORKS_API_KEY")
    if not key: raise RuntimeError("FIREWORKS_API_KEY not set")
    return {"Authorization": f"Bearer {key}"}


def _sleep_with_jitter(seconds: float) -> None:
    jitter = min(0.5, seconds * 0.25)
    time.sleep(max(0.0, seconds) + random.random() * jitter)


def _post_json_with_retries(url: str, headers: dict, json_payload: dict | None, max_attempts: int = 6, base_delay: float = 1.0, params: dict | None = None) -> httpx.Response:
    attempt = 0
    delay_seconds = base_delay
    last_exc: Exception | None = None
    while attempt < max_attempts:
        attempt += 1
        try:
            with httpx.Client(timeout=60.0) as client:
                if json_payload is None:
                    resp = client.post(url, headers=headers, params=params)
                else:
                    resp = client.post(url, headers=headers, json=json_payload, params=params)
            if 200 <= resp.status_code < 300:
                return resp
            if resp.status_code in (429, 408) or resp.status_code >= 500:
                ra = resp.headers.get("retry-after") or resp.headers.get("Retry-After")
                try:
                    ra_s = float(ra) if ra is not None else None
                except Exception:
                    ra_s = None
                wait = max(delay_seconds, ra_s or 0.0)
                _sleep_with_jitter(wait)
                delay_seconds = min(delay_seconds * 2.0, 16.0)
                continue
            # Non-retryable error; attach helpful context if account id looks invalid
            if 400 <= resp.status_code < 500:
                hint = ""
                if "/accounts/" in url and "@" in url:
                    hint = "\nHint: FIREWORKS_ACCOUNT_ID appears to be an email. Set it to your account slug (e.g., 'my-team'), not an email."
                raise httpx.HTTPStatusError(f"{resp.status_code} error for {url}: {resp.text}{hint}", request=resp.request, response=resp)
            resp.raise_for_status()
            return resp
        except httpx.RequestError as e:
            last_exc = e
            _sleep_with_jitter(delay_seconds)
            delay_seconds = min(delay_seconds * 2.0, 16.0)
        except Exception as e:
            last_exc = e
            _sleep_with_jitter(delay_seconds)
            delay_seconds = min(delay_seconds * 2.0, 16.0)
    if last_exc:
        raise last_exc
    raise RuntimeError(f"POST {url} failed after {max_attempts} attempts")


def _post_multipart_with_retries(url: str, headers: dict, files: dict, max_attempts: int = 6, base_delay: float = 1.0) -> httpx.Response:
    attempt = 0
    delay_seconds = base_delay
    last_exc: Exception | None = None
    while attempt < max_attempts:
        attempt += 1
        try:
            # Rewind any file-like objects so each retry re-sends the full payload
            try:
                for key, value in list(files.items()):
                    # Common shapes per httpx: {"file": (<name>, <fileobj>, <content_type>)}
                    if isinstance(value, (list, tuple)) and len(value) >= 2:
                        file_obj = value[1]
                        if hasattr(file_obj, "seek"):
                            try:
                                file_obj.seek(0)
                            except Exception:
                                pass
                    elif hasattr(value, "seek"):
                        try:
                            value.seek(0)
                        except Exception:
                            pass
            except Exception:
                # Best-effort rewind only; never fail the upload loop because of this
                pass
            with httpx.Client(timeout=None) as client:
                resp = client.post(url, headers=headers, files=files)
            if 200 <= resp.status_code < 300:
                return resp
            if resp.status_code in (429, 408) or resp.status_code >= 500:
                ra = resp.headers.get("retry-after") or resp.headers.get("Retry-After")
                try:
                    ra_s = float(ra) if ra is not None else None
                except Exception:
                    ra_s = None
                wait = max(delay_seconds, ra_s or 0.0)
                _sleep_with_jitter(wait)
                delay_seconds = min(delay_seconds * 2.0, 16.0)
                continue
            # Non-retryable error; attach helpful context if account id looks invalid
            if 400 <= resp.status_code < 500:
                hint = ""
                if "/accounts/" in url and "@" in url:
                    hint = "\nHint: FIREWORKS_ACCOUNT_ID appears to be an email. Set it to your account slug (e.g., 'my-team'), not an email."
                raise httpx.HTTPStatusError(f"{resp.status_code} error for {url}: {resp.text}{hint}", request=resp.request, response=resp)
            resp.raise_for_status()
            return resp
        except httpx.RequestError as e:
            last_exc = e
            _sleep_with_jitter(delay_seconds)
            delay_seconds = min(delay_seconds * 2.0, 16.0)
        except Exception as e:
            last_exc = e
            _sleep_with_jitter(delay_seconds)
            delay_seconds = min(delay_seconds * 2.0, 16.0)
    if last_exc:
        raise last_exc
    raise RuntimeError(f"POST {url} failed after {max_attempts} attempts")


def _get_dataset(account_id: str, dataset_id: str) -> dict:
    url = f"{DATASETS_BASE}/accounts/{account_id}/datasets/{dataset_id}"
    with httpx.Client(timeout=30.0) as client:
        resp = client.get(url, headers=auth_headers())
    if resp.status_code == 404:
        raise httpx.HTTPStatusError(
            f"404 Not Found for {url}", request=resp.request, response=resp
        )
    if 200 <= resp.status_code < 300:
        try:
            return resp.json()
        except Exception:
            return {}
    # Non-retryable error; attach helpful context if account id looks invalid
    if 400 <= resp.status_code < 500:
        hint = "\nHint: FIREWORKS_ACCOUNT_ID appears to be an email. Set it to your account slug (e.g., 'my-team'), not an email." if ("/accounts/" in url and "@" in url) else ""
        raise httpx.HTTPStatusError(f"{resp.status_code} error for {url}: {resp.text}{hint}", request=resp.request, response=resp)
    resp.raise_for_status()
    return {}


def _get_dataset_state(account_id: str, dataset_id: str) -> str | None:
    try:
        ds = _get_dataset(account_id, dataset_id)
        return ds.get("state")
    except httpx.HTTPStatusError as e:
        if e.response is not None and e.response.status_code == 404:
            return None
        raise


def _wait_until_ready(account_id: str, dataset_id: str, timeout_s: float = 900.0, poll_s: float = 2.0) -> None:
    deadline = time.time() + timeout_s
    last_state: str | None = None
    while time.time() < deadline:
        try:
            state = _get_dataset_state(account_id, dataset_id)
        except Exception:
            state = None
        if state == "READY":
            return
        # Some APIs may briefly report UNKNOWN/PROCESSING/UPLOADING; keep polling
        if state != last_state and state is not None:
            last_state = state
            print(f"Dataset {dataset_id} state: {state}")
        _sleep_with_jitter(poll_s)
    raise TimeoutError(f"Dataset {dataset_id} did not become READY within {int(timeout_s)}s")


def _request_signed_url(account_id: str, dataset_id: str, filename: str, size_bytes: int) -> str:
    url = f"{DATASETS_BASE}/accounts/{account_id}/datasets/{dataset_id}:getUploadEndpoint"
    payload = {"filenameToSize": {os.path.basename(filename): int(size_bytes)}}
    r = _post_json_with_retries(url, headers=auth_headers(), json_payload=payload)
    body = {}
    try:
        body = r.json()
    except Exception:
        pass
    # Response shape: {"filenameToSignedUrls": {"file": "https://..."}}
    mapping = body.get("filenameToSignedUrls") or body.get("filename_to_signed_urls") or {}
    signed = mapping.get(os.path.basename(filename))
    if not signed:
        raise RuntimeError("Signed URL not returned by getUploadEndpoint")
    return signed


def _required_signed_headers_from_url(signed_url: str) -> list[str]:
    """Return the lowercased list of headers that the signed URL requires."""
    q = parse_qs(urlparse(signed_url).query)
    sh = q.get("X-Goog-SignedHeaders") or q.get("x-goog-signedheaders") \
        or q.get("X-Amz-SignedHeaders") or q.get("x-amz-signedheaders") or []
    if not sh:
        return []
    return [h.strip().lower() for h in sh[0].split(";") if h.strip()]


def _put_file_to_signed_url(signed_url: str, local_path: str) -> None:
    # Note: No Fireworks auth header when PUT-ing to the storage signed URL
    size_bytes = os.stat(local_path).st_size
    required = set(_required_signed_headers_from_url(signed_url))

    # Default content type that matches common signing on Fireworks for datasets
    content_type = "application/octet-stream"

    headers: dict[str, str] = {}
    # Only include headers that were part of the signature. Values must match.
    if "content-type" in required:
        headers["Content-Type"] = content_type

    # GCS enforces range when this header is signed; must be present with the signed value.
    if "x-goog-content-length-range" in required:
        # For Fireworks-signed GCS URLs, this must be "<size>,<size>"
        headers["x-goog-content-length-range"] = f"{size_bytes},{size_bytes}"

    # Not part of the signature, but needed so the range check can be evaluated
    headers["Content-Length"] = str(size_bytes)

    # Brief log before PUT with minimal required headers and inputs
    try:
        print(
            f"Uploading via signed URL. Size: {size_bytes} bytes. "
            f"Required signed headers: {sorted(list(required))}. "
            f"Sending headers: {sorted(list(headers.keys()))}",
            file=sys.stderr,
        )
    except Exception:
        pass

    # Send the bytes
    with open(local_path, "rb") as f, httpx.Client(timeout=None) as client:
        resp = client.put(signed_url, data=f, headers=headers)

    if not (200 <= resp.status_code < 300):
        detail = resp.text[:2000] if resp.text else ""
        raise httpx.HTTPStatusError(
            f"Signed PUT failed with status {resp.status_code}. Body: {detail}",
            request=resp.request, response=resp
        )
    try:
        print(f"Signed PUT succeeded with status {resp.status_code}", file=sys.stderr)
    except Exception:
        pass


def _validate_upload(account_id: str, dataset_id: str) -> None:
    url = f"{DATASETS_BASE}/accounts/{account_id}/datasets/{dataset_id}:validateUpload"
    _post_json_with_retries(url, headers=auth_headers(), json_payload={})


def create_dataset(display_name: str, account_id: str) -> str:
    url = f"{DATASETS_BASE}/accounts/{account_id}/datasets"
    base_id = normalize_dataset_id(display_name)

    # Existence check (idempotent): if the dataset already exists, skip creation
    exists_url = f"{DATASETS_BASE}/accounts/{account_id}/datasets/{base_id}"
    try:
        with httpx.Client(timeout=30.0) as client:
            g = client.get(exists_url, headers=auth_headers())
        if g.status_code == 200:
            # It already exists; ensure it's not marked as newly created
            try:
                _newly_created_datasets.discard(base_id)
            except Exception:
                pass
            return base_id
        if g.status_code not in (200, 404):
            hint = "\nHint: FIREWORKS_ACCOUNT_ID appears to be an email. Set it to your account slug (e.g., 'my-team'), not an email." if ("/accounts/" in exists_url and "@" in exists_url) else ""
            raise httpx.HTTPStatusError(f"{g.status_code} error for {exists_url}: {g.text}{hint}", request=g.request, response=g)
    except httpx.RequestError:
        # Fall through to attempt creation; transient network error on GET shouldn't block creation
        pass

    # Per docs, send datasetId in the request body and include displayName
    payload = {
        "dataset": {
            "displayName": display_name[:63],
            # Mark as user uploaded so we can upload files into it
            "userUploaded": {},
        },
        "datasetId": base_id,
    }

    # Attempt creation; if 409 Conflict, append a numeric suffix and retry a few times defensively
    suffix = 1
    max_suffix = 5
    while True:
        try:
            r = _post_json_with_retries(url, headers=auth_headers(), json_payload=payload, params=None)
            try:
                _newly_created_datasets.add(base_id)
            except Exception:
                pass
            break
        except httpx.HTTPStatusError as e:
            if e.response is not None and e.response.status_code == 409:
                # Collision; adjust id/display and retry
                if suffix >= max_suffix:
                    # Give up retrying suffixes; treat as success with the original id
                    try:
                        _newly_created_datasets.discard(base_id)
                    except Exception:
                        pass
                    return base_id
                suffix += 1
                new_display = f"{display_name}-{suffix}"
                base_id = normalize_dataset_id(new_display)
                payload["dataset"]["displayName"] = new_display
                payload["datasetId"] = base_id
                continue
            raise

    body = r.json() if r.headers.get("content-type", "").lower().startswith("application/json") else {}
    # Common shapes: {"name": "accounts/<acc>/datasets/<id>"} or {"id": "<id>"}
    name = (
        body.get("name")
        or body.get("id")
        or (body.get("dataset") or {}).get("name")
        or ""
    )
    if not name:
        # Try to infer from Location header
        loc = r.headers.get("location") or r.headers.get("Location")
        if loc:
            name = loc
    if name:
        return name.split("/")[-1]
    # Fallback to the intended id if API returns 200 with an empty body
    return base_id


def upload_dataset_file(account_id: str, dataset_id: str, local_path: str, filename=None):
    """Upload a dataset file, choosing the correct workflow by size.

    - <=150MB: single-request multipart upload to :upload
    - >150MB: signed URL + validate flow

    Datasets already in READY are skipped. If a dataset exists but is not READY
    (e.g., stuck in UPLOADING), the upload/validate flow will attempt to finalize it.
    """
    # If dataset already READY, do nothing
    try:
        state = _get_dataset_state(account_id, dataset_id)
    except Exception:
        state = None
    if state == "READY":
        print(f"Dataset {dataset_id} is READY; skipping upload.", file=sys.stderr)
        return {"status": "skipped", "reason": "ready", "datasetId": dataset_id}

    size_bytes = os.stat(local_path).st_size
    fname = filename or os.path.basename(local_path)

    if size_bytes <= MB_150:
        # Single-request upload
        url = f"{DATASETS_BASE}/accounts/{account_id}/datasets/{dataset_id}:upload"
        with open(local_path, "rb") as f:
            files = {"file": (fname, f, "application/jsonl")}
            try:
                _post_multipart_with_retries(url, headers=auth_headers(), files=files)
            except httpx.HTTPStatusError as e:
                # Treat "already uploaded" as success for idempotency
                if e.response is not None and e.response.status_code == 400:
                    msg = ""
                    try:
                        body = e.response.json()
                        msg = body.get("error", {}).get("message", "") or body.get("message", "")
                    except Exception:
                        msg = e.response.text or ""
                    if "already uploaded" in msg.lower():
                        print(f"Dataset {dataset_id} already uploaded; skipping.", file=sys.stderr)
                    else:
                        raise
                else:
                    raise
            except Exception as e:
                # Fallback: if multipart repeatedly fails (e.g., persistent 429/5xx),
                # switch to the signed URL + validate flow for robustness.
                try:
                    print(
                        f"Multipart upload failed for {dataset_id} with {type(e).__name__}: {e}. "
                        f"Falling back to signed URL flow...",
                        file=sys.stderr,
                    )
                except Exception:
                    pass
                signed = _request_signed_url(account_id, dataset_id, fname, size_bytes)
                _put_file_to_signed_url(signed, local_path)
                _validate_upload(account_id, dataset_id)
                _wait_until_ready(account_id, dataset_id)
                return {"status": "uploaded", "method": "fallback_signed_url", "datasetId": dataset_id}
        # Confirm transition to READY
        _wait_until_ready(account_id, dataset_id)
        return {"status": "uploaded", "method": "multipart", "datasetId": dataset_id}

    # >150MB: presigned URL + validate
    signed = _request_signed_url(account_id, dataset_id, fname, size_bytes)
    try:
        print(
            f"Obtained signed URL for dataset {dataset_id}. File: {fname}, Size: {size_bytes} bytes.",
            file=sys.stderr,
        )
    except Exception:
        pass
    _put_file_to_signed_url(signed, local_path)
    _validate_upload(account_id, dataset_id)
    try:
        print(f"validateUpload succeeded for dataset {dataset_id}.", file=sys.stderr)
    except Exception:
        pass
    _wait_until_ready(account_id, dataset_id)
    try:
        print(f"Dataset {dataset_id} is READY.", file=sys.stderr)
    except Exception:
        pass
    return {"status": "uploaded", "method": "signed_url", "datasetId": dataset_id}


def main():
    load_dotenv()
    ap = argparse.ArgumentParser()
    ap.add_argument("--account", required=True)
    ap.add_argument("--name", required=True)
    ap.add_argument("--file", required=True)
    args = ap.parse_args()
    dsid = create_dataset(args.name, args.account)
    result = upload_dataset_file(args.account, dsid, args.file)
    # Machine-readable output; keep stdout clean of extra logs
    print(json.dumps(result))


if __name__ == "__main__":
    main()
