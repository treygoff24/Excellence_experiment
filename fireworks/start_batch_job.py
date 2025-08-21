from __future__ import annotations
import os, sys, json, httpx, argparse, random, time
from dotenv import load_dotenv
API_BASE = os.environ.get("FIREWORKS_API_BASE", "https://api.fireworks.ai")
BATCH_BASE = f"{API_BASE}/v1"
def auth_headers():
    key = os.environ.get("FIREWORKS_API_KEY")
    if not key: raise RuntimeError("FIREWORKS_API_KEY not set")
    return {"Authorization": f"Bearer {key}"}
def _sleep_with_jitter(seconds: float) -> None:
    jitter = min(0.5, seconds * 0.25)
    time.sleep(max(0.0, seconds) + random.random() * jitter)
def _post_json_with_retries(url: str, headers: dict, json_payload: dict | None, max_attempts: int = 6, base_delay: float = 1.0) -> httpx.Response:
    attempt = 0
    delay_seconds = base_delay
    last_exc: Exception | None = None
    while attempt < max_attempts:
        attempt += 1
        try:
            with httpx.Client(timeout=60.0) as client:
                if json_payload is None:
                    resp = client.post(url, headers=headers)
                else:
                    resp = client.post(url, headers=headers, json=json_payload)
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
            # Non-retryable error; attach helpful context and include response text for 4xx
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
def create_batch_job(account_id: str, model: str, input_dataset_id: str, display_name=None,
                     temperature=None, max_tokens=None, top_p=None, top_k=None, stop=None):
    # Normalize account: accept "accounts/<slug>[/...]" or bare slug; extract just the slug
    acc = (account_id or "").strip()
    if "/" in acc:
        try:
            # Prefer first segment after "accounts/" if present; else take first path segment
            acc = (acc.split("accounts/", 1)[1] if "accounts/" in acc else acc).split("/", 1)[0]
        except Exception:
            acc = acc.split("/", 1)[0]

    url = f"{BATCH_BASE}/accounts/{acc}/batchInferenceJobs"

    # Build inferenceParameters strictly in camelCase per API; put extras like stop into extraBody (JSON string)
    inference_params: dict[str, object] = {}
    if max_tokens is not None:
        inference_params["maxTokens"] = int(max_tokens)
    if temperature is not None:
        inference_params["temperature"] = float(temperature)
    if top_p is not None:
        inference_params["topP"] = float(top_p)
    if top_k is not None:
        inference_params["topK"] = int(top_k)
    extra: dict[str, object] = {}
    if stop:
        # Ensure list of strings; API expects this under extraBody as a stringified JSON object
        extra["stop"] = list(stop)
    if extra:
        inference_params["extraBody"] = json.dumps(extra)

    # Assemble payload; omit null/empty optional fields
    payload: dict[str, object] = {
        "model": model,
        "inputDatasetId": input_dataset_id,
        "inferenceParameters": inference_params,
    }
    if display_name:
        payload["displayName"] = display_name

    # Remove any empty nested dicts if present
    def _prune_empty(d: dict) -> dict:
        return {k: v for k, v in d.items() if v not in (None, {}, [], "")}
    payload = _prune_empty(payload)
    if "inferenceParameters" in payload:
        payload["inferenceParameters"] = _prune_empty(payload["inferenceParameters"])  # type: ignore[index]

    # Prepare headers and log a safe, redacted view of the request for debugging (no secrets)
    headers = auth_headers() | {"Content-Type": "application/json"}
    safe_headers = {"Authorization": "Bearer ****", "Content-Type": headers.get("Content-Type")}
    try:
        print(f"POST {url}", file=sys.stderr)
        print(f"Headers: {safe_headers}", file=sys.stderr)
        print(f"Payload: {json.dumps(payload, indent=2)}", file=sys.stderr)
    except Exception:
        # Never fail due to logging
        pass

    # Before creating the batch job, ensure the dataset is in "ready" state
    def get_dataset_state(account_id: str, dataset_id: str) -> str:
        acc = (account_id or "").strip()
        if "/" in acc:
            try:
                acc = (acc.split("accounts/", 1)[1] if "accounts/" in acc else acc).split("/", 1)[0]
            except Exception:
                acc = acc.split("/", 1)[0]
        url = f"{BATCH_BASE}/accounts/{acc}/datasets/{dataset_id}"
        headers = auth_headers()
        with httpx.Client(timeout=15.0) as client:
            resp = client.get(url, headers=headers)
            resp.raise_for_status()
            data = resp.json()
        return data.get("state", "")

    def wait_for_dataset_ready(account_id: str, dataset_id: str, timeout_sec: int = 900, poll_interval_sec: int = 5) -> bool:
        import time
        start = time.monotonic()
        while (time.monotonic() - start) < timeout_sec:
            state = get_dataset_state(account_id, dataset_id)
            print(f"Dataset {dataset_id} state: {state}")
            if state == "READY":
                return True
            time.sleep(poll_interval_sec)
        return False

    if not wait_for_dataset_ready(account_id, input_dataset_id):
        raise RuntimeError(f"Dataset {input_dataset_id} is not ready after waiting")

    try:
        r = _post_json_with_retries(url, headers=headers, json_payload=payload)
        return r.json()
    except httpx.HTTPStatusError as e:
        # Surface server message for 400s and other client errors
        msg = e.response.text if getattr(e, "response", None) is not None else str(e)
        raise httpx.HTTPStatusError(f"Failed to create batch job: {msg}", request=getattr(e, "request", None), response=getattr(e, "response", None))
def main():
    load_dotenv()
    ap = argparse.ArgumentParser()
    ap.add_argument("--account", required=True)
    ap.add_argument("--model", required=True)
    ap.add_argument("--input_dataset_id", required=True)
    ap.add_argument("--name", default=None)
    ap.add_argument("--temperature", type=float, default=None)
    ap.add_argument("--max_tokens", type=int, default=None)
    ap.add_argument("--top_p", type=float, default=None)
    ap.add_argument("--top_k", type=int, default=None)
    ap.add_argument("--stop", nargs='*', default=None)
    args = ap.parse_args()
    job = create_batch_job(
        args.account,
        args.model,
        args.input_dataset_id,
        args.name,
        args.temperature,
        args.max_tokens,
        args.top_p,
        args.top_k,
        args.stop,
    )
    print(json.dumps(job, indent=2))
if __name__ == "__main__":
    main()
