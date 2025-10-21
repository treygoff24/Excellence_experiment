

# Alternative Batch Pipeline for GPT‑5 (OpenAI) and Claude (Anthropic)

> **Scope**: This guide shows how to add an **alternative experiment pipeline**—without modifying your current Fireworks-based pipeline—that reuses nearly all stages of your workflow while enabling experiments to run via **OpenAI’s Batch API** (for GPT‑5) and **Anthropic’s Message Batches API** (for Claude). It emphasizes asynchronous **batch** execution for cost, throughput, and reproducibility.

---

## 1) Introduction

You already operate a deterministic, manifest-driven evaluation pipeline built around Fireworks batch inference. The goal here is to introduce a **parallel** pipeline that targets (a) OpenAI’s GPT‑5 via the OpenAI API and (b) Anthropic’s Claude via the Anthropic API—**without touching the existing pipeline**. We will **reuse** your preparation, batching, scoring, statistics, cost summarization, and reporting stages by introducing **backend adapters** and a thin orchestrator that swaps in model‑provider–specific “submit/poll/parse” steps. This design preserves idempotence, manifest logging, and artifacts while unlocking model diversity and batch discounts. 

**Validation (step result):** The existing pipeline exposes clean seams at “submit → poll → parse,” and its parsers already expect **OpenAI‑compatible** JSON. That’s sufficient to route OpenAI batches directly and to normalize Anthropic results into the same shape for downstream reuse. Proceed.

---

## 2) Prerequisites

* **Accounts & Keys**

  * OpenAI API account with access to **GPT‑5** models and the **Batch API**. (Batch pricing shows **~50% discount** vs synchronous, with 24h completion window). ([OpenAI][1])
  * Anthropic API account with access to **Claude** models and **Message Batches** (GA). Anthropic batches permit up to **10,000 requests per batch** and **50% lower cost** vs standard calls, with <24h completion. ([Anthropic][2])
* **SDKs**

  * `openai` (latest) for Python (the **Batch API** is exposed as `client.batches.*`; files live under `client.files.*`). ([OpenAI Cookbook][3])
  * `anthropic` (latest) for Python (`client.messages.batches.*` streaming `.results` to JSONL). ([Claude Docs][4])
* **Foundational knowledge**

  * Your repo structure, manifests, and phase boundaries: **prepare → build → submit → poll → parse → score → stats → costs → report → archive**. 

**Validation:** Both providers offer official batch modes with JSONL results and SDK bindings. Download and auth steps are supported. Proceed.

---

## 3) Overview of the Existing Pipeline (and how we’ll reuse it)

Your current architecture is explicitly stageful and idempotent. The **only** Fireworks‑specific code lives under the “integration” layer (dataset upload, start batch, queue manager, poll/download, parse), while the orchestrator, data prep, batching, scoring, and stats are generic. We will:

* **Call your existing**: data prep, batch build (JSONL shards), scoring, statistics, cost summarization, reporting, archiving.
* **Swap only**: submission/polling/parsing code by adding **new provider adapters** (OpenAI, Anthropic) and a thin **alternative orchestrator** that uses your manifest v2 and run state utilities to mimic stage gating. 

**Validation:** The repo already isolates Fireworks functions in their own modules; manifests and parsers are stable. Proceed.

---

## 4) API Research Summary (with stepwise validation)

### 4.1 OpenAI (GPT‑5) — Batch processing

**Key findings (from official OpenAI docs and cookbook):**

* **Batch API** submits a JSONL **input file** (each line: `custom_id`, HTTP `method`, `url`, and `body` mirroring the target endpoint), creates a **batch job**, and returns an **`output_file_id`** for results. Supported endpoints include **`/v1/responses`** and **`/v1/chat/completions`**; results may be in **different order**—use `custom_id` to rejoin. ([OpenAI Cookbook][3])
* Create job via `client.batches.create(input_file_id, endpoint="/v1/... ", completion_window="24h")`; retrieve with `client.batches.retrieve(id)`; download with `client.files.content(output_file_id)`. ([OpenAI Cookbook][3])
* **Pricing**: OpenAI advertises **~50% lower cost** for batch vs synchronous; completion window ~**24h**; Batch has **separate rate limits** from real‑time. ([OpenAI][1])
* **Model**: **GPT‑5** models are available in the Responses API; “reasoning effort” is configurable (e.g., minimal/low/medium/high) in Responses. (We’ll use `reasoning: {effort: "high"}` for parity with your instruction.) ([OpenAI Platform][5])

**Validation (OpenAI):** Official docs confirm the Batch JSONL shape, endpoints, `custom_id` contract, SDK calls, and 50% cost guidance. We will target `/v1/responses` for GPT‑5; fall back to `/v1/chat/completions` if needed. Proceed.

---

### 4.2 Anthropic (Claude) — Message Batches

**Key findings (official docs):**

* **Message Batches API** accepts an array of **`requests`**, each with a `custom_id` and **Messages API** parameters (`model`, `max_tokens`, `messages`, etc.), processes up to **10,000 requests per batch**, and exposes a **`results_url`** for streaming `.jsonl` results where order is **not guaranteed**; join by `custom_id`. ([Claude Docs][4])
* Batch completion is polled via `client.messages.batches.retrieve(id)` until `processing_status=="ended"`, then streamed via `client.messages.batches.results(id)`. ([Claude Docs][4])
* Launch announcement confirms **50% discount** and **<24h** processing. ([Anthropic][2])
* Rate limits for **Message Batches** are tracked separately (queue size; RPM). ([Claude Docs][6])

**Validation (Anthropic):** SDK and docs provide first‑class Message Batches with `custom_id` and JSONL results; economics and limits are documented. Proceed.

---

## 5) Designing the Alternative Pipeline

### 5.1 High‑level architecture

Introduce **`backends/`** with two adapters:

* `backends/openai/` — `upload_batch_input.py`, `start_batch_job.py`, `poll_and_download.py`, `parse_results.py`
* `backends/anthropic/` — `start_message_batch.py`, `poll_and_stream.py`, `parse_results.py`

Add **`scripts/alt_run_all.py`** (new, not replacing anything) that:

* Reuses your **prepare** and **build** phases.
* Switches to **OpenAI** or **Anthropic** adapters for **submit → poll → parse**.
* Drops normalized outputs (combined JSONL + CSV) under the **same trial directory layout** used today, then calls your **score → stats → costs → report → archive** as‑is. 
* Keeps the **shared control registry** flow intact so control shards are produced once, reused across trials, and refreshed on resume just like the Fireworks path.

### 5.2 Integration points with the current pipeline

* **Reuse**: `scripts/prepare_data.py`, `scripts/build_batches.py`, scoring modules, statistics, reporting, archiving, manifest utilities, run‑state utilities. 
* **Swap**:

  * **Submit**: `fireworks/start_batch_job.py` is replaced by `openai/start_batch_job.py` or `anthropic/start_message_batch.py`.
  * **Poll**: replace `fireworks/poll_and_download.py` with per‑provider pollers.
  * **Parse**: reuse `fireworks/parse_results.py` for **OpenAI**; add a **thin normalizer** to convert Anthropic messages into the same OpenAI‑compatible shape, then call your existing parser unchanged. 

### 5.3 Extensibility for future models/APIs

* The **adapter interface** is intentionally small: `build_requests() → submit() → poll() → materialize_results() → normalize()`. New providers implement this interface and drop into `alt_run_all`.
* Keep **provider config blocks** isolated so you can add **Azure OpenAI Batch** (same semantics, different base URL/headers) or others later. ([Microsoft Learn][7])

**Validation (design):** The adapter seam aligns with your current Fireworks isolation; normalizing to one JSON schema allows full reuse downstream. Proceed.

---

## 6) Implementation Steps

### 6.1 Setup & configuration (new files only—no edits to the original pipeline)

1. **Create alt orchestrator**
   `scripts/alt_run_all.py` — imports existing `prepare`/`build`/`score`/`stats`/`report` functions; selects a backend by CLI flag (`--backend {openai,anthropic}`) and calls that adapter for submit/poll/parse. It writes **manifest v2** entries identical to Fireworks’ stages (stage keys: `submitted`, `downloaded`, `parsed`) and persists provider metadata (e.g., `batch_id`, `output_file_id`, `results_uri`) alongside those markers so resume/debug tooling has parity with the Fireworks manifests. 

2. **Add config (new)**
   `config/alt_eval_config.yaml` with a provider stanza (do **not** modify existing config):

   ```yaml
   backend: alt            # {openai|anthropic} set at runtime via CLI flag
   provider:
     name: openai          # or 'anthropic'
     model: gpt-5          # e.g., 'gpt-5' or 'claude-sonnet-4-5'
     batch:
       endpoint: /v1/responses   # OpenAI only; '/v1/chat/completions' fallback
       completion_window: 24h     # OpenAI Batch
       max_tokens: 1024
       temperature: 0.0
       # Anthropic ignores endpoint/completion_window; uses Messages params
   pricing:
     openai:
       gpt-5:
         input_per_mtok: 1.25
         output_per_mtok: 10.00
         batch_discount: 0.5   # informational; OpenAI prices publish 50% via Batch API
     anthropic:
       claude-sonnet-4-5:
         input_per_mtok_batch: 1.50
         output_per_mtok_batch: 7.50
   ```

   (Use your existing `scripts/summarize_costs.py` mapping to reconcile OpenAI `usage.prompt_tokens/completion_tokens` and Anthropic `usage.input_tokens/output_tokens`.) ([OpenAI][1])

3. **Directory layout**

   * `backends/openai/*` and `backends/anthropic/*` mirror the Fireworks integration layer file structure to keep mental symmetry. 

**Validation:** New files avoid touching the original pipeline; CLI chooses backend; cost accounting stays explicit. Proceed.

---

### 6.2 Building provider‑specific batch inputs

**OpenAI** (uses **JSONL** upload; results via `output_file_id`):

* Convert each of your built shard lines (which already package system/user messages) into **Batch JSONL records**:

```json
{
  "custom_id": "t0_control_p01_i0000001",
  "method": "POST",
  "url": "/v1/responses",
  "body": {
    "model": "gpt-5",
    "input": [
      {"role": "developer", "content": "<system prompt here>"},
      {"role": "user", "content": "<task instruction + question here>"}
    ],
    "temperature": 0.0,
    "max_output_tokens": 1024,
    "reasoning": { "effort": "high" }
  }
}
```

* If you prefer **Chat Completions**, set `url` to `"/v1/chat/completions"` and move messages under `body.messages=[...]`. The Batch API accepts the same parameters as the target endpoint; always supply a unique `custom_id`. Results are **not ordered**, so always join by `custom_id`. ([OpenAI Cookbook][3])
* Keep shard sizes within the provider limits—OpenAI currently caps batch inputs at **≤10,000 requests** per submission and **≤100 MB** per uploaded JSONL file, so gate per-shard manifest construction accordingly. ([OpenAI Cookbook][3])

**Anthropic** (uses **requests array**; results streamed):

* Build a `requests` list where each item has a `custom_id` and **Messages API** params:

```python
# backends/anthropic/build_requests.py
from anthropic.types.message_create_params import MessageCreateParamsNonStreaming
from anthropic.types.messages.batch_create_params import Request

def build_requests(rows, model="claude-sonnet-4-5", max_tokens=1024, temperature=0.0):
    reqs = []
    for r in rows:
        reqs.append(Request(
            custom_id=r["custom_id"],  # preserved from your shard
            params=MessageCreateParamsNonStreaming(
                model=model,
                max_tokens=max_tokens,
                temperature=temperature,
                messages=[{"role":"user","content": r["user_payload"]}],
                system=r["system_prompt"]
            )
        ))
    return reqs
```

Anthropic’s batch API supports the same features as the **Messages** API and produces a **`.jsonl`** results stream; join by `custom_id`. Message Batches currently accept up to **10,000 requests** per batch, so mirror the sharding guardrails you use for OpenAI. ([Claude Docs][4])

**Validation:** Both providers confirm the **`custom_id`** contract and JSONL flow. Proceed.

---

### 6.3 Submitting batches

**OpenAI (Python SDK)**

```python
# backends/openai/start_batch_job.py
from openai import OpenAI
client = OpenAI()

# Write JSONL input file to disk
input_path = shard_input_path  # produced by your build step
batch_file = client.files.create(file=open(input_path, "rb"), purpose="batch")

batch_job = client.batches.create(
    input_file_id=batch_file.id,
    endpoint="/v1/responses",     # or "/v1/chat/completions"
    completion_window="24h",
    metadata={"trial": trial_slug, "part": part_key}
)
# Persist batch_job.id into manifest
```

Download results once `status == "completed"`:

```python
import io, zipfile, pathlib

job = client.batches.retrieve(batch_job.id)
result_bytes = client.files.content(job.output_file_id).content
with zipfile.ZipFile(io.BytesIO(result_bytes)) as archive:
    archive.extractall(pathlib.Path(results_jsonl_path).parent)
# OpenAI returns results as a ZIP archive; reuse the Fireworks extractor helper to
# land `<something>.jsonl` before flowing into the existing parser.
```

Official cookbook shows the **file upload → create → poll → download** cycle and the JSONL record shape. ([OpenAI Cookbook][3])

**Anthropic (Python SDK)**

```python
# backends/anthropic/start_message_batch.py
import anthropic, time
client = anthropic.Anthropic()

message_batch = client.messages.batches.create(requests=build_requests(...))

# Poll until ended
while True:
    b = client.messages.batches.retrieve(message_batch.id)
    if b.processing_status == "ended":
        break
    time.sleep(30)

# Stream results to a JSONL on disk
with open(results_jsonl_path, "w") as w:
    for result in client.messages.batches.results(message_batch.id):
        w.write(result.model_dump_json() + "\n")
```

Docs show `create → retrieve (poll) → results (stream)` and that results are **unordered**. ([Claude Docs][4])

**Validation:** Both flows are SDK‑first and idiomatic. Proceed.

---

### 6.4 Parsing & normalization

* **OpenAI**: Your current parser (“reads Fireworks JSONL (OpenAI‑compatible responses) → predictions.csv”) should work on Batch results that mirror **Chat** or **Responses** bodies; if using Responses, adapt a small shim to read `response.body.output_text` (or `choices[0].message.content` for Chat) into the same `"text"` field your parser expects. 
* **Anthropic**: Add a **thin normalizer** (`backends/anthropic/normalize_to_openai.py`) that maps:

  * Anthropic message → synthetic OpenAI‑like object:

    ```json
    {
      "custom_id": "...",
      "response": {
        "body": {
          "choices": [{
            "message": {"content": "<anthropic_text>"},
            "finish_reason": "<stop_reason>"
          }],
          "usage": {
            "prompt_tokens": <input_tokens>,
            "completion_tokens": <output_tokens>
          }
        }
      }
    }
    ```
  * Then pipe that JSONL into your existing `parse_results.py` for CSV emission.

**Validation:** This preserves downstream compatibility and idempotence. Proceed.

---

### 6.5 Cost accounting and pricing

* **OpenAI**: Use the published pricing for GPT‑5; apply **Batch API** discount per OpenAI’s pricing page (50% vs synchronous). Keep a provider→model→rate table and map token fields from the result usage objects. ([OpenAI][1])
* **Anthropic**: Use the **Message Batches** discounted rates (e.g., Sonnet 3.5/4.5 figures) and Anthropic usage fields. ([Anthropic][2])
* Reuse your `scripts/summarize_costs.py` plumbing; add provider guards that read **OpenAI `usage.prompt_tokens/completion_tokens`** vs **Anthropic `usage.input_tokens/output_tokens`**. 

**Validation:** Both providers publish explicit batch pricing; your cost summarizer just needs a field map. Proceed.

---

### 6.6 Example end‑to‑end calls (minimal)

**OpenAI batch with GPT‑5 (Responses API)**
*Build `input.jsonl` and run:*

```python
from openai import OpenAI, BadRequestError
client = OpenAI()

# Upload JSONL of {custom_id, method, url="/v1/responses", body: {...}}
f = client.files.create(file=open("input.jsonl", "rb"), purpose="batch")
job = client.batches.create(input_file_id=f.id, endpoint="/v1/responses", completion_window="24h")

# poll
job = client.batches.retrieve(job.id)
# later...
result = client.files.content(job.output_file_id).content
open("results.jsonl","wb").write(result)
```

Batch guides/cookbook confirm structure and flow. ([OpenAI Cookbook][3])

**Anthropic batch with Claude (Messages API)**

```python
import anthropic, time
from anthropic.types.messages.batch_create_params import Request
from anthropic.types.message_create_params import MessageCreateParamsNonStreaming

client = anthropic.Anthropic()
reqs = [
  Request(
    custom_id="item-0001",
    params=MessageCreateParamsNonStreaming(
      model="claude-sonnet-4-5",
      max_tokens=512,
      messages=[{"role":"user","content":"Explain selective risk metrics in one paragraph."}],
      system="You are a careful evaluator."
    )
  )
]
batch = client.messages.batches.create(requests=reqs)
while client.messages.batches.retrieve(batch.id).processing_status != "ended":
    time.sleep(10)

with open("results.jsonl","w") as w:
    for r in client.messages.batches.results(batch.id):
        w.write(r.model_dump_json()+"\n")
```

This matches Anthropic’s example semantics. ([Claude Docs][4])

**Validation:** Samples align with docs and are minimal; your build/orchestrator will wrap these. Proceed.

---

## 7) Error Handling (robustness playbook)

**Classes of failure and recommended handling**

* **Validation failures** (bad JSONL, wrong URL/endpoint):

  * **OpenAI** returns validation status before running; job may be `failed`. Record in manifest and re‑emit the shard with corrected JSONL. Use docs’ `custom_id` to surgically retry failed rows only. ([OpenAI Cookbook][3])
* **Partial results / unordered results**:

  * Both batch APIs can return subset and unordered lines. Always index by `custom_id` and join to the shard manifest to know which rows to re‑enqueue. ([OpenAI Cookbook][3])
* **Timeout / expiration**:

  * OpenAI SLA: batch completes **within ~24h**; if expired, completed work is returned; you’re charged only for completed requests. Annotate manifest and re‑batch residuals. ([OpenAI Help Center][8])
* **429 / rate limits**:

  * **OpenAI** Batch has a **separate pool**; Anthropic’s Message Batches has a separate queue and RPM. Implement exponential backoff on polling and **respect `Retry-After`** where present. ([OpenAI Help Center][8])
* **HTTP 5xx / transient CDN**:

  * Mirror your Fireworks retry/backoff logic; for OpenAI, re‑fetch `output_file_id` on transient errors; for Anthropic, resume streaming results (idempotent). ([OpenAI Cookbook][3])
* **Usage/tokens mismatch**:

  * Normalize both providers’ usage into your cost schema; log both raw and normalized usage in your trial manifest to aid audits. 
* **Reasoning settings**:

  * For GPT‑5 via Responses, prefer `reasoning: {effort: "high"}` for parity with your instruction here; verify it is accepted by the model class and record in the manifest. ([OpenAI Platform][5])

**Validation:** Official docs cover batch lifecycle, pricing/SLAs, and limits; proposed handlers map to your existing queue manager semantics. Proceed.

---

## 8) Testing and Validation

1. **Smoke test** (one shard, N=50 items, `samples_per_item=1`, `temperature=0.0`):

   * Run OpenAI and Anthropic backends separately; confirm:

     * `results_combined.jsonl` produced; CSV extract matches expected schema.
     * `per_item_scores.csv` renders; `significance.json` deterministic. 
2. **Golden outputs**:

   * Freeze a **golden CSV** per backend for a tiny dataset; assert exact string matches across runs (idempotence).
3. **Failure injection**:

   * Corrupt 10 lines in a JSONL (OpenAI): ensure `failed` status captured; retry path re‑emits only failed `custom_id`s.
   * Kill Anthropic poll loop mid‑run: ensure resume logic picks up from `processing_status`.
4. **Cost sanity**:

   * Compare cost deltas with/without batch (OpenAI) and with Anthropic Message Batches; verify **~50%** reduction vs synchronous for same token usage (within rounding). ([OpenAI][1])
5. **Statistical parity**:

   * Verify that B/A metrics and McNemar/bootstrap outputs appear under the same keys, preserving your reporting scripts. 

**Validation:** The plan exercises batch lifecycle, determinism, and cost claims with concrete checks. Proceed.

---

## 9) Conclusion

By adding a **provider‑adapter layer** and a **thin alternative orchestrator**, you can run **GPT‑5** and **Claude** experiments on top of your **unchanged** evaluation pipeline. Both OpenAI’s Batch API and Anthropic’s Message Batches give you **bulk throughput** and **~50% lower cost** for non‑urgent workloads, and both return **JSONL** keyed by `custom_id`, allowing clean joins back into your deterministic manifests and scoring stack. This approach keeps your **idempotent**, **resumable**, ** statistically rigorous** workflow intact while broadening your model surface area.

---

## 10) References

* **Your pipeline architecture & manifests** — Excellence Experiment Developer Onboarding Guide. 
* **OpenAI**

  * Batch API guide & API reference (input JSONL, `custom_id`, 24h window; results unordered): **official**. ([OpenAI Cookbook][3])
  * Batch API pricing/positioning (≈50% discount; separate limits; 24h window): **official**. ([OpenAI][1])
  * GPT‑5 model pages and Responses API / reasoning effort parameter: **official**. ([OpenAI Platform][9])
* **Anthropic**

  * Message Batches API (create → retrieve → results; `custom_id`; unordered results): **official**. ([Claude Docs][4])
  * Launch note (10k requests/batch; 50% discount; <24h): **official**. ([Anthropic][2])
  * Rate limits for Message Batches (RPM; queue limits): **official**. ([Claude Docs][6])

---

### Appendix — Minimal code templates (drop‑in files)

> These are deliberately short and provider‑specific; wire them behind `scripts/alt_run_all.py`.

**`backends/openai/submit_and_collect.py`**

```python
from openai import OpenAI
import json, time, pathlib

def submit_and_collect(input_jsonl_path: str, out_path: str, endpoint="/v1/responses"):
    client = OpenAI()
    # Upload batch input
    up = client.files.create(file=open(input_jsonl_path, "rb"), purpose="batch")
    job = client.batches.create(input_file_id=up.id, endpoint=endpoint, completion_window="24h")
    # Poll
    while True:
        j = client.batches.retrieve(job.id)
        if j.status in ("completed", "failed", "expired", "canceled"):
            break
        time.sleep(30)
    if j.status != "completed":
        return {"status": j.status, "output_file_id": None}
    # Download results JSONL
    b = client.files.content(j.output_file_id).content
    pathlib.Path(out_path).write_bytes(b)
    return {"status": j.status, "output_file_id": j.output_file_id}
```

(Flow mirrors the official cookbook.) ([OpenAI Cookbook][3])

**`backends/anthropic/submit_and_collect.py`**

```python
import anthropic, time, pathlib

def submit_and_collect(requests, out_path: str):
    client = anthropic.Anthropic()
    batch = client.messages.batches.create(requests=requests)
    while True:
        cur = client.messages.batches.retrieve(batch.id)
        if cur.processing_status == "ended":
            break
        time.sleep(30)
    with open(out_path, "w") as w:
        for res in client.messages.batches.results(batch.id):
            w.write(res.model_dump_json() + "\n")
    return {"status": "completed", "batch_id": batch.id}
```

(Flow mirrors Anthropic’s examples; `.results` streams JSONL.) ([Claude Docs][4])

**`backends/anthropic/normalize_to_openai.py`** (sketch)

```python
import json

def normalize_line(anthropic_result_line: dict) -> dict:
    cid = anthropic_result_line.get("custom_id")
    res = anthropic_result_line["result"]
    if res["type"] != "succeeded":
        return {"custom_id": cid, "error": res["type"]}
    msg = res["message"]
    text_chunks = [c["text"] for c in msg["content"] if c["type"] == "text"]
    text = "".join(text_chunks)
    usage = msg.get("usage", {})
    return {
        "custom_id": cid,
        "response": {
            "body": {
                "choices": [{
                    "message": {"content": text},
                    "finish_reason": msg.get("stop_reason", "stop")
                }],
                "usage": {
                    "prompt_tokens": usage.get("input_tokens"),
                    "completion_tokens": usage.get("output_tokens")
                }
            }
        }
    }
```

(Bridges Anthropic Messages output to your OpenAI‑compatible parser.)

---

**Final validation:** All required capabilities—OpenAI **Batch API** and Anthropic **Message Batches**—are present, documented, and compatible with a normalization shim. The architecture avoids touching your existing pipeline while maximizing reuse of manifests, scoring, and reporting. Proceed to implement.

---

## 10) Verification & Smoke Coverage (2025-10)

To keep the alternative backends shippable without real API keys, the repo now includes a repeatable smoke harness and recorded fixtures:

* `scripts.alt_run_all --replay_dir <fixtures>` hydrates canned OpenAI/Anthropic results and runs the parse stage end-to-end. Failure fixtures (`replay_validation`, `replay_partial`, `replay_expired`) simulate validation blow-ups, partial data, and expired batches without live calls.
* `python -m scripts.alt_smoke --mode dry-run` executes dual-provider dry runs on fixture shards (submit+poll). `--mode replay` replays goldens to exercise parsing. `make alt-smoke` integrates the dry-run path for CI and local spot checks.
* `pytest tests/smoke/test_alt_backends.py` asserts the recorded replay goldens (JSONL + CSV) and verifies failure-injection + resume semantics for both providers.

Use these commands as the default verification loop before requesting production API access or touching adapter internals.

[1]: https://openai.com/api/pricing/?utm_source=chatgpt.com "API Pricing"
[2]: https://www.anthropic.com/news/message-batches-api "Introducing the Message Batches API \ Anthropic"
[3]: https://cookbook.openai.com/examples/batch_processing "Batch processing with the Batch API"
[4]: https://docs.claude.com/en/api/messages-batch-examples "Message Batches examples - Claude Docs"
[5]: https://platform.openai.com/docs/guides/latest-model?utm_source=chatgpt.com "Using GPT-5 - OpenAI API"
[6]: https://docs.claude.com/en/api/rate-limits "Rate limits - Claude Docs"
[7]: https://learn.microsoft.com/en-us/azure/ai-foundry/openai/how-to/batch?utm_source=chatgpt.com "Getting started with Azure OpenAI batch deployments"
[8]: https://help.openai.com/en/articles/9197833-batch-api-faq?utm_source=chatgpt.com "Batch API FAQ"
[9]: https://platform.openai.com/docs/models/gpt-5?utm_source=chatgpt.com "Model - OpenAI API"
