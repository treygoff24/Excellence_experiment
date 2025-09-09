
# Windows + Local LLM Port — Full Action Plan (Executable Tickets)

**Purpose.** Adapt the “Excellence Experiment” prompt-eval framework to run **entirely on Windows**, using **local, open‑source LLMs** on an **RTX 5080 (16 GB VRAM) laptop GPU** + **Intel Ultra 9 Series 2 CPU**, while preserving existing artifacts and evaluation contract (datasets → batch inputs → predictions.csv → scoring/stats/report). We will introduce a **pluggable inference backend** and add a **Windows‑native developer workflow** without changing scoring/statistics logic or report generation. According to the current README, the orchestrator phases are `prepare → build → submit → poll → parse → score → stats → costs → report`; we will keep this interface and outputs stable. fileciteturn0file0

**How to use.** Assign each ticket to an agent. Tickets are sequenced; do them in order unless marked parallelizable. Each ticket includes **Objective**, **Why**, **Inputs**, **Steps**, **Acceptance Criteria**, **Risks/Mitigations**, and **Validation (goals + hardware)**. All commands assume **PowerShell** on Windows.

---

## Hardware/OS Assumptions (Read First)

- OS: Windows 11 (PowerShell 7+ recommended).
- GPU: **NVIDIA RTX 5080 (16 GB VRAM)** laptop variant.
- CPU: **Intel Ultra 9 Series 2** (many efficiency/performance cores; hyperthreading enabled).
- Python: 3.11+ (CPython).
- CUDA drivers installed (for llama.cpp CUDA build or Ollama’s CUDA backend).
- Disk: ≥ 50 GB free (models + datasets + results).
- Network: Not required for inference once models are pulled; required for first-time model downloads.
- Antivirus: Exclude the repo’s `experiments/`, `results/`, and `data/` directories to reduce IO contention (optional but recommended).

**Initial perf defaults (safe on 16 GB VRAM):**
- Model: 7–8B Instruct **Q4_K_M** (throughput) or **Q6_K** (quality).
- Context: 4k tokens (start), `max_new_tokens ≤ 1024`.
- Concurrency: `max_concurrent_requests = 1` (start), try `2` after smoke tests.
- Scoring/stats threads (CPU): 8 (tune per core count).

---

## Ticket 0 — Windows Bootstrap & Developer Ergonomics

**Objective.** Provide a one-command Windows setup and reproducible environment.

**Why.** The current docs assume Unix tools; Windows needs PowerShell and path/process fixes. fileciteturn0file0

**Inputs.** Root repo; `requirements.txt`.

**Steps.**
1. Create PowerShell bootstrap script at `tools\bootstrap.ps1`:
   ```powershell
   # tools/bootstrap.ps1
   param([switch]$Recreate)

   if ($Recreate -and (Test-Path .venv)) { Remove-Item .venv -Recurse -Force }
   py -3.11 -m venv .venv
   .\.venv\Scripts\Activate.ps1
   python -m pip install --upgrade pip
   pip install -r requirements.txt
   Write-Host "✔ Virtualenv ready. Run '.\.venv\Scripts\Activate.ps1' to activate."
   ```
2. Add `tools\tasks.ps1` (Windows equivalent to Make targets referenced in README) mapping to the same phases (data/build/eval/parse/score/stats/report/plan/smoke). Each function calls the Python entry points used by the orchestrator.
3. Document usage in `docs\windows.md` with copy/paste commands.

**Acceptance Criteria.**
- Running `powershell -ExecutionPolicy Bypass -File tools\bootstrap.ps1` creates a working venv and installs deps.
- `tools\tasks.ps1 -List` shows available tasks (eval, smoke, score, stats, report, plan).

**Risks/Mitigations.**
- _Risk:_ Execution policy blocks scripts → _Mitigate:_ instruct `-ExecutionPolicy Bypass`.
- _Risk:_ Python not on PATH → _Mitigate:_ use `py -3.11` launcher.

**Validation (goals + hardware).**
- Aligns with repo’s phase names; no change to Python packages or versions. CPU/GPU untouched. fileciteturn0file0

---

## Ticket 1 — Introduce Pluggable Inference Backend Interface

**Objective.** Decouple inference from Fireworks by adding a minimal backend interface and refactoring Fireworks code under `backends\fireworks\`.

**Why.** We need a **local** inference path while preserving orchestration, manifests, and downstream scoring. fileciteturn0file0

**Inputs.** `scripts\run_all.py`, `fireworks\*.py` (submission/poll/parse), config loader.

**Steps.**
1. Create `backends\__init__.py` with an abstract `InferenceClient` and `BatchExecutor` protocol:
   ```python
   # backends/interfaces.py
   from typing import Protocol, Iterable, Dict, Any

   class InferenceClient(Protocol):
       def generate(self, *, messages: list[dict[str, str]] | None = None,
                         prompt: str | None = None,
                         model: str = "", params: dict[str, Any] | None = None
                   ) -> dict[str, Any]:  # {text, finish_reason, usage?}
           ...

   class BatchExecutor(Protocol):
       def run_jsonl(self, input_path: str, output_path: str, max_concurrent: int = 1) -> None:
           ...
   ```
2. Move Fireworks-specific modules into `backends\fireworks\` and expose an adapter that implements the same `BatchExecutor` surface (no functional change).
3. Add `backends\local\` package with empty stubs for now: `ollama_client.py`, `llama_cpp_client.py`, `local_batch.py`, `parse_results.py`.

4. Update config schema (see Ticket 6) to include `backend: "fireworks" | "local"` and local engine fields.

5. In `scripts\run_all.py`, switch on `config.backend` to call `backends.local.local_batch` instead of Fireworks when set to `"local"`. Do not change the phase names or manifests.

**Acceptance Criteria.**
- `python -m scripts.run_all --plan_only --backend local` prints a plan that includes the same phases but uses the local executor for `submit/poll/parse` equivalently.
- Fireworks path still works when `--backend fireworks` (no regressions).

**Risks/Mitigations.**
- _Risk:_ Interface drift with downstream parse/score → _Mitigate:_ enforce identical `predictions.csv` schema in Ticket 5.

**Validation.**
- Preserves orchestrator phases and artifacts as in README. fileciteturn0file0

---

## Ticket 2 — Local Engine (Preferred): Ollama HTTP Client

**Objective.** Implement an Ollama-backed `InferenceClient` using the local HTTP API.

**Why.** Ollama provides an easy Windows install, stable API, and broad GGUF support; ideal for 7–8B Q4/Q6 on 16 GB VRAM.

**Inputs.** Ollama installed (`ollama serve`), model pulled (e.g., an 8B Instruct in Q4_K_M).

**Steps.**
1. New module `backends\local\ollama_client.py`:
   ```python
   import requests, uuid, time

   class OllamaClient:
       def __init__(self, base_url: str = "http://127.0.0.1:11434", model: str = ""):
           self.base = base_url.rstrip("/")
           self.model = model

       def generate(self, *, messages=None, prompt=None, model=None, params=None):
           body = {
               "model": model or self.model,
               "stream": False,
           }
           if messages: body["messages"] = messages
           if prompt: body["prompt"] = prompt
           if params: body.update(params or {})
           t0 = time.time()
           r = requests.post(f"{self.base}/api/chat" if messages else f"{self.base}/api/generate", json=body, timeout=600)
           r.raise_for_status()
           j = r.json()
           text = (j.get("message", {}) or {}).get("content") or j.get("response", "")
           usage = j.get("eval_count")
           return {
               "text": text,
               "finish_reason": "stop" if not j.get("done_reason") else j["done_reason"],
               "usage": {"completion_tokens": usage} if usage is not None else None,
               "request_id": str(uuid.uuid4()),
               "latency_s": time.time() - t0,
           }
   ```
2. Support both `messages=[{role:"system"|"user"|"assistant",content:"..."}]` and raw `prompt` modes. Ensure the **system prompt** is injected faithfully for this experiment (see Ticket 6 prompt adapter).

3. Parameter mapping (read from config): `temperature`, `top_p`, `top_k`, `num_predict` (maps to `max_new_tokens`), `stop` (if defined).

4. Add a small health check (`/api/tags`) and model presence check; instruct users to `ollama pull <model>` if missing.

**Acceptance Criteria.**
- `python - <<<'from backends.local.ollama_client import OllamaClient as C; print(C().generate(prompt="Hello", model="<your-model>")["text"][:10])'` prints text.
- When used via Ticket 4, batch generation completes for a 20‑item slice without errors.

**Risks/Mitigations.**
- _Risk:_ Ollama not installed/running → _Mitigate:_ health check with actionable error message.
- _Risk:_ Model mismatch → _Mitigate:_ config validation that the model id exists locally.

**Validation.**
- Compatible with system prompts; supports Q4/Q6 quantizations suitable for 16 GB VRAM.

---

## Ticket 3 — Local Engine (Alternative): llama.cpp (llama‑cpp‑python)

**Objective.** Provide an in‑process client using `llama-cpp-python` with CUDA/cuBLAS wheels.

**Why.** Offers lower overhead and more control; useful if avoiding an HTTP daemon or using custom GPU offload strategies.

**Inputs.** `pip install llama-cpp-python==*` with CUDA wheels; GGUF model path on disk.

**Steps.**
1. Module `backends\local\llama_cpp_client.py`:
   ```python
   from llama_cpp import Llama
   import uuid, time

   class LlamaCppClient:
       def __init__(self, model_path: str, n_ctx: int = 4096, n_gpu_layers: int = -1, n_threads: int = 8):
           self.llm = Llama(model_path=model_path, n_ctx=n_ctx, n_gpu_layers=n_gpu_layers, verbose=False)
           self.n_threads = n_threads

       def generate(self, *, messages=None, prompt=None, model=None, params=None):
           args = {"temperature": params.get("temperature", 0.0), "top_p": params.get("top_p", 1.0),
                   "max_tokens": params.get("max_new_tokens", 1024)}
           t0 = time.time()
           if messages:
               out = self.llm.create_chat_completion(messages=messages, **args)
               text = out["choices"][0]["message"]["content"]
               finish = out["choices"][0].get("finish_reason", "stop")
               usage = out.get("usage")
           else:
               out = self.llm(prompt, **args)
               text = out["choices"][0]["text"]
               finish = out["choices"][0].get("finish_reason", "stop")
               usage = out.get("usage")
           return {
               "text": text, "finish_reason": finish,
               "usage": usage, "request_id": str(uuid.uuid4()), "latency_s": time.time() - t0
           }
   ```
2. Expose CUDA offload settings: `n_gpu_layers=-1` to push all layers if VRAM permits; otherwise allow `gpu_split` list for fine control.
3. Add a quick model-size sanity check: warn if 13B at Q6 on 16 GB (may be tight).

**Acceptance Criteria.**
- A small prompt returns text with no exceptions; VRAM usage visible via `nvidia-smi`.
- Swap between Ollama and llama.cpp by changing config only.

**Risks/Mitigations.**
- _Risk:_ CUDA wheel mismatch → _Mitigate:_ document exact wheel versions in `docs/windows.md`.
- _Risk:_ OOM at long contexts → _Mitigate:_ start at 4k context, concurrency 1; instruct reducing context if OOM.

**Validation.**
- Provides a daemon‑less option; good for offline/air‑gapped runs.

---

## Ticket 4 — Local Batch Executor (reads JSONL → writes combined outputs)

**Objective.** Replace “upload/submit/poll/download” with a local **work queue** that consumes per‑part JSONL inputs and writes a combined JSONL of results per part (same locations the parser expects).

**Why.** Preserve the **contract** with downstream parsing/scoring while running inference locally. fileciteturn0file0

**Inputs.** `data\batch_inputs\*.jsonl` parts from `scripts.build_batches`.

**Steps.**
1. `backends\local\local_batch.py`:
   - Read one `...pXX.jsonl` at a time.
   - Create a bounded **async** or thread pool queue with `max_concurrent_requests` from config.
   - For each item, render the prompt/messages via the **Prompt Adapter** (Ticket 6), call client, collect `{custom_id, response_text, finish_reason, request_id, usage?, latency_s}`.
   - Write a per‑part `results\raw\<trial>\<part>.jsonl` (or mirror cloud layout used by the existing parser).
2. Add idempotent resume: if output for a part exists and validates (same number of lines as input), skip.
3. Record a lightweight per‑part state file with counts and timings for debugging.

**Acceptance Criteria.**
- For a tiny run (e.g., 50 items), outputs are produced per part and combined without errors.
- On rerun, parts already completed are skipped.

**Risks/Mitigations.**
- _Risk:_ Uncaught exceptions stall queue → _Mitigate:_ wrap task with retries, mark failures, continue; write an errors file.
- _Risk:_ Over-concurrency → _Mitigate:_ hard cap at 2 on 16 GB by default.

**Validation.**
- Mirrors “submit/poll/download” semantics locally; downstream parser sees the same structure. fileciteturn0file0

---

## Ticket 5 — Parser: Local Outputs → `predictions.csv` (Schema Parity)

**Objective.** Convert local combined JSONL to the **exact** `predictions.csv` schema consumed by scoring/stats.

**Why.** Avoid any change to scoring/statistics and reports. fileciteturn0file0

**Inputs.** Outputs from Ticket 4; existing Fireworks parser reference.

**Steps.**
1. Implement `backends\local\parse_results.py` with the same column set:
   - `custom_id, dataset, item_id, condition, temp, sample_index, type, request_id, finish_reason, response_text, prompt_tokens, completion_tokens, total_tokens`.
2. Pull `custom_id` and metadata from the **input JSONL** record echoed in outputs, or embed the echo in Ticket 4 to guarantee availability.
3. Token accounting: if the engine returns usage, map it; otherwise leave blank (or fill via Ticket 8).
4. Validate row count equals unique `custom_id`s; log any duplicates or missing IDs.

**Acceptance Criteria.**
- Running only `parse_results.py` on local JSONL yields a `predictions.csv` identical (columns and row count) to the Fireworks path for the same inputs.
- `scoring.score_predictions` then runs and writes `per_item_scores.csv` with no code changes.

**Risks/Mitigations.**
- _Risk:_ Metadata loss from JSONL → _Mitigate:_ echo input fields verbatim into each output line in Ticket 4.

**Validation.**
- Guarantees downstream compatibility; keeps the scientific comparison valid. fileciteturn0file0

---

## Ticket 6 — Config Schema & Prompt Adapter

**Objective.** Extend config to support local backend settings and centralize **system prompt** rendering for A/B experiments.

**Why.** The experiment’s treatment/control hinge on the **system prompt**; this must be injected identically across backends. fileciteturn0file0

**Inputs.** `config\schema.py`, `config\eval_config.local.yaml` (new), prompt files under `config\prompts\`.

**Steps.**
1. Extend Pydantic schema with:
   ```yaml
   backend: local | fireworks
   local_engine: ollama | llama_cpp
   local_endpoint: "http://127.0.0.1:11434"   # for Ollama; null for llama.cpp
   local_model: "<ollama-tag or gguf-path>"
   max_concurrent_requests: 1
   tokenizer: "llama" | "hf:<repo>"           # optional for token accounting
   ```
2. Create `scripts\prompt_adapter.py`:
   - Inputs: prepared item, condition (`control|treatment`), prompt set, task type (`open_book|closed_book`), temps, max_new_tokens.
   - Output: either `messages=[{role:"system", content:<system>}, {role:"user", content:<task>} ]` (preferred) or a single `prompt` string.
   - For **open-book**, append the provided **context** to the user message. For closed-book, include only question text.
3. Ensure prompt tokens are length-audited by a utility (`scripts\audit_prompts.py`) (optional).

**Acceptance Criteria.**
- Rendering a sample item in both conditions shows only the system prompt differs, as expected; user content remains identical except for context presence (open-book).

**Risks/Mitigations.**
- _Risk:_ Prompt leakage between conditions → _Mitigate:_ unit test that control and treatment system prompts hash differently while user messages hash identically per item.

**Validation.**
- Protects experiment integrity across engines and keeps config single‑source‑of‑truth. fileciteturn0file0

---

## Ticket 7 — Windows Pathing & Multiprocessing Fixes

**Objective.** Ensure all scripts run on Windows: path separators, spawn semantics, and thread/process pools.

**Why.** Windows uses `spawn`; processes must be guarded; paths must be `pathlib`‑safe. fileciteturn0file0

**Inputs.** `scripts\*.py`, any code using `multiprocessing`/`subprocess`/file paths.

**Steps.**
1. Replace `os.path` with `pathlib.Path` throughout touched modules.
2. Guard all script entry points with:
   ```python
   if __name__ == "__main__":
       import multiprocessing as mp
       mp.freeze_support()
       main()
   ```
3. Prefer `ThreadPoolExecutor` or `asyncio` for local HTTP concurrency; keep process pools small for CPU-bound scoring only.
4. Add Windows-friendly temp file handling (no exclusive locks; ensure newline universal mode `newline=""` for CSV).

**Acceptance Criteria.**
- Orchestrator runs on Windows start‑to‑finish for a 50‑item trial with backend=local.

**Risks/Mitigations.**
- _Risk:_ Deadlocks from nested pools → _Mitigate:_ keep nesting shallow; document best‑practice in `docs/windows.md`.

**Validation.**
- Stability on Windows without altering Linux/macOS behavior.

---

## Ticket 8 — Token Accounting & Optional Local Cost Telemetry

**Objective.** Capture token counts when available; otherwise approximate; optionally record latency and GPU/VRAM telemetry.

**Why.** The framework tracks cost/usage; even if “$ cost” is irrelevant locally, tokens and latency improve analysis. fileciteturn0file0

**Inputs.** Engine responses; NVML (optional).

**Steps.**
1. If engine reports usage, pass through to parser (Ticket 5).
2. If not, add `scripts\estimate_tokens.py`:
   - For Llama-family, use llama.cpp tokenizer or HF tokenizer (`tokenizers`/`transformers`) for approximation.
   - Count prompt vs completion separately using rendered messages.
3. Optional: Add `telemetry\nvml.py` using `pynvml` to sample `gpu_util`, `mem_used`, `power_draw` every 250ms during generation (feature‑flagged).
4. Persist `local_costs.json` per trial with summary stats (avg latency, p50/p95, avg mem). Keep separate from core results.

**Acceptance Criteria.**
- `predictions.csv` includes token columns when possible; otherwise blank.
- A telemetry JSON is written when the flag is on; the eval completes without telemetry too.

**Risks/Mitigations.**
- _Risk:_ Tokenizer mismatch → _Mitigate:_ document it as **estimate** and keep separate from significance metrics.
- _Risk:_ NVML unavailable → _Mitigate:_ optional feature with graceful fallback.

**Validation.**
- Adds observability without changing core significance results. fileciteturn0file0

---

## Ticket 9 — Windows Run Recipes & Smoke Tests

**Objective.** Provide one‑liners to validate orchestration locally before full runs.

**Why.** Quick feedback, mirrors README flow. fileciteturn0file0

**Inputs.** `tools\tasks.ps1`, new `config\eval_config.local.yaml` (Ticket 10).

**Steps.**
1. In `tools\tasks.ps1`, implement:
   - `Invoke-Plan`, `Invoke-Smoke`, `Invoke-Data`, `Invoke-Build`, `Invoke-Eval`, `Invoke-Parse`, `Invoke-Score`, `Invoke-Stats`, `Invoke-Report`.
2. Smoke flow:
   ```powershell
   # plan only
   python -m scripts.run_all --config config\eval_config.local.yaml --plan_only

   # tiny offline slice
   python -m scripts.run_all --config config\eval_config.local.yaml --dry_run --prompt_sets operational_only --temps 0.0 --limit_items 200 --parts_per_dataset 3 --max_concurrent_jobs 2

   # end-to-end small run (backend=local)
   python -m scripts.run_all --config config\eval_config.local.yaml --archive --limit_items 200 --parts_per_dataset 3 --max_concurrent_jobs 1
   ```

**Acceptance Criteria.**
- All three commands complete; final command produces `predictions.csv`, `per_item_scores.csv`, `significance.json`, and a report under `results/` or `experiments/…` (depending on config).

**Risks/Mitigations.**
- _Risk:_ Large artifacts slow on first run → _Mitigate:_ keep `--limit_items` small initially.

**Validation.**
- End-to-end behavior matches the documented phases. fileciteturn0file0

---

## Ticket 10 — Local Example Configs

**Objective.** Provide turnkey configs for Ollama and llama.cpp.

**Why.** Reduce setup time and avoid mistakes.

**Inputs.** `config\eval_config.local.yaml` (Ollama), `config\eval_config.local.llamacpp.yaml`.

**Steps.**
1. Create `config\eval_config.local.yaml`:
   ```yaml
   backend: local
   local_engine: ollama
   local_endpoint: "http://127.0.0.1:11434"
   local_model: "llama3.1:8b-instruct-q4_K_M"
   temps: [0.0]
   samples_per_item: {"0.0": 1}
   max_new_tokens: {closed_book: 1024, open_book: 1024}
   max_concurrent_requests: 1
   prompt_sets:
     default:
       control: config/prompts/control_system.txt
       treatment: config/prompts/treatment_system.txt
   default_prompt_set: default
   ```
2. Create `config\eval_config.local.llamacpp.yaml` with `local_engine: llama_cpp` and `local_model: "C:\\models\\llama3.1-8b-instruct-q4_k_m.gguf"`, plus `n_ctx`/`n_gpu_layers` fields as needed (the client can read them).

**Acceptance Criteria.**
- Both configs validate via `python -c "from config.schema import load_config; load_config('config/eval_config.local.yaml')"` (adapt to actual loader).

**Risks/Mitigations.**
- _Risk:_ Model tag/path typos → _Mitigate:_ validation routine checks connectivity/path exists.

**Validation.**
- Minimizes user error; reproducible runs. fileciteturn0file0

---

## Ticket 11 — Documentation: `docs/windows.md` + README Edits

**Objective.** Document Windows setup, local engines, run recipes, and troubleshooting; update README with a “Windows + Local” quick‑start pointer.

**Why.** Developer discoverability. fileciteturn0file0

**Inputs.** Prior tickets content.

**Steps.**
1. Write `docs\windows.md` covering: prerequisites; bootstrap; Ollama vs llama.cpp; configs; commands; telemetry; OOM remediation; antivirus exclusions; known issues.
2. Add a “Windows + Local” section to README (brief) linking to `docs/windows.md`.

**Acceptance Criteria.**
- A new Windows contributor can complete a small run without back‑and‑forth.

**Risks/Mitigations.**
- _Risk:_ Docs drift → _Mitigate:_ include commands that are tested in CI (where possible).

**Validation.**
- Aligns with existing quick‑start and phase naming. fileciteturn0file0

---

## Ticket 12 — Performance Tuning & VRAM Budget Guide

**Objective.** Provide clear, hardware‑aware tuning presets and OOM/latency playbook.

**Why.** 16 GB VRAM requires sensible defaults; KV cache grows with context; concurrency must be managed.

**Inputs.** Empirical tests from Tickets 2–4.

**Steps.**
1. Add `docs\performance.md`:
   - **Presets** for RTX 5080 (16 GB):  
     - _Throughput:_ 8B Q4_K_M, context 4k–8k, `max_new_tokens` 512–1024, concurrency 1→2.  
     - _Quality:_ 8B Q6_K, context 4k, `max_new_tokens` 512–1024, concurrency 1.  
     - _Stretch:_ 13B Q4_K_M, context 4k, `max_new_tokens` 512, concurrency 1.
   - **Symptoms → Fixes**: OOM (reduce context/concurrency/quantization), slow tokens (lower `top_k`/increase `temperature` modestly, or switch engine), frequent `length` finishes (increase `max_new_tokens` or refine stop sequences).
   - **Telemetry reading**: how to interpret NVML samples (utilization, memory, power).

**Acceptance Criteria.**
- Users can resolve OOM or poor throughput by following the guide; add quick decision tree.

**Risks/Mitigations.**
- _Risk:_ Model‑specific quirks → _Mitigate:_ note variability and advise measuring with `nvidia-smi`.

**Validation.**
- Guidance matches hardware constraints; safe defaults first.

---

## Ticket 13 — CI Smoke (Optional, if Windows runner available)

**Objective.** Add a small GitHub Actions workflow that lint‑checks and runs a **dry‑run** orchestration smoke on Windows.

**Why.** Prevent regressions in Windows pathing.

**Inputs.** GitHub Actions, small cache budgets.

**Steps.**
1. `.github\workflows\windows-smoke.yml`: setup Python; cache pip; run `tools\bootstrap.ps1`; execute `python -m scripts.run_all --config config\eval_config.local.yaml --dry_run --limit_items 10` (no model download).
2. Skip Ollama/llama.cpp installation to keep the job fast; ensure it only tests orchestration — not inference.

**Acceptance Criteria.**
- Workflow passes and catches syntax/path errors on PRs.

**Risks/Mitigations.**
- _Risk:_ Lack of Windows runner → _Mitigate:_ treat as optional; keep local scripts robust.

**Validation.**
- Guards against Windows-only drift.

---

## Ticket 14 — PR Plan & Change Isolation

**Objective.** Stage changes to minimize risk.

**Steps.**
- **PR 1:** Backend interface + Fireworks move + config flags + Windows bootstrap/tasks.
- **PR 2:** Local engine clients (Ollama + llama.cpp stubs) + Prompt Adapter.
- **PR 3:** Local Batch Executor + Parser + token accounting.
- **PR 4:** Docs (`windows.md`, performance), example configs, telemetry (optional).
- **PR 5 (opt):** CI smoke on Windows.

**Acceptance Criteria.**
- Each PR is shippable and independently testable; no PR breaks Fireworks path by default.

---

## Ticket 15 — Troubleshooting Playbook

**Objective.** Provide fast answers for common failures.

**Playbook.**
- **Symptom:** `Connection refused` to Ollama → **Fix:** `ollama serve`; verify `Invoke-WebRequest http://127.0.0.1:11434/api/tags`.
- **Symptom:** OOM during generation → **Fix:** reduce context; lower to Q4_K_M; set `max_concurrent_requests=1`; close other GPU apps.
- **Symptom:** `finish_reason=length` spikes → **Fix:** raise `max_new_tokens`; add `stop` sequences if needed.
- **Symptom:** Parser row mismatch → **Fix:** ensure outputs echo `custom_id` and metadata; re-run Ticket 4 validation.
- **Symptom:** Slow scoring → **Fix:** set `set MKL_NUM_THREADS=8`, `set OMP_NUM_THREADS=8`, `set NUMEXPR_MAX_THREADS=8`; avoid running inference simultaneously.

**Acceptance Criteria.**
- Issues resolved within one edit based on guidance.

---

## Appendix A — Command Recipes (Copy/Paste)

**Install & activate**
```powershell
powershell -ExecutionPolicy Bypass -File tools\bootstrap.ps1
.\.venv\Scripts\Activate.ps1
```

**Run Ollama (separate terminal)**
```powershell
ollama serve
ollama pull llama3.1:8b-instruct-q4_K_M
```

**Plan → tiny dry-run → small end-to-end**
```powershell
python -m scripts.run_all --config config\eval_config.local.yaml --plan_only

python -m scripts.run_all --config config\eval_config.local.yaml --dry_run `
  --prompt_sets operational_only --temps 0.0 --limit_items 200 `
  --parts_per_dataset 3 --max_concurrent_jobs 2

python -m scripts.run_all --config config\eval_config.local.yaml --archive `
  --limit_items 200 --parts_per_dataset 3 --max_concurrent_jobs 1
```

**Manual: parse → score → stats → report (useful for debugging)**
```powershell
python -m backends.local.parse_results --results_jsonl results\raw\combined.jsonl --out_csv results\predictions.csv
python -m scoring.score_predictions --pred_csv results\predictions.csv --prepared_dir data\prepared --out_dir results
python -m scoring.stats --per_item_csv results\per_item_scores.csv --config config\eval_config.local.yaml --out_path results\significance.json
python -m scripts.generate_report --results_dir results
```

---

## Definition of Done (Project)

- Runs on Windows end‑to‑end with **backend=local** (Ollama or llama.cpp) and produces **identical schemas** for `predictions.csv` and downstream artifacts.
- README contains a **Windows + Local quick‑start** with link to `docs/windows.md`.
- Example configs present and validated.
- Fireworks path still works unchanged when selected.
- Basic performance guidance and troubleshooting included.

---

## Notes on Scientific Integrity

- Keep dataset preparation, scoring, statistics, and report generation unchanged; we are modifying **only the inference substrate** and orchestration glue needed for local execution. This preserves causal comparisons of system prompts and the interpretation of hallucination/accuracy metrics as documented. fileciteturn0file0

