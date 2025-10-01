# Windows + Local LLM Port — Completion & Remediation Guide

**Audience.** Codex agents or contributors resuming the Windows + local LLM migration. This playbook captures the current repository state (as of commit `9cda383`) and provides explicit instructions to close remaining gaps so the pipeline runs reliably on Windows with Ollama or llama.cpp backends.

**Scope.** Focus on finishing the migration plan documented in `docs/planning/PORT_WINDOWS_LOCAL_LLM_ACTION_PLAN.md` and resolving issues identified during the latest assessment (see `codex/logs/124.md` and the readiness review).

---

## 1. Repository Snapshot

- **Branch parity:** `main` and `windows-and-local-llm-migration` both point at `9cda383` (“Handle local dry-run and align Windows tasks”).
- **Core deliverables present:**
  - Backend abstraction (`backends/interfaces.py`) and Fireworks shims (`backends/fireworks/`).
  - Local engines: Ollama HTTP client (`backends/local/ollama_client.py`) and llama.cpp client (`backends/local/llama_cpp_client.py`).
  - Local batch executor (`backends/local/local_batch.py`) and parser (`backends/local/parse_results.py`).
  - Windows bootstrap (`tools/bootstrap.ps1`), task runner (`tools/tasks.ps1`), Windows docs (`docs/windows.md`), troubleshooting (`docs/troubleshooting_windows_local.md`), and performance guide (`docs/performance.md`).
  - Example configs: `config/eval_config.local.yaml`, `config/eval_config.local.llamacpp.yaml`.

- **Planning artifacts:**
  - Action plan (`docs/planning/PORT_WINDOWS_LOCAL_LLM_ACTION_PLAN.md`).
  - PR plan (`codex/pr-plan/PR_SEQUENCE.md`, `PR_TEMPLATES.md`, `VALIDATION_CHECKLIST.md`).
  - Ticket logs (`codex/logs/110-116,122,124,125.md`).

---

## 2. Done vs Pending (Ticket Mapping)

| Ticket | Status | Evidence / File(s) |
| --- | --- | --- |
| 0 – Windows bootstrap | ✅ | `tools/bootstrap.ps1`, `docs/windows.md`, log 110 |
| 1 – Backend interface | ✅ | `backends/interfaces.py`, log 111 |
| 2 – Ollama client | ✅ | `backends/local/ollama_client.py`, log 112 |
| 3 – llama.cpp client | ✅ | `backends/local/llama_cpp_client.py`, log 113 |
| 4 – Local batch executor | ✅ (core) | `backends/local/local_batch.py`, log 114 |
| 5 – Local parser | ✅ | `backends/local/parse_results.py`, log 115 |
| 6 – Config schema & prompt adapter | ✅ | `config/schema.py`, `scripts/prompt_adapter.py`, log 116 |
| 7 – Windows pathing & multiprocessing | ⚠️ Partial | `scripts/run_all.py`, `scripts/state_utils.py` (missing `freeze_support`) |
| 8 – Token accounting & telemetry | ❌ Pending | No `scripts/estimate_tokens.py`/`telemetry/` implementations |
| 9 – Windows run recipes & smoke tests | ✅ | `tools/tasks.ps1`, `docs/windows.md`
| 10 – Example configs | ✅ | `config/eval_config.local*.yaml`
| 11 – Documentation & README edits | ✅ | `docs/windows.md`, `README.md`
| 12 – Performance guide | ✅ | `docs/performance.md`
| 13 – Windows CI smoke | ❌ Pending | No workflow under `.github/workflows/`
| 14 – PR plan | ✅ | `codex/pr-plan/`
| 15 – Troubleshooting playbook | ✅ | `docs/troubleshooting_windows_local.md`

---

## 3. Outstanding Issues & Risks

1. **Manifest job statuses not updated for local runs**  
   - Evidence: `scripts/run_all.py:1207-1239` initializes `trial_manifest["job_status"]`, but the completion hook (`queue.progress_cb`) only runs on Fireworks jobs (`scripts/run_all.py:1387-1419`). Local runs exit `queue.run_queue` without updating status, leaving entries stuck at `"pending"`.
   - Risk: Resume logic and diagnostics misinterpret local runs as incomplete.

2. **llama.cpp concurrency safety**  
   - Local queue clamps `max_concurrent_requests` to `min(requested, 2)` (`backends/local/local_batch.py:172-174`).  
   - `LlamaCppClient` reuses a single llama.cpp instance (`backends/local/llama_cpp_client.py:69-139`) which is not thread-safe; running more than one worker can segfault or corrupt outputs.
   - Risk: Non-deterministic crashes or empty generations when concurrency > 1.

3. **Missing `multiprocessing.freeze_support()` guards**  
   - Plan requires guarding entry points for Windows spawn semantics (`docs/planning/PORT_WINDOWS_LOCAL_LLM_ACTION_PLAN.md:323-331`).  
   - Current CLIs (e.g., `scripts/run_all.py:1960`, `scripts/build_batches.py:1-210`) lack the guard.  
   - Risk: Running CLI scripts in frozen environments (PyInstaller) or certain PowerShell contexts can hang or crash on Windows.

4. **Token accounting & telemetry (Ticket 8) unimplemented**  
   - No helpers for estimating tokens or GPU telemetry.  
   - Risk: Local runs generate blank cost metrics, reducing parity with Fireworks path.

5. **Windows CI smoke (Ticket 13) missing**  
   - No `.github/workflows/windows-smoke.yml`.  
   - Risk: Regressions in Windows-specific code go unnoticed.

6. **Documentation gaps**  
   - `docs/windows.md` references CUDA/cuBLAS wheels but lacks install commands and troubleshooting specifically for llama.cpp wheels on Windows.  
   - Risk: Contributors may fail to set up GPU-accelerated llama.cpp.

7. **Operational validation incomplete**  
   - No recorded evidence of `python -m scripts.run_all --config config\eval_config.local.yaml ...` full run on Windows hardware.

---

## 4. Remediation Tasks (Step-by-Step)

### 4.1 Fix Manifest Job Status for Local Runs

**Goal:** Ensure `trial_manifest.json` reflects completed parts when using the local backend.

**Files:**
- `scripts/run_all.py`
- `backends/local/local_batch.py`

**Implementation Steps:**
1. Extend `LocalQueueManager.run_queue` to emit progress events compatible with the `_progress_cb` used for Fireworks jobs. Options:
   - Add a callback interface: if `self.progress_cb` is set, call it with `{"job_key": job_key, "event": "completed"}` after each part finishes (`backends/local/local_batch.py:198-312`).
   - Alternatively, return a list of completed `job_key`s and update manifest in `run_all` right after `queue.run_queue(results_dir)`.
2. In `scripts/run_all.py`, when `is_local_backend` is true:
   - Before queue execution, set `queue.progress_cb = _local_progress_cb` (akin to Fireworks path) that updates `trial_manifest["job_status"][jkey] = "completed"` and persists via `mf.write_manifest(...)`.
   - After `queue.run_queue`, verify counts by comparing number of input rows vs output rows and log discrepancies.
3. Re-run the dry-run branch of the local path (`args.dry_run`) to ensure it mirrors the real-path logic.

**Validation:**
- Execute a small local run (`--limit_items 20 --parts_per_dataset 2`).  
- Open `experiments/run_<ID>/<slug>/results/trial_manifest.json` and confirm `job_status` values are `"completed"` with correct part entries.

### 4.2 Enforce Safe Concurrency for llama.cpp

**Goal:** Prevent multi-threaded access to a single llama.cpp instance.

**Files:**
- `backends/local/local_batch.py`
- `backends/local/llama_cpp_client.py`
- Documentation (`docs/windows.md`, `docs/performance.md`)

**Implementation Steps:**
1. In `_build_client`, if `engine == "llama_cpp"`, log a warning that concurrency is forced to 1 unless multiple model instances are spun up.  
   - Example: `if engine == "llama_cpp" and max_concurrent > 1: max_concurrent = 1; print("For llama_cpp, max_concurrent_requests forced to 1 to avoid thread-safety issues")`.
2. In `LocalQueueManager.__init__`, branch on engine type: store `self.local_engine = config.get("local_engine")`. When computing `self.max_concurrent`, cap llama.cpp to 1 explicitly.
3. Update docs (`docs/windows.md`, `docs/performance.md`) to explain concurrency constraints per engine.
4. Optional enhancement: instantiate multiple `LlamaCppClient` objects (one per worker) when concurrency > 1 is requested, but only if you handle model loading overhead. Document the trade-offs if implementing this.

**Validation:**
- Run a local llama.cpp job with `max_concurrent_requests: 2`. Confirm warning is printed and only one worker is used.
- Observe stable outputs with repeated runs.

### 4.3 Add `freeze_support()` Guard to CLI Entry Points (Ticket 7 Completion)

**Goal:** Align with Windows spawn requirements for all CLI scripts.

**Files:**
- Every `scripts/*.py` entry point (at minimum: `run_all.py`, `build_batches.py`, `prepare_data.py`, `generate_report.py`, `smoke_orchestration.py`, `smoke_test.py`, `resume_run.py`, `unsupported_sensitivity.py`, etc.).

**Implementation Steps:**
1. For each script with `if __name__ == "__main__": main()`, change to:
   ```python
   if __name__ == "__main__":
       import multiprocessing as mp
       mp.freeze_support()
       main()
   ```
2. Maintain import-local pattern to avoid reordering top-level imports unnecessarily.
3. Run `python -m scripts.SMALL_SCRIPT --help` on Windows to ensure there are no regressions.

**Validation:**
- On a Windows machine (or via WSL+PowerShell), invoke `python -m scripts.run_all --help` and other CLI commands; no regressions expected on POSIX systems.

### 4.4 Implement Token Accounting & Optional Telemetry (Ticket 8)

**Goal:** Provide token usage estimates and optional GPU telemetry for local runs, keeping parity with Fireworks outputs.

**Files (proposed):**
- `scripts/estimate_tokens.py`
- `telemetry/nvml.py` (optional)
- `backends/local/local_batch.py`
- `docs/performance.md`, `docs/windows.md`

**Implementation Steps:**
1. **Token estimation:**
   - Create `scripts/estimate_tokens.py` exposing `estimate_tokens(messages: list[dict], tokenizer_hint: str | None) -> dict`.
   - Use `tiktoken` or llama.cpp tokenizer bridging when possible. Fallback to counting characters if no tokenizer available.
   - In `LocalQueueManager._run_one_part`, when engine response lacks `usage`, call `estimate_tokens` on the request payload and populate `usage` in the response object.
   - Expose tokenizer hints via `config.schema` (`tokenizer` field already exists).
2. **Telemetry (optional):**
   - Add `telemetry/nvml.py` with a context manager that samples NVIDIA stats using `pynvml` if available.  
   - Wrap local generation calls to record `latency_ms`, `gpu_mem_mb`, `gpu_utilization` (if desired) and write per-part `state.json` enhancements or a new `local_costs.json` artifact.
3. Update docs and README to explain how to enable/disable telemetry and note prerequisites (NVML, CUDA drivers).

**Validation:**
- Run local jobs; inspect `results/predictions.csv` for populated `prompt_tokens`, `completion_tokens`, `total_tokens`.
- Confirm telemetry artifacts (if enabled) are written without crashing when NVML is absent (graceful degrade).

### 4.5 Add Windows CI Smoke Workflow (Ticket 13)

**Goal:** Ensure PRs trigger a Windows dry-run to catch path regressions.

**Files:**
- `.github/workflows/windows-smoke.yml`

**Implementation Steps:**
1. Create `.github/workflows/windows-smoke.yml` with the following outline:
   ```yaml
   name: Windows Smoke

   on:
     pull_request:
       paths:
         - '**.py'
         - 'tools/**'
         - 'config/**'
         - 'docs/**'

   jobs:
     smoke:
       runs-on: windows-latest
       steps:
         - uses: actions/checkout@v4
         - name: Set up Python
           uses: actions/setup-python@v5
           with:
             python-version: '3.11'
         - name: Install repo requirements
           shell: pwsh
           run: |
             powershell -ExecutionPolicy Bypass -File tools\bootstrap.ps1
         - name: Dry-run orchestration
           shell: pwsh
           run: |
             .\.venv\Scripts\Activate.ps1
             python -m scripts.run_all --config config\eval_config.local.yaml --dry_run --limit_items 10 --parts_per_dataset 2
   ```
2. Ensure the command avoids real model downloads (`--dry_run`).
3. Optionally, add caching (`actions/cache`) for pip to improve runtime.

**Validation:**
- Push a test branch to trigger the workflow.
- Confirm the job passes; adjust timeouts if necessary (<10 minutes target).

### 4.6 Expand Documentation for llama.cpp Wheels

**Goal:** Provide actionable setup steps for Windows users installing GPU-enabled llama.cpp.

**Files:**
- `docs/windows.md`
- `docs/performance.md`
- `docs/troubleshooting_windows_local.md`

**Implementation Steps:**
1. Add a “Installing llama-cpp-python on Windows” section covering:
   - Using prebuilt CUDA wheels: `pip install llama-cpp-python==<version> --extra-index-url https://jllllll.github.io/llama-cpp-python-cuBLAS-wheels/whl/cu121`
   - Verifying installation: `python -c "from llama_cpp import Llama; print('OK')"`
   - Common errors and remedies (e.g., CUDA version mismatch).
2. Update troubleshooting guide with wheel-related errors (e.g., `DLL load failed` → reinstall with matching CUDA version).
3. Cross-link from README quick start.

**Validation:**
- Run through the documented commands on a Windows VM or machine to ensure they work.

### 4.7 Final Operational Validation

**Goal:** Demonstrate end-to-end success on Windows with both local backends.

**Steps:**
1. **Ollama path:**
   ```powershell
   powershell -ExecutionPolicy Bypass -File tools\bootstrap.ps1
   .\.venv\Scripts\Activate.ps1
   ollama serve
   ollama pull llama3.1:8b-instruct-q4_K_M
   python -m scripts.run_all --config config\eval_config.local.yaml --archive --limit_items 200 --parts_per_dataset 3
   ```
   - Record duration, confirm `predictions.csv`, `per_item_scores.csv`, `significance.json`, and `reports/report.md` exist.
2. **llama.cpp path:**
   ```powershell
   powershell -ExecutionPolicy Bypass -File tools\bootstrap.ps1
   .\.venv\Scripts\Activate.ps1
   pip install llama-cpp-python --extra-index-url <CUDA-wheel-url>
   python -m scripts.run_all --config config\eval_config.local.llamacpp.yaml --archive --limit_items 100 --parts_per_dataset 2
   ```
   - Verify outputs and check GPU usage.
3. Run `python -m scripts.smoke_orchestration --n 3 --prompt_set operational_only --dry_run` to ensure the smoke harness remains functional.
4. Capture logs or summarize in `codex/logs/<new ticket>.md` for traceability.

---

## 5. Suggested Work Sequencing

1. **Foundation fixes:** Manifest status + concurrency guard (Sections 4.1, 4.2).
2. **Windows compatibility:** Add `freeze_support()` guards (Section 4.3).
3. **Feature parity:** Token estimation & optional telemetry (Section 4.4).
4. **Automation:** Add Windows CI workflow (Section 4.5).
5. **Docs refresh:** llama.cpp installation guidance (Section 4.6).
6. **Validation runs:** Execute end-to-end tests; log results (Section 4.7).

If time-constrained, prioritize sections 4.1–4.3 and 4.5 to eliminate functional blockers, then follow with telemetry/docs.

---

## 6. Reporting & Checkpointing

- After each major step, add a log entry under `codex/logs/` (e.g., `codex/logs/126.md`) capturing:
  - Summary of changes.
  - Files touched.
  - Validation commands & outcomes.
  - Follow-up work.
- Keep the PR plan updated if deviations arise (`codex/pr-plan/PR_SEQUENCE.md`).
- For each PR:
  - Attach `git diff` summary.
  - Include `tools/tasks.ps1 -Task Smoke` or `python -m scripts.run_all --plan_only` output as evidence.
  - Note Windows-specific validation results.

---

## 7. Quick Reference

| Topic | Command / Path |
| --- | --- |
| Validate local config | `python -c "from config.schema import load_config; load_config('config/eval_config.local.yaml')"` |
| Local dry run (Ollama) | `python -m scripts.run_all --config config\eval_config.local.yaml --dry_run --limit_items 20 --parts_per_dataset 2` |
| Local real run (Ollama) | `python -m scripts.run_all --config config\eval_config.local.yaml --archive --limit_items 200 --parts_per_dataset 3` |
| llama.cpp install (CUDA 12.1 example) | `pip install llama-cpp-python --extra-index-url https://jllllll.github.io/llama-cpp-python-cuBLAS-wheels/whl/cu121` |
| Windows smoke task | `powershell -ExecutionPolicy Bypass -File tools\tasks.ps1 -Task Smoke` |
| Manifest location | `experiments/run_<ID>/<trial-slug>/results/trial_manifest.json` |
| Combined results | `.../results/results_combined.jsonl` |

---

## 8. Exit Criteria

The Windows + local LLM port is considered complete when:

1. Local backend runs (Ollama & llama.cpp) produce full evaluation artifacts and update manifests accurately.
2. Concurrency is controlled safely; documentation reflects engine limitations.
3. CLI scripts comply with Windows multiprocessing requirements.
4. Token usage fields populate in `predictions.csv` for local runs; optional telemetry is either implemented or explicitly deferred.
5. Windows CI smoke workflow passes on PRs.
6. Documentation enables new contributors to bootstrap, run, and troubleshoot without external context.
7. Validation logs or run outputs confirm successful Windows runs using the example configs.

Once these criteria are met, update the README “Windows + Local” section with a brief status note (“Supported & validated”) and close the outstanding tickets.

---

*Prepared for hand-off: use this guide as a checklist and update it (or the planning docs) as work progresses.*
