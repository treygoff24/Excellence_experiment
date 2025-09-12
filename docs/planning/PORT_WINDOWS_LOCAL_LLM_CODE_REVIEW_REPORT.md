# Windows + Local LLM Port — Code Review Report

Date: 2025-09-12
Reviewer: Codex CLI Agent

## Checklist (what I did)
- Reviewed the full plan in `docs/planning/PORT_WINDOWS_LOCAL_LLM_ACTION_PLAN.md`.
- Scanned repository for all plan-related code paths, configs, and docs.
- Audited orchestrator integration and Windows compatibility hotspots.
- Verified local backends (Ollama, llama.cpp) and batch executor implementations.
- Identified issues, gaps, and suboptimal areas; drafted improvements and next steps.

## Executive Summary
The implementation is substantially progressed but not yet “ready for deployment and testing” on Windows with a local backend:
- Pluggable backend interfaces and local clients exist and look solid.
- Windows bootstrap and task runner scripts are present.
- Parsing/scoring/stats/report paths remain intact and schema-compatible.
- Critical gaps remain: the orchestrator (`scripts/run_all.py`) still rejects `backend=local`, and multiple modules import `fcntl` unconditionally, which breaks on Windows. Example local configs are missing, and the local pipeline lacks an integrated combine step for per‑part outputs.

Hard blockers to resolve before a Windows + Local end‑to‑end run:
- Add local backend dispatch in `scripts/run_all.py` (submit/poll/parse path).
- Remove or guard `fcntl` imports/usage for Windows compatibility.

## Findings by Ticket (Plan → Code Reality)

### Ticket 0 — Windows Bootstrap & Developer Ergonomics
- Status: Implemented.
  - `tools/bootstrap.ps1` and `tools/tasks.ps1` exist and are usable.
  - `docs/windows.md` present (introductory; minimal content).
- Improvements:
  - Expand `docs/windows.md` per plan (more detailed setup, examples, troubleshooting pointers, model pulls).

### Ticket 1 — Pluggable Inference Backend Interface
- Status: Implemented.
  - `backends/interfaces.py` defines `InferenceClient`/`BatchExecutor` protocols.
  - Fireworks code shims under `backends/fireworks/` keep the default path intact.

### Ticket 2 — Local Engine: Ollama (HTTP)
- Status: Implemented.
  - `backends/local/ollama_client.py` maps params (`temperature`, `top_p`, `top_k`, `max_new_tokens→num_predict`, `stop`, `seed`).
  - Helpful errors for connectivity and missing model (`ollama pull …`).
- Improvements:
  - Optional: expose a simple CLI health check command in docs.

### Ticket 3 — Local Engine: llama.cpp (in‑process)
- Status: Implemented.
  - `backends/local/llama_cpp_client.py` creates chat/raw completions; includes basic VRAM heuristics and error messaging for CUDA/cuBLAS wheel issues.
- Improvements:
  - Docs: detail recommended wheels and CUDA version notes for Windows.

### Ticket 4 — Local Batch Executor (JSONL → per‑part results)
- Status: Implemented (per‑part generation), integration incomplete.
  - `backends/local/local_batch.py` reads input parts, runs with bounded concurrency, echoes `custom_id`, and emits per‑part `results.jsonl`.
  - Resume logic and atomic writes present; good.
- Gaps:
  - No orchestration wiring: `scripts/run_all.py` never calls the local queue path.
  - No built‑in “combine” step to produce `results_combined.jsonl` expected by parsers/manifest.
- Improvements:
  - In `run_all`, when `backend=local`, use local `split_jsonl`/`create_queue` and after per‑part completion, write a combined `results_combined.jsonl` (mirroring Fireworks path).

### Ticket 5 — Parser: Local Outputs → predictions.csv
- Status: Implemented.
  - `backends/local/parse_results.py` outputs the exact schema expected downstream.
  - Note: `fireworks/parse_results.py` is also compatible with the local result shape, so either can be used once a combined JSONL exists.

### Ticket 6 — Config Schema & Prompt Adapter
- Status: Partially implemented.
  - `config/schema.py` includes `backend`, `local_engine`, `local_endpoint`, `local_model`, `max_concurrent_requests`, `tokenizer`.
  - Dedicated `prompt_adapter.py` not present; however, `scripts/build_batches.py` already injects the system prompt and produces message‑based inputs, which is sufficient for schema parity.
- Improvements:
  - Either add the adapter as planned (for centralization/testing) or document that `build_batches.py` is the single source of truth for prompt rendering.

### Ticket 7 — Windows Pathing & Multiprocessing Fixes
- Status: Incomplete (Windows blockers present).
  - `scripts/run_all.py` imports `fcntl` at module import time. Windows lacks `fcntl`, causing immediate ImportError.
  - `scripts/state_utils.py` also imports and uses `fcntl` unguarded.
- Fixes (suggested):
  - Guard `fcntl` imports and calls (try/except); on Windows, skip locking or use a cross‑platform file lock (e.g., `msvcrt` or `portalocker`).
  - Keep `newline=""` for CSV (already present) and prefer `pathlib.Path` in new code where feasible.

### Ticket 8 — Token Accounting & Optional Telemetry
- Status: Not implemented (optional per plan).
  - No `scripts/estimate_tokens.py` or `telemetry/nvml.py` present.
  - Ollama client passes usage when available; llama.cpp usage passthrough is present.
- Improvements:
  - If desired, add token estimation and NVML telemetry behind a feature flag; otherwise, remove/mark as deferred in docs to avoid confusion.

### Ticket 9 — Windows Run Recipes & Smoke Tests
- Status: Partially implemented.
  - `tools/tasks.ps1` provides task aliases; smoke orchestration exists.
  - Lacks explicit examples for `backend=local` until `run_all` wiring is added.

### Ticket 10 — Local Example Configs
- Status: Missing.
  - `config/eval_config.local.yaml` and `config/eval_config.local.llamacpp.yaml` are not present.
- Improvements:
  - Add both example configs and ensure `python -c "from config.schema import load_config; load_config('…')"` validates.

### Ticket 11 — Documentation: Windows + README Edits
- Status: Partial.
  - `docs/windows.md` is minimal; `docs/troubleshooting_windows_local.md` is good; `docs/performance.md` is comprehensive.
  - README lacks a “Windows + Local” quick‑start pointer.
- Improvements:
  - Expand `docs/windows.md` and add a README pointer with a minimal local quick‑start.

### Ticket 12 — Performance Tuning & VRAM Budget Guide
- Status: Implemented.
  - `docs/performance.md` provides presets and troubleshooting, aligned with plan.

### Ticket 13 — CI Smoke (Windows)
- Status: Not present (optional).

### Ticket 14 — PR Plan & Change Isolation
- Status: Implemented.
  - `codex/pr-plan` and `codex/logs` contain the staged PR plan and validation checklist.

### Ticket 15 — Troubleshooting Playbook
- Status: Implemented.
  - `docs/troubleshooting_windows_local.md` provides actionable fixes.

## Additional Observations & Suggestions
- Orchestrator default behavior is preserved (Fireworks); good for safety. However, please add local backend wiring under explicit opt‑in.
- Local concurrency cap of 2 on 16GB VRAM is sensible; consider making it configurable with a documented default.
- Consider adding lightweight unit tests for `parse_custom_id` and output schema validation to catch regressions early.
- Logging: local batch errors are aggregated but not always surfaced with context; a brief per‑part summary in logs would help debugging.

## Validation Summary
How I evaluated the implementation:
- Reviewed the plan document end‑to‑end for intended scope and acceptance criteria.
- Inspected code locations added/modified by the plan (backends/, scripts/, config/, docs/).
- Verified presence and basic behavior of new modules (Ollama/llama.cpp clients, local batch executor, local parser, schema updates, Windows scripts).
- Checked orchestrator pathways for backend switching and artifact expectations (per‑part vs combined JSONL).
- Audited Windows blockers (imports, file locking) and cross‑checked doc coverage (setup, troubleshooting, performance).

Next recommended steps:
1) Wire `backend=local` in `scripts/run_all.py`:
   - Use `backends.local.local_batch.split_jsonl` for splitting and `create_queue(...).run_queue(results_dir)` for execution.
   - After parts finish, combine into `results_combined.jsonl` under the trial `results/` directory.
   - Reuse existing parse/score/stats/report phases unmodified.
2) Fix Windows compatibility by guarding `fcntl` imports/usage in `scripts/run_all.py` and `scripts/state_utils.py`.
3) Add `config/eval_config.local.yaml` and `config/eval_config.local.llamacpp.yaml` with conservative defaults (Ollama endpoint and 8B Q4_K_M model; llama.cpp GGUF path example).
4) Update README with a “Windows + Local” quick‑start and expand `docs/windows.md` with copy/paste recipes (including `ollama serve`/pulls).
5) Optional: implement token estimation and NVML telemetry behind a feature flag, or mark as deferred in the docs.

Once these are done, run `tools/tasks.ps1 -Task Smoke` and a small end‑to‑end local run to validate parity of outputs and downstream scoring/statistics.
