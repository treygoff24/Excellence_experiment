# PR Sequence for Windows + Local LLM Port

**Purpose.** Stage the Windows + Local LLM port across 5 separate PRs to minimize risk and keep the Fireworks path working by default throughout the migration.

**Key Principles:**
- Each PR is independently mergeable and testable
- Fireworks remains the default backend until users opt into `backend=local`
- No breaking changes to existing workflows or output schemas
- Each PR includes validation evidence (command outputs, test results)

---

## PR 1: Backend Interface Foundation + Windows Bootstrap

**Scope:** Core abstractions and Windows development environment.

**Changes:**
- `backends/` package with abstract `InferenceClient` and `BatchExecutor` protocols
- Move existing Fireworks code to `backends/fireworks/` (adapter pattern)
- `tools/bootstrap.ps1` (Windows environment setup)
- `tools/tasks.ps1` (PowerShell equivalents of Make targets)
- `docs/windows.md` (basic Windows setup instructions)
- Config schema extension for `backend` field
- Update `scripts/run_all.py` to switch on backend (Fireworks remains default)

**Files Modified/Added:**
```
+ backends/__init__.py
+ backends/interfaces.py
+ backends/fireworks/__init__.py
+ backends/fireworks/batch_executor.py        # moved from fireworks/
+ backends/fireworks/upload_dataset.py       # moved from fireworks/
+ backends/fireworks/start_batch_job.py      # moved from fireworks/
+ backends/fireworks/poll_and_download.py    # moved from fireworks/
+ backends/fireworks/parse_results.py        # moved from fireworks/
+ backends/local/__init__.py                 # empty stubs for now
+ backends/local/ollama_client.py            # stub
+ backends/local/llama_cpp_client.py         # stub  
+ backends/local/local_batch.py              # stub
+ backends/local/parse_results.py            # stub
+ tools/bootstrap.ps1                        # Windows setup script
+ tools/tasks.ps1                           # PowerShell Make equivalents
+ docs/windows.md                           # Windows setup guide
~ config/schema.py                          # add backend field
~ scripts/run_all.py                        # add backend switching logic
```

**Acceptance Criteria:**
- `python -m scripts.run_all --config config/eval_config.yaml --plan_only` works (Fireworks path unchanged)
- `python -m scripts.run_all --config config/eval_config.yaml --backend fireworks --plan_only` works
- `python -m scripts.run_all --config config/eval_config.yaml --backend local --plan_only` shows plan but gracefully fails with "local backend not implemented"
- Windows: `powershell -ExecutionPolicy Bypass -File tools\bootstrap.ps1` creates working venv
- Windows: `tools\tasks.ps1 -List` shows available commands

**Testing:** 
- Existing Fireworks smoke tests pass
- Windows bootstrap validation
- Backend switching logic validation (plan-only mode)

---

## PR 2: Local Engine Clients + Prompt Adapter

**Scope:** Implement Ollama and llama.cpp inference clients with centralized prompt rendering.

**Changes:**
- Complete `backends/local/ollama_client.py` (HTTP API client)
- Complete `backends/local/llama_cpp_client.py` (in-process client)
- `scripts/prompt_adapter.py` (centralized system prompt injection)
- Health checks and model validation
- Parameter mapping for temperature, top_p, max_new_tokens, etc.

**Files Modified/Added:**
```
~ backends/local/ollama_client.py            # implement OllamaClient
~ backends/local/llama_cpp_client.py         # implement LlamaCppClient  
+ scripts/prompt_adapter.py                 # centralized prompt rendering
~ config/schema.py                          # add local_engine, local_model, etc.
```

**Acceptance Criteria:**
- `python -c "from backends.local.ollama_client import OllamaClient; print('OK')"` imports successfully
- `python -c "from backends.local.llama_cpp_client import LlamaCppClient; print('OK')"` imports successfully
- Prompt adapter correctly renders control vs treatment system prompts
- Health checks detect missing Ollama service or model
- Parameter mapping works for common settings (temperature, max_tokens)

**Testing:**
- Unit tests for prompt adapter (control vs treatment rendering)
- Integration test with local Ollama (if available)
- Graceful fallback when local engines unavailable

---

## PR 3: Local Batch Executor + Parser + Token Accounting

**Scope:** Complete local inference pipeline with identical output schema to Fireworks.

**Changes:**
- Complete `backends/local/local_batch.py` (work queue, async/threading)
- Complete `backends/local/parse_results.py` (JSONL → predictions.csv)
- Token accounting and usage estimation
- Idempotent resume for local batches
- Integration with prompt adapter from PR 2

**Files Modified/Added:**
```
~ backends/local/local_batch.py              # implement local batch processing
~ backends/local/parse_results.py           # implement local result parsing
+ scripts/estimate_tokens.py                # token counting utilities
~ config/schema.py                          # add max_concurrent_requests, etc.
~ scripts/run_all.py                        # integrate local batch executor
```

**Acceptance Criteria:**
- `python -m scripts.run_all --config config/eval_config.local.yaml --dry_run --limit_items 50` completes
- Output `predictions.csv` has identical schema to Fireworks version
- `scoring.score_predictions` runs successfully on local outputs
- Resume functionality works (skip completed parts)
- Token accounting populated when available from engine

**Testing:**
- End-to-end smoke test with local backend (50 items)
- Schema validation (local vs Fireworks predictions.csv)
- Resume functionality validation
- Concurrent request limiting

---

## PR 4: Documentation, Example Configs, and Optional Telemetry

**Scope:** Complete user experience with docs, configs, and optional performance monitoring.

**Changes:**
- Complete `docs/windows.md` with full Windows setup guide
- Complete `docs/performance.md` with VRAM tuning guide  
- Example local configs (`eval_config.local.yaml`, `eval_config.local.llamacpp.yaml`)
- Optional telemetry (GPU monitoring, latency tracking)
- Troubleshooting playbook
- Update main README with Windows + Local quickstart

**Files Modified/Added:**
```
~ docs/windows.md                           # complete Windows guide
+ docs/performance.md                       # VRAM/performance tuning
+ config/eval_config.local.yaml            # Ollama example config
+ config/eval_config.local.llamacpp.yaml   # llama.cpp example config
+ telemetry/nvml.py                         # optional GPU monitoring
+ scripts/troubleshoot.py                   # diagnostic utilities
~ README.md                                 # add Windows + Local section
```

**Acceptance Criteria:**
- New Windows user can complete full setup following docs
- Example configs validate via schema loader
- Optional telemetry can be enabled/disabled via config
- Performance guide provides actionable VRAM optimization steps
- README includes clear Windows + Local quickstart

**Testing:**
- Documentation walkthrough validation
- Config validation tests
- Optional telemetry feature toggle
- Performance guide validation on target hardware

---

## PR 5: CI Windows Smoke (Optional)

**Scope:** Add Windows CI validation to prevent regressions.

**Changes:**
- GitHub Actions workflow for Windows
- Dry-run orchestration smoke (no model downloads)
- Windows-specific path and process handling validation

**Files Modified/Added:**
```
+ .github/workflows/windows-smoke.yml       # Windows CI workflow
~ tools/tasks.ps1                           # CI-friendly commands
```

**Acceptance Criteria:**
- Windows CI workflow passes on PR submissions
- Dry-run orchestration completes without errors
- Windows path handling validated
- No model downloads required (keeps CI fast)

**Testing:**
- CI workflow validation
- Windows-specific regression testing
- Dry-run orchestration on Windows runner

---

## Change Isolation Strategy

**Default Behavior Preservation:**
- `backend=fireworks` remains the default throughout all PRs
- Existing `make eval` and `python -m scripts.run_all` commands unchanged
- All output schemas (predictions.csv, significance.json, reports) remain identical
- No changes to scoring, statistics, or report generation logic

**Feature Flags:**
- Local backend only activated when `backend=local` explicitly set
- Local engines only used when local backend enabled
- All new Windows tools are additive (no replacement of existing Unix tools)

**Testing Strategy:**
- Each PR includes both positive tests (new functionality) and negative tests (no regression)
- Fireworks path validated in each PR to ensure no breakage
- Schema validation ensures output compatibility
- Resume/idempotency tested at each stage

**Rollback Plan:**
- Each PR is independently revertible
- Feature flags allow disabling local backend if issues arise
- Fireworks path preserved as fallback throughout migration