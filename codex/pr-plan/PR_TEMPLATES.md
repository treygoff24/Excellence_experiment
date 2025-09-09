# PR Templates for Windows + Local LLM Port

## PR 1 Template: Backend Interface Foundation + Windows Bootstrap

### Title
```
feat: Add pluggable inference backend interface + Windows bootstrap

- Introduce abstract InferenceClient and BatchExecutor protocols
- Move Fireworks code to backends/fireworks/ (adapter pattern)  
- Add Windows development environment setup
- Preserve Fireworks as default backend throughout
```

### Description Template
```
## Summary
This PR introduces a pluggable backend interface to decouple inference from Fireworks while preserving all existing functionality. It also adds Windows development environment support.

## Changes
- **Backend Interface**: Abstract `InferenceClient` and `BatchExecutor` protocols in `backends/interfaces.py`
- **Fireworks Adapter**: Moved existing Fireworks code to `backends/fireworks/` with no functional changes
- **Local Stubs**: Empty stubs in `backends/local/` for future implementation
- **Windows Support**: Bootstrap script and PowerShell task runner
- **Config Extension**: Added `backend` field to schema (defaults to "fireworks")
- **Orchestrator Update**: Backend switching logic in `scripts/run_all.py`

## Testing Evidence
- [ ] Fireworks path unchanged: `python -m scripts.run_all --config config/eval_config.yaml --plan_only`
- [ ] Backend switching works: `python -m scripts.run_all --backend fireworks --plan_only` 
- [ ] Local backend fails gracefully: `python -m scripts.run_all --backend local --plan_only`
- [ ] Windows bootstrap: `powershell -ExecutionPolicy Bypass -File tools\bootstrap.ps1`
- [ ] Windows tasks: `tools\tasks.ps1 -List`

## Validation Commands
```bash
# Verify no regression in Fireworks path
make smoke
python -m scripts.run_all --config config/eval_config.yaml --plan_only

# Test backend switching
python -m scripts.run_all --backend fireworks --plan_only  
python -m scripts.run_all --backend local --plan_only

# Windows validation (on Windows machine)
powershell -ExecutionPolicy Bypass -File tools\bootstrap.ps1
.\\.venv\\Scripts\\Activate.ps1
tools\\tasks.ps1 -List
```

## Rollback Plan
If issues arise, revert this PR. Fireworks functionality is preserved in `backends/fireworks/` with identical behavior.
```

---

## PR 2 Template: Local Engine Clients + Prompt Adapter

### Title
```
feat: Implement Ollama and llama.cpp clients with centralized prompt adapter

- Add OllamaClient (HTTP API) and LlamaCppClient (in-process)
- Centralized system prompt rendering in scripts/prompt_adapter.py
- Health checks and parameter mapping for local engines
- Preserve Fireworks as default; local engines only active when backend=local
```

### Description Template
```
## Summary
Implements local inference clients (Ollama and llama.cpp) with centralized prompt handling to ensure identical system prompt injection across backends.

## Changes
- **Ollama Client**: HTTP API client with health checks and model validation
- **llama.cpp Client**: In-process client with CUDA/cuBLAS support
- **Prompt Adapter**: Centralized system prompt rendering for A/B experiments
- **Parameter Mapping**: Temperature, top_p, max_new_tokens, stop sequences
- **Config Extension**: Added local_engine, local_model, endpoint settings

## Testing Evidence
- [ ] Ollama client imports: `python -c "from backends.local.ollama_client import OllamaClient; print('OK')"`
- [ ] llama.cpp client imports: `python -c "from backends.local.llama_cpp_client import LlamaCppClient; print('OK')"`
- [ ] Prompt adapter unit tests pass
- [ ] Health checks detect missing services
- [ ] Parameter mapping validation

## Validation Commands
```bash
# Import validation
python -c "from backends.local.ollama_client import OllamaClient; print('Ollama OK')"
python -c "from backends.local.llama_cpp_client import LlamaCppClient; print('LlamaCpp OK')"

# Prompt adapter testing
python -m pytest tests/test_prompt_adapter.py -v

# Health check validation (requires Ollama running)
python -c "from backends.local.ollama_client import OllamaClient; OllamaClient().health_check()"
```

## Dependencies
- For full testing: Ollama installed and running with a model pulled
- llama-cpp-python with CUDA wheels for GPU testing
- No dependencies required for basic import/unit tests
```

---

## PR 3 Template: Local Batch Executor + Parser + Token Accounting

### Title
```
feat: Complete local inference pipeline with schema-compatible outputs

- Implement local batch processing with async/threading
- JSONL → predictions.csv parser matching Fireworks schema exactly
- Token accounting and usage estimation
- Idempotent resume functionality for local batches
```

### Description Template
```
## Summary
Completes the local inference pipeline by implementing batch processing, result parsing, and token accounting while maintaining identical output schemas to Fireworks.

## Changes
- **Batch Executor**: Async work queue with concurrency limiting and resume support
- **Result Parser**: Converts local JSONL to predictions.csv with identical schema
- **Token Accounting**: Usage tracking and estimation when engines don't provide counts
- **Integration**: Connects with prompt adapter and local clients from PR 2
- **Resume Logic**: Skip completed parts on rerun

## Testing Evidence
- [ ] End-to-end smoke: `python -m scripts.run_all --config config/eval_config.local.yaml --dry_run --limit_items 50`
- [ ] Schema validation: Local predictions.csv matches Fireworks schema exactly
- [ ] Scoring compatibility: `python -m scoring.score_predictions` runs on local outputs
- [ ] Resume functionality: Rerun skips completed parts
- [ ] Concurrency limiting: Respects max_concurrent_requests setting

## Validation Commands
```bash
# End-to-end local pipeline (requires Ollama + model)
python -m scripts.run_all --config config/eval_config.local.yaml --dry_run --limit_items 50

# Schema validation
python -m scripts.validate_schema results/predictions.csv

# Resume test
python -m scripts.run_all --config config/eval_config.local.yaml --resume --limit_items 50

# Scoring validation
python -m scoring.score_predictions --pred_csv results/predictions.csv --prepared_dir data/prepared
```

## Output Schema Guarantee
The local pipeline produces predictions.csv with these exact columns matching Fireworks:
`custom_id, dataset, item_id, condition, temp, sample_index, type, request_id, finish_reason, response_text, prompt_tokens, completion_tokens, total_tokens`
```

---

## PR 4 Template: Documentation, Example Configs, and Optional Telemetry

### Title
```
docs: Complete Windows + Local LLM user experience

- Comprehensive Windows setup and performance tuning guides
- Turnkey example configs for Ollama and llama.cpp
- Optional GPU telemetry and troubleshooting utilities
- Update README with Windows + Local quickstart
```

### Description Template
```
## Summary
Completes the user experience with comprehensive documentation, example configurations, and optional performance monitoring for the Windows + Local LLM port.

## Changes
- **Documentation**: Complete Windows setup guide and VRAM performance tuning
- **Example Configs**: Turnkey configs for Ollama and llama.cpp workflows
- **Telemetry**: Optional GPU monitoring and latency tracking (feature-flagged)
- **Troubleshooting**: Diagnostic utilities and common issue resolutions
- **README Update**: Windows + Local quickstart section

## Testing Evidence
- [ ] New user can complete Windows setup following docs
- [ ] Example configs validate: `python -c "from config.schema import load_config; load_config('config/eval_config.local.yaml')"`
- [ ] Telemetry toggles work: Enable/disable via config without breaking core functionality
- [ ] Performance guide provides actionable optimization steps
- [ ] README quickstart is accurate and complete

## Validation Commands
```bash
# Config validation
python -c "from config.schema import load_config; load_config('config/eval_config.local.yaml')"
python -c "from config.schema import load_config; load_config('config/eval_config.local.llamacpp.yaml')"

# Documentation walkthrough
# (Manual: Follow docs/windows.md on clean Windows system)

# Telemetry feature toggle
python -m scripts.run_all --config config/eval_config.local.yaml --telemetry --dry_run --limit_items 10
python -m scripts.run_all --config config/eval_config.local.yaml --dry_run --limit_items 10
```

## Documentation Completeness
- Windows setup: Python, venv, dependencies, execution policies
- Local engines: Ollama vs llama.cpp trade-offs, installation, model management  
- Performance tuning: VRAM budgets, context limits, concurrency settings, OOM resolution
- Troubleshooting: Common errors and step-by-step fixes
```

---

## PR 5 Template: CI Windows Smoke (Optional)

### Title
```
ci: Add Windows validation workflow

- Windows GitHub Actions runner with dry-run orchestration smoke
- Validates Windows path handling and multiprocessing 
- No model downloads (keeps CI fast and reliable)
- Prevents Windows-specific regressions
```

### Description Template
```
## Summary
Adds Windows CI validation to prevent regressions in path handling, multiprocessing, and orchestration logic without requiring model downloads.

## Changes
- **CI Workflow**: `.github/workflows/windows-smoke.yml` with Windows runner
- **Dry-run Smoke**: Orchestration validation without inference
- **Path Validation**: Windows-specific path separator and temp file handling
- **Process Validation**: spawn vs fork multiprocessing differences

## Testing Evidence
- [ ] Windows CI workflow passes
- [ ] Dry-run orchestration completes: Windows smoke test
- [ ] Path handling validated: No hardcoded Unix paths
- [ ] Process spawning works: No multiprocessing deadlocks

## Validation Commands
```powershell
# Windows CI equivalent (run on Windows)
powershell -ExecutionPolicy Bypass -File tools\bootstrap.ps1
.\\.venv\\Scripts\\Activate.ps1
python -m scripts.run_all --config config\\eval_config.local.yaml --dry_run --limit_items 10
```

## CI Strategy
- Use GitHub Windows runners (windows-latest)
- Install Python 3.11 via setup-python action
- Cache pip dependencies for faster runs
- Run dry-run orchestration only (no model downloads)
- Validate Windows-specific path and process handling
```

---

## General PR Guidelines

### All PRs Should Include:
1. **Validation Evidence**: Command outputs showing functionality works
2. **Regression Testing**: Confirm Fireworks path still works  
3. **Schema Compatibility**: Ensure output formats unchanged
4. **Documentation Updates**: Keep docs in sync with code changes
5. **Rollback Plan**: Clear steps to revert if issues arise

### Merge Requirements:
- All CI checks pass (existing + new Windows CI in PR 5)
- Manual validation commands run successfully
- Code review approval from team
- No breaking changes to existing Fireworks workflow
- Schema validation passes for any output format changes

### Testing Strategy Per PR:
- **Unit Tests**: Core functionality and error handling
- **Integration Tests**: End-to-end workflows with realistic data
- **Regression Tests**: Fireworks path unchanged
- **Platform Tests**: Windows-specific validation where applicable