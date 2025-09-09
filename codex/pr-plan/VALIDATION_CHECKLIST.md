# Validation Checklist for Windows + Local LLM PRs

## Pre-Merge Requirements (All PRs)

### Code Quality
- [ ] All CI checks pass (linting, tests, type checking)
- [ ] Code review approved by at least one team member
- [ ] No merge conflicts with target branch
- [ ] No hardcoded paths or platform-specific assumptions
- [ ] Error handling includes actionable error messages
- [ ] No secrets or credentials committed

### Fireworks Path Preservation
- [ ] **Critical**: `python -m scripts.run_all --config config/eval_config.yaml --plan_only` works unchanged
- [ ] **Critical**: `make smoke` passes (if Makefile exists)
- [ ] Existing output schemas unchanged (predictions.csv, significance.json)
- [ ] No changes to scoring/statistics/report generation logic
- [ ] Default behavior remains `backend=fireworks` (no opt-out required)

### Documentation
- [ ] All new functionality documented
- [ ] Command examples tested and accurate
- [ ] Breaking changes clearly marked (should be none)
- [ ] README updates reflect actual functionality

---

## PR 1 Specific: Backend Interface Foundation

### Functionality Validation
```bash
# Fireworks path unchanged
python -m scripts.run_all --config config/eval_config.yaml --plan_only
make smoke  # if available

# Backend switching logic
python -m scripts.run_all --backend fireworks --plan_only
python -m scripts.run_all --backend local --plan_only  # should fail gracefully

# Import validation
python -c "from backends.fireworks import BatchExecutor; print('OK')"
python -c "from backends.interfaces import InferenceClient; print('OK')"
```

### Windows Validation
```powershell
# Windows bootstrap (run on Windows)
powershell -ExecutionPolicy Bypass -File tools\bootstrap.ps1

# Activation test
.\\.venv\\Scripts\\Activate.ps1
python --version  # should be 3.11+

# Task runner
tools\\tasks.ps1 -List  # should show available commands
```

### File Structure Validation
- [ ] `backends/fireworks/` contains moved Fireworks code
- [ ] `backends/local/` contains stub files only
- [ ] `backends/interfaces.py` defines protocols clearly
- [ ] `config/schema.py` includes backend field with default "fireworks"
- [ ] No broken imports after file moves

### Rollback Test
- [ ] Revert commit and confirm Fireworks path still works
- [ ] Re-apply commit and confirm no regression

---

## PR 2 Specific: Local Engine Clients

### Import Validation
```bash
# Basic imports (no dependencies required)
python -c "from backends.local.ollama_client import OllamaClient; print('Ollama OK')"
python -c "from backends.local.llama_cpp_client import LlamaCppClient; print('LlamaCpp OK')"
python -c "from scripts.prompt_adapter import render_prompt; print('Adapter OK')"
```

### Prompt Adapter Validation
```bash
# Unit tests for prompt consistency
python -m pytest tests/test_prompt_adapter.py -v

# Manual prompt validation
python -c "
from scripts.prompt_adapter import render_prompt
control = render_prompt('test item', 'control', 'open_book')
treatment = render_prompt('test item', 'treatment', 'open_book')
print('Control vs Treatment system prompts differ:', control != treatment)
"
```

### Health Check Validation (Optional - requires setup)
```bash
# Ollama health check (if Ollama available)
python -c "
from backends.local.ollama_client import OllamaClient
try:
    client = OllamaClient()
    client.health_check()
    print('Ollama health check passed')
except Exception as e:
    print('Ollama not available (expected):', e)
"
```

### Configuration Validation
- [ ] Config schema validates local engine settings
- [ ] Parameter mapping works (temperature, top_p, max_new_tokens)
- [ ] Graceful fallback when engines unavailable

---

## PR 3 Specific: Local Batch Executor + Parser

### End-to-End Pipeline Validation
```bash
# Dry-run smoke (requires Ollama + model for full test)
python -m scripts.run_all --config config/eval_config.local.yaml --dry_run --limit_items 50

# Schema validation
python -m scripts.validate_schema results/predictions.csv  # if validator exists
head -1 results/predictions.csv  # check column headers match Fireworks
wc -l results/predictions.csv    # check row count matches input
```

### Resume Functionality
```bash
# Initial run
python -m scripts.run_all --config config/eval_config.local.yaml --limit_items 100 --parts_per_dataset 2

# Resume test (should skip completed parts)
python -m scripts.run_all --config config/eval_config.local.yaml --resume --limit_items 100 --parts_per_dataset 2
```

### Downstream Compatibility
```bash
# Scoring should work unchanged
python -m scoring.score_predictions --pred_csv results/predictions.csv --prepared_dir data/prepared --out_dir results

# Statistics should work unchanged  
python -m scoring.stats --per_item_csv results/per_item_scores.csv --config config/eval_config.local.yaml
```

### Schema Verification
- [ ] predictions.csv has exact columns: `custom_id, dataset, item_id, condition, temp, sample_index, type, request_id, finish_reason, response_text, prompt_tokens, completion_tokens, total_tokens`
- [ ] All custom_ids from input present in output
- [ ] No duplicate custom_ids in output
- [ ] Token counts populated when available

---

## PR 4 Specific: Documentation & Configs

### Configuration Validation
```bash
# Example configs validate
python -c "from config.schema import load_config; load_config('config/eval_config.local.yaml'); print('Ollama config OK')"
python -c "from config.schema import load_config; load_config('config/eval_config.local.llamacpp.yaml'); print('LlamaCpp config OK')"
```

### Documentation Testing
- [ ] Windows setup guide: Manual walkthrough on clean Windows system
- [ ] Performance guide: Commands and recommendations tested
- [ ] Troubleshooting: Common issues and fixes validated
- [ ] README quickstart: Commands tested and accurate

### Telemetry Validation (Optional)
```bash
# Telemetry toggle test
python -m scripts.run_all --config config/eval_config.local.yaml --telemetry --dry_run --limit_items 10
python -m scripts.run_all --config config/eval_config.local.yaml --dry_run --limit_items 10
```

### Content Validation
- [ ] All command examples in docs are tested and accurate
- [ ] Performance recommendations match target hardware (RTX 5080, 16GB VRAM)
- [ ] Troubleshooting covers common failure modes
- [ ] README Windows section links to correct docs

---

## PR 5 Specific: CI Windows Smoke

### CI Workflow Validation
- [ ] Windows workflow runs successfully in CI
- [ ] All steps complete without errors
- [ ] Dry-run orchestration passes
- [ ] No model downloads required (keeps CI fast)

### Local Windows Testing
```powershell
# Equivalent to CI workflow (run on Windows)
powershell -ExecutionPolicy Bypass -File tools\bootstrap.ps1
.\\.venv\\Scripts\\Activate.ps1
python -m scripts.run_all --config config\\eval_config.local.yaml --dry_run --limit_items 10
```

### Platform-Specific Validation
- [ ] Path separators handled correctly (Windows vs Unix)
- [ ] Multiprocessing works (spawn vs fork)
- [ ] Temp files created and cleaned up properly
- [ ] No hardcoded Unix assumptions in CI workflow

---

## Emergency Rollback Procedures

### PR 1 Rollback
```bash
# If backend interface breaks Fireworks path
git revert <PR1_COMMIT_HASH>
# Test: python -m scripts.run_all --config config/eval_config.yaml --plan_only
```

### PR 2-3 Rollback
```bash
# If local implementation breaks orchestrator
git revert <PR_COMMIT_HASH>
# Or disable via config:
# Set backend: fireworks in config files
```

### PR 4 Rollback
```bash
# If docs/configs cause confusion
git revert <PR4_COMMIT_HASH>
# Remove example configs if they cause validation errors
```

### PR 5 Rollback
```bash
# If CI workflow causes issues
git revert <PR5_COMMIT_HASH>
# Or disable workflow:
# Comment out workflow triggers in .github/workflows/
```

---

## Cross-PR Integration Testing

After merging all PRs, validate complete integration:

### Full Pipeline Validation
```bash
# Fireworks path (should be unchanged)
python -m scripts.run_all --config config/eval_config.yaml --limit_items 200 --archive

# Local path (end-to-end)
python -m scripts.run_all --config config/eval_config.local.yaml --limit_items 200 --archive

# Compare outputs
diff results/predictions.csv experiments/run_*/*/results/predictions.csv  # should have same schema
```

### Schema Compatibility Verification
```bash
# Run both backends and compare schemas
python -m scripts.compare_schemas \
  --fireworks experiments/run_fireworks/trial1/results/predictions.csv \
  --local experiments/run_local/trial1/results/predictions.csv

# Verify downstream processing works identically
python -m scoring.stats --per_item_csv results/per_item_scores.csv --config config/eval_config.yaml
```

### Performance Validation
- [ ] Local backend performs reasonably on target hardware
- [ ] Memory usage within VRAM constraints (16GB RTX 5080)
- [ ] Concurrency limits respected
- [ ] Resume functionality prevents duplicate work

---

## Success Criteria Summary

A successful Windows + Local LLM port should achieve:

1. **Zero Regression**: Fireworks path works identically before/after all PRs
2. **Feature Parity**: Local backend produces same output schemas
3. **Platform Support**: Full Windows development environment
4. **Documentation**: Complete setup and troubleshooting guides
5. **CI Coverage**: Automated validation prevents regressions
6. **Rollback Safety**: Each PR can be safely reverted if needed

### Definition of Done
- [ ] All 5 PRs merged successfully
- [ ] Both backends produce identical output schemas
- [ ] Windows development environment fully functional
- [ ] Documentation enables new contributor success
- [ ] CI prevents platform-specific regressions
- [ ] Emergency rollback procedures tested and documented