# Local Multi-Prompt Sweep (Local backend)

**Date:** October 10, 2025  
**Issue:** Historically, multi-prompt sweeps on local backend could skip trials or mix outputs.  
**Status:** Orchestrator improved; CLI can override config sweeps; `--no_sweep` available.

---

## Problem Description

### Expected Behavior (current)

When running a multi-prompt sweep with the local backend:

```powershell
python -m scripts.run_all --config config\eval_config.local.yaml --limit_items 250 --parts_per_dataset 2
```

The orchestrator will:
1. Execute all prompt sets from `sweep.prompt_sets` once in a single run (unless `--no_sweep`).
2. Isolate outputs per trial under `experiments/run_<RUN_ID>/<trial-slug>/{results,reports}/`.
3. Produce a top-level `aggregate_report.md` for the run.

### Prior Behavior (pre-fix)

- Some trials could be skipped due to shared `results/` gating.
- Archiving could leave shared state between trials.

This has been addressed by isolating per-trial outputs under `experiments/` and improving sweep handling.

---

## Root Cause Analysis

### Architecture Issue

The `run_all.py` orchestrator was designed primarily for the Fireworks cloud backend, where:
- Each trial gets a unique job ID
- Results are downloaded to trial-specific directories
- The `--archive` flag works seamlessly because each trial naturally isolates its outputs

For the **local backend**:
- All trials share the same `results/` working directory
- The resume/gating logic checks for existing artifacts in `results/`
- Once the first trial completes, subsequent trials see the artifacts and skip
- The `--archive` flag attempts to move `results/` but doesn't clean between trials

### Specific Code Flow

1. Trial 1 (e.g., `operational_only`):
   - Runs inference → creates `results/`
   - Completes scoring/stats
   - Archives (but leaves `results/` in place)

2. Trial 2 (e.g., `structure_without_content`):
   - Checks for `results/` → **FOUND**
   - Gating logic: "already done"
   - Skips all stages
   - Reports completion in 0.0 minutes

3. Trials 3-8: Same skip behavior

---

## Recommended Usage

- To run all prompts once from config sweep (recommended):

```powershell
python -m scripts.run_all --config config\eval_config.local.yaml --limit_items 250 --parts_per_dataset 2
```

- To force-disable config sweep and run only prompts/models provided on CLI:

```powershell
python -m scripts.run_all --config config\eval_config.local.yaml --no_sweep --prompt_sets operational_only,structure_without_content
```

- If both config `sweep` and CLI lists are present, CLI overrides the sweep lists. The run will log overrides.

---

## Usage

### Quick Validation Run (250 items, ~55 minutes)

```powershell
python -m scripts.run_all --config config\eval_config.local.yaml --limit_items 250 --parts_per_dataset 2 --max_concurrent_jobs 1
```

### Overnight Full Run (5000 items, ~8-10 hours)

Edit the script:
```powershell
# Change line 15
$limitItems = 5000
```

Then run:
```powershell
.\run_all_prompts.ps1
```

### Multi-Model Support

The script can be adapted for multiple models. See "Multi-Model Experimentation" section below.

---

## Output Structure

Results are archived to timestamped directories:

```
experiments/
└── run_r20251010HHMMSS/
    ├── aggregate_report.md
    ├── batch_inputs/
    ├── <model>-<prompt>-tp...-tk...-mx.../
    │   ├── results/
    │   │   ├── predictions.csv
    │   │   ├── per_item_scores.csv
    │   │   ├── significance.json
    │   │   ├── costs.json
    │   │   └── ...
    │   └── reports/
    │       └── report.md
    └── ... (8 trial directories)
```

---

## Multi-Model Experimentation

### Models Tested (October 2025)

The following models were tested on an NVIDIA RTX 5080 (16GB VRAM) with this framework:

1. **llama31-8b-q4k-gpu** (~4.7GB VRAM)
   - Meta's Llama 3.1, 8B parameters, Q4 quantized
   - Baseline model for comparison
   - Custom model with `num_gpu=999` for maximum GPU offloading

2. **mistral-7b-q4k-gpu** (~4.4GB VRAM)
   - Mistral AI's 7B model with sliding window attention
   - Different architecture from Llama
   - Custom model with `num_gpu=999` for maximum GPU offloading

3. **qwen25-7b-q4k-gpu** (~4.4GB VRAM)
   - Alibaba's Qwen 2.5, trained on multilingual data
   - Strong benchmark performance for 7B class
   - Custom model with `num_gpu=999` for maximum GPU offloading

4. **gemma2-9b-q4k-gpu** (~5.5GB VRAM)
   - Google's Gemma 2, 9B parameters with grouped-query attention
   - Knowledge distilled from Gemini
   - Custom model with `num_gpu=999` for maximum GPU offloading

5. **gpt-oss-20b-gpu** (~11-12GB VRAM) ⚠️
   - 20B parameter model (GPT-OSS variant)
   - **Near VRAM limit** - monitor for OOM errors
   - Custom model with `num_gpu=999` for maximum GPU offloading

**Note:** All models are custom-configured with `num_gpu=999` to force all layers onto GPU for optimal performance on RTX 5080 (16GB VRAM).

### Multi-Model Script Extension

To run the same prompt sweep across multiple models, extend the script:

```powershell
$models = @(
    "llama3.1:8b-instruct-q4_K_M",
    "mistral:7b-instruct-q4_K_M",
    "qwen2.5:7b-instruct-q4_K_M",
    "gemma2:9b-instruct-q4_K_M"
)

foreach ($model in $models) {
    Write-Host "`n🤖 Running model: $model" -ForegroundColor Magenta
    
    # Update config with current model
    # (requires config modification or model-specific configs)
    
    foreach ($prompt in $prompts) {
        # ... existing prompt loop ...
        $archiveDir = "experiments\run_$timestamp`_$model`_$prompt"
        # ... rest of archiving ...
    }
}
```

**Note:** Model selection currently requires either:
- Separate config files per model (e.g., `eval_config.local.llama.yaml`)
- Dynamic config editing in the script
- Manual model switching between runs

---

## Performance Characteristics

### Observed Timing (RTX 5080, 250 items/prompt)

| Model | Items/Min | Minutes/Prompt | Total (8 prompts) |
|-------|-----------|----------------|-------------------|
| Llama 3.1 8B | 72.7 | 6.9 | 55 min |
| Mistral 7B | ~75 | 6.7 | 53 min |
| Qwen 2.5 7B | ~73 | 6.8 | 54 min |
| Gemma 2 9B | ~65 | 7.7 | 62 min |
| GPT-OSS 20B | ~35-45 | 11-14 | 88-112 min |

**5000 items/prompt estimates:**
- 7-9B models: 8-10 hours (all 8 prompts)
- 20B models: 18-22 hours (all 8 prompts)

---

## Future Improvements

### Short-Term

1. **Integrate into main orchestrator** - Modify `run_all.py` to properly handle multi-trial local runs
2. **Better model switching** - Support `--model_id` override without config edits
3. **Parallel processing** - Run multiple small models concurrently if VRAM allows

### Long-Term

1. **Unified backend handling** - Abstract trial isolation logic to work seamlessly across backends
2. **Smart caching** - Allow selective reuse of prepared data and batch inputs
3. **Resume support** - Enable per-prompt resumability within a multi-prompt run
4. **Progress tracking** - Real-time dashboard for long multi-model/multi-prompt runs

---

## Troubleshooting

### Problem: Script expands to more prompts than expected

**Symptoms:** A single CLI prompt still runs all prompts.

**Fix:** Use `--no_sweep` to disable config sweep. Or remove/clear `sweep` in your config.

---

### Problem: Archive directories not created

**Symptoms:** No `experiments/` directory after completion.

**Fix:** Check PowerShell execution policy and directory creation permissions:
```powershell
New-Item -ItemType Directory -Path "experiments\test" -Force
```

---

### Problem: GPT-OSS 20B out of memory

**Symptoms:** CUDA out of memory errors, driver crashes, or empty responses.

**Fixes:**
1. Close all other GPU applications
2. Reduce `max_new_tokens` in config (1024 → 512)
3. Set `max_concurrent_requests: 1` (should already be set)
4. Monitor with `nvidia-smi -l 1` to watch VRAM usage
5. If persistent, fall back to 14B models (Phi 3 Medium, Mistral-Nemo)

---

## Related Files

- **Main orchestrator:** `scripts/run_all.py`
- **Workaround script:** `run_all_prompts.ps1`
- **Local config:** `config/eval_config.local.yaml`
- **Backend implementation:** `backends/local/local_batch.py`

---

## Contributing

If you improve the multi-prompt local workflow:

1. Test with at least 3 prompt sets and 2 models
2. Verify archive isolation (no cross-contamination)
3. Document any config changes required
4. Update this guide with new approaches

---

**Prepared by:** AI assistant during Windows local LLM setup  
**Last Updated:** October 9, 2025  
**Status:** Production workaround, pending upstream fix

