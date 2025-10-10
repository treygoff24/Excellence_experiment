# Local Multi-Prompt Sweep Workaround

**Date:** October 9, 2025  
**Issue:** Multi-prompt sweep with `--archive` flag fails on local backend  
**Status:** Workaround implemented via `run_all_prompts.ps1`

---

## Problem Description

### Expected Behavior

When running a multi-prompt sweep with the local backend:

```powershell
python -m scripts.run_all --config config\eval_config.local.yaml --archive --limit_items 250
```

The orchestrator should:
1. Execute all prompt sets defined in `sweep.prompt_sets`
2. Archive each trial to separate directories under `experiments/run_<RUN_ID>/`
3. Generate unique results for each prompt set

### Actual Behavior

What actually happens:

1. **Trial planning succeeds** - All 8 prompt sets are identified and queued
2. **Batch building succeeds** - Input files created for all prompts in `data/batch_inputs/`
3. **Only ONE trial executes** - First prompt runs inference
4. **Remaining trials skip** - All subsequent prompts find cached `results/` directory and skip execution with "already done" messages
5. **No proper archiving** - The `--archive` flag doesn't isolate results between trials for local backend

**Result:** Only 1 of 8 prompt sets actually runs, all share the same results.

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

## Workaround Solution

### Approach

Created `run_all_prompts.ps1` that:
1. Runs each prompt set individually in sequence
2. Cleans working directories between runs
3. Manually archives results to timestamped directories
4. Preserves all outputs in isolated experiment folders

### Implementation

**File:** `run_all_prompts.ps1`

**Key features:**

```powershell
foreach ($prompt in $prompts) {
    # 1. Clean working directories before each run
    Remove-Item results, data\batch_inputs, reports -Recurse -Force
    
    # 2. Run single prompt set
    python -m scripts.run_all --config config\eval_config.local.yaml `
        --limit_items $limitItems `
        --parts_per_dataset $partsPerDataset `
        --prompt_sets $prompt
    
    # 3. Archive manually with timestamp
    $timestamp = Get-Date -Format "yyyyMMddHHmmss"
    $archiveDir = "experiments\run_$timestamp`_$prompt"
    Copy-Item results -Destination "$archiveDir\results" -Recurse
    Copy-Item reports -Destination "$archiveDir\reports" -Recurse
}
```

**Benefits:**
- ✅ Each prompt set runs fresh
- ✅ Results isolated in separate directories
- ✅ No cross-contamination between trials
- ✅ Resume-safe (each prompt is independent)
- ✅ Easy to identify which experiment is which

---

## Usage

### Quick Validation Run (250 items, ~55 minutes)

```powershell
.\run_all_prompts.ps1
```

Processes all 8 prompt sets with 250 items per condition.

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
├── run_20251009183052_operational_only/
│   ├── results/
│   │   ├── predictions.csv
│   │   ├── per_item_scores.csv
│   │   ├── significance.json
│   │   ├── costs.json
│   │   └── ...
│   └── reports/
│       └── report.md
├── run_20251009183759_structure_without_content/
│   ├── results/
│   └── reports/
├── run_20251009184512_philosophy_without_instructions/
│   ├── results/
│   └── reports/
... (8 total directories, one per prompt set)
```

---

## Multi-Model Experimentation

### Models Tested (October 2025)

The following models were tested on an NVIDIA RTX 5080 (16GB VRAM) with this framework:

1. **llama3.1:8b-instruct-q4_K_M** (~4.7GB VRAM)
   - Meta's Llama 3.1, 8B parameters, Q4 quantized
   - Baseline model for comparison

2. **mistral:7b-instruct-q4_K_M** (~4.4GB VRAM)
   - Mistral AI's 7B model with sliding window attention
   - Different architecture from Llama

3. **qwen2.5:7b-instruct-q4_K_M** (~4.4GB VRAM)
   - Alibaba's Qwen 2.5, trained on multilingual data
   - Strong benchmark performance for 7B class

4. **gemma2:9b-instruct-q4_K_M** (~5.5GB VRAM)
   - Google's Gemma 2, 9B parameters with grouped-query attention
   - Knowledge distilled from Gemini

5. **gpt-oss:20b** (~11-12GB VRAM) ⚠️
   - 20B parameter model (GPT-OSS variant)
   - **Near VRAM limit** - monitor for OOM errors

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

### Problem: Script skips prompts even with workaround

**Symptoms:** Still seeing "already done" messages after first prompt.

**Fix:** Ensure cleanup is running:
```powershell
# Manually verify
dir results, data\batch_inputs, reports
# Should not exist between runs
```

Check script has execute permissions and Force flag is working.

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

