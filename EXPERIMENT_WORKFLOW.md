# Experiment Workflow Guide

This guide explains the experiment management system and the orchestrator controls that provide clean separation between runs and robust resumability.

## Overview

The system organizes each run under `experiments/run_<RUN_ID>/`. Within a run, each trial (model × prompt set × decoding) has its own slugged directory.

## Directory Structure

```
experiments/
  run_<RUN_ID>/
    gpt-oss-120b-operational_only-tp1-tk50-mx1024-1024/
      results/              # predictions.csv, per_item_scores.csv, significance.json, costs.json, etc.
      reports/              # report.md
    ... other trial slugs ...
    multi_trial_manifest.json
    aggregate_report.md
```

## Running Experiments

### Standard Workflow

1. **Clean workspace** (optional, if previous run isn't archived):
   ```bash
   python scripts/clean_workspace.py --dry_run  # Preview what will be cleaned
   python scripts/clean_workspace.py           # Actually clean
   ```

2. **Run experiment** with orchestrator controls:
  ```bash
  # Creates experiments/run_<RUN_ID>/ with per‑trial folders
  python -m scripts.run_all --config config/eval_config.yaml --archive

  # Treatment only, decoupled splitting (24 parts), 4 concurrent jobs, resumable
  python -m scripts.run_all --config config/eval_config.yaml \
    --condition=treatment --parts_per_dataset=24 --max_concurrent_jobs=4 --resume --archive
  ```

3. **List all experiments**:
   ```bash
   python scripts/list_runs.py           # Compact view
   python scripts/list_runs.py -v        # Detailed view
   ```

4. **Compare experiments**:
   ```bash
   python scripts/compare_runs.py --run1 t0_r20250818 --run2 t1_r20250821
   ```

### Advanced Options

**Custom run ID:**
```bash
python scripts/run_all.py --run_id my_custom_run --archive
```

**Skip certain steps:**
```bash
python -m scripts.run_all --skip_prepare --skip_build --archive
```

**Manual archiving:**
```bash
python scripts/archive_run.py --experiment_name t15_custom_experiment
```

## Configuration

The config file supports an `experiments_dir` path and prompt sets:

```yaml
paths:
  # ... other paths ...
  experiments_dir: "experiments"
prompt_sets:
  operational_only:
    control: config/prompts/control_system.txt
    treatment: config/prompts/operational_only_system.md
  # ... other sets ...
```

## Script Reference

### `scripts/run_all.py`
Main experiment runner with new isolation features:
- `--run_id`: Custom run identifier
- `--archive`: Automatically archive results after completion
- Automatically creates experiment-specific directories when `experiments/` exists

### `scripts/list_runs.py`
List and inspect archived experiments:
- `--experiments_dir`: Custom experiments directory
- `-v, --verbose`: Show detailed information

### `scripts/compare_runs.py`
Compare results between two experiments:
- `--run1`, `--run2`: Experiment names or paths to compare
- Shows configuration differences, performance metrics, and costs

### `scripts/archive_run.py`
Archive a completed run to experiments directory:
- `--experiment_name`: Custom archive name
- `--dry_run`: Preview what will be archived

### `scripts/clean_workspace.py`
Clean working directories for new experiments:
- `--dry_run`: Preview what will be cleaned
- `--clean_prepared`: Also clean prepared data (not recommended)

## Migration from Old System

Your existing experiments have been archived as:
- `t0_r20250818`: Original temp=0 experiment (batch inputs only)
- `t1_r20250821`: Recent temp=1.0 experiment (complete results)

## Best Practices

1. **Always use `--archive`** when running `scripts/run_all.py` for automatic organization
2. **Clean workspace** between experiments using `scripts/clean_workspace.py`
3. **Preserve prepared data** - it's reusable across experiments
4. **Use descriptive run IDs** for important experiments
5. **Compare results** regularly using `scripts/compare_runs.py`

## Example Workflow

```bash
# List existing experiments
python scripts/list_runs.py

# Clean workspace for new experiment  
python scripts/clean_workspace.py

# Run new experiment with temperature=0.5
# (Edit config/eval_config.yaml to set temps: [0.5] first)
python scripts/run_all.py --archive

# Compare with previous run
python scripts/compare_runs.py --run1 t1_r20250821 --run2 t05_r20250825123456
```

This system ensures clean separation between experiments and provides easy tools for managing and comparing results.
