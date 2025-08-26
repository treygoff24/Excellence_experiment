# Experiment Workflow Guide

This guide explains how to use the new experiment management system that provides clean separation between different experimental runs.

## Overview

The system now organizes experiments into separate directories under `experiments/` with the naming convention `t{temperature}_{run_id}`, preventing data mixing between runs.

## Directory Structure

```
experiments/
├── t0_r20250818/          # Temperature=0.0 run from 2025-08-18
│   ├── batch_inputs/
│   ├── results/
│   ├── reports/
│   └── archive_manifest.json
├── t1_r20250821/          # Temperature=1.0 run from 2025-08-21
│   ├── batch_inputs/
│   ├── results/
│   ├── reports/
│   └── archive_manifest.json
└── current -> t1_r20250821  # Optional symlink to active experiment
```

## Running Experiments

### Standard Workflow

1. **Clean workspace** (optional, if previous run isn't archived):
   ```bash
   python scripts/clean_workspace.py --dry_run  # Preview what will be cleaned
   python scripts/clean_workspace.py           # Actually clean
   ```

2. **Run experiment** with automatic run isolation:
   ```bash
   # Will create experiment directory automatically if experiments/ exists
   python scripts/run_all.py --archive
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
python scripts/run_all.py --skip_prepare --skip_build --archive
```

**Manual archiving:**
```bash
python scripts/archive_run.py --experiment_name t15_custom_experiment
```

## Configuration

The config file now supports an `experiments_dir` path:

```yaml
paths:
  # ... other paths ...
  experiments_dir: "experiments"
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