# Excellence Experiment

A controlled A/B evaluation framework for testing **system prompts** on Fireworks AI using `accounts/fireworks/models/gpt-oss-120b`.

This project provides a complete pipeline for measuring whether treatment system prompts have statistically significant causal impacts on hallucination rates and factual accuracy. All parameters (model, decoding, datasets, scoring) are held constant except the system prompt.

## Quick Start

```bash
# Setup environment
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

# Configure credentials and prompts (see Setup section)
# Then run:
python -m scripts.run_all --archive
# or
make eval
```

> **New in 2025**: Run isolation system automatically organizes experiments into separate directories. See [EXPERIMENT_WORKFLOW.md](EXPERIMENT_WORKFLOW.md) for details.

## Features

- **Data Processing**: Automatic download and normalization of SQuAD 2.0, TriviaQA, and Natural Questions datasets into canonical JSONL format
- **Batch Inference**: Integrated Fireworks AI batch processing for cost-effective evaluation (~50% cheaper than synchronous calls)
- **Multiple Conditions**: Support for temperature variations (T=0.0 for deterministic, T=0.7 with K=5 replicates)
- **Comprehensive Scoring**: 
  - Exact Match (EM) and F1 scoring
  - Abstention rate tracking
  - False answer detection on unanswerables
  - Unsupported claim detection for open-book tasks
- **Statistical Analysis (2025 upgrade)**:
  - Exact McNemar test with odds ratio + 95% CI (Baptista–Pike)
  - Paired bootstrap CIs for deltas (EM, F1, abstain_rate, false_answer_rate, unsupported_rate)
  - Effect sizes: Hodges–Lehmann, Cohen’s d, Cliff’s delta; optional permutation p-values
  - Multiple comparisons: Benjamini–Hochberg FDR q-values across temps/types/subgroups
  - Subgroups: per-dataset effects (TriviaQA, NQ-Open, SQuAD v2)
  - Selective-risk: risk–coverage points and AURC; Non‑inferiority (TOST) for EM/F1
  - Mixed-effects robustness (optional): logistic GEE for EM, cluster‑robust linear model for F1
- **Cost Tracking**: Token usage monitoring with batch discount application
- **Reproducibility**: Complete run manifests with prompts, parameters, and job IDs

## Setup

### Prerequisites
- Python 3.10+
- Fireworks AI account with API access

### Installation

1. **Clone and setup environment**:
   ```bash
   git clone <repository-url>
   cd Excellence_experiment
   python -m venv .venv && source .venv/bin/activate
   pip install -r requirements.txt
   ```

2. **Configure credentials**:
   Create `.env` from `ENV_TEMPLATE.txt`:
   ```bash
   cp ENV_TEMPLATE.txt .env
   ```
   Fill in your credentials:
   - `FIREWORKS_API_KEY` - Your Fireworks API key
   - `FIREWORKS_ACCOUNT_ID` - Your account slug (e.g., `my-team`)
   - `FIREWORKS_API_BASE` - Optional, defaults to `https://api.fireworks.ai`

3. **Add your prompts**:
   - `config/prompts/control_system.txt` - Baseline system prompt
   - `config/prompts/treatment_system.txt` - Experimental system prompt

4. **Configure evaluation**:
   Edit `config/eval_config.yaml` to adjust:
   - Model parameters (temperature, tokens, etc.)
   - Dataset sizes and sampling
   - Cost calculation settings
   
   Configuration is validated by `config/schema.py` on startup.

## Usage

### Full Evaluation Pipeline

```bash
python -m scripts.run_all
# or
make eval
```

The pipeline executes these steps automatically:

1. **Data Preparation** - Downloads and normalizes datasets → `data/prepared/*.jsonl`
2. **Batch Input Generation** - Creates evaluation batches → `data/batch_inputs/*.jsonl`
3. **Dataset Upload** - Uploads to Fireworks AI
4. **Job Execution** - Starts batch inference jobs for all conditions
5. **Results Processing** - Downloads, parses, and scores results
6. **Report Generation** - Creates comprehensive evaluation report → `reports/report.md`

### Individual Scripts

For development or debugging, run individual pipeline components:

```bash
# Prepare datasets only
python -m scripts.prepare_data

# Build batch inputs
python -m scripts.build_batches

# Run smoke test (small subset)
python -m scripts.smoke_test

# Generate cost summary
python -m scripts.summarize_costs
```

### Smoke Test (Offline, Full Flow)

Exercise the entire workflow locally on a tiny subset (no external API calls):

```
# Makefile target
make smoke

# Or directly
python -m scripts.smoke_test --mode flow --n 2 --out_dir results/smoke
```

This runs: build_batches → simulate results → parse → score → stats → cost, writing outputs under `results/smoke/<timestamp>/results/`.

## Project Structure

```
Excellence_experiment/
├── config/                 # Configuration files
│   ├── eval_config.yaml   # Main evaluation settings
│   ├── prompts/           # System prompts for A/B testing
│   └── task_instructions/ # Task-specific instructions
├── data/                  # Dataset storage
│   ├── raw/              # Original downloaded datasets
│   ├── prepared/         # Normalized JSONL files
│   └── batch_inputs/     # Batch inference input files
├── fireworks/            # Fireworks AI integration
├── scoring/              # Evaluation metrics and scoring
├── scripts/              # Main pipeline scripts
├── results/              # Evaluation outputs and manifests
└── reports/              # Generated evaluation reports
```

## Experiment History

### Temperature = 1.0 Experiment (Current - temp-1-experiment branch)

**Model**: `accounts/fireworks/models/gpt-oss-120b` @ T=1.0  
**Status**: In Progress  
**Key Changes**:
- Temperature increased from 0.0 to 1.0 for stochastic sampling
- Samples per item increased from 1 to 10 replicates for variance estimation
- Max tokens doubled from 512 to 1024 for both task types
- Enhanced robustness features: dataset upload retries, extended timeouts, file rewind logic
- Added standard deviation metrics and 95% confidence intervals to scoring
- Improved batch splitting with proper .jsonl extensions

**Configuration**:
```yaml
temps: [1.0]
samples_per_item:
  "1.0": 10    # 10 replicates for variance estimation
max_new_tokens:
  closed_book: 1024
  open_book: 1024
```

### Temperature = 0.0 Baseline Results

**Model**: `accounts/fireworks/models/gpt-oss-120b` @ T=0.0  
**Last Run**: 2025-08-19 (based on run manifest)

### Performance Summary

| Task Type | Control EM | Treatment EM | Improvement |
|-----------|------------|--------------|-------------|
| Closed-book | 47.4% | 53.5% | **+6.1pp** |
| Open-book | 0.4% | 0.7% | +0.3pp |

### Statistical Significance
- **McNemar Test**: p < 0.001 (highly significant)
- **Effect Size**: Treatment shows statistically significant improvement in closed-book tasks

### Cost Analysis
- **Total Tokens**: 53.1M (44.8M prompt + 8.3M completion)
- **Estimated Cost**: $5.84 (with 50% batch discount)
- **Token Efficiency**: Treatment prompt is significantly longer (1,391 vs 18 tokens)

*See `reports/report.md` for detailed analysis and `results/` directory for raw data.*

## Configuration

Key settings in `config/eval_config.yaml` (current temp=1.0 experiment):

```yaml
model_id: "accounts/fireworks/models/gpt-oss-120b"
temps: [1.0]  # Stochastic sampling for variance estimation
samples_per_item:
  "1.0": 10   # 10 replicates for statistical analysis
max_new_tokens:
  closed_book: 1024  # Doubled from 512
  open_book: 1024    # Doubled from 512
use_batch_api: true  # 50% cost savings
```

**Previous baseline configuration (T=0.0)**:
```yaml
temps: [0.0]  # Deterministic
samples_per_item:
  "0.0": 1    # Single sample per item
max_new_tokens:
  closed_book: 512
  open_book: 512
```

## Datasets

### Additional configuration keys (new)

Augmented statistical and unsupported detection configuration:

```yaml
stats:
  bootstrap_samples: 5000
  permutation_samples: 5000
  random_seed: 1337
  enable_permutation: true
  enable_fdr: true
  risk_thresholds: [0.0, 0.25, 0.5, 0.75, 1.0]
  tost_alpha: 0.05
  tost_margins: { em: 0.01, f1: 0.01 }
unsupported:
  strategy: overlap   # baseline|overlap|nli (nli is a conservative placeholder)
  threshold: 0.5
  min_token_overlap: 0.6
```

### New artifacts and scripts

- `results/significance.json` (schema_version=2):
  - Per temp/type: `mcnemar{b,c,p_exact,odds_ratio,or_ci_95,q_value}`
  - `metrics[k]` with `delta_mean`, `ci_95`, `wilcoxon{W,p_value,q_value}`, `hodges_lehmann`, `cohens_d`, `cliffs_delta`, `perm_p_value`
  - `subgroups.dataset[...]` (per-dataset effects)
  - `selective_risk{thresholds,points,aurc}`
  - `tost{em,f1}` non-inferiority summaries
- `scripts/unsupported_sensitivity.py` → `results/unsupported_sensitivity.json`
- `scripts/mixed_effects.py` → `results/mixed_models.json` (requires `statsmodels`)
- `scripts/power_analysis.py` → `results/power_analysis.json`
- `scripts/cost_effectiveness.py` → `results/cost_effectiveness.json`

### Report updates

`scripts/generate_report.py` now includes:
- Significance with OR+CI, Δ with 95% CIs, and FDR q-values
- Dataset subgroups; selective-risk AURC and points; TOST
- Mixed-effects (if available), unsupported sensitivity, power/MDE, and cost-effectiveness

### Git hygiene for large artifacts

- This repo intentionally ignores any `results/` and `reports/` directories (at any path depth) to avoid committing large data.
- Keep artifacts local or publish externally (e.g., a release asset or object storage). If needed, use git‑lfs for huge files.

The framework evaluates on three standard datasets:

1. **SQuAD 2.0** - Open-book QA with unanswerables (passage-based)
2. **TriviaQA** - Closed-book factual questions (rc.nocontext subset)
3. **Natural Questions Open** - Closed-book real user queries

All datasets are automatically downloaded and normalized into consistent JSONL format.

## Development

### Running Tests
```bash
# Smoke test with small subset
python -m scripts.smoke_test

# Check configuration validation
python -c "from config.schema import EvalConfig; EvalConfig.from_file('config/eval_config.yaml')"
```

### Adding New Datasets
1. Add dataset download logic to `scripts/prepare_data.py`
2. Implement scoring function in `scoring/` directory
3. Update `config/eval_config.yaml` with size limits

### Modifying Prompts
Edit files in `config/prompts/` and `config/task_instructions/`. The system will:
- Track prompt token counts
- Include prompt hashes in reproducibility manifest
- Report token cost differences between conditions

## Flexible experimentation (models, prompts, sweeps)

- Config additions in `config/eval_config.yaml`:
  - `model_aliases`: map short names to full model ids
  - `prompt_sets`: named sets with `control` and `treatment` paths
  - `default_prompt_set`: which set to use when not specified
  - `models`: optional list of models to sweep (simple)
  - `sweep`: optional cartesian sweep across `models`, `prompt_sets`, `temps`, `top_p`, `top_k`, and `max_new_tokens`
  - `trials`: explicit list of trial objects with per-trial overrides

Example:

```yaml
model_aliases:
  mixtral8x7b: accounts/fireworks/models/mixtral-8x7b-instruct
  llama38b: accounts/fireworks/models/llama-v3p1-8b-instruct
prompt_sets:
  default:
    control: config/prompts/control_system.txt
    treatment: config/prompts/treatment_system.txt
  concise:
    control: config/prompts/control_concise.txt
    treatment: config/prompts/treatment_concise.txt
default_prompt_set: default
sweep:
  models: [mixtral8x7b, llama38b]
  prompt_sets: [default, concise]
  temps: [0.2, 0.7]
  top_p: [0.9, 1.0]
  max_new_tokens:
    open_book: [512, 1024]
    closed_book: [512]
```

- Build inputs once per prompt set and temperatures:

```bash
python -m scripts.build_batches --config config/eval_config.yaml --prompt_set default --temps 0.2,0.7
```

- Run all trials (sweep or explicit):

```bash
python -m scripts.run_all --config config/eval_config.yaml --models mixtral8x7b,llama38b --prompt_sets default,concise --temps 0.2,0.7
```

Outputs:
- `experiments/run_<RUN_ID>/batch_inputs/` shared across trials
- Per-trial: `experiments/run_<RUN_ID>/<trial-slug>/{results,reports}/`
- Per-trial manifest: `trial_manifest.json`
- Multi-trial summary: `experiments/run_<RUN_ID>/multi_trial_manifest.json`
- Aggregate comparison: `experiments/run_<RUN_ID>/aggregate_report.md`

Backwards compatibility: if only `model_id` and `temps` are set, behavior matches prior single-model single-prompt runs.
