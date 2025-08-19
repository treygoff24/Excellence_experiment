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
python -m scripts.run_all
# or
make eval
```

## Features

- **Data Processing**: Automatic download and normalization of SQuAD 2.0, TriviaQA, and Natural Questions datasets into canonical JSONL format
- **Batch Inference**: Integrated Fireworks AI batch processing for cost-effective evaluation (~50% cheaper than synchronous calls)
- **Multiple Conditions**: Support for temperature variations (T=0.0 for deterministic, T=0.7 with K=5 replicates)
- **Comprehensive Scoring**: 
  - Exact Match (EM) and F1 scoring
  - Abstention rate tracking
  - False answer detection on unanswerables
  - Unsupported claim detection for open-book tasks
- **Statistical Analysis**: McNemar and Wilcoxon tests for significance testing
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

## Latest Results

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

Key settings in `config/eval_config.yaml`:

```yaml
model_id: "accounts/fireworks/models/gpt-oss-120b"
temps: [0.0]  # Add 0.7 for stochastic evaluation
samples_per_item:
  "0.0": 1    # Deterministic
  "0.7": 5    # K=5 samples for averaging
max_new_tokens:
  closed_book: 512
  open_book: 512
use_batch_api: true  # 50% cost savings
```

## Datasets

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
