# Excellence Experiment

A controlled A/B evaluation framework for testing system prompts on Fireworks AI. The pipeline builds batch datasets, runs batch inference, parses and scores outputs, and produces statistical reports that quantify causal impacts of system prompts on hallucination and accuracy metrics.

Highlights
- Fully scripted, resumable orchestrator with per-trial manifest v2 and run-level state.
- Exact causal tests (McNemar), paired bootstrap CIs, effect sizes, optional permutation tests, FDR.
- Open/closed-book scoring with abstention, false answer (on unanswerables), and unsupported detection.
- Cost tracking and cost-effectiveness; optional mixed-effects robustness models and sensitivity sweeps.

See `EXPERIMENT_WORKFLOW.md` for a walkthrough of the execution model and run layouts.

## Quick Start

```bash
# Setup environment
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

# Configure credentials and prompts (see Setup)
python -m scripts.run_all --config config/eval_config.yaml --archive
# or
make eval
```

New in 2025
- Decoupled dataset splitting from concurrency; `--parts_per_dataset` vs `--max_concurrent_jobs`.
- Per-trial manifest v2 with stage_status and artifact paths; idempotent phase gating and resume.
- Dry-run orchestration smoke with STOP/resume injection and automatic cleanup.

## Architecture Overview

- config/
  - schema.py: Pydantic-validated EvalConfigModel + `load_config()`. Paths, pricing, sizes, prompt sets, optional analyses, stats settings; normalizes env vars and defaults.
  - eval_config.yaml: main experiment config (model(s), temps, tokens, prompt sets, sweep/trials).
  - prompts/: control and treatment system prompts. task_instructions/: task-specific additions when applicable.
- scripts/
  - run_all.py: end-to-end orchestrator (prepare → build → submit → poll → parse → score → stats → costs → report → archive). STOP/resume, plan-only, from/to gating, per-trial manifests.
  - prepare_data.py, build_batches.py: data normalization and JSONL batch generation.
  - smoke_test.py, smoke_orchestration.py: offline flow and dry-run orchestration smokes.
  - generate_report.py, summarize_costs.py, compare_runs.py, resume_run.py, state_utils.py, manifest_v2.py, archive_run.py.
  - Post-processing: unsupported_sensitivity.py, mixed_effects.py, power_analysis.py, cost_effectiveness.py.
- fireworks/
  - upload_dataset.py, start_batch_job.py, poll_and_download.py: dataset + batch job control with retries.
  - batch_queue_manager.py: concurrency-limited submission/polling/download queue for parts.
  - parse_results.py: normalize Fireworks batch results JSONL → predictions.csv.
- scoring/
  - score_predictions.py: per-item aggregation from predictions.csv → per_item_scores.csv with stdevs.
  - stats.py: McNemar, bootstrap CIs, effect sizes, selective-risk, FDR, TOST, meta-analysis → significance.json.
  - squad_v2.py, triviaqa.py, nq_open.py, unsupported.py, normalize.py.

Artifacts live under `results/` for simple runs or `experiments/run_<RUN_ID>/<trial-slug>/{results,reports}/` for orchestrated runs. Large artifacts are .gitignored by default.

## Project Structure

```
Excellence_experiment/
├── config/                 # Evaluation config, schema, and prompts
│   ├── eval_config.yaml    # Main configuration (validated on load)
│   ├── schema.py           # Pydantic schema + load_config()
│   ├── prompts/            # System prompts for A/B testing
│   └── task_instructions/  # Optional task-specific instructions
├── data/
│   ├── raw/                # Downloaded sources (ignored)
│   ├── prepared/           # Canonical JSONL inputs (ignored)
│   └── batch_inputs/       # Per-temp/cond inputs and parts (ignored)
├── fireworks/              # Fireworks dataset+batch helpers and parsing
├── scoring/                # Scoring, unsupported detection, statistics
├── scripts/                # Orchestrator and utilities
├── experiments/            # Multi-trial runs under run_<RUN_ID>/ (ignored)
├── results/                # Simple-run outputs (ignored)
└── reports/                # Simple-run reports (ignored)
```

Trial slug example: `gpt-oss-120b-operational_only-tp1-tk50-mx1024-1024`.

## Setup

Prerequisites
- Python 3.10+
- Fireworks AI account with API access

Environment
```bash
python -m venv .venv && source .venv/bin/activate && pip install -r requirements.txt
cp ENV_TEMPLATE.txt .env
```
Populate `.env`:
- `FIREWORKS_API_KEY` — API key
- `FIREWORKS_ACCOUNT_ID` — account slug (e.g., `my-team`)
- `FIREWORKS_API_BASE` — optional; defaults to `https://api.fireworks.ai`

Prompts
- `config/prompts/control_system.txt` — baseline
- `config/prompts/treatment_system.txt` — experiment (or use named prompt sets in config)

Validate config
```bash
python -c "from config.schema import load_config; load_config('config/eval_config.yaml')"
```

## Run Recipes

Make targets
- `make venv` — create venv and install deps.
- `make data` — prepare datasets into `data/prepared/`.
- `make build` — build batch inputs into `data/batch_inputs/`.
- `make eval` — full pipeline via `scripts/run_all.py`.
- `make smoke` — dry-run orchestration smoke.
- `make audit` — prompt token lengths and input-cost deltas.
- `make parse | score | stats | report` — run post-processing stages for simple runs.
- `make plan` — show phase plan/gating and exit (no submissions).

Full pipeline
```bash
python -m scripts.run_all --config config/eval_config.yaml --archive
```

Orchestrator knobs
- `--condition {control,treatment,both}`
- `--parts_per_dataset N` or `--lines_per_part N`
- `--max_concurrent_jobs M` (throughput control)
- `--resume`, `--plan_only`, `--only_step X`, `--from_step X`, `--to_step Y`
- `--limit_items N`, `--skip_prepare`, `--skip_build`, `--dry_run`

Examples
```bash
# Treatment-only; split into 24 parts, fill up to 4 concurrent jobs, resume-safe
python -m scripts.run_all --config config/eval_config.yaml \
  --condition=treatment --parts_per_dataset=24 --max_concurrent_jobs=4 --resume --archive

# Plan only (no network)
python -m scripts.run_all --config config/eval_config.yaml --plan_only

# Quick offline iteration (no network), tiny slice
python -m scripts.run_all --config config/eval_config.yaml --dry_run \
  --prompt_sets operational_only --temps 0.0 --limit_items 200 --parts_per_dataset 3 --max_concurrent_jobs 2
```

Smoke tests
- Classic flow: `python -m scripts.smoke_test --mode flow --n 2`
- Orchestration dry-run with STOP/resume: `python -m scripts.smoke_orchestration --n 3 --prompt_set operational_only --dry_run [--keep]`

Environment notes
- `.env` auto-loads; prefer `--account_id=<slug>` if needed. Avoid passing empty shell expansions.

Individual scripts (useful for debugging)
```bash
# Prepare datasets only
python -m scripts.prepare_data --config config/eval_config.yaml

# Build batch inputs
python -m scripts.build_batches --config config/eval_config.yaml

# Parse results into CSV
python -m fireworks.parse_results --results_jsonl results/results_combined.jsonl --out_csv results/predictions.csv

# Score predictions → per_item_scores.csv
python -m scoring.score_predictions --pred_csv results/predictions.csv --prepared_dir data/prepared --out_dir results

# Compute significance
python -m scoring.stats --per_item_csv results/per_item_scores.csv --config config/eval_config.yaml --out_path results/significance.json
```

## Datasets and Preparation

Built-in datasets
- SQuAD 2.0 (open-book; unanswerables with passages)
- TriviaQA (closed-book; rc.nocontext)
- Natural Questions Open (closed-book)

Prepared JSONL schemas (under `data/prepared/`)
- `open_book.jsonl`: `{ id, dataset, question, context, answers, is_unanswerable }`
- `closed_book.jsonl`: `{ id, dataset, question, answers }`

Batch inputs (under `data/batch_inputs/`)
- Files per temp and condition: `t{temp_label}[_<prompt_set>]_control|treatment.jsonl`
- Optional parts for concurrency: `...p01.jsonl`, `...p02.jsonl`, ...

## Fireworks Integration

Submission and polling
- Datasets uploaded per part; batch jobs created per part using `custom_id = dataset|item_id|condition|temp|sample_index|type`.
- `fireworks.batch_queue_manager.QueueManager` enforces `--max_concurrent_jobs`, maintains state, and downloads outputs when complete.
- `scripts/run_all.py` writes per-trial manifests (schema_version=2) with `jobs{}`, `job_status{}`, `stage_status{}` and artifacts.

Parsing results
- `fireworks.parse_results` extracts normalized fields to `predictions.csv`:
  - Columns: `custom_id, dataset, item_id, condition, temp, sample_index, type, request_id, finish_reason, response_text, prompt_tokens, completion_tokens, total_tokens`.
- Idempotent: validates row count equals unique `custom_id`s in the combined JSONL bundle.

Manual CLI alternative (optional)
- Install: `brew install fw-ai/firectl/firectl` (macOS)
- Verify: `firectl version`; auth via `firectl auth login` or `FIREWORKS_API_KEY` env var.
- Prefer account selection by slug: pass `--account-id=<slug>` when needed (avoid empty shell expansions).
- Common commands: `firectl list datasets|batch-jobs|deployments|models`, `firectl get dataset|batch-job <id>`, `firectl create dataset`, `firectl create batch-inference-job`.
- Workflow mapping:
  1) Build inputs: `make build`.
  2) Upload JSONL: `firectl create-dataset ...` (capture dataset ID).
  3) Start batch inference: `firectl create batch-inference-job ... --dataset-id <id> ...`.
  4) Monitor: `firectl get batch-job <id>`.
  5) Download outputs: `firectl download dataset <outputDatasetId> --output-dir results/raw_download` (optionally add `--download-lineage`), then parse via `python -m fireworks.parse_results --results_jsonl <combined.jsonl> --out_csv results/predictions.csv`.
- Orchestrated run with pre-uploaded data: `python -m scripts.run_all --skip_prepare --skip_build` (or use `fireworks/process_existing_datasets.py`).

## Scoring and Statistics

Per-item aggregation
- `scoring.score_predictions` reads `predictions.csv` and writes `per_item_scores.csv` with replicate aggregation and stdevs.
- Columns include: `item_key, type, condition, temp, em, em_std, [f1, f1_std], abstain_rate, abstain_rate_std, [false_answer_rate, false_answer_rate_std], [unsupported_rate, unsupported_rate_std]`.

Significance and effect sizes
- `scoring.stats` writes `results/significance.json` (schema_version=2) per temp and type with:
  - McNemar stats: `b, c, p_exact, odds_ratio, or_ci_95`.
  - Metrics deltas: `delta_mean`, `ci_95`, `wilcoxon{W,p_value}`, `hodges_lehmann`, `cohens_d`, `cliffs_delta`, optional `perm_p_value`.
  - FDR q-values across temps/types/subgroups when enabled.
  - Dataset subgroups; selective-risk AURC and points; TOST non-inferiority for EM/F1.
  - Optional meta-analysis (fixed/random; Q, I², τ²) across datasets.

Post-processing extras
- Unsupported sensitivity: `python -m scripts.unsupported_sensitivity --pred_csv results/predictions.csv --config config/eval_config.yaml --out_path results/unsupported_sensitivity.json`.
- Mixed-effects robustness (requires `statsmodels`): `python -m scripts.mixed_effects --pred_csv results/predictions.csv --out_path results/mixed_models.json`.
- Power/MDE (false_answer_rate on unanswerables): `python -m scripts.power_analysis --per_item_csv results/per_item_scores.csv --out_path results/power_analysis.json`.
- Cost-effectiveness: `python -m scripts.cost_effectiveness --pred_csv results/predictions.csv --per_item_csv results/per_item_scores.csv --out_path results/cost_effectiveness.json`.

Report
- `python -m scripts.generate_report --results_dir <...>` renders a comprehensive markdown report including significance, subgroups, selective-risk, optional models and sensitivity, power/MDE, and cost-effectiveness.

## Configuration

Minimal example (default single-model run)
```yaml
model_id: "accounts/fireworks/models/gpt-oss-120b"
temps: [0.0]
samples_per_item:
  "0.0": 1
max_new_tokens:
  closed_book: 1024
  open_book: 1024
use_batch_api: true
prompt_sets:
  default:
    control: config/prompts/control_system.txt
    treatment: config/prompts/treatment_system.txt
default_prompt_set: default
```

Sweeps and multi-trial runs
- Use `model_aliases`, `sweep{models,prompt_sets,temps,top_p,top_k,max_new_tokens}`, or explicit `trials` for per-trial overrides. Orchestrator expands these into trial slugs and runs each independently.

Unsupported detection configuration
```yaml
unsupported:
  strategy: overlap  # baseline|overlap|nli
  threshold: 0.5
  min_token_overlap: 0.6
```

Statistics configuration
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
```

## Reproducibility and Resume

Per-trial manifest v2
- `experiments/run_<RUN_ID>/<trial-slug>/results/trial_manifest.json` tracks prompts (sha256 and token counts), temps, datasets, job names, job_status, stage_status with artifact paths.
- Idempotent done-predicates: parse/score/stats/costs/report phases skip when artifacts are present and validated.

Run-level state
- `experiments/run_<RUN_ID>/run_state.json` tracks phase statuses with timestamps; updated atomically with a run-level lock.
- STOP semantics: create `experiments/run_<RUN_ID>/STOP_REQUESTED` or send SIGINT/SIGTERM to request a cooperative stop between phases. Resume with `--resume`.

Repair and resilience
- Manifests auto-upgrade to v2 when legacy; orchestrator can recover from missing downloads via recorded `OUTPUT_DATASET_ID` and CLI fallback.

## Large Files Policy

- Do not commit `results/` or `reports/` anywhere; ignored recursively by `.gitignore`.
- Per-run directories under `experiments/run_*/` and `experiments/**/batch_inputs/` are also ignored.
- Keep artifacts local or publish externally; use release attachments or object storage for sharing.

## Development and Testing

Primary checks
- `make smoke` for an end-to-end dry-run sanity check.
- Config validation: `python -c "from config.schema import load_config; load_config('config/eval_config.yaml')"`.
- Unit tests (focus on scoring/parsing utilities): `pytest -q`.

Determinism and seeds
- Keep random seeds fixed; start with small `--n` in smokes for quick iterations.

Coding style and conventions
- PEP 8 with 4-space indents; type hints for public functions.
- Naming: snake_case for modules/functions, CapWords for classes, UPPER_SNAKE_CASE for constants.
- CLI pattern: `main()` with `argparse` and `if __name__ == '__main__': main()`.
- Prefer absolute imports within top-level packages (`scripts`, `scoring`, `fireworks`).

Security
- Copy `ENV_TEMPLATE.txt` to `.env`; never commit secrets. `python-dotenv` auto-loads `.env` at runtime.

Commit and PR guidelines
- Commits: concise, imperative (e.g., "Improve batch splitting").
- PRs: include what changed and why, config diffs (`config/eval_config.yaml`), prompt paths, and evidence (smoke output, artifact paths, updated reports).

## Troubleshooting and Tips

- Use `--plan_only` to see gating; phases show selected/skipped with reasons.
- Prefer `--account_id=<slug>` or omit (loaded from `.env`); avoid passing empty shell expansions.
- If Fireworks API returns incomplete metadata, run_all falls back to `firectl download dataset <OUTPUT_DATASET_ID>`.
- For manual control over existing datasets, see `fireworks/process_existing_datasets.py`.

## Docs and Prompt Resources

- Guides
  - `docs/guides/gpt5-prompting-best-practices-guide.md` — Responses API, tool calling, structured outputs, MCP, long-context prompting, observability/evals, safety.
  - `docs/guides/gpt5-agentic-workflow-guide.md` — Agents SDK design patterns, verbosity/reasoning controls, tracing/guardrails, worked examples.
- Prompts and templates
  - `docs/prompts/agent-implementation.md`, `docs/prompts/agent-reviewer.md`, `docs/prompts/agent-release-manager.md`, `docs/prompts/run-ticket-example.txt`.
  - `codex/TICKET_TEMPLATE.md`, `codex/LOG_TEMPLATE.md` (if present) standardize tickets and operational logs.
- Orchestration planning
  - `docs/planning/stop_resume_design.md` — run_state.json schema, per-trial manifest v2, idempotent gating.
