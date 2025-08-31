# Claude Agent Guide

This document orients Claude (and similar LLM agents) to the project’s structure, workflows, and the upgraded statistical analysis so you can contribute effectively and safely.

## Goals (tl;dr)
- Evaluate control vs treatment system prompts on Fireworks AI with rigorous, reproducible statistics.
- Center hallucination outcomes (false answers on unanswerables; unsupported claims), alongside accuracy and abstention.
- Produce machine‑readable outputs and a readable report.

## Repo Map
- `config/` — `eval_config.yaml` (validated), prompts, task instructions.
- `scripts/` — pipeline and analysis CLIs (`run_all.py`, `generate_report.py`, etc.).
- `fireworks/` — data upload/poll/parse utilities.
- `scoring/` — metrics + `stats.py` (core inference) and `unsupported.py` (support detection).
- `experiments/` — archived runs: `run_<RUN_ID>/<trial-slug>/{results,reports}/`.
- `results/`, `reports/` — top‑level outputs for simple runs (ignored in Git).

## How to Run
- Environment (activate venv first): `python -m venv .venv && source .venv/bin/activate`
- Full pipeline: `python -m scripts.run_all --config config/eval_config.yaml --archive` (or `make eval`).
- Offline smoke: `make smoke` — generates a tiny end‑to‑end run locally.
- Orchestrator smoke (dry‑run, auto‑cleanup): `python -m scripts.smoke_orchestration --n 3 --prompt_set operational_only [--keep]`.
- Post‑processing (run any subset):
  - `python -m scoring.stats --per_item_csv results/per_item_scores.csv --config config/eval_config.yaml --out_path results/significance.json`
  - `python -m scripts.unsupported_sensitivity --pred_csv results/predictions.csv --prepared_dir data/prepared --config config/eval_config.yaml --out_path results/unsupported_sensitivity.json`
  - `python -m scripts.mixed_effects --pred_csv results/predictions.csv --prepared_dir data/prepared --config config/eval_config.yaml --out_path results/mixed_models.json`
  - `python -m scripts.power_analysis --per_item_csv results/per_item_scores.csv --prepared_dir data/prepared --config config/eval_config.yaml --out_path results/power_analysis.json`
  - `python -m scripts.cost_effectiveness --pred_csv results/predictions.csv --per_item_csv results/per_item_scores.csv --config config/eval_config.yaml --out_path results/cost_effectiveness.json`
  - `python -m scripts.generate_report --config config/eval_config.yaml --results_dir results --reports_dir reports`

## Orchestrator Controls
- `--condition {control,treatment,both}`
- `--parts_per_dataset N` or `--lines_per_part N` (decoupled from concurrency)
- `--max_concurrent_jobs M`
- `--resume` (skip completed parts using per‑trial manifest)
- `--limit_items N`, `--skip_prepare`, `--skip_build`
- `--dry_run` (synthesize results locally for offline iteration)

Environment: `.env` is loaded via python‑dotenv. Prefer `--account_id=slug` or omit entirely. Avoid `--account_id "$FIREWORKS_ACCOUNT_ID"` if the shell variable may be unset.

## Key Config (eval_config.yaml)
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
  strategy: overlap   # baseline|overlap|nli (nli is placeholder)
  threshold: 0.5
  min_token_overlap: 0.6
```

## Outputs (schemas)
- `significance.json` (schema_version=2):
  - `results["<temp>"]["open|closed"]` → `mcnemar{b,c,p_exact,odds_ratio,or_ci_95,q_value}`, `metrics[...]`, `subgroups`, `selective_risk`, `tost`.
  - Optional `meta{em,f1}` per type: `fixed{delta_mean,ci_95}`, `random{delta_mean,ci_95,tau2}`, `heterogeneity{Q,df,p_value,I2}`.
- Optional: `unsupported_sensitivity.json`, `mixed_models.json`, `power_analysis.json`, `cost_effectiveness.json`.
- Report: `reports/report.md` includes all of the above.

## Do/Don’t
- Do keep seeds fixed and respect validated config schemas.
- Do use `scripts.s​moke_test.py` or `scripts.smoke_orchestration` for quick iteration.
- Don’t commit large artifacts — any `results/`/`reports/` dirs are ignored recursively.
 - Per‑run under `experiments/run_<RUN_ID>/` is also ignored; reference paths in PRs instead of committing.
- Don’t change output schema keys casually; keep backward compatibility or update docs accordingly.

## Common Tasks
- “Upgrade stats” → edit `scoring/stats.py`, then update `scripts/generate_report.py` and README/AGENTS.
- “Adjust unsupported detection” → tweak `scoring/unsupported.py` and `unsupported` config.
- “Add a metric” → extend `scoring/score_predictions.py`, plumb through `stats.py`, and render in the report.

## Optional Dependencies
- `statsmodels` for mixed‑effects robustness: `pip install statsmodels`.

## Contact
- See EXPERIMENT_WORKFLOW.md and README for additional context on runs and organization.

## Fireworks CLI (firectl)
- Purpose: Command‑line control for Fireworks resources (datasets, batch inference jobs, deployments, models, LoRA). Handy for quick iteration and manual inspection alongside Python scripts. Reference: https://fireworks.ai/docs/tools-sdks/firectl/firectl
- Install: `brew install fw-ai/firectl/firectl` (Homebrew). Confirm with `firectl version`.
- Authenticate: `firectl auth login` or set env `FIREWORKS_API_KEY` (auto‑loaded from `.env`). Prefer passing `--account-id=<slug>` explicitly when switching orgs.
- Core commands (discover via `firectl --help`):
  - Datasets: `firectl create dataset`, `firectl list datasets`, `firectl get dataset <id_or_name>`.
  - Batch inference: `firectl create batch-inference-job` (if available), `firectl list batch-jobs`, `firectl get batch-job <id_or_name>`.
  - Deployments/Models: `firectl create deployment|model`, `firectl list deployments|models`, `firectl update ...`, `firectl delete ...`.
  - LoRA: `firectl load-lora`, `firectl unload-lora`.
- Using firectl with this repo:
  - As a drop‑in for upload/start: you can upload prepared JSONL with `firectl create-dataset` instead of `fireworks/upload_dataset.py`, then start a job with `firectl create-batch-inference-job` instead of `fireworks/start_batch_job.py`.
  - Monitoring: use `firectl list-resources batch-jobs` and `firectl get-resources batch-job <id>` for status; logs via `firectl get-batch-job-logs <id>`.
  - Parsing results: keep using `python -m fireworks.parse_results` and downstream `scoring/*` to normalize, score, and analyze outputs; firectl doesn’t replace our parsing/statistics.
- Example flow (manual smoke):
  - `make build` → generate `data/batch_inputs/*.jsonl`.
  - `firectl create-dataset ...` → note dataset ID.
  - `firectl create batch-inference-job --dataset-id <id> --model <model_or_deployment> ...`.
  - Wait for success (`firectl get batch-job <id>`), then download results with `firectl download dataset <outputDatasetId> --output-dir results/raw_download`. Parse with `python -m fireworks.parse_results --job_id <id> --out_dir results/` and continue with `make parse score stats report`.
- Tips:
  - Keep schemas consistent with our scorers; CLI won’t reshape inputs.
  - For reproducible A/B runs, prefer `scripts.run_all` for splitting, concurrency control, and archiving; use firectl for spot‑checks or ad‑hoc jobs.
  - Clean up with `firectl delete-resources` to avoid cluttering accounts with trial artifacts.

### Downloading results via CLI
- Fast path: `firectl download dataset <outputDatasetId> --output-dir results/raw_download`
  - Optionally add `--download-lineage` to pull all related datasets.
- Python helper (all‑in‑one): `python -m fireworks.poll_and_download --account <account_slug> --job_name <job_id_or_name> --out_dir results/raw_download` to poll, resolve `outputDatasetId`, download, extract, and combine into `results.jsonl`.
