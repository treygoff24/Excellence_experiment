# WARP.md

This file provides guidance to WARP (warp.dev) when working with code in this repository.

Project: Excellence Experiment — controlled A/B evaluation of system prompts on Fireworks AI (OpenAI-compatible API) across multiple QA datasets. The pipeline automates data prep, batch inference, parsing, scoring, significance testing, cost analysis, and reporting.

Prerequisites
- Python 3.10+
- Fireworks AI credentials in a .env file (see README): FIREWORKS_API_KEY, FIREWORKS_ACCOUNT_ID (account slug, not an email), optional FIREWORKS_API_BASE (defaults to https://api.fireworks.ai)

Environment setup
- python -m venv .venv && source .venv/bin/activate
- pip install -r requirements.txt
- cp ENV_TEMPLATE.txt .env  # then fill in values

Common commands
- One-shot full pipeline (prepare → build batches → upload/schedule → poll → parse/score/stats/costs → report)
  - python -m scripts.run_all
  - make eval  # wrapper for python -m scripts.run_all

- Prepare datasets only (writes data/prepared/*.jsonl)
  - python -m scripts.prepare_data --config config/eval_config.yaml
  - make data

- Build batch inputs only (writes data/batch_inputs/*.jsonl)
  - python -m scripts.build_batches --config config/eval_config.yaml
  - make build

- Upload prebuilt batch inputs to Fireworks datasets (expects ACCOUNT env to be set to your account slug)
  - ACCOUNT=$FIREWORKS_ACCOUNT_ID make upload

- Parse downloaded results JSONL → CSV
  - python -m fireworks.parse_results --results_jsonl results/results_combined.jsonl --out_csv results/predictions.csv
  - make parse

- Score predictions and compute significance/costs
  - python -m scoring.score_predictions --pred_csv results/predictions.csv --prepared_dir data/prepared --out_dir results --config config/eval_config.yaml
  - python -m scoring.stats --per_item_csv results/per_item_scores.csv --metric em --out_path results/significance.json
  - python -m scripts.summarize_costs --pred_csv results/predictions.csv --config config/eval_config.yaml --out_path results/costs.json
  - make score; make stats; make report

- Regenerate the Markdown report from existing artifacts
  - python -m scripts.generate_report --config config/eval_config.yaml --results_dir results --reports_dir reports

Development loop (no formal unit tests configured)
- Quick smoke run against a small sample (writes results/smoke/*)
  - python -m scripts.smoke_test --config config/eval_config.yaml --n 50 --temp 0.0 --type both --condition both --out_dir results/smoke
  - make smoke
- Run a minimal single-item check (useful as a “single test” during iteration)
  - python -m scripts.smoke_test --config config/eval_config.yaml --n 1 --temp 0.0 --type closed --condition control --out_dir results/smoke
  - For open-book only: set --type open
- Validate configuration schema quickly
  - python -c "from config.schema import EvalConfigModel; import yaml; print(EvalConfigModel(**yaml.safe_load(open('config/eval_config.yaml'))).model_dump())"

Important environment notes
- FIREWORKS_ACCOUNT_ID must be a slug (e.g., my-team). Do not use an email. Several commands will hint or normalize this, but setting it correctly avoids 4xx API errors.
- The smoke_test client uses FIREWORKS_API_BASE and will default to the OpenAI-compatible inference path (/inference/v1) if needed.
- Batch input files may be split automatically if >500MB; the orchestrator handles creating multiple datasets/jobs per condition/temperature.

Big-picture architecture
- Configuration and validation (config/schema.py)
  - Pydantic models define the experiment (model_id, temps, samples_per_item, max_new_tokens, pricing, paths, unsupported_threshold). load_config normalizes paths and validates samples_per_item keys/temps.
- Data preparation (scripts/prepare_data.py)
  - Downloads/normalizes datasets via HuggingFace datasets:
    - SQuAD v2 (open-book; includes context and unanswerable flag)
    - TriviaQA rc.nocontext and NQ-Open (closed-book)
  - Produces canonical JSONL files: data/prepared/open_book.jsonl and closed_book.jsonl.
- Batch input construction (scripts/build_batches.py)
  - Reads canonical JSONL and constructs Fireworks batch rows per temp and condition using:
    - System prompts from config/prompts/{control_system.txt,treatment_system.txt}
    - Task instructions from config/task_instructions/{closed_book.txt,open_book.txt}
  - Writes data/batch_inputs/t{temp}_{condition}.jsonl with custom_id encoding: dataset|item_id|condition|temp|sample_index|type.
- Fireworks integration (fireworks/*)
  - upload_dataset.py: Idempotently creates datasets, uploads files (≤150MB via multipart; >150MB via signed URL + validate), waits for READY, and handles transient errors. Strong hints if account id looks like an email.
  - start_batch_job.py: Waits for dataset READY, then creates batchInferenceJobs with inferenceParameters (temperature, maxTokens, topP/topK, optional stop in extraBody). Account id normalization and robust retrying.
  - poll_and_download.py: Polls job state to COMPLETED, fetches output dataset metadata, downloads the external bundle with retries, and extracts JSONL results.
  - parse_results.py: Normalizes Fireworks/OpenAI response shapes, extracts message content, token usage, and writes results/predictions.csv.
- Scoring and statistics (scoring/*)
  - score_predictions.py: Joins predictions to canonical data, computes per-item metrics:
    - open-book: EM, F1, abstain rate, false answer rate (for unanswerables), unsupported_rate heuristic (response not grounded in passage by normalized substring check, gated by unsupported_threshold).
    - closed-book: EM and abstain rate via dataset-specific scorers (triviaqa.py, nq_open.py). Normalization helpers in scoring/normalize.py.
  - stats.py: Paired analysis across conditions for each temp:
    - McNemar’s test (binary win/loss on thresholded metric)
    - Wilcoxon signed-rank (continuous deltas with guards for degenerate cases)
    - Outputs results/significance.json
  - summarize_costs.py: Aggregates prompt/completion tokens from predictions.csv and computes USD using config.pricing with optional batch_discount.
- Orchestration and reporting
  - scripts/run_all.py: End-to-end flow: prepare → build → upload (with 500MB-aware splitting) → start jobs → poll/download → combine results → parse → score → stats → costs → write reports/report.md. Also writes results/run_manifest.json with prompt hashes and token counts.
  - scripts/generate_report.py: Rebuilds the Markdown report from existing artifacts (per_item_scores.csv, significance.json, costs.json, manifest).
- Makefile targets
  - venv, data, build, upload, parse, score, stats, report, eval, smoke map to the Python entry points above. upload expects ACCOUNT=<slug>.

Key files and outputs
- Config: config/eval_config.yaml (validated by config/schema.py)
- Prompts: config/prompts/control_system.txt, config/prompts/treatment_system.txt
- Instructions: config/task_instructions/closed_book.txt, config/task_instructions/open_book.txt
- Prepared data: data/prepared/{open_book.jsonl, closed_book.jsonl}
- Batch inputs: data/batch_inputs/t{temp}_{condition}.jsonl
- Results: results/{results_combined.jsonl, predictions.csv, per_item_scores.csv, significance.json, costs.json, run_manifest.json}
- Report: reports/report.md

Notes derived from README.md and code
- Full pipeline commands and dataset/task coverage mirror the README Quick Start and Usage sections.
- The model defaults and temperature settings are controlled in config/eval_config.yaml (temps array and samples_per_item mapping with stringified temp keys).
- Costs in README assume batch API discount; adjust config.pricing if your account has different rates.

