# Excellence Experiment – Developer Onboarding Guide

This document brings new developers up to speed on the Excellence Experiment codebase. It explains why the project exists, how the end-to-end evaluation pipeline works, and where to modify the system when building new features. Read it once before you start writing code; it is intentionally exhaustive so you can begin contributing immediately.

---

## 1. Mission & Core Concepts

- **Goal:** Run controlled, repeatable A/B experiments on Fireworks AI to measure how alternative system prompts affect factual accuracy, hallucination rates, and downstream cost/quality trade-offs.
- **Design principles:** Idempotent stages, deterministic batching, resumable orchestration, explicit manifests for reproducibility, and rich statistical analysis (McNemar, bootstrap, meta-analysis, etc.).
- **Primary experimental unit:** For each dataset item, compare `control` vs `treatment` system prompts under matched decoding settings (temperature, top-p, top-k, max tokens, samples per item).
- **Backends:** Fireworks batch inference is the default, but the pipeline can run locally via Ollama or llama.cpp for smoke testing or offline iteration.

---

## 2. Architecture at a Glance

| Layer | Responsibility | Key Modules |
| --- | --- | --- |
| **Configuration** | Structured experiment settings, prompt sets, pricing | `config/schema.py`, `config/eval_config.yaml`, `config/prompts/`, `config/task_instructions/` |
| **Data prep** | Download & normalize datasets into canonical JSONL | `scripts/prepare_data.py`, `data/prepared/` |
| **Batch build** | Combine prompts + instructions + questions into Fireworks-ready JSONL shards | `scripts/build_batches.py`, `data/batch_inputs/`, shard manifests |
| **Orchestration** | Stateful prepare → build → submit → poll → parse → score → stats → costs → report → archive | `scripts/run_all.py`, `scripts/state_utils.py`, `scripts/manifest_v2.py`, `scripts/shared_controls.py` |
| **Fireworks integration** | Dataset upload, job submission, concurrency control, polling & download | `fireworks/upload_dataset.py`, `fireworks/start_batch_job.py`, `fireworks/batch_queue_manager.py`, `fireworks/poll_and_download.py`, `fireworks/parse_results.py` |
| **Scoring & analytics** | Normalize predictions, compute metrics & significance, optional robustness analyses | `scoring/score_predictions.py`, `scoring/normalize.py`, `scoring/squad_v2.py`, `scoring/triviaqa.py`, `scoring/nq_open.py`, `scoring/stats.py`, `scripts/unsupported_sensitivity.py`, `scripts/mixed_effects.py`, `scripts/power_analysis.py`, `scripts/cost_effectiveness.py` |
| **Artifacts & reporting** | Store per-trial outputs, manifests, aggregated reports | `results/`, `reports/`, `experiments/run_<RUN_ID>/...`, `scripts/generate_report.py`, `scripts/summarize_costs.py` |
| **Tooling & tests** | Smoke tests, manifest/plan validation, utilities | `scripts/smoke_orchestration.py`, `scripts/smoke_test.py`, `tests/` |

---

## 3. Repository Tour

- `README.md` — High-level summary; use `EXPERIMENT_WORKFLOW.md` for orchestration details.
- `config/`
  - `eval_config.yaml` — Default experiment configuration. Validated by `config/schema.py`.
  - `schema.py` — Pydantic models (`EvalConfigModel`, nested models) defining every config field. Handles defaults, validation, environment-variable expansion, and backward compatibility.
  - `prompts/` — System prompts for control / various treatments (§6).
  - `task_instructions/` — Per-task user instruction blocks injected during batch build (closed- vs open-book).
- `data/` — Working directory for datasets (ignored by git).
  - `raw/` — Original downloads.
  - `prepared/` — Canonical JSONL used for batching plus `prepared_manifest.json`.
  - `batch_inputs/` — Fireworks-ready prompt JSONL shards per temperature × condition × prompt set, plus `build_manifest.json`.
- `scripts/` — CLI entry points and orchestrator helpers (§7).
- `fireworks/` — API clients, queue manager, parsing logic (§8).
- `scoring/` — Metric implementations, scoring pipeline, statistics (§9).
- `experiments/` — Archived multi-trial runs (ignored by git); structure documented in `EXPERIMENT_WORKFLOW.md`.
- `results/` & `reports/` — Outputs for simple runs (ignored by git).
- `telemetry/` — NVML sampling for local backends.
- `tests/` — Pytest suite covering orchestration gating, manifest upgrades, STOP/resume behavior.
- `docs/` — Supplemental docs (Windows, performance, multi-prompt workaround, etc.).

Keep `.gitignore` exclusions in mind: never commit `results/`, `reports/`, `experiments/run_*`, or `data/**` artifacts.

---

## 4. End-to-End Pipeline

The orchestrator (`scripts/run_all.py`) executes deterministic phases. Each phase writes a manifest entry so resumes and dry runs can skip work safely.

| Phase | CLI entry point | Inputs | Outputs / Side Effects | Idempotency Cues |
| --- | --- | --- | --- | --- |
| `prepare` | `python -m scripts.prepare_data` | Hugging Face datasets, `config.evals.sizes` | `data/prepared/{open_book,closed_book}.jsonl`, `prepared_manifest.json` | Validates SHA256 + item counts |
| `build` | `python -m scripts.build_batches` | Prepared JSONL, prompts, instructions, `samples_per_item`, `temps` | `data/batch_inputs/t{temp}{_prompt_set}_{condition}.jsonl`, `build_manifest.json` | Manifest tracks SHA256/lines per shard |
| `submit` | `fireworks.upload_dataset`, `fireworks.start_batch_job`, queue manager | Batch shards, Fireworks account credentials | Datasets created, jobs dispatched (control + treatment per part) | `manifest_v2` stores job metadata, display names |
| `poll` | `fireworks.batch_queue_manager`, `fireworks.poll_and_download` | Submitted job IDs | Downloads per-part results JSONL + combined file | Stage status recorded in trial manifest |
| `parse` | `python -m fireworks.parse_results` | Combined JSONL bundle | `results/predictions.csv`, `results/prompt_usage.json`, per-part metadata | Checks timestamps to skip reruns |
| `score` | `python -m scoring.score_predictions` | Predictions CSV, canonical prepared data | `results/per_item_scores.csv`, reused control rows hydrated | Skips when output newer than predictions |
| `stats` | `python -m scoring.stats` | Per-item scores, config stats settings | `results/significance.json` (schema_version=2) | Validates schema, writes deterministic JSON |
| `costs` | `python -m scripts.summarize_costs` | Predictions CSV, pricing | `results/costs.json` | Aggregates token usage × pricing |
| `report` | `python -m scripts.generate_report` | Results, significance, optional analyses | `reports/report.md`, optional `aggregate_report.md` (multi-trial) | Reads manifest to include paths |
| `archive` | `scripts.archive_run` | Run outputs | Moves trial artifacts to `experiments/run_<RUN_ID>/...` | Controlled by `--archive` flag |

Each phase writes timestamps and status into:
- `experiments/run_<RUN_ID>/run_state.json` — Top-level phase machine (§7).
- `experiments/run_<RUN_ID>/<trial-slug>/results/trial_manifest.json` — Stage-level manifest v2 per trial (§7.4).

The CLI `--plan_only` flag prints the phase plan without executing anything; tests in `tests/test_gating_plan_only.py` ensure selection logic is stable.

---

## 5. Configuration Deep Dive (`config/schema.py`)

### 5.1 Core fields

- `model_id`: Fireworks model path (e.g., `accounts/fireworks/models/gpt-oss-120b`).
- `temps`: List of temperatures to evaluate. Validator coerces everything to floats.
- `samples_per_item`: Mapping of temperature → number of replicates per item. Keys normalized to `"0.0"`, `"0.7"`, etc. Missing entries default to the first supplied value.
- `max_new_tokens`: Separate limits for `closed_book` and `open_book` tasks.
- `top_p`, `top_k`, `stop`: Decoding controls; `stop` applies only when `use_batch_api` is `False` (row-level stop support).
- `sizes`: Optional per-dataset caps used by `scripts.prepare_data` (fields like `open_book_max_items`, `triviaqa_max_items`).
- `paths`: Location hints for working directories and manifests. On load, environment variables and `~` expansions are normalized.
- `pricing`: Per-million token costs and batch discount. Used by `scripts.summarize_costs.py`.
- `use_batch_api`: Toggle to switch between Fireworks batch endpoints vs single completion endpoints.

### 5.2 Unsupported detection & optional analyses

- `unsupported_threshold` (legacy) plus `unsupported` nested model:
  - `strategy`: `"baseline"`, `"overlap"`, or `"nli"` determine unsupported detection algorithm (`scoring/unsupported.py`).
  - `threshold`, `min_token_overlap`, `nli_model`: Control heuristics.
- `optional`: Boolean toggles to run extra scripts from `scripts/unsupported_sensitivity`, `scripts/mixed_effects`, `scripts.power_analysis`, `scripts.cost_effectiveness`.

### 5.3 Statistics

- `stats.bootstrap_samples`, `stats.permutation_samples`: Must be positive; used in `scoring/stats.py` bootstrap/permutation loops.
- `stats.enable_permutation`, `stats.enable_fdr`: Flags for optional p-value adjustments.
- `stats.random_seed`: Controls deterministic sampling.
- `stats.risk_thresholds`: Sorted list clamped to `[0, 1]` for selective risk metrics.
- `stats.tost_alpha`, `stats.tost_margins`: Parameters for two one-sided tests (non-inferiority).

### 5.4 Prompt sets & sweeps

- `prompt_sets`: Mapping name → `{control, treatment}` prompt files. `load_config` ensures at least one default prompt set exists.
- `default_prompt_set`: Fallback when CLI omits `--prompt_set`.
- `sweep`: Optional grid search definition for multi-prompt/temperature sweeps. Fields: `models`, `prompt_sets`, `temps`, `top_p`, `top_k`, `max_new_tokens`.
- `trials`: Explicit trial definitions (model, prompt set, temps) when orchestrating complex multi-factor runs.

### 5.5 Local backend support

- `backend`: `"fireworks"` or `"local"`.
- `local_engine`: `"ollama"` or `"llama_cpp"`.
- `local_endpoint`, `local_model`, `tokenizer`, `max_concurrent_requests`, `enable_local_telemetry`, `local_n_ctx`, `local_n_gpu_layers`: Control scripts under `scripts/local_*` and telemetry recorders (§11).

Validate any config change with:
```bash
python -c "from config.schema import EvalConfigModel; EvalConfigModel.from_file('config/eval_config.yaml')"
```

---

## 6. Prompts & Prompt Sets

- Control prompt: `config/prompts/control_system.txt` — 18-token minimalist instruction.
- Treatment variants:
  - `config/prompts/treatment_system.txt` — Full “Ethics of Excellence” manifesto (1,391 tokens).
  - Additional treatments for ablations (`operational_only`, `structure_without_content`, `length_matched_*`, `excellence_*_percent`).
- Prompt sets combine control & treatment files; `scripts/build_batches.py` injects system + user messages per dataset item.
- Task instructions: `config/task_instructions/closed_book.txt` and `open_book.txt` enforce abstention behavior.
- To add a new prompt:
  1. Create the file under `config/prompts/`.
  2. Add entry to `config/eval_config.yaml` under `prompt_sets`.
  3. (Optional) Update `sweep.prompt_sets` to include it in sweeps.
  4. Re-run `make build` to regenerate batch shards.

---

## 7. Orchestrator Internals (`scripts/run_all.py`)

### 7.1 High-level flow

`main()` loads config, resolves prompt sets, and constructs a run directory at `experiments/run_<RUN_ID>/`. Phases are executed sequentially unless filtered via CLI flags (`--only_step`, `--from_step`, `--to_step`). Each phase updates `run_state.json` using helpers in `scripts/state_utils.py`.

### 7.2 State management (`scripts/state_utils.py`)

- `RunStateLock` provides cross-process locking via `fcntl` when available.
- `run_state.json` schema:
  ```json
  {
    "schema_version": 1,
    "run_id": "t1_r20250821",
    "phases": {
      "prepare": {"status": "completed", "started_at": "...", "updated_at": "...", "last_error": null},
      ...
    },
    "stop_token": {"requested_at": null, "reason": null}
  }
  ```
- STOP/resume: `StopToken` monitors for `STOP_REQUESTED` sentinel files or OS signals. `--resume` verifies config hash (`compute_config_hash`) before skipping completed phases.

### 7.3 Trial slug & manifest handling

- Trial slug format: `<model-short>-<prompt-set>-tp<top_p>-tk<top_k>-mx<closed>-<open>`. Example: `gpt-oss-120b-operational_only-tp1-tk50-mx1024-1024`.
- Each trial writes under `experiments/run_<RUN_ID>/<slug>/` with:
  - `results/` — outputs per stage.
  - `reports/` — trial-level report.
  - `state/` — optional debugging metadata.

### 7.4 Manifest v2 (`scripts/manifest_v2.py`)

- Guarantees schema_version=2 with `stage_status` entries per stage:
  ```json
  "stage_status": {
    "downloaded": {
      "status": "completed",
      "artifacts": {
        "combined_path": "results_combined.jsonl",
        "parts": [{"job_key": "t0_control_p01", "path": "...", "size": 12345}],
        "n_results": 25424
      }
    },
    ...
  }
  ```
- Auto-upgrades legacy manifests, synthesizes new ones if corrupt, and records artifact metadata (file sizes, counts).

### 7.5 Shared control cache (`scripts/shared_controls.py`)

- Control runs are deterministic; the orchestrator caches control responses in `experiments/run_<RID>/shared_controls/<key>/`.
- Registry file: `control_registry.json` with entries `{key: {status, shared_rel, files, counts}}`.
- During resume or multi-trial sweeps, the orchestrator reuses existing control outputs instead of re-submitting jobs. `scoring.score_predictions` rehydrates those rows when parsing results to keep per-item metrics aligned.
- `scripts/shared_controls.refresh_registry` repairs registry entries and ensures on-disk artifacts exist; missing files reset status to `pending`.

---

## 8. Fireworks Integration Layer

### 8.1 Dataset upload (`fireworks/upload_dataset.py`)

- Wraps Fireworks dataset creation via HTTP (with retries).
- Records dataset IDs, handles `--resume` by checking existing manifests.
- Validates dataset display names (<64 chars) via `_make_dataset_display_name`.

### 8.2 Batch job submission (`fireworks/start_batch_job.py`)

- Creates batch inference jobs with desired decoding parameters.
- Accepts stop sequences, temperature, top-p, top-k, and max token overrides.

### 8.3 Queue manager (`fireworks/batch_queue_manager.py`)

- Coordinates multi-part submissions with concurrency control (`max_concurrent_jobs` CLI flag).
- `QueueManager` tracks `JobInfo` dataclasses: status transitions, retries, quota backoff.
- Integrates STOP tokens to pause submissions gracefully.
- Emits progress callbacks for logging/telemetry.

### 8.4 Polling & download (`fireworks/poll_and_download.py`)

- Polls job state (`get_batch_job` → `_normalize_state`) until completion/failure with exponential backoff.
- Resolves `outputDatasetId` → `externalUrl` and downloads bundles (ZIP/TAR/GZIP).
- Extracts JSONL files to `results/<job_key>/results.jsonl`, then concatenates to `results/results_combined.jsonl`.
- Robust to transient CDN errors (429/403/404), uses `Retry-After` header.

### 8.5 Result parsing (`fireworks/parse_results.py`)

- Reads Fireworks JSONL (OpenAI-compatible responses) → `predictions.csv`.
- Extracts usage metrics, finish reasons, request IDs.
- Appends metadata to manifests for traceability (prompt hashes, token counts).

---

## 9. Scoring & Statistical Analysis

### 9.1 Per-item scoring (`scoring/score_predictions.py`)

- Loads canonical ground truth via `load_canonical` from `data/prepared/*.jsonl`.
- Aggregates replicates per `(dataset|item, condition, temp, type)`:
  - Closed-book scorers: `scoring/triviaqa.py`, `scoring/nq_open.py`.
  - Open-book scorer: `scoring/squad_v2.py` (EM, F1, false answer handling).
- Computes abstention, EM, F1, false answer, unsupported metrics, plus standard deviations for replicate runs.
- Rehydrates reused control rows from shared control cache when necessary.

### 9.2 Statistics (`scoring/stats.py`)

- Consumes `per_item_scores.csv` along with configuration to produce `results/significance.json` (schema_version=2).
- Metrics:
  - Exact McNemar (counts `b`, `c`, `p_exact`, odds ratio with CI).
  - Paired bootstrap confidence intervals for EM/F1 deltas, effect sizes (Hedges’ g, Cohen’s d, Cliff’s delta).
  - Optional permutation p-values (`enable_permutation`).
  - FDR correction across metrics/temps/subgroups (`enable_fdr`).
  - Selective-risk AURC, risk operating points.
  - TOST non-inferiority tests (EM/F1 margins).
  - Dataset-level and meta-analytic aggregation (fixed/random effects with heterogeneity stats `Q`, `I²`, `τ²`).
- Writes deterministic JSON keyed by temperature labels (`"0.0"`, `"1.0"`).

### 9.3 Optional analyses

- `scripts/unsupported_sensitivity.py`: Sweeps unsupported thresholds, records `unsupported_sensitivity.json`.
- `scripts/mixed_effects.py`: Runs GEE logistic / cluster-robust OLS via `statsmodels`.
- `scripts/power_analysis.py`: Minimum detectable effect calculation, required sample sizes.
- `scripts.cost_effectiveness.py`: Cost per percentage-point gains.
- Manifest entries note whether optional analyses ran successfully.

---

## 10. Reporting & Artifacts

- Trial-level report: `scripts/generate_report.py` composes metrics, significance highlights, cost summaries, and optional analyses into Markdown.
- Aggregate report: For multi-trial runs, `experiments/run_<RID>/aggregate_report.md` synthesizes comparisons.
- `scripts.summarize_costs.py` uses pricing config + token usage to produce `costs.json`.
- Artifact layout (per trial):
  ```
  experiments/run_<RID>/<slug>/
    results/
      predictions.csv
      per_item_scores.csv
      significance.json
      costs.json
      trial_manifest.json
      shared_controls/...
    reports/
      report.md
  ```
- For single-run workflows (without `--archive`), the same files live under `results/` and `reports/`.

---

## 11. Local Backend Support & Telemetry

- `config.backend` = `"local"` enables local inference drivers.
- `scripts/local_backend_runner.py` (see `scripts/` directory) handles request scheduling against:
  - **Ollama**: HTTP endpoint specified via `local_endpoint`.
  - **llama.cpp**: CLI invocation via `local_model` path and optional `local_n_ctx` / `local_n_gpu_layers`.
- `telemetry/nvml.py` samples GPU usage when `enable_local_telemetry=True`.
- Windows-specific bootstrap instructions live in `docs/windows.md` and `tools/bootstrap.ps1`. Multi-prompt sweeps on Windows require `run_all_prompts.ps1` (documented in `docs/local_multi_prompt_workaround.md`).

---

## 12. Development Workflow

1. **Clone & environment setup**
   ```bash
   python -m venv .venv && source .venv/bin/activate
   pip install -r requirements.txt            # or `make venv`
   cp ENV_TEMPLATE.txt .env                   # populate Fireworks credentials
   ```
2. **Sanity checks**
   - Validate config: `python -c "from config.schema import load_config; load_config('config/eval_config.yaml')"`
   - Dry run: `make smoke` (`scripts.smoke_orchestration` → dry-run with auto-cleanup).
3. **Full pipeline (Fireworks)**
   ```bash
   python -m scripts.run_all --config config/eval_config.yaml --archive
   ```
4. **Iterative development tips**
   - Use `--dry_run`, `--limit_items`, and reduced `samples_per_item` to iterate quickly.
   - Stage gating: run `python -m scripts.run_all --only_step parse --resume` to rerun a single phase.
   - Use `--plan_only` to confirm what will execute before making network calls.
5. **Testing**
   - Unit tests: `pytest` (focus on `tests/` modules).
   - Smoke orchestrator: `python -m scripts.smoke_orchestration --n 3 --prompt_set operational_only --dry_run`.
6. **Style guidelines**
   - PEP 8, 4-space indents, type hints on public functions.
   - Prefer absolute imports (e.g., `from scripts.state_utils import ...`).
   - Comments only where logic is non-obvious; avoid filler comments.
   - Keep deterministic seeds (see `scripts/prepare_data.RANDOM_SEED=2025`, `stats.random_seed`).
7. **Makefile shortcuts**
   - `make data`, `make build`, `make eval`, `make parse`, `make score`, `make stats`, `make report`.
   - `make audit` prints prompt token counts & cost deltas.

---

## 13. Extending the System

### 13.1 Adding a new dataset

1. Update `scripts/prepare_data.py` to load and normalize the dataset (add new `@dataclass` if needed).
2. Extend `load_*` functions and add entries to the canonical JSONL outputs.
3. Expand scoring logic (`scoring/normalize.py` or create new scorer module) with EM/F1 or custom metrics.
4. Update `scoring/score_predictions.py` to recognize the dataset and route to the new scorer.
5. Adjust `config/schema.SizesModel` if dataset-specific limits are required.

### 13.2 New metrics or statistical tests

1. Implement metric computation in appropriate scorer module.
2. Include aggregation logic and standard deviations in `score_predictions`.
3. Update `scoring/stats.py` to compute significance, bootstrap intervals, and include outputs in `significance.json`.
4. Document new fields in `README.md` / relevant docs and ensure backward compatibility.

### 13.3 New prompt variants or sweeps

- Add prompt file, register under `config/eval_config.yaml`.
- If running multi-prompt sweeps, ensure the slug abbreviation (`_abbr_prompt_set`) yields unique <64 character dataset names.
- Consider update to `scripts/run_all` if new parameters (e.g., nucleus sampling variations) should be part of the slug.

### 13.4 Supporting another backend

- Implement API wrappers analogous to `fireworks/upload_dataset.py` and `fireworks/start_batch_job.py`.
- Extend orchestrator to branch on `config.backend`; reuse manifest writing to keep resumes consistent.
- Update `config/schema.py` to validate and normalize new backend-specific fields.

---

## 14. Troubleshooting & Observability

- **Resume logic:** If a run stops unexpectedly, rerun `python -m scripts.run_all --resume --archive`. The orchestrator verifies config hash and restarts incomplete phases only.
- **Shared control cache issues:** Use `python -m scripts.shared_controls.refresh_registry <run_dir>` or delete specific `shared_controls/<key>/` directories to force regeneration.
- **Manifest corruption:** `scripts/manifest_v2.load_manifest` auto-backs up corrupt files and synthesizes replacements. Check `.backup.*` and `.corrupt.*` in trial directories.
- **Quota errors:** Queue manager sets `quota_blocked=True` when Fireworks returns quota errors; once a running job completes, submissions resume automatically.
- **Partial downloads:** `manifest_v2.compute_stage_statuses` marks `downloaded.status` as `in_progress` if part sizes are zero; rerun `--from_step poll`.
- **Unsupported detection drift:** Tune `config.unsupported.threshold` or run `python -m scripts.unsupported_sensitivity` after adjusting heuristics.
- **STOP requests:** Create `experiments/run_<RID>/STOP_REQUESTED` or send SIGINT. The orchestrator stops between phases; rerun with `--resume` later.

---

## 15. Quick Reference Commands

```bash
# Environment
make venv

# Smoke test (dry run, auto cleanup)
make smoke

# Full pipeline with archiving
python -m scripts.run_all --config config/eval_config.yaml --archive

# Resume from parse onwards
python -m scripts.run_all --config config/eval_config.yaml --from_step parse --resume

# Validate prompts & costs
make audit

# Inspect experiment history
python scripts/list_runs.py -v

# Compare two runs
python scripts/compare_runs.py --run1 <run_a> --run2 <run_b>
```

---

## 16. Onboarding Checklist for New Developers

1. **Read** this guide plus `README.md` and `EXPERIMENT_WORKFLOW.md`.
2. **Set up** environment and run `make smoke` to confirm local wiring.
3. **Review** `config/eval_config.yaml` and prompts to understand current experimental settings.
4. **Explore** `scripts/run_all.py` and `scoring/stats.py` to familiarize yourself with orchestration and analytics.
5. **Run** a limited dry run (e.g., `--dry_run --limit_items 100 --parts_per_dataset 2`) to walkthrough outputs.
6. **Inspect** existing experiments (`python scripts/list_runs.py -v`) to see artifact structure.
7. **Pick a starter task** (e.g., add a new prompt variant, tweak unsupported detection, extend reporting) and trace the relevant code path using the directory references in this guide.

Once you have completed the checklist, you should be comfortable owning new features and contributing to the Excellence Experiment pipeline.

---

## 17. Glossary

- **Control** — Baseline system prompt condition.
- **Treatment** — Experimental system prompt being evaluated.
- **Shared control cache** — On-disk reuse of deterministic control responses across trials.
- **Manifest v2** — Per-trial JSON document tracking stage completion and artifacts.
- **Samples per item** — Number of replicate completions per question per temperature.
- **Selective risk** — Metrics evaluating abstention vs false positives at various thresholds.
- **TOST** — Two one-sided t-test, used for non-inferiority checks.

---

For questions or deeper dives, start by examining the modules referenced in each section. The codebase is organized so that each stage has a single point of entry and explicit manifests—use those as anchor points when making changes.
