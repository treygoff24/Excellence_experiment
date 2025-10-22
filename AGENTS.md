# Repository Guidelines

## Project Structure & Module Organization
- Root: Python 3.10+ project for A/B evaluating system prompts on Fireworks AI.
- Key dirs:
  - `config/` — `eval_config.yaml`, prompt files; schema validation on load.
  - `scripts/` — CLI entrypoints (e.g., `run_all.py`, `prepare_data.py`).
  - `fireworks/` — dataset upload, job control, result parsing.
  - `scoring/` — normalization, metrics, statistics.
  - `data/` — `raw/`, `prepared/`, `batch_inputs/` artifacts.
  - `experiments/` — `run_<RUN_ID>/<trial-slug>/{results,reports}/` outputs.
  - `results/`, `reports/` — top‑level summaries for simple runs.

## Build, Test, and Development Commands
- Venv: `python -m venv .venv && source .venv/bin/activate`
- Environment: `python -m venv .venv && source .venv/bin/activate && pip install -r requirements.txt`
- Make targets:
  - `make venv` — create venv and install deps.
  - `make data` — download/prepare datasets to `data/prepared/`.
- `make build` — create batch inputs in `data/batch_inputs/`.
- `make eval` — full pipeline via `scripts/run_all.py`.
- `make smoke` — small end‑to‑end run for sanity checks.
- `make alt-smoke` — exercises the OpenAI/Anthropic replay fixtures; run `make venv` first so `.venv/bin/python` exists, or invoke `python3 -m scripts.alt_smoke` manually.
- Orchestrator smoke (dry‑run, auto‑cleanup):
  - `python -m scripts.smoke_orchestration --n 3 --prompt_set operational_only` (add `--keep` to retain outputs)
- `make parse | score | stats | report` — post‑processing utilities.
- New post‑processing:
  - `python -m scripts.unsupported_sensitivity ...` → sensitivity sweep for unsupported detection.
  - `python -m scripts.mixed_effects ...` → GEE logistic + cluster‑robust OLS for F1 (requires `statsmodels`).
  - `python -m scripts.power_analysis ...` → MDE and required N for primary endpoint.
  - `python -m scripts.cost_effectiveness ...` → $ per 1pp gains and cost deltas.
- Direct run: `python -m scripts.run_all --config config/eval_config.yaml --archive`.
  - Orchestrator knobs: `--condition`, `--parts_per_dataset`/`--lines_per_part`, `--max_concurrent_jobs`, `--resume`, `--limit_items`, `--skip_prepare`, `--skip_build`, `--dry_run`.
  - Example (treatment‑only, fill up to 4 concurrent, resume safe):
    - `python -m scripts.run_all --config config/eval_config.yaml --condition=treatment --parts_per_dataset=24 --max_concurrent_jobs=4 --resume --archive`
- Validate config: `python -c "from config.schema import EvalConfig; EvalConfig.from_file('config/eval_config.yaml')"`.

## Coding Style & Naming Conventions
- Style: PEP 8, 4‑space indents, type hints required for public functions.
- Naming: `snake_case` for modules/functions, `CapWords` for classes, `UPPER_SNAKE_CASE` for constants. Filenames match module purpose (e.g., `build_batches.py`).
- CLI pattern: define `main()` with `argparse` and use `if __name__ == '__main__': main()` when applicable.
- Imports: prefer absolute imports within top‑level packages (`scripts`, `scoring`, `fireworks`).

## Testing Guidelines
- Primary checks: `make smoke` and config validation (see above).
- Determinism: keep sampling seeds fixed; start with small `--n` for quick iteration.
- If adding unit tests, use `pytest` under `tests/` with `test_*.py`; aim for coverage on `scoring/` and parsing utilities.

## Statistical Analysis (what to expect)
- `scoring/stats.py` computes:
  - Exact McNemar (b,c,p_exact) with odds ratio and CI.
  - Paired bootstrap CIs for deltas across metrics and effect sizes (HL, Cohen’s d, Cliff’s delta).
  - Optional permutation p-values.
  - FDR q-values across temps/types/subgroups.
  - Subgroups per dataset; selective‑risk AURC and points; TOST non‑inferiority for EM/F1.
  - Meta-analysis across datasets (EM, F1 open-book): fixed/random effects with heterogeneity (Q, I², τ²).
- Optional robustness models: `scripts/mixed_effects.py` (GEE/OLS).

## Artifacts and Reports
- `results/significance.json` (schema_version=2) consolidates all stats.
- Optional extras: `unsupported_sensitivity.json`, `mixed_models.json`, `power_analysis.json`, `cost_effectiveness.json`.
- `scripts/generate_report.py` renders all sections. Use the smoke config (temps) to align significance sections with outputs.

## Large Files Policy
- Do not commit `results/` or `reports/` directories. These are ignored recursively by `.gitignore`.
- Per‑run directories under `experiments/run_*/` and `experiments/**/batch_inputs/` are also ignored.
- If you must share artifacts, upload them externally and link paths in PRs.

## Local Backend (Windows + Ollama/llama.cpp)
- **Bootstrap:** `powershell -ExecutionPolicy Bypass -File tools\bootstrap.ps1` (Windows only).
- **Configs:** `config/eval_config.local.yaml` (Ollama), `config/eval_config.local.llamacpp.yaml` (llama.cpp).
- **Multi-prompt sweep workaround:** Use `run_all_prompts.ps1` for local backend; the built-in `--archive` multi-trial sweep has isolation issues (see `docs/local_multi_prompt_workaround.md`).
- **Tested models (October 2025, 16GB VRAM):**
  - `llama31-8b-q4k-gpu` - Meta baseline (custom GPU-optimized)
  - `mistral-7b-q4k-gpu` - Mistral AI, different architecture (custom GPU-optimized)
  - `qwen25-7b-q4k-gpu` - Alibaba, multilingual training (custom GPU-optimized)
  - `gemma2-9b-q4k-gpu` - Google, grouped-query attention (custom GPU-optimized)
  - `gpt-oss-20b-gpu` - 20B parameters (custom GPU-optimized, near VRAM limit)
  - **Note:** Custom models configured with `num_gpu=999` to force maximum GPU layer offloading for optimal performance
- **Performance:** ~72 items/min (7-9B models), ~35-45 items/min (20B models) on RTX 5080.
- **Docs:** `docs/windows.md`, `docs/troubleshooting_windows_local.md`, `docs/performance.md`, `docs/local_multi_prompt_workaround.md`.

## Agent Tips
- Prefer `make smoke` and `scripts.smoke_orchestration` to validate changes quickly; keep seeds/config deterministic.
- Multi-trial sweeps write shared control batches once per `(model, dataset partition, prompt, sampling config)`. Downstream steps hydrate controls via the registry in `control_registry.json` rather than re-running control inference per trial.
- When editing stats/report schemas, document fields in README and keep backward compatibility when feasible.
- Avoid network actions unless necessary; ask for approval when required.
- For local backend multi-model/multi-prompt runs on Windows, use `run_all_prompts.ps1` to avoid trial isolation issues.
### Shared Control Cache
- Registry refresh on startup/resume cleans/prunes stale entries before submissions.
- Producers export control JSONL + metadata into `experiments/run_<RID>/shared_controls/<key>/` and mark registry entries `completed`.
- When manifest entries hit an existing key, the submit step marks `mode: reuse` and skips new control jobs; downstream consumers hydrate rows directly from the shared cache (no duplicate scoring rows).
- Resume sanitization re-queues only missing controls. To force regeneration, delete `shared_controls/<key>/` plus the matching registry entry and rerun with `--resume`.
- Smoke validation: run two trials sharing a control configuration and confirm only one control job launches; rerun with `--resume` after deleting a cached JSONL to ensure regeneration occurs.
- Use `scripts.shared_controls.refresh_registry` in ad-hoc tooling if you need to inspect or repair registries outside the orchestrator.

## Fireworks CLI (firectl)
- Overview: Fireworks’ official CLI for managing datasets, batch inference jobs, deployments, models, and LoRA adapters. Useful for quick, manual ops or debugging alongside our scripted pipeline. Docs: https://fireworks.ai/docs/tools-sdks/firectl/firectl
- Install:
  - macOS/Homebrew: `brew install fw-ai/firectl/firectl` (or `brew tap fw-ai/firectl && brew install firectl`).
  - Verify: `firectl version`.
- Auth:
  - `firectl auth login` to authenticate via browser, or set `FIREWORKS_API_KEY` in env (loaded automatically from `.env`).
  - Prefer account selection by slug: pass `--account-id=<slug>` when needed (avoid empty shell expansions).
- Common commands (discover via `firectl --help`):
  - Datasets: `firectl create dataset`, `firectl list datasets`, `firectl get dataset <id_or_name>`.
  - Batch jobs: `firectl list batch-jobs`, `firectl get batch-job <id_or_name>` (names may vary by version).
  - Deployments/Models: `firectl create deployment|model`, `firectl list deployments|models`, `firectl update ...`, `firectl delete ...`.
  - LoRA: `firectl load-lora`, `firectl unload-lora`.
- How it maps to this repo:
  - Dataset upload: alternative to `python -m fireworks.upload_dataset`. Use `firectl create-dataset` to upload prepared JSONL and capture the dataset ID.
  - Start batch inference: alternative to `python -m fireworks.start_batch_job`. Use `firectl create batch-inference-job` (or your version’s equivalent) with your model/deployment and dataset ID.
  - Monitor jobs: `firectl list batch-jobs` and `firectl get batch-job <id>`; logs may be available via a logs subcommand in some versions.
  - Download results: our scripts `fireworks/poll_and_download.py` and `fireworks/parse_results.py` remain the primary path to pull and normalize outputs. Use CLI to retrieve job IDs and status if preferred.
- Suggested workflows:
  - Quick manual run:
    1) `make build` (or `python -m scripts.build_batches ...`).
    2) `firectl create-dataset ...` to upload the JSONL.
    3) `firectl create batch-inference-job ... --dataset-id <id> ...`.
    4) Poll with `firectl get batch-job <id>`. When complete, download results with `firectl download dataset <outputDatasetId> --output-dir results/raw_download`, then parse via `python -m fireworks.parse_results --job_id <id> --out_dir results/`.
  - Orchestrated run (CLI‑assisted): run `python -m scripts.run_all ... --skip_prepare --skip_build` if you’ve already uploaded the dataset via CLI; provide the dataset/job IDs with the script flags if applicable.
- Tips and caveats:
  - Keep prepared inputs in the same JSONL schema our scorers expect; the CLI does not transform inputs.
  - Use `--max-concurrent-jobs` in our orchestrator to control throughput; firectl itself doesn’t orchestrate multi‑part splitting for you.
  - Cleanup: `firectl delete-resources` supports removing obsolete datasets/deployments/jobs; prefer deleting test artifacts after smoke runs.

### Downloading results via CLI
- Fast path: `firectl download dataset <outputDatasetId> --output-dir results/raw_download`
  - Optional: `--download-lineage` to download the entire lineage chain of related datasets.
- Python helper (does polling + extraction): `python -m fireworks.poll_and_download --account <account_slug> --job_name <job_id_or_name> --out_dir results/raw_download`.
  - This polls, resolves `outputDatasetId`, downloads from `externalUrl`, extracts JSONL(s), and writes a combined `results.jsonl`.

## Commit & Pull Request Guidelines
- Commits: concise, imperative summaries (e.g., "Improve batch splitting", "Update documentation for temperature=1.0 experiment").
- PRs must include:
  - What changed and why; link related issues.
  - Affected config diff (`config/eval_config.yaml`) and prompt file paths.
  - Evidence: `make smoke` output, paths to generated artifacts (e.g., `experiments/run_<RUN_ID>/…`), and any updated reports.
- Security: copy `ENV_TEMPLATE.txt` to `.env`; never commit secrets. `python-dotenv` loads env vars at runtime.
  - `.env` is auto‑loaded; avoid passing empty shell expansions. Prefer `--account_id=slug` or omit.
