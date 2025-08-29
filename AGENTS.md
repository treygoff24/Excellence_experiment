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
- Environment: `python -m venv .venv && source .venv/bin/activate && pip install -r requirements.txt`
- Make targets:
  - `make venv` — create venv and install deps.
  - `make data` — download/prepare datasets to `data/prepared/`.
  - `make build` — create batch inputs in `data/batch_inputs/`.
  - `make eval` — full pipeline via `scripts/run_all.py`.
- `make smoke` — small end‑to‑end run for sanity checks.
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
- Optional robustness models: `scripts/mixed_effects.py` (GEE/OLS).

## Artifacts and Reports
- `results/significance.json` (schema_version=2) consolidates all stats.
- Optional extras: `unsupported_sensitivity.json`, `mixed_models.json`, `power_analysis.json`, `cost_effectiveness.json`.
- `scripts/generate_report.py` renders all sections. Use the smoke config (temps) to align significance sections with outputs.

## Large Files Policy
- Do not commit `results/` or `reports/` directories. These are ignored recursively by `.gitignore`.
- Per‑run directories under `experiments/run_*/` and `experiments/**/batch_inputs/` are also ignored.
- If you must share artifacts, upload them externally and link paths in PRs.

## Agent Tips
- Prefer `make smoke` and `scripts.smoke_orchestration` to validate changes quickly; keep seeds/config deterministic.
- When editing stats/report schemas, document fields in README and keep backward compatibility when feasible.
- Avoid network actions unless necessary; ask for approval when required.

## Commit & Pull Request Guidelines
- Commits: concise, imperative summaries (e.g., "Improve batch splitting", "Update documentation for temperature=1.0 experiment").
- PRs must include:
  - What changed and why; link related issues.
  - Affected config diff (`config/eval_config.yaml`) and prompt file paths.
  - Evidence: `make smoke` output, paths to generated artifacts (e.g., `experiments/run_<RUN_ID>/…`), and any updated reports.
- Security: copy `ENV_TEMPLATE.txt` to `.env`; never commit secrets. `python-dotenv` loads env vars at runtime.
  - `.env` is auto‑loaded; avoid passing empty shell expansions. Prefer `--account_id=slug` or omit.
