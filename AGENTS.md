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
  - `make parse | score | stats | report` — post‑processing utilities.
- Direct run: `python -m scripts.run_all --config config/eval_config.yaml --archive`.
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

## Commit & Pull Request Guidelines
- Commits: concise, imperative summaries (e.g., "Improve batch splitting", "Update documentation for temperature=1.0 experiment").
- PRs must include:
  - What changed and why; link related issues.
  - Affected config diff (`config/eval_config.yaml`) and prompt file paths.
  - Evidence: `make smoke` output, paths to generated artifacts (e.g., `experiments/run_<RUN_ID>/…`), and any updated reports.
- Security: copy `ENV_TEMPLATE.txt` to `.env`; never commit secrets. `python-dotenv` loads env vars at runtime.
