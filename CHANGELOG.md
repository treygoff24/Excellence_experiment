# Changelog

All notable changes to this repository will be documented in this file.

## 2025-08-29 — Orchestrator robustness + docs

Enhancements
- Decoupled dataset splitting from concurrency in `scripts/run_all.py`.
  - New flags: `--parts_per_dataset N` and `--lines_per_part N`.
  - `--max_concurrent_jobs M` now controls only concurrency, not the number of parts.
- Resumable runs with early per‑trial manifests.
  - Manifests are written before queue execution and updated via progress callbacks.
  - `--resume` skips already completed parts/conditions.
- Improved download workflow with breadcrumbs.
  - Writes `*_OUTPUT_DATASET_ID.txt` when external URL isn’t immediately available; polling/combining can be re‑run.
- New dry‑run mode for offline iteration.
  - `--dry_run` synthesizes per‑part results from batch inputs; runs full parse/score/report locally (no network calls).
- Orchestration smoke test with auto‑cleanup.
  - `python -m scripts.smoke_orchestration --n 3 --prompt_set operational_only [--keep]`.
- Safer environment handling.
  - `.env` is auto‑loaded; prefer `--account_id=slug` or omit; avoid empty shell expansions (e.g., `--account_id "$FIREWORKS_ACCOUNT_ID"` if the shell var is unset).
- Git hygiene.
  - Ignored `experiments/run_*/` and `experiments/**/batch_inputs/` to avoid committing large artifacts.

Usage examples
- Treatment only across all prompt sets; split into 24 parts; run 4 at a time; resume‑safe; archive on finish:
  ```bash
  python -m scripts.run_all --config config/eval_config.yaml \
    --condition=treatment --parts_per_dataset=24 --max_concurrent_jobs=4 --resume --archive
  ```
- Plan only (no submissions):
  ```bash
  python -m scripts.run_all --config config/eval_config.yaml --plan_only
  ```
- Local, offline iteration (dry‑run):
  ```bash
  python -m scripts.run_all --config config/eval_config.yaml --dry_run \
    --prompt_sets operational_only --temps 0.0 --limit_items 200 --parts_per_dataset 3 --max_concurrent_jobs 2
  ```
- Orchestration smoke (end‑to‑end, auto‑cleanup by default):
  ```bash
  python -m scripts.smoke_orchestration --n 3 --prompt_set operational_only
  # add --keep to inspect outputs
  ```

Documentation
- Updated README.md with orchestrator controls and examples.
- Updated CLAUDE.md and AGENTS.md for agent guidance and tips.
- Updated EXPERIMENT_WORKFLOW.md to reflect run layout and controls.

Notes
- Some optional analyses (stats, power) require SciPy/Statsmodels; in environments without them the pipeline falls back gracefully for the smoke/dry‑run paths.
