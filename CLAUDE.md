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
- Full pipeline: `python -m scripts.run_all --config config/eval_config.yaml --archive` (or `make eval`).
- Offline smoke: `make smoke` — generates a tiny end‑to‑end run locally.
- Post‑processing (run any subset):
  - `python -m scoring.stats --per_item_csv results/per_item_scores.csv --config config/eval_config.yaml --out_path results/significance.json`
  - `python -m scripts.unsupported_sensitivity --pred_csv results/predictions.csv --prepared_dir data/prepared --config config/eval_config.yaml --out_path results/unsupported_sensitivity.json`
  - `python -m scripts.mixed_effects --pred_csv results/predictions.csv --prepared_dir data/prepared --config config/eval_config.yaml --out_path results/mixed_models.json`
  - `python -m scripts.power_analysis --per_item_csv results/per_item_scores.csv --prepared_dir data/prepared --config config/eval_config.yaml --out_path results/power_analysis.json`
  - `python -m scripts.cost_effectiveness --pred_csv results/predictions.csv --per_item_csv results/per_item_scores.csv --config config/eval_config.yaml --out_path results/cost_effectiveness.json`
  - `python -m scripts.generate_report --config config/eval_config.yaml --results_dir results --reports_dir reports`

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
- Optional: `unsupported_sensitivity.json`, `mixed_models.json`, `power_analysis.json`, `cost_effectiveness.json`.
- Report: `reports/report.md` includes all of the above.

## Do/Don’t
- Do keep seeds fixed and respect validated config schemas.
- Do use `scripts/smoke_test.py` for quick iteration.
- Don’t commit large artifacts — any `results/`/`reports/` dirs are ignored recursively.
- Don’t change output schema keys casually; keep backward compatibility or update docs accordingly.

## Common Tasks
- “Upgrade stats” → edit `scoring/stats.py`, then update `scripts/generate_report.py` and README/AGENTS.
- “Adjust unsupported detection” → tweak `scoring/unsupported.py` and `unsupported` config.
- “Add a metric” → extend `scoring/score_predictions.py`, plumb through `stats.py`, and render in the report.

## Optional Dependencies
- `statsmodels` for mixed‑effects robustness: `pip install statsmodels`.

## Contact
- See EXPERIMENT_WORKFLOW.md and README for additional context on runs and organization.
