id: 201
slug: alt-orchestrator
title: Ticket 201 — Introduce alternative orchestrator with backend dispatch
branch: feat/gpt5-201-alt-orchestrator
ticket_file: ./codex/tickets/sprint-openai-anthropic-batch/201-alt-orchestrator.md
log_file: ./codex/logs/201.md

## Objective
- Stand up `scripts/alt_run_all.py` to reuse prepare/build phases, dispatch per-provider submit/poll/parse flows, persist provider IDs in manifests, and honor the shared control registry semantics.

## Scope
- Create `scripts/alt_run_all.py` exposing CLI parity with `scripts/run_all.py` plus `--backend {openai,anthropic}` and passthrough options (prepare/build/score/stats/report toggles).
- Wire in existing manifest/run-state helpers so stages `submitted`, `downloaded`, and `parsed` mirror Fireworks behavior while storing provider metadata (`batch_id`, `output_file_id`, `results_uri`).
- Integrate shared control registry hydration/reuse so control shards are produced once and reused across trials when the alternative orchestrator runs.
- Provide dry-run and resume semantics consistent with the current pipeline (`--dry_run`, `--resume`, `--skip_*`).
- Add argparse help/docs and ensure module imports remain absolute.

## Out of Scope
- Provider-specific submit/poll implementations (Tickets 202 & 203).
- Pricing tables or cost summarizer changes (Ticket 204).
- Smoke/golden validations that rely on external APIs (Ticket 205).

## Acceptance
- `python -m scripts.alt_run_all --help` exits 0 and lists backend flag plus prepare/build options.
- Running `python -m scripts.alt_run_all --backend openai --config config/alt_eval_config.yaml --dry_run --skip_prepare --skip_build --limit_items 2` produces manifest entries with provider metadata keys while leaving Fireworks pipeline untouched.
- Shared control registry updates occur exactly once per control key in a multi-trial dry run (add unit test or integration check using fixture registry).
- Lint (`ruff check scripts/alt_run_all.py`) and type check (`pyright scripts/alt_run_all.py`) pass.

## Deliverables
- Branch: feat/gpt5-201-alt-orchestrator
- Files: `scripts/alt_run_all.py`, updates to manifest/run-state utilities if needed, CLI docs (`docs/README.md` or dedicated guide reference).
- Log: ./codex/logs/201.md
