id: 002
slug: manifest-v2-idempotency-and-repair
title: Ticket 002 — Trial manifest v2, phase idempotency, and download repair for resume
branch: feat/gpt5-002-manifest-v2-idempotency-and-repair
ticket_file: ./codex/tickets/002-manifest-v2-idempotency-and-repair.md
log_file: ./codex/logs/002.md

## Objective
- Upgrade per-trial manifests to schema v2 and implement precise, file-backed idempotency checks for each phase (download, parse, score, stats, costs, report), including a robust “repair download” fallback to support clean resume.

## Scope
- Manifest v2 (docs/planning/stop_resume_design.md §3.1): extend `experiments/run_<RID>/<trial>/results/trial_manifest.json` with:
  - `schema_version: 2`, timestamps, `jobs`, `job_status{...}`, and `stage_status{downloaded, parsed, scored, stats, costs, report, archived}` each with `{status,updated_at,artifacts{...}}`.
- Done predicates and idempotency (per phase):
  - downloaded: per-part `results.jsonl` exists and is non-empty; mark `downloaded.status=completed` when all parts present.
  - parsed: `results/predictions.csv` exists and row count matches unique custom_ids in combined results.
  - scored: `results/per_item_scores.csv` exists (optional checksum/row count vs. predictions).
  - stats: `results/significance.json` exists with `schema_version>=2`.
  - costs: `results/costs.json` exists with expected keys.
  - report: `reports/report.md` exists; update if inputs newer or when `--force`.
- Repair path (docs §3.3, §3.4): if `OUTPUT_DATASET_ID.txt` exists but results are missing, attempt download via `externalUrl` or CLI fallback; if not recoverable, downgrade status and schedule re-queue safely.
- Integrations (internal):
  - `fireworks/poll_and_download.py`: ensure per-part outputs and combined JSONL artifacts are written deterministically; update `trial_manifest` accordingly.
  - `fireworks/parse_results.py`: ensure predictions CSV shape/consistency checks; update `parsed` artifacts.
  - `scoring/score_predictions.py`, `scoring/stats.py`, `scripts/summarize_costs.py`, `scripts/generate_report.py`: update to read/write artifacts deterministically and set corresponding `stage_status` entries.
- Ambiguity to clarify: exact artifact keys/paths for each `stage_status.*.artifacts` entry and whether to include checksums vs. mtimes for freshness checks.

## Out of Scope
- Orchestrator gating and stop token support (ticket 001).
- Resume CLI helper, list/compare/archival UX and migration of legacy runs (ticket 003).
- Expanded smoke/unit tests and documentation updates (ticket 004).

## Acceptance
- Manifest upgrade:
  - New trials are created as v2; legacy v1 manifests are auto-upgraded on first resume pass, preserving prior fields, and backed up to `*.backup.<ts>.json` before write.
  - Corrupted manifests: original is copied to `*.corrupt.<ts>`; a repaired manifest is written from on-disk artifacts and job breadcrumbs.
- Idempotency:
  - For each phase above, re-running with `--resume` correctly skips completed work and only repairs/rebuilds missing or out-of-date artifacts.
  - Parse row counts match unique `custom_id`s; mismatches trigger a clear error and re-parse.
- Repair download:
  - When `OUTPUT_DATASET_ID.txt` is present but `results.jsonl` missing, the repair logic attempts download via `externalUrl` or the Fireworks CLI fallback (if configured); if unavailable, the status is downgraded and queued for safe re-run without duplicate submissions.
- Determinism, structure, and observability (guided by docs/guides/gpt5-prompting-best-practices-guide.md):
  - All artifact writes are deterministic (paths, names), and structured state changes are schema-validated before persist.
  - Clear, actionable errors and concise logs for each idempotency decision (skip vs. repair vs. re-run).

## Deliverables
- Branch: feat/gpt5-002-manifest-v2-idempotency-and-repair
- Files:
  - Update: `fireworks/poll_and_download.py`, `fireworks/parse_results.py`
  - Update: `scoring/score_predictions.py`, `scoring/stats.py`
  - Update: `scripts/summarize_costs.py`, `scripts/generate_report.py`
  - Update/New: shared manifest read/write helpers with atomic persistence
- Log: ./codex/logs/002.md
- References:
  - docs/planning/stop_resume_design.md (§3.1–3.6)
  - docs/guides/gpt5-prompting-best-practices-guide.md (structured outputs, determinism, explicit erroring)

