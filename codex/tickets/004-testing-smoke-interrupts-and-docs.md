id: 004
slug: testing-smoke-interrupts-and-docs
title: Ticket 004 — Smoke interrupts, unit tests, and docs for resume
branch: feat/gpt5-004-testing-smoke-interrupts-and-docs
ticket_file: ./codex/tickets/004-testing-smoke-interrupts-and-docs.md
log_file: ./codex/logs/004.md

## Objective
- Validate and document the resume/stop system: add smoke tests with forced interruptions, focused unit tests for state/gating/manifest upgrade, and documentation updates (README, EXPERIMENT_WORKFLOW.md) covering new flags, state files, and troubleshooting.

## Scope
- Orchestrator smoke (docs/planning/stop_resume_design.md §3.7):
  - Extend `scripts/smoke_orchestration.py` to support `--dry_run`, inject a stop signal mid-queue, and verify that `--resume` completes without duplicate submissions or missing outputs.
  - Add checkpoints to interrupt after build, mid-submission, mid-polling, and pre-parse; capture artifacts for assertions.
- Unit tests (pytest):
  - State machine transitions and gating from `--from_step/--to_step/--only_step`.
  - Manifest v1→v2 upgrade from files; idempotency predicates for parse/score/stats; StopToken behavior under signals.
  - Place tests under `tests/` and keep them fast/deterministic (fixed seeds), skipping network operations by default.
- Documentation and DX:
  - Update README.md and EXPERIMENT_WORKFLOW.md with: new CLI flags, stop semantics, state/manifest file locations, plan-only mode, and migration guidance.
  - Add quick tips to use Fireworks CLI as a fallback for downloads; emphasize deterministic seeds and structured outputs per prompting best practices.
- Make targets:
  - Update `make smoke` to run the dry-run orchestration path; optionally add `make plan` for `--plan_only` preview.
- Ambiguity to clarify: acceptable runtime budget for smoke (default N and concurrency), and whether to gate network-dependent checks behind a feature flag/marker.

## Out of Scope
- Core orchestrator resume logic (ticket 001) and manifest/idempotency changes (ticket 002).
- Resume CLI helper and migration (ticket 003).

## Acceptance
- Smoke orchestration:
  - `python -m scripts.smoke_orchestration --n 3 --prompt_set operational_only --dry_run` runs, injects a stop, resumes, and finishes without duplicate submissions; artifacts under a temp run root pass presence/shape checks.
- Unit tests:
  - New pytest suite passes locally (no network); includes tests for gating logic, manifest upgrade, idempotency detection, and StopToken.
  - If optional deps (e.g., SciPy) are missing, tests skip gracefully; captured in test logs.
- Docs:
  - README and EXPERIMENT_WORKFLOW describe new flags (`--resume`, `--from_step/--to_step/--only_step`, `--plan_only`) and state/manifest schemas; troubleshooting includes config drift and corrupted manifest repair steps.
- Best-practice alignment (docs/guides/gpt5-prompting-best-practices-guide.md):
  - Tests and docs emphasize deterministic seeds, structured outputs, explicit acceptance checks, and a plan–execute–verify loop.

## Deliverables
- Branch: feat/gpt5-004-testing-smoke-interrupts-and-docs
- Files:
  - Update: `scripts/smoke_orchestration.py`
  - New: `tests/` for focused unit tests (state, gating, manifest upgrade, idempotency)
  - Update: `README.md`, `EXPERIMENT_WORKFLOW.md`, and `Makefile`
- Log: ./codex/logs/004.md
- References:
  - docs/planning/stop_resume_design.md (§3.7)
  - docs/guides/gpt5-prompting-best-practices-guide.md (determinism, observability, explicit acceptance)

