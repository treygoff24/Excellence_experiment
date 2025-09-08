id: 001
slug: resume-state-and-gating
title: Ticket 001 — Run state, step gating, and graceful stop for robust resume
branch: feat/gpt5-001-resume-state-and-gating
ticket_file: ./codex/tickets/001-resume-state-and-gating.md
log_file: ./codex/logs/001.md

## Objective
- Implement a durable run-level state machine (`run_state.json`), step gating flags (`--from_step/--to_step/--only_step`), and a cooperative stop mechanism to enable precise, idempotent resume across the full pipeline.

## Scope
- Orchestrator: extend `scripts/run_all.py` with ordered phases, gating flags, and `--resume` semantics that derive the minimal worklist from state + files (see docs/planning/stop_resume_design.md §3.1–3.6).
- State persistence: write `experiments/run_<RUN_ID>/run_state.json` (schema_version=1) with `phases{...}`, `last_checkpoint`, `stop_requested`, and `config_hash`; atomic writes with a run-level lock file.
- Graceful stop: introduce `StopToken` and honor both a `STOP_REQUESTED` file and SIGINT/SIGTERM; stop queuing new work, let in-flight jobs finish, persist state.
- Plan view: add `--plan_only` to render gated phases and the computed resume worklist without executing.
- Queue integration: pass `stop_event` into the queue path (e.g., `fireworks/batch_queue_manager.py`) to respect soft-stop logic.
- Dependencies (internal): `scripts/run_all.py`, `fireworks/batch_queue_manager.py`. Create `scripts/stop_run.py` (touches STOP_REQUESTED for a given `--run_id`).
- Ambiguity to clarify: required lock strategy (fcntl vs. portalocker), and exact run-level lock file name/location; acceptable behavior on missing `run_state.json` when `--resume` is passed (warn vs. fail vs. auto-migrate—see below Acceptance).

## Out of Scope
- Per-phase done predicates and `trial_manifest` v2 upgrade (ticket 002).
- Resume CLI helper, migration utilities, and list/compare integrations (ticket 003).
- Expanded testing/docs (ticket 004).

## Acceptance
- CLI gating: the following all work and produce the expected gated execution and plan view.
  - `python -m scripts.run_all --config config/eval_config.yaml --run_id <RID> --resume`
  - `python -m scripts.run_all --config config/eval_config.yaml --run_id <RID> --only_step parse --resume`
  - `python -m scripts.run_all --config config/eval_config.yaml --run_id <RID> --from_step stats --to_step report --resume`
  - `python -m scripts.run_all --config config/eval_config.yaml --run_id <RID> --plan_only`
- run_state schema and idempotency:
  - Creates/updates `experiments/run_<RID>/run_state.json` with `schema_version=1`, `phases{status,started_at,updated_at,last_error}`, `last_checkpoint{phase,trial_slug,resume_key,timestamp}`, and `config_hash` (sha256 of effective config).
  - Atomic writes: no partial JSON on crash; concurrent writers are prevented via a lock file. If a write fails, the file remains valid JSON.
- Graceful stop behavior:
  - `python -m scripts.stop_run --run_id <RID>` (or touch STOP_REQUESTED) triggers soft-stop: no new submissions; running jobs are allowed to complete; final state persisted with `phases[<phase>].status == 'stopped'` or `completed` as applicable.
- Determinism & safety (guided by docs/guides/gpt5-prompting-best-practices-guide.md):
  - Resume is idempotent: re-running with `--resume` does not submit duplicate jobs or redo completed work; outputs remain unchanged byte-for-byte when inputs and state are unchanged.
  - Errors are explicit and actionable (invalid step names list allowed values; missing `run_state.json` provides next steps).
- Missing-state policy (ambiguity resolution path):
  - When `--resume` is passed and `run_state.json` is absent, the command prints a clear message and exits non-zero, instructing to use the migration flow (ticket 003) or `--plan_only`. If product decision favors auto-migration, document and implement that instead.

## Deliverables
- Branch: feat/gpt5-001-resume-state-and-gating
- Files:
  - Update: `scripts/run_all.py` (phase gating, resume controller, plan_only, state writes)
  - Update: `fireworks/batch_queue_manager.py` (accept/observe stop event)
  - New: `scripts/stop_run.py` (create STOP_REQUESTED for a run)
  - New/Update: locking helper for atomic JSON writes (either local util or inline helper)
- Log: ./codex/logs/001.md
- References:
  - docs/planning/stop_resume_design.md (§2–3.6)
  - docs/guides/gpt5-prompting-best-practices-guide.md (structured outputs, plan/execute/verify loops, determinism, observability)

