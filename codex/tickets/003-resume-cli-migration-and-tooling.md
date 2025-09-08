id: 003
slug: resume-cli-migration-and-tooling
title: Ticket 003 — Resume CLI, legacy migration, and run tooling integrations
branch: feat/gpt5-003-resume-cli-migration-and-tooling
ticket_file: ./codex/tickets/003-resume-cli-migration-and-tooling.md
log_file: ./codex/logs/003.md

## Objective
- Provide a first-class resume entrypoint and operational tooling: a `scripts/resume_run.py` helper, legacy run migration into `run_state.json`, config drift detection, and UX in `list_runs`, `compare_runs`, and `archive_run` that understands run state.

## Scope
- Resume helper:
  - `scripts/resume_run.py --run_id <RID> [--only_step/--from_step/--to_step] [--force]`
  - Discovers `experiments/run_<RID>/effective_config.yaml` (or falls back to provided `--config`), computes/compares `config_hash`, and dispatches to `scripts.run_all` with `--resume` and proper gating flags.
- Legacy migration (docs/planning/stop_resume_design.md §4):
  - For runs without `run_state.json`, scan per-trial directories for manifests and artifacts; synthesize `run_state.json` with `migrated: true` and best-effort phase statuses; warn if uncertain.
- Config drift policy:
  - Store `config_hash` in `run_state.json` at creation; on resume, if mismatch and `--force` not set, print a clear warning and exit non-zero; with `--force`, proceed and record the drift in state.
- Operational UX integrations:
  - `scripts/list_runs.py`: display the high-level phase statuses derived from `run_state.json` (e.g., completed/in_progress/stopped/failed), and whether `migrated`.
  - `scripts/compare_runs.py`: include compatibility checks (config hash equality, manifest schema versions) and highlight discrepancies.
  - `scripts/archive_run.py`: read `run_state.json` to ensure only completed or explicitly requested archival proceeds; write an `archive_manifest.json` with links to per-trial artifacts.
- Ambiguity to clarify: canonical location for `effective_config.yaml` under the run root and whether to enforce its presence strictly vs. allowing `--config` override.

## Out of Scope
- State model, stop token, and gating in the orchestrator (ticket 001).
- Per-phase idempotency and manifest v2 implementation (ticket 002).
- Extended testing/docs (ticket 004).

## Acceptance
- Resume helper works:
  - `python -m scripts.resume_run --run_id <RID>` resumes from the correct phase(s) with no duplicate submissions.
  - `--only_step`, `--from_step/--to_step` are passed through correctly; `--force` overrides config drift after a clear warning.
- Migration works:
  - On runs lacking `run_state.json`, invoking `resume_run.py` creates a valid state file (`migrated: true`) by inspecting existing manifests/files; corrupted manifests are backed up and partially reconstructed; any uncertainties are logged.
- Tooling awareness:
  - `list_runs` shows per-run high-level statuses; `compare_runs` flags config/schema mismatches; `archive_run` respects state and emits `archive_manifest.json`.
- Best-practice alignment (docs/guides/gpt5-prompting-best-practices-guide.md):
  - Explicit, structured outputs (state JSON, archive manifest) and clear error messages; deterministic behavior; encourage clarifying signals (warnings with next-step guidance) instead of silent failure.

## Deliverables
- Branch: feat/gpt5-003-resume-cli-migration-and-tooling
- Files:
  - New: `scripts/resume_run.py`
  - Update: `scripts/list_runs.py`, `scripts/compare_runs.py`, `scripts/archive_run.py`
  - Update/New: shared config hashing helper (sha256 canonicalization)
- Log: ./codex/logs/003.md
- References:
  - docs/planning/stop_resume_design.md (§3.6–§4)
  - docs/guides/gpt5-prompting-best-practices-guide.md (structured outputs, explicit controls, observability)

