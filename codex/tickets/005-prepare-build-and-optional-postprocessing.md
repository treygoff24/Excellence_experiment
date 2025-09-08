id: 005
slug: prepare-build-and-optional-postprocessing
title: Ticket 005 — Prepare/build idempotency and optional post-processing resume integration
branch: feat/gpt5-005-prepare-build-and-optional-postprocessing
ticket_file: ./codex/tickets/005-prepare-build-and-optional-postprocessing.md
log_file: ./codex/logs/005.md

## Objective
- Complete resume coverage by making `prepare` and `build` phases idempotent and integrating optional post-processing scripts with artifact-based done predicates so they participate cleanly in resume and step gating.

## Scope
- Prepare/build idempotency:
  - `scripts/prepare_data.py`: write deterministic outputs to `data/prepared/` and emit a minimal manifest (e.g., `prepared_manifest.json`) with dataset list, counts, and checksums; skip when present and valid.
  - `scripts/build_batches.py`: deterministic sharding to `data/batch_inputs/` (stable names/ordering); write `build_manifest.json` with shard counts and checksums; resume: rebuild missing shards only.
  - Update `scripts/run_all.py` to set phase status for `prepare` and `build` using these manifests.
- Optional post-processing integration:
  - Ensure `scripts/unsupported_sensitivity.py`, `scripts/mixed_effects.py`, `scripts/power_analysis.py`, and `scripts/cost_effectiveness.py` are idempotent via clear inputs/outputs and are discoverable by `run_all` gating as optional steps (feature-flagged in config) or remain standalone with documented resume commands.
  - For gated integration (if enabled), add done predicates (JSON outputs in `experiments/run_<RID>/<trial>/results/`): `unsupported_sensitivity.json`, `mixed_models.json`, `power_analysis.json`, `cost_effectiveness.json` as applicable.
- Ambiguity to clarify: whether optional analyses should be first-class gated phases vs. invoked ad hoc; default enablement per config.

## Out of Scope
- Core orchestrator state/gating (ticket 001), manifest v2 and per-phase idempotency (ticket 002), resume CLI/migration (ticket 003), and tests/docs scaffolding (ticket 004).

## Acceptance
- Prepare/build:
  - Re-running `prepare`/`build` with `--resume` performs no work when manifests and outputs are valid; missing shards/files are rebuilt only.
  - Manifests include dataset/shard counts and checksums; invalid or partial outputs trigger a clear repair/rebuild, not a full rebuild.
- Optional post-processing:
  - When enabled in config, `run_all` recognizes done predicates for each optional analysis and resumes/skips accordingly.
  - When not enabled, standalone invocations succeed and write deterministic JSON artifacts; re-running is idempotent.
- Best-practice alignment (docs/guides/gpt5-prompting-best-practices-guide.md):
  - Deterministic outputs (stable naming), structured JSON artifacts, explicit acceptance checks, and concise, actionable logs.

## Deliverables
- Branch: feat/gpt5-005-prepare-build-and-optional-postprocessing
- Files:
  - Update: `scripts/prepare_data.py`, `scripts/build_batches.py`
  - Update: `scripts/run_all.py` (phase wiring for prepare/build; optional analyses if gated)
  - Update: `scripts/unsupported_sensitivity.py`, `scripts/mixed_effects.py`, `scripts/power_analysis.py`, `scripts/cost_effectiveness.py` (idempotent outputs)
- Log: ./codex/logs/005.md
- References:
  - docs/planning/stop_resume_design.md (§3.1 phases; idempotency guidance)
  - docs/guides/gpt5-prompting-best-practices-guide.md (structured outputs, determinism)

