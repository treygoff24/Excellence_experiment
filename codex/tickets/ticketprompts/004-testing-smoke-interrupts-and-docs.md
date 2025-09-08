Execution Prompt — Ticket 004: Smoke interrupts, unit tests, and docs for resume

You are a GPT‑5 coding agent in the Codex CLI. Use plan–execute–verify loops, tools to inspect/modify files, and complete all acceptance checks. Emphasize determinism, structured outputs, and explicit, actionable errors.

Critical rules (repeat):
- Persist until done; read before changing; no guessing.
- Use `update_plan`, `shell`, and `apply_patch` to work; keep changes focused.
- Determinism: fixed seeds, fast tests, network skipped by default.
- Style: PEP 8, type hints for public functions, absolute imports.
- Repo hygiene: do not commit `results/` or `reports/`; never commit secrets.

Branching
- Create and work on `feat/gpt5-004-testing-smoke-interrupts-and-docs` based on the prior ticket branch, NOT main.
  - Example: `git fetch && git checkout -b feat/gpt5-004-testing-smoke-interrupts-and-docs feat/gpt5-003-resume-cli-migration-and-tooling`

Goal
- Validate and document the resume/stop system: add smoke tests with forced interruptions, focused unit tests for state/gating/manifest upgrade, and documentation updates covering new flags, state files, and troubleshooting.

Primary files to add/modify
- Update: `scripts/smoke_orchestration.py` — `--dry_run`, inject stop mid‑queue, assert clean resume
- New: `tests/` — unit tests for state machine transitions, gating flags, manifest v1→v2 upgrade, idempotency predicates, and StopToken behavior
- Update: `README.md`, `EXPERIMENT_WORKFLOW.md` — new flags, semantics, file locations, plan‑only, migration guidance
- Update: `Makefile` — update `make smoke` to run dry‑run orchestration; optional `make plan` for `--plan_only`

Smoke orchestration
- Extend to support `--dry_run`; inject a stop after specific checkpoints (post‑build, mid‑submission, mid‑polling, pre‑parse).
- Ensure `--resume` completes without duplicates or missing outputs; capture minimal artifacts for assertions.

Unit tests (pytest)
- Fast, deterministic, no network by default. Skip tests gracefully when optional deps are missing.
- Cover:
  - State machine transitions and gating from `--from_step/--to_step/--only_step`.
  - Manifest v1→v2 upgrade from files.
  - Idempotency predicates for parse/score/stats.
  - StopToken behavior under signals.

Documentation and DX
- Update README and EXPERIMENT_WORKFLOW with new flags (`--resume`, `--from_step/--to_step/--only_step`, `--plan_only`), state/manifest schema locations, stop semantics, and migration guidance.
- Add quick tips for Fireworks CLI fallback download; emphasize deterministic seeds and structured outputs.

Implementation plan
1) Add/extend tests under `tests/` focused on logic you introduced in tickets 001–003.
2) Extend `smoke_orchestration.py` with `--dry_run`, injected stop, and assertions for presence/shape of artifacts.
3) Update docs and Makefile targets (`make smoke`, optional `make plan`).
4) Run tests locally and fix failures; ensure skips are explicit and logged.

Acceptance checks (must pass)
- `python -m scripts.smoke_orchestration --n 3 --prompt_set operational_only --dry_run` runs, injects a stop, resumes, and finishes without duplicates; temp run artifacts pass checks.
- New pytest suite passes locally (no network); tests for gating logic, manifest upgrade, idempotency detection, and StopToken.
- Docs updated with new flags, semantics, schemas, plan‑only, and troubleshooting.
- Best practices: determinism, observability, explicit acceptance.

Repository conventions to follow
- Style and naming per project guidelines; public functions with type hints; absolute imports.
- Do not commit artifacts; keep seeds deterministic.

Validation steps to run before finishing
- `pytest -q`
- `python -m scripts.smoke_orchestration --n 3 --prompt_set operational_only --dry_run`
- `make smoke` if applicable

Deliverables
- Updated `scripts/smoke_orchestration.py`, docs, Makefile; new `tests/` suite.
- Log progress in `./codex/logs/004.md` with key commands and outcomes.

Commit and PR guidance
- Small, focused commits; include summary of test runs in commit footers when useful.
