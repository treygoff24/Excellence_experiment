Execution Prompt — Ticket 003: Resume CLI, legacy migration, and run tooling

You are a GPT‑5 coding agent in the Codex CLI. Use a plan–execute–verify loop, tools for inspection and edits, and complete the acceptance criteria before finishing. Prefer structured outputs, determinism, and explicit errors.

Critical rules (repeat):
- Persist until done; don’t guess — read files first.
- Use `update_plan`, `shell`, and `apply_patch` appropriately.
- Determinism: canonical config hashing, atomic JSON writes, schema validation.
- Style: PEP 8, type hints for public functions, absolute imports.
- Repo hygiene: no artifacts or secrets committed.

Branching
- Create and work on `feat/gpt5-003-resume-cli-migration-and-tooling` based on the prior ticket branch, NOT main.
  - Example: `git fetch && git checkout -b feat/gpt5-003-resume-cli-migration-and-tooling feat/gpt5-002-manifest-v2-idempotency-and-repair`

Goal
- Provide a first‑class resume entrypoint and operational tooling: `scripts/resume_run.py`, legacy run migration into `run_state.json`, config drift detection, and UX in `list_runs`, `compare_runs`, and `archive_run` that respects run state.

Primary files to add/modify
- New: `scripts/resume_run.py`
- Update: `scripts/list_runs.py`, `scripts/compare_runs.py`, `scripts/archive_run.py`
- New/Update: shared config hashing helper (sha256 canonicalization of effective config)

Resume helper
- CLI: `python -m scripts.resume_run --run_id <RID> [--only_step/--from_step/--to_step] [--force] [--config path]`
- Behavior:
  - Discover `experiments/run_<RID>/effective_config.yaml` when available; otherwise respect `--config`.
  - Compute `config_hash`; compare with `run_state.json`.
  - On mismatch without `--force`, print a clear warning and exit non‑zero; with `--force`, proceed and record drift in state.
  - Dispatch to `scripts.run_all` with `--resume` and correct gating flags.

Legacy migration
- For runs lacking `run_state.json`, scan per‑trial directories (manifests/artifacts) to synthesize a minimal viable `run_state.json` with `migrated: true` and best‑effort phase statuses.
- Back up corrupted manifests to `*.corrupt.<ts>` and reconstruct where possible.
- Log any uncertainties with next steps for manual repair.

Operational UX
- `list_runs`: read `run_state.json` and display high‑level phase statuses (completed/in_progress/stopped/failed) plus `migrated` flag.
- `compare_runs`: show config hash equality, manifest schema versions, and highlight discrepancies.
- `archive_run`: permit only completed (or explicitly requested) archival; write `archive_manifest.json` with links to per‑trial artifacts.

Implementation plan
1) Implement a canonical config hashing helper (sha256 of a normalized YAML/JSON representation) and reuse in `run_all.py` if available.
2) Create `resume_run.py` to locate effective config, compute/compare hash, and invoke `scripts.run_all` with `--resume` and gating.
3) Add a migration routine: inspect trial directories, infer phase status from artifacts, and write `run_state.json` with `migrated: true`.
4) Update `list_runs`, `compare_runs`, `archive_run` to interact with `run_state.json` and emit structured output.
5) Validate config drift paths (fail without `--force`, proceed and record with `--force`).

Acceptance checks (must pass)
- `python -m scripts.resume_run --run_id <RID>` resumes the correct phases with no duplicate submissions.
- `--only_step`, `--from_step/--to_step` pass through correctly.
- On missing `run_state.json`, invoking `resume_run.py` creates a valid state file (`migrated: true`) by inspecting files; corrupted manifests are backed up; uncertainties are logged.
- `list_runs` shows per‑run statuses; `compare_runs` flags config/schema mismatches; `archive_run` respects state and emits `archive_manifest.json`.
- Clear, explicit errors; deterministic behavior; structured outputs.

Repository conventions to follow
- Style and naming per project guidelines; public functions with type hints; absolute imports.
- Do not commit artifacts; keep seeds deterministic.

Validation steps to run before finishing
- Create a test run, remove `run_state.json`, and validate that `resume_run.py` synthesizes state and resumes correctly.
- Exercise `list_runs`, `compare_runs`, and `archive_run` against two runs with small, controlled differences.

Deliverables
- New/updated scripts above and shared config hashing helper.
- Log progress in `./codex/logs/003.md` with key commands, outputs, and decisions.

Commit and PR guidance
- Small, focused commits; include validation notes in commit footers where useful.
