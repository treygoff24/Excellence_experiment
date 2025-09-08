Execution Prompt — Ticket 001: Run state, step gating, and graceful stop

You are a GPT‑5 coding agent working in the Codex CLI on this repository. Follow a plan–execute–verify loop, use tools to inspect and modify files, and keep going until the acceptance criteria pass. Prefer deterministic, structured outputs and explicit error messages.

Critical rules (repeat):
- Persist: plan, implement, validate, and iterate until done.
- Tools: use `update_plan`, `shell` for reading/running, and `apply_patch` for edits; don’t guess file contents.
- Determinism: stable paths, atomic writes, schema validation; avoid network unless required.
- Style: PEP 8, 4‑space indents, type hints for public functions, absolute imports.
- Repo hygiene: do not commit `results/`, `reports/`, or run artifacts; never commit secrets; `.env` is auto‑loaded.

Branching
- Create and work on `feat/gpt5-001-resume-state-and-gating` based on `main`.
  - Example: `git checkout main && git pull && git checkout -b feat/gpt5-001-resume-state-and-gating`

Goal
- Implement a durable run-level state machine (`run_state.json`), step gating flags (`--from_step/--to_step/--only_step`), a `--plan_only` view, and a cooperative stop mechanism that enables precise, idempotent resume across the pipeline.

Primary files to modify
- `scripts/run_all.py` — ordered phases, gating flags, `--resume`, `--plan_only`, state writes
- `fireworks/batch_queue_manager.py` — accept and honor a stop token/event
- New: `scripts/stop_run.py` — touch STOP_REQUESTED for a given `--run_id`
- New/Update: shared helper for atomic JSON writes and run-level lock handling

State model and schema (target; validate on save)
- `experiments/run_<RID>/run_state.json` (schema_version=1):
  - `schema_version: 1`
  - `run_id: <RID>`
  - `config_hash: <sha256 of effective config>`
  - `phases: {
      prepare: {status, started_at, updated_at, last_error},
      build: {…},
      submit: {…},
      poll: {…},
      parse: {…},
      score: {…},
      stats: {…},
      costs: {…},
      report: {…}
    }`
  - `last_checkpoint: {phase, trial_slug, resume_key, timestamp}`
  - `stop_requested: bool`

Atomicity and locking (no new deps)
- Writes: `json.dumps` → temp file in run root → `os.replace` to target for atomic swap.
- Lock: create a run-level `.state.lock` file in `experiments/run_<RID>/` and acquire an exclusive advisory lock.
  - Use `fcntl.flock` on POSIX; on non‑POSIX, use best‑effort exclusive create with retry backoff. Always release locks.

Gating and plan view
- Add argparse flags: `--resume`, `--plan_only`, `--only_step`, `--from_step`, `--to_step`.
- Compute a minimal worklist from current `run_state.json` plus on‑disk artifacts.
- `--plan_only` prints the ordered phases with computed gated actions and exits 0.

Graceful stop
- Introduce `StopToken` and read both a file sentinel `STOP_REQUESTED` under the run root and OS signals (SIGINT/SIGTERM).
- On stop, cease queuing new work, allow in‑flight jobs to finish, persist phase status as `stopped` or `completed` accordingly.

Missing state behavior
- When `--resume` is passed and `run_state.json` is missing: print a clear, actionable error and exit non‑zero, instructing the user to use the migration flow (ticket 003) or `--plan_only`.

Logging and errors
- Log concise, explicit messages for gating decisions (skip/execute), state writes, and stop handling; include allowed step names on invalid input.

Implementation plan
1) Inspect current orchestrator and queue code; outline ordered phases in `run_all.py` and a minimal state writer/reader util.
2) Implement state schema with atomic writes and lock helper; add `config_hash` computation.
3) Implement gating flags and minimal worklist construction consistent with on‑disk artifacts.
4) Implement `--plan_only` rendering, including per‑phase computed action.
5) Add `StopToken`, STOP_REQUESTED file watcher, and SIGINT/SIGTERM handling in `run_all.py`.
6) Pass stop token into `fireworks/batch_queue_manager.py`; ensure new submissions stop when requested.
7) Add `scripts/stop_run.py` to write STOP_REQUESTED for a `--run_id`.
8) Validate acceptance with smoke‑level runs; ensure idempotency on re‑runs.

Acceptance checks (must pass)
- CLI gating works and produces expected plan view:
  - `python -m scripts.run_all --config config/eval_config.yaml --run_id <RID> --resume`
  - `python -m scripts.run_all --config config/eval_config.yaml --run_id <RID> --only_step parse --resume`
  - `python -m scripts.run_all --config config/eval_config.yaml --run_id <RID> --from_step stats --to_step report --resume`
  - `python -m scripts.run_all --config config/eval_config.yaml --run_id <RID> --plan_only`
- State file semantics: contents and atomicity as described above; lock prevents concurrent writers; JSON is always valid.
- Graceful stop: `python -m scripts.stop_run --run_id <RID>` soft‑stops queuing, lets in‑flight jobs finish, updates phase status, and persists final state.
- Determinism & safety: resume is idempotent; explicit errors for invalid steps; missing state policy enforced.

Repository conventions to follow
- Style and naming per project guidelines; public functions with type hints; absolute imports.
- Do not commit run artifacts or secrets; keep seeds deterministic; prefer `make smoke` for quick validations.

Validation steps to run before finishing
- `python -m scripts.run_all --config config/eval_config.yaml --run_id <RID> --plan_only`
- Trigger STOP during a dry run and confirm clean stop + subsequent `--resume` continues correctly.
- `make smoke` (if applicable) to confirm no regressions.

Deliverables
- Updated `scripts/run_all.py`, `fireworks/batch_queue_manager.py`.
- New `scripts/stop_run.py` and a local lock/atomic JSON util.
- Log progress in `./codex/logs/001.md` using concise, timestamped entries.

Commit and PR guidance
- Make small, focused commits with imperative summaries (e.g., “Add run_state.json writer with atomic swap”).
- Include a short test log snippet or commands run in the commit message footer when useful.
