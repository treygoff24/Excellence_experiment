Task Checklist

- Review orchestration, manifests, and queue lifecycle in the current codebase.
- Identify architectural updates for a run-level state machine and per-trial stage status.
- Specify CLI surface for resume/stop and step-scoped execution (start/end/only).
- Propose persistent state schemas and recovery/idempotency strategies.
- Define error handling for failed resumes, invalid steps, and corrupted state.
- Outline testing/validation including smoke interrupts and migration from older runs.

### 1. Introduction

Robust stop/resume is essential for long, multi-trial experiments that orchestrate dataset uploads, batch job submission, polling/downloading, and post-processing across parsing, scoring, statistics, costs, and reporting. Interruptions may come from user-initiated stops, failures, or preemptions. The resumption must be precise: no duplicate submissions, no skipped work, and no rework beyond what is strictly necessary.

Goals
- Simple command to resume a specific run exactly at the interruption point.
- Ability to resume any specified individual step within the `run_all` process (e.g., only parse/score/stats).
- Idempotent execution at every phase; safe to re-run with `--resume` without duplicating jobs or corrupting outputs.
- Graceful stop: finish in-flight safe operations, persist state, do not submit new jobs.
- Clear observability via run-level and trial-level manifests with validated schemas and atomic writes.

Non-goals
- Canceling remote jobs (can be added later; initial implementation treats stop as local-only and lets remote jobs complete).
- Deep external orchestration (e.g., external workflow engines). The solution stays within the repo’s orchestration.

### 2. High-Level Approach

Architecture updates
- Introduce an explicit run-level state machine (Run Phases) and a durable `run_state.json` at `experiments/run_<RUN_ID>/`.
- Extend per-trial `trial_manifest.json` to v2 with stage statuses beyond job submission (downloaded, parsed, scored, stats, costs, report, archived), plus timestamps and output paths.
- Add a cooperative stop mechanism: a `StopToken` checked across long loops and a `stop_run` file flag; SIGINT/SIGTERM set the token.
- Enhance `QueueManager` to respect a `stop_event` (do not submit more work; allow running jobs to complete and download their results; persist states).
- Standardize idempotency checks for each phase using deterministic file paths, content checksums, and schema-validated manifests.
- Step gating in `run_all.py`: `--from_step/--to_step/--only_step` to execute a subset of phases; `--resume` uses manifests to compute the precise worklist.

CLI surface
- Resume a run end-to-end from the exact interruption point: `python -m scripts.run_all --config ... --run_id <RID> --resume`.
- Resume a specific step or range: `--only_step parse` or `--from_step poll --to_step stats`.
- Graceful stop: `python -m scripts.stop_run --run_id <RID>` (or `touch experiments/run_<RID>/STOP_REQUESTED`).

Idempotency and recovery
- For every phase, derive a precise “already done” predicate from state + files. Rebuild/repair when files are missing but state indicates completion.
- When manifests are missing or corrupted, reconstruct state from on-disk outputs and remote job metadata; preserve the corrupted file aside and write a repaired manifest.

### 3. Step-by-Step Implementation Plan

3.1 State model and schemas

Run phases (ordered)
1) prepare
2) build
3) submit (dataset upload + batch job creation)
4) poll (poll jobs) + download (fetch results) [treated as one composite phase in gating]
5) parse
6) score
7) stats
8) costs
9) report
10) archive

Run-level state file: `experiments/run_<RUN_ID>/run_state.json`
- schema_version: 1
- run_id: string
- config_path: string, and `config_hash` (sha256 of canonicalized effective config)
- phases: map of phase -> {status: not_started|in_progress|completed|failed|stopped, started_at, updated_at, last_error}
- trials: optional summary list of trial slugs with per-trial current phase and counts
- last_checkpoint: object with {phase, trial_slug, resume_key, timestamp}
- stop_requested: bool (derived from STOP_REQUESTED file if present)
- locks: optional info to aid debugging (who wrote last)

Per-trial manifest v2: `experiments/run_<RID>/<trial-slug>/results/trial_manifest.json`
- schema_version: 2
- created_utc, run_id, trial metadata (existing fields retained)
- temps, samples_per_item, prompts (existing)
- datasets: {"t{temp}_{cond}": [dataset_id_per_part]}
- jobs: {"t{temp}_{cond}": [job_name_per_part]}
- job_status: {"t{temp}_{cond}_pXX": pending|submitted|running|completed|failed|downloaded}
- stage_status: map with keys: downloaded, parsed, scored, stats, costs, report, archived → {status, updated_at, artifacts: {...}}
  - downloaded.artifacts: per-part results presence summary
  - parsed.artifacts: predictions_csv path
  - scored.artifacts: per_item_scores_csv path
  - stats.artifacts: significance_json path
  - costs.artifacts: costs_json path
  - report.artifacts: report_md path

Atomic persistence
- Reuse `write_manifest()` (already atomic with `fcntl`) for both `trial_manifest.json` and `run_state.json` (add a sibling `write_state()` helper or reuse the same function for arbitrary state files).
- Always bump `updated_at`; record `last_error` on failures.

Migration and compatibility
- If `trial_manifest.json` lacks `schema_version`, treat as v1 and upgrade in-memory by inferring `stage_status` from files and `job_status`.
- `run_state.json` is new; generate it on first run or on first resume if missing.

3.2 Cooperative stop

StopToken
- Implement a `StopToken` with: `set()`, `is_set()`, and `check()` (raises `StopRequested` for fast unwind in outer loops). Backed by an in-memory flag updated by:
  - OS signals (SIGINT, SIGTERM) → handler sets token and writes `STOP_REQUESTED` file under run root.
  - Presence of `experiments/run_<RID>/STOP_REQUESTED` file (polled periodically).

QueueManager updates
- Add an optional `stop_event: threading.Event | None` (or a callable `should_stop() -> bool`).
- Before submitting a new job: if `stop_event.is_set()`: do not submit; break out to the next loop tick.
- In the main loop: if stop is set and there are no running jobs: exit immediately. If jobs are running, keep polling and downloading until all currently running jobs are terminal, then exit.
- Emit a progress event `{event: 'stopped'}` when exiting due to a stop; `run_all` updates manifests/run_state accordingly.

CLI for stop
- New `scripts/stop_run.py --run_id <RID>`:
  - Resolves `experiments/run_<RID>`, writes `STOP_REQUESTED`, and prints status.
  - Optional `--force` to set a second flag `STOP_NOW` that tells the orchestrator to exit without waiting for running jobs (best-effort persistence still applied).

3.3 CLI design and examples

Augment `scripts/run_all.py` with phase gating
- New flags:
  - `--from_step {prepare|build|submit|poll|parse|score|stats|costs|report|archive}`
  - `--to_step {prepare|build|submit|poll|parse|score|stats|costs|report|archive}`
  - `--only_step <step>` (shorthand for from==to)
  - `--resume` (existing) gains stronger semantics: probe `run_state.json` + per-trial manifests to construct the minimal worklist; skip already done work at every phase.
  - `--trial_filter <slug_regex>` (optional) to scope work to specific trials.
  - `--cond_filter {control|treatment}` (optional) to scope work to a condition.

Examples
- Resume an entire interrupted run: `python -m scripts.run_all --config config/eval_config.yaml --run_id r20250905142300 --resume`
- Only re-run stats and report: `python -m scripts.run_all --config config/eval_config.yaml --run_id r... --from_step stats --to_step report --resume`
- Re-run parse/score for treatment only: `python -m scripts.run_all --config config/eval_config.yaml --run_id r... --only_step parse --cond_filter treatment --resume`

Convenience wrapper (optional)
- New `scripts/resume_run.py --run_id <RID> [--only_step ...]` that discovers `effective_config.yaml` under the run root and dispatches to `scripts.run_all` with appropriate flags.

3.4 Phase-specific implementation details

General
- Compute `effective_config.yaml` (already present) and `config_hash = sha256(canonical_yaml)` once; store in `run_state.json` for mismatch warnings.
- At start, if `--resume`, load `run_state.json` (if exists) and all per-trial manifests; derive which trials/conditions/parts and phases remain.
- Always persist phase transitions in `run_state.json`: set phase status to `in_progress` on entry, to `completed` on success, to `failed` on exception; include `last_error`.

prepare
- Target: `scripts/prepare_data.py` via existing subprocess call.
- Done predicate: presence of `prepared_dir/open_book.jsonl` and `prepared_dir/closed_book.jsonl` with non-zero size.
- Resume behavior: if done and `--resume`, skip; else re-run.

build
- Target: `scripts/build_batches.py` per prompt set with the union of temps across trials (existing logic).
- Done predicate: for each needed `(prompt_set, temp)` the files `t{temp}{suffix}_control.jsonl` and `t{temp}{suffix}_treatment.jsonl` exist and match expected counts (optional: store file hashes in `run_state.json` to avoid false positives).
- Resume: skip already built; re-build missing.

submit (dataset upload + job creation)
- Target: Extend existing per-trial loop in `scripts/run_all.py`.
- Done predicate: for each `resume_key = t{temp}_{cond}`, all parts (as determined by split) have `job_status[jkey]=='submitted'|'running'|'completed'|'downloaded'` and `jobs[resume_key]` has names for each part.
- Resume: per-part skipping already completed+downloaded (existing code checks both status and results file). For parts marked completed but missing results, reattempt download or re-submit if job name is missing.
- Persist: update `trial_manifest.stage_status.submit = {status: completed}` when all parts have at least submitted; update to `downloaded` later.

poll + download
- Target: `QueueManager.run_queue()` and `_process_job()` path in `run_all.py`.
- Behavior changes: pass `stop_event`; upon completion of a job, record `downloaded` status in `job_status`; write results under `results/t{temp}_{cond}_pXX/results.jsonl`.
- Done predicate: for each `jkey`, `results.jsonl` exists with >0 lines; set `stage_status.downloaded.status=completed` when all parts meet this.
- Resume: if `OUTPUT_DATASET_ID.txt` exists but no results are present, call a repair path to download via `externalUrl` or CLI fallback (`firectl`), then re-check.

parse
- Target: `fireworks.parse_results` once per trial; input is combined JSONL already produced (existing aggregator writes `results_combined.jsonl`).
- Done predicate: `results/predictions.csv` exists and matches the number of unique custom_ids in combined JSONL.
- Persist: `stage_status.parsed` with artifact path.
- Resume: if `predictions.csv` missing or mismatched, re-run parse.

score
- Target: `scoring.score_predictions` per trial.
- Done predicate: `results/per_item_scores.csv` exists; optional checksum to guard corruption.
- Persist: `stage_status.scored` with CSV path.
- Resume: re-run if missing; idempotent.

stats
- Target: `scoring.stats` per trial (with try/except in case optional dependencies are absent, as today).
- Done predicate: `results/significance.json` exists with `schema_version>=2`.
- Persist: `stage_status.stats` with JSON path.
- Resume: re-run if missing or legacy schema.

costs
- Target: `scripts/summarize_costs.py` or equivalent in-run computation; persist to `results/costs.json`.
- Done predicate: `results/costs.json` exists with expected keys.
- Resume: re-run if missing.

report
- Target: `scripts/generate_report.py` per trial.
- Done predicate: `reports/report.md` exists; update if inputs newer (mtime comparison or hash based) or if `--force`.
- Resume: re-run on demand.

archive
- Target: `scripts/archive_run.py` when `--archive` set, or as an explicit `only_step`.
- Done predicate: `experiments/run_<RID>/aggregate_report.md` and per-trial folders populated; `archive_manifest.json` present.
- Resume: safe to re-run; script moves artifacts under experiments root.

3.5 Resume controller flow in `scripts/run_all.py`

1) Parse flags; compute `phases_to_run` from `from_step/to_step/only_step`.
2) Resolve `run_root` under `experiments/` and load `run_state.json` (if present). Compute `config_hash` for mismatch warning.
3) Enumerate trials via `_expand_trials(cfg, args)` and reconcile with on-disk trials under the run root. If `--trial_filter`, limit set.
4) For each phase in order:
   - If `--resume`, compute worklist from trial manifests and file existence. Mark run_state[phase]=in_progress.
   - Execute the phase while periodically calling `StopToken.check()` to allow graceful stop.
   - On success, mark run_state[phase]=completed; on exception, mark failed and exit non-zero.
5) At the end, write a multi-trial summary as today and set run_state to completed for all phases actually run.

3.6 Error handling and recovery

Failed resumes / invalid steps
- Validate step names early; print allowed values and exit 2 on invalid input.
- If `--resume` but neither `run_state.json` nor any `trial_manifest.json` exists for the given `--run_id`, print a helpful message and suggest `--plan_only` or a fresh run.

Corrupted manifests
- On JSON parse error: backup the broken file to `*.corrupt.<ts>`; attempt reconstruction from on-disk outputs and partial manifests.
- If reconstruction is incomplete (e.g., missing job names), mark `job_status` as unknown and schedule a re-fetch of job metadata using `OUTPUT_DATASET_ID.txt` or skip with a warning if impossible.

Remote/job inconsistencies
- If a part is `completed` in manifest but has no `results.jsonl`, attempt best-effort download via `outputDatasetId` or CLI; otherwise downgrade to `submitted` and re-queue if inputs are available and re-submission is safe.

Stop during phases
- If stopping during submission: finish the current submission call, persist, and stop queuing new parts.
- If stopping during polling: do not abandon running jobs; keep polling until jobs finish (soft-stop). If `STOP_NOW` is set, exit after persisting current state.
- Always persist `run_state.last_checkpoint` and per-trial `stage_status`.

3.7 Testing and validation

Smoke and orchestration
- Extend `scripts/smoke_orchestration.py` to: (a) run in `--dry_run`, (b) send a stop signal mid-queue, (c) assert that resumption with `--resume` completes without re-submitting or duplicating outputs.

Unit tests (pytest; focused scope)
- State machine: transitions, gating logic from `from_step/to_step/only_step`.
- Manifest upgrade: v1 → v2 inference from files and existing fields.
- Idempotency predicates per phase (e.g., parse/score done detection).
- StopToken behavior: signals set the flag; loops observe and break.

Integration tests (optional, behind a marker)
- End-to-end dry-run with forced interruption at multiple checkpoints (after build, mid-submission, mid-polling, pre-parse) and resume to completion.

Operational validation
- `make smoke` and `scripts.smoke_orchestration` runs with `--keep` for manual inspection.
- Config validation remains as-is; add a `python -m scripts.run_all --plan_only` view that reflects phase gating and resume-derived worklists.

Documentation
- Update README and EXPERIMENT_WORKFLOW.md: new CLI flags, stop/resume semantics, state file locations, and troubleshooting.

### 4. Potential Challenges and Mitigations

- Concurrency and race conditions: Use `write_manifest()` with `fcntl` locks for atomic writes; avoid partial state; ensure one writer per run by acquiring a run-level lock file under the run root.
- Duplicate submissions on resume: Derive per-part work strictly from `job_status` + results presence; require both to treat a part as done. Keep dataset/job naming stable using existing `display_name` patterns.
- Legacy runs without `run_state.json`: Auto-generate `run_state.json` on first resume by scanning per-trial manifests and files. Tag as `migrated: true` for awareness.
- Corrupted or partial downloads: Keep `OUTPUT_DATASET_ID.txt` breadcrumbs; add a “repair download” pass before re-submission; tolerate both `externalUrl` and CLI paths.
- Config drift across resumes: Store and compare `config_hash`; warn and require `--force` to proceed if hash mismatch is detected.
- Optional dependencies (e.g., SciPy/statsmodels): Preserve current try/except flow; record partial completion in `stage_status` with `last_error` for missing extras; allow resume once dependencies are installed.

Appendix: Snippets

Example: stop-aware loop in QueueManager (conceptual)
```
while True:
    if stop_event and stop_event.is_set():
        if not running_jobs:
            break
        # poll running jobs to completion, then break
    submit_more_if_possible()
    completed = check_running_jobs()
    for j in completed: download_results(j)
    if all_done: break
    sleep(...)
```

Example: CLI usage
```
# Resume a run fully
python -m scripts.run_all --config config/eval_config.yaml --run_id r20250905142300 --resume

# Resume only stats and reporting
python -m scripts.run_all --config config/eval_config.yaml --run_id r20250905142300 --from_step stats --to_step report --resume

# Request a graceful stop from another terminal
python -m scripts.stop_run --run_id r20250905142300
```

