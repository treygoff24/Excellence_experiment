Execution Prompt — Ticket 002: Manifest v2, phase idempotency, and download repair

You are a GPT‑5 coding agent in the Codex CLI. Execute with a plan–execute–verify loop, use tools to inspect/modify files, and continue until acceptance criteria pass. Prefer structured outputs, deterministic behavior, and explicit errors.

Critical rules (repeat):
- Persist and iterate until done; no guessing.
- Use `update_plan`, `shell`, `apply_patch` prudently.
- Determinism: stable filenames/paths, atomic JSON writes, schema validation.
- Style: PEP 8, type hints for public functions, absolute imports.
- Repo hygiene: do not commit `results/` or `reports/`; never commit secrets.

Branching
- Create and work on `feat/gpt5-002-manifest-v2-idempotency-and-repair` based on the prior ticket branch, NOT main.
  - Example: `git fetch && git checkout -b feat/gpt5-002-manifest-v2-idempotency-and-repair feat/gpt5-001-resume-state-and-gating`

Goal
- Upgrade per‑trial manifests to schema v2 and add precise, file‑backed idempotency checks for each phase (download, parse, score, stats, costs, report), plus a robust repair path for downloads to support resume.

Primary files to modify
- `fireworks/poll_and_download.py` — deterministic per‑part outputs; update manifest
- `fireworks/parse_results.py` — predictions CSV checks; update manifest
- `scoring/score_predictions.py`, `scoring/stats.py` — deterministic outputs; manifest stage updates
- `scripts/summarize_costs.py`, `scripts/generate_report.py` — deterministic outputs; manifest stage updates
- Shared manifest read/write helpers with atomic persistence (new/update)

Manifest v2 schema (target; validate on save)
- `experiments/run_<RID>/<trial>/results/trial_manifest.json`:
  - `schema_version: 2`
  - `timestamps` (created_at, updated_at)
  - `jobs` and `job_status{...}` (if applicable)
  - `stage_status`:
    - `downloaded|parsed|scored|stats|costs|report|archived: {status: oneof[pending,in_progress,completed,stopped,failed], updated_at, artifacts:{...}}`
  - For each phase, populate `artifacts` with deterministic, relative paths and (where helpful) checksums or sizes.

Done predicates and idempotency (implement as explicit checks)
- downloaded: all per‑part `results.jsonl` exist and are non‑empty; mark completed when all present.
- parsed: `results/predictions.csv` exists and row count matches the unique `custom_id`s in combined results.
- scored: `results/per_item_scores.csv` exists; optionally verify row count aligns with predictions.
- stats: `results/significance.json` exists with `schema_version >= 2`.
- costs: `results/costs.json` exists with expected top‑level keys.
- report: `reports/report.md` exists; update if inputs are newer or when `--force` is set.

Repair path for downloads
- If `OUTPUT_DATASET_ID.txt` exists but expected outputs are missing, attempt to resolve via `externalUrl` or a Fireworks CLI fallback (if configured). If unrecoverable, downgrade status and schedule safe re‑queue without duplicate submissions.

Observability and errors
- Each idempotency decision logs: reason to skip/repair/re‑run; errors are explicit and actionable.

Implementation plan
1) Define a small manifest helper to read/validate/write schema v2 with atomic swap and timestamps; include backup on upgrade.
2) In `poll_and_download.py`, ensure deterministic per‑part file naming, combined JSONL writing, and proper `stage_status.downloaded` updates; implement repair logic.
3) In `parse_results.py`, output `predictions.csv` deterministically; compute and validate row counts; update `parsed` artifacts.
4) Wire `score_predictions.py`, `stats.py`, `summarize_costs.py`, `generate_report.py` to set/update the respective `stage_status` entries and ensure deterministic outputs.
5) Implement auto‑upgrade from v1 to v2 with `.backup.<ts>.json` and `.corrupt.<ts>` flows.
6) Re‑run phases with `--resume` to confirm only missing/out‑of‑date artifacts are rebuilt.

Acceptance checks (must pass)
- New trials are created as v2; legacy v1 manifests auto‑upgrade with a backup; corrupted files get copied to `*.corrupt.<ts>` and repaired manifests are written from artifacts.
- Re‑runs with `--resume` correctly skip completed phases and only repair/rebuild what’s missing/out‑of‑date.
- Parse row counts match unique `custom_id`s; mismatches trigger clear errors and re‑parse.
- Download repair: when `OUTPUT_DATASET_ID.txt` exists but outputs are missing, attempt recovery via `externalUrl` or CLI; otherwise downgrade status and schedule safe re‑run without duplicates.
- Determinism, structure, observability: deterministic artifact paths/names; schema‑validated writes; clear logs.

Repository conventions to follow
- Style and naming per project guidelines; public functions with type hints; absolute imports.
- Do not commit artifacts; maintain deterministic seeds; prefer `make smoke` for quick validations.

Validation steps to run before finishing
- Exercise `--resume` flows to confirm idempotent skipping across all phases.
- Intentionally delete a per‑part file to trigger repair and validate expected behavior.

Deliverables
- Updated files listed above and shared manifest helpers.
- Log progress in `./codex/logs/002.md` (timestamped entries; commands, outcomes, and follow‑ups).

Commit and PR guidance
- Small, focused commits with imperative summaries; include short validation notes in commit footers when useful.
