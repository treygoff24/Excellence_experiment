Execution Prompt — Ticket 005: Prepare/build idempotency and optional post‑processing

You are a GPT‑5 coding agent in the Codex CLI. Use plan–execute–verify loops, inspect and modify files with tools, and keep going until acceptance is met. Emphasize deterministic outputs, structured artifacts, and explicit, actionable logs.

Critical rules (repeat):
- Persist until done; read before changing; no guessing.
- Use `update_plan`, `shell`, and `apply_patch` for work.
- Determinism: stable shard naming, manifests, checksums; atomic writes.
- Style: PEP 8, type hints for public functions, absolute imports.
- Repo hygiene: do not commit `data/` artifacts, `results/`, or `reports/`; never commit secrets.

Branching
- Create and work on `feat/gpt5-005-prepare-build-and-optional-postprocessing` based on the prior ticket branch, NOT main.
  - Example: `git fetch && git checkout -b feat/gpt5-005-prepare-build-and-optional-postprocessing feat/gpt5-004-testing-smoke-interrupts-and-docs`

Goal
- Complete resume coverage by making `prepare` and `build` idempotent and integrating optional post‑processing scripts with artifact‑based done predicates so they participate cleanly in resume and step gating.

Primary files to add/modify
- `scripts/prepare_data.py` — deterministic outputs to `data/prepared/`; emit `prepared_manifest.json` with dataset list, counts, checksums; skip when valid.
- `scripts/build_batches.py` — deterministic sharding to `data/batch_inputs/`; emit `build_manifest.json` with shard counts and checksums; resume: rebuild missing shards only.
- `scripts/run_all.py` — set/read phase status for `prepare` and `build` using the new manifests.
- Optional analyses: `scripts/unsupported_sensitivity.py`, `scripts/mixed_effects.py`, `scripts/power_analysis.py`, `scripts/cost_effectiveness.py` — ensure idempotent outputs and (if gated) discoverable done predicates.

Manifests
- `prepared_manifest.json`: input datasets, item counts, checksums of outputs, created_at/updated_at.
- `build_manifest.json`: shard naming scheme, counts, checksums per shard, created_at/updated_at.
- Use atomic write then `os.replace` and validate JSON on save.

Optional post‑processing integration
- If enabled in config, `run_all` recognizes completion via JSON artifacts written under `experiments/run_<RID>/<trial>/results/`:
  - `unsupported_sensitivity.json`, `mixed_models.json`, `power_analysis.json`, `cost_effectiveness.json`
- If not enabled, standalone invocations still write deterministic outputs and are idempotent.

Implementation plan
1) Add manifest writers/readers for `prepared_manifest.json` and `build_manifest.json` with checksums and timestamps.
2) Make `prepare_data.py` and `build_batches.py` deterministic; on resume, skip when manifests are valid; repair only missing/invalid shards.
3) Update `run_all.py` to set/read phase status for `prepare`/`build` based on manifests.
4) Add/adjust optional analyses to be idempotent and (if configured) integrated with gating.
5) Validate idempotency: re‑run `--resume` and confirm minimal, targeted work.

Acceptance checks (must pass)
- Re‑running `prepare`/`build` with `--resume` performs no work when manifests/outputs are valid; missing shards/files are rebuilt only.
- Manifests include dataset/shard counts and checksums; invalid/partial outputs trigger clear repair/rebuild.
- When enabled in config, `run_all` recognizes done predicates for optional analyses and resumes/skips accordingly; when disabled, standalone tools are idempotent and deterministic.
- Determinism and logs: stable naming, structured JSON artifacts, explicit logs.

Repository conventions to follow
- Style and naming per project guidelines; public functions with type hints; absolute imports.
- Do not commit artifacts; keep seeds deterministic.

Validation steps to run before finishing
- Run `prepare` and `build`, then re‑run with `--resume` to verify no‑op behavior; delete a shard to confirm targeted rebuild.
- If optional analyses are gated, toggle config to validate skip/execute behavior.

Deliverables
- Updated scripts listed above; manifests and idempotent logic.
- Log progress in `./codex/logs/005.md` with key commands and outcomes.

Commit and PR guidance
- Small, focused commits; include brief validation notes in commit footers where useful.
