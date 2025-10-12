# Shared Control Reuse Initiative

## Background & Problem Statement

Multi-trial prompt sweeps currently re-run the control arm for every trial, even when the control inputs, prompt, model, and sampling parameters are identical. Because control runs use deterministic settings (temperature 0) and identical inputs, repeated execution wastes GPU time and blocks other work. The goal is to execute each unique control configuration exactly once, cache the artifacts, and fan them out to every trial that needs them, while still supporting resume semantics, downstream statistics, and archival.

Key requirements:

- Avoid duplicate control inference work across trials within a sweep.
- Preserve per-item pairing between treatment outputs and the shared control outputs.
- Make the pipeline resumable: if a run is interrupted, we must detect existing shared control outputs and skip recomputation.
- Keep post-processing (parse/score/stats/report) deterministic by pointing each trial at the canonical cache.
- Ensure archiving and cleanup routines understand the new shared-control directory.

## Implementation Plan

1. **Shared Control Registry (state layer)**
   - Extend `scripts/state_utils.py` with helpers to read/write a `control_registry.json` alongside `run_state.json`.
   - Each entry is keyed by `(backend, model_id, temp, prompt hash, input hash)` and stores metadata about the shared control directory and status (pending/completed).

2. **Batch Build & Manifest Enhancements**
   - During `scripts.build_batches`, continue producing a single control shard per prompt set + temperature combination (already happening today); record shard hashes in the build manifest so we can detect duplicates.
   - No behavioral change yet, but the manifest info lets the submit phase compute consistent input hashes.

3. **Submit/Execution Refactor (`scripts.run_all.py`)**
   - For each trial temperature/condition pair, compute the control key and consult the shared registry.
   - If a completed entry exists, skip control submissions; emit lightweight linkage metadata in the trial manifest.
   - If not, queue the control batch exactly once, mark registry status `pending`, and, upon completion, export control artifacts into `shared_controls/<key>/` (combined results JSONL plus trial manifest metadata).
   - Update trial manifests with `control_registry` pointers so downstream phases know where to fetch control data.

4. **Post-Processing Consumers**
   - Update `fireworks/parse_results.py`, `scoring/score_predictions.py`, `scoring/stats.py`, and `scripts/generate_report.py` to detect `control_registry` pointers. When present, load control rows from the shared directory instead of the per-trial `predictions.csv`.
   - Ensure per-item pairing is preserved by joining on `dataset|item_id|temp` keys.

5. **Resume & Error Handling**
   - On `--resume`, resurrect the registry and skip any control work already marked `completed` with intact files. If files are missing, fall back to re-running the control batch.
   - Provide registry sanitation: remove stale entries whose shared directories no longer exist.

6. **Archival & Documentation**
   - Update `scripts/archive_run.py` (and any IO mover scripts) to include `shared_controls/` in the archived payload and restore `control_registry.json`.
   - Document usage in `AGENTS.md`, `docs/*`, and note troubleshooting steps (e.g., clearing the registry when control prompts change).

7. **Testing**
   - Add or adapt smoke tests to cover: single trial (no sharing), multi-trial sharing, and a resume scenario where control is skipped on rerun.
   - Validate with `make smoke` plus a targeted multi-trial dry run.

## Current Progress

- Registry helpers (`load_control_registry`, `write_control_registry`) exist and are being used to persist run-scoped control state. Control keys now include `run_id`, so caches are isolated per run.
- `scripts/run_all.py` submit phase now:
  - Computes control keys from prompt/input hashes and consults the registry before launching control jobs.
  - Records per-trial `control_registry` entries that distinguish producer vs. reuse cases.
  - Skips re-submission when a completed entry exists, and wires producer trials to export control artifacts.
  - Flushes registry updates as the submit phase progresses.
- Control producers now export filtered control JSONL + metadata into `shared_controls/<run_key>/` during the parse/download phase, and registry entries are updated to `completed` with file metadata.
- Trial manifests are hydrated with shared-control metadata so downstream tools can eventually consume shared artifacts.

## Final Status

- **Shared registry lifecycle** is now fully self-healing. The orchestrator refreshes `control_registry.json` when a run restarts, repairing or pruning stale entries before any submission work begins.
- **Downstream consumers** (parse, score, stats, reporting) transparently hydrate cached control rows. If the CSV already contains the shared control data, scorers reuse it; otherwise, they rehydrate the minimal delta directly from the shared cache without duplicating rows.
- **Archival and resume flows** are control-cache aware. Shared artifacts are restored alongside trial results, and subsequent resumes reuse or regenerate controls deterministically.
- **Documentation** across `AGENTS.md` and this plan reflects the completed workflow, detailing how to clear caches, interpret registry entries, and verify reuse in smoke runs.
- **Testing guidance** now includes a concrete smoke checklist covering reuse, resume after partial completion, and cache repair scenarios.
- **Local backend parity** verified via multi-prompt Ollama sweep; shared controls hydrate correctly without Fireworks dependencies.

The shared-control reuse initiative is complete and operating in production runs.

