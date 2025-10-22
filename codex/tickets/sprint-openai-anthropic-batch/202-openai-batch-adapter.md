id: 202
slug: openai-batch-adapter
title: Ticket 202 — Implement OpenAI Batch backend adapter
branch: feat/gpt5-202-openai-batch
ticket_file: ./codex/tickets/sprint-openai-anthropic-batch/202-openai-batch-adapter.md
log_file: ./codex/logs/202.md

## Objective
- Deliver a production-ready OpenAI Batch adapter that builds request JSONL from existing shards, submits jobs, polls for completion, extracts ZIP results, and normalizes outputs into the existing parser format.

## Scope
- Add `backends/openai/build_inputs.py` converting prepared shard rows into Batch API JSONL records (support `/v1/responses` by default, `/v1/chat/completions` override).
- Implement `backends/openai/start_batch_job.py`, `poll_and_download.py`, and supporting utilities using the official SDK, including metadata plumbing back to the manifest.
- Ensure download step unzips the output file and drops JSONL into the expected trial directory, reusing shared extraction helpers where possible.
- Provide unit/integration tests with recorded fixtures covering submission payload generation, polling state machine, and ZIP extraction (no live API calls).
- Update `scripts/alt_run_all.py` dispatch to call the OpenAI adapter and surface provider metadata (`batch_id`, `output_file_id`).

## Out of Scope
- Anthropic backend work (Ticket 203).
- Config/pricing updates (Ticket 204).
- End-to-end smoke orchestration (Ticket 205).

## Acceptance
- `pytest tests/backends/openai/test_batch_adapter.py` passes (include fixture ZIP with multiple unordered results).
- Dry-run invocation `python -m scripts.alt_run_all --backend openai --config config/alt_eval_config.yaml --dry_run --skip_prepare --skip_build --limit_items 2` writes OpenAI Batch payloads under `experiments/run_*/batch_inputs/` and records manifest metadata without contacting the network.
- Result normalization produces JSONL compatible with `fireworks/parse_results.py` (covered by unit test that pipes sample output into parser).
- Ruff and pyright succeed on new modules (`ruff check backends/openai`, `pyright backends/openai`).

## Deliverables
- Branch: feat/gpt5-202-openai-batch
- Files: `backends/openai/*`, tests under `tests/backends/openai/`, updates to shared utilities/helpers if needed.
- Log: ./codex/logs/202.md
