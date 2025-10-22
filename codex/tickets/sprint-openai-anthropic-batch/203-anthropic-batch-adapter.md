id: 203
slug: anthropic-batch-adapter
title: Ticket 203 — Implement Anthropic Message Batches backend adapter
branch: feat/gpt5-203-anthropic-batch
ticket_file: ./codex/tickets/sprint-openai-anthropic-batch/203-anthropic-batch-adapter.md
log_file: ./codex/logs/203.md

## Objective
- Provide an Anthropic Message Batches adapter that builds request payloads, submits and polls batches, streams results, and normalizes outputs into the OpenAI-compatible schema required by the downstream parser.

## Scope
- Add `backends/anthropic/build_requests.py` to transform shard rows into `Request` objects (respecting ≤10k requests per batch) with configurable model/temperature/max_tokens.
- Implement `backends/anthropic/start_message_batch.py`, `poll_and_stream.py`, and `normalize_to_openai.py`, capturing provider metadata and writing JSONL outputs keyed by `custom_id`.
- Ensure result normalization stitches text segments, maps usage fields, and handles failure cases (`errored`, `expired`) for manifest logging and retry surfacing.
- Provide unit tests verifying request batching, normalization correctness, and streaming behavior using canned SDK responses (no live API traffic).
- Hook the Anthropic adapter into `scripts/alt_run_all.py` backend dispatch.

## Out of Scope
- OpenAI adapter implementation (Ticket 202).
- Config/pricing updates (Ticket 204).
- Cross-provider smoke validation and docs (Ticket 205).

## Acceptance
- `pytest tests/backends/anthropic/test_message_batches.py` passes with fixtures covering success and failure rows, including normalization into OpenAI-style JSONL.
- Dry-run invocation `python -m scripts.alt_run_all --backend anthropic --config config/alt_eval_config.yaml --dry_run --skip_prepare --skip_build --limit_items 2` produces request payload previews and manifest metadata without network calls.
- Normalized outputs feed through `fireworks/parse_results.py` without modification (unit test assertion).
- Ruff and pyright succeed on new modules (`ruff check backends/anthropic`, `pyright backends/anthropic`).

## Deliverables
- Branch: feat/gpt5-203-anthropic-batch
- Files: `backends/anthropic/*`, tests under `tests/backends/anthropic/`, updates to shared helpers if needed.
- Log: ./codex/logs/203.md
