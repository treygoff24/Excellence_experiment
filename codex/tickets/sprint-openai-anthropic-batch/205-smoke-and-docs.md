id: 205
slug: smoke-and-docs
title: Ticket 205 — Validate alt backends with smoke runs and documentation
branch: feat/gpt5-205-smoke-docs
ticket_file: ./codex/tickets/sprint-openai-anthropic-batch/205-smoke-and-docs.md
log_file: ./codex/logs/205.md

## Objective
- Establish verification coverage for the alternative OpenAI/Anthropic backends (smoke runs, golden outputs, failure drills) and document the workflows for other engineers.

## Scope
- Create smoke configs (or CLI wrappers) that run `scripts.alt_run_all` against fixture shards for both providers with `--dry_run` plus a recorded replay mode using canned results.
- Produce golden CSV/JSONL artifacts (small dataset) for OpenAI and Anthropic paths and wire them into automated assertions (e.g., `tests/smoke/test_alt_backends.py`).
- Add failure-injection tests simulating validation errors, partial results, and expired batches to confirm resume/retry behavior without live traffic.
- Update docs (`docs/README.md`, `docs/planning/OpenAI_Anthropic_API_plan_V1.md`, or new HOWTO) detailing smoke commands, shared control reuse expectations, and ZIP extraction nuances.
- Ensure CI wiring (e.g., `make smoke` or dedicated `make alt-smoke`) runs dry smokes for both providers.

## Out of Scope
- Core adapter or config code (Tickets 201–204).
- Live batch submissions to OpenAI/Anthropic.

## Acceptance
- `pytest tests/smoke/test_alt_backends.py` passes, covering golden outputs and failure-injection scenarios.
- `make alt-smoke` (or documented equivalent) executes dual-provider dry-run smokes and exits 0.
- Documentation accurately reflects smoke steps, shared control expectations, and troubleshooting tips; `markdownlint` or `ruff check docs` passes.
- CI configuration updates (if any) are documented in the ticket log.

## Deliverables
- Branch: feat/gpt5-205-smoke-docs
- Files: tests under `tests/smoke/`, golden fixtures under `tests/fixtures/alt_backends/`, updates to `Makefile`/CI, documentation edits.
- Log: ./codex/logs/205.md
