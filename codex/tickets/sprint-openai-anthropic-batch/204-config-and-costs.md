id: 204
slug: config-and-costs
title: Ticket 204 — Add alt backend config and pricing integration
branch: feat/gpt5-204-config-costs
ticket_file: ./codex/tickets/sprint-openai-anthropic-batch/204-config-and-costs.md
log_file: ./codex/logs/204.md

## Objective
- Introduce configuration and pricing support for the alternative OpenAI/Anthropic backends, ensuring cost summaries and manifests capture provider economics without disturbing the Fireworks defaults.

## Scope
- Add `config/alt_eval_config.yaml` (and optional prompt config references) describing provider stanza, batch settings, and pricing tables per the planning doc.
- Extend `config/schema.py` (or related dataclasses) to validate the new provider metadata and pricing fields.
- Update `scripts/summarize_costs.py` (and any helpers) to ingest OpenAI `usage.prompt_tokens/completion_tokens` and Anthropic `usage.input_tokens/output_tokens`, applying batch discounts and emitting comparable cost summaries.
- Ensure manifests/run records include pricing provider keys needed by downstream reporting.
- Update docs/README or relevant guides explaining when to use `alt_eval_config.yaml` and how to supply model credentials.

## Out of Scope
- Implementing backend adapters (Tickets 202–203).
- Test harness/smoke orchestration updates (Ticket 205).

## Acceptance
- `python -c "from config.schema import EvalConfig; EvalConfig.from_file('config/alt_eval_config.yaml')"` succeeds.
- `make report` (or `python -m scripts.summarize_costs --config config/alt_eval_config.yaml --dry_run`) operates without regression on existing Fireworks runs and produces provider-specific cost entries when fed fixture usage JSON.
- Unit tests covering cost mapping for OpenAI vs Anthropic pass (`pytest tests/scoring/test_cost_summary_alt_backends.py`).
- Documentation changes reviewed and linted (`markdownlint` or `ruff check docs`).

## Deliverables
- Branch: feat/gpt5-204-config-costs
- Files: `config/alt_eval_config.yaml`, updates to `config/schema.py`, `scripts/summarize_costs.py`, relevant docs and tests.
- Log: ./codex/logs/204.md
