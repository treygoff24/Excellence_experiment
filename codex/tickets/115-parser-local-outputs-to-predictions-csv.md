id: 115
slug: parser-local-outputs-to-predictions-csv
title: Ticket 115 — Parser: local outputs → predictions.csv (schema parity)
ticket_file: ./codex/tickets/115-parser-local-outputs-to-predictions-csv.md
log_file: ./codex/logs/115.md

## Objective
- Convert local combined JSONL to the exact `predictions.csv` schema consumed by scoring/stats with strict column parity and row-count validation.

## Scope
- New `backends/local/parse_results.py` producing columns:
  - `custom_id, dataset, item_id, condition, temp, sample_index, type, request_id, finish_reason, response_text, prompt_tokens, completion_tokens, total_tokens`.
- Pull `custom_id` and metadata from echoed input in JSONL; enforce unique `custom_id` and 1:1 with inputs.
- If engine usage present, map token counts; else leave blank (Ticket 118 will estimate optionally).

## Out of Scope
- Token estimation and telemetry — Ticket 118.

## Acceptance
- Running only the parser on local JSONL yields a `predictions.csv` identical in columns and row count to the Fireworks path for the same inputs.
- `python -m scoring.score_predictions --pred_csv results/predictions.csv --prepared_dir data/prepared --out_dir results` runs cleanly.
- Determinism & safety: CSV written with `newline=""`; robust to Windows newlines; explicit error on duplicates/missing IDs.

## Deliverables
- Files:
  - New: `backends/local/parse_results.py`
- Log: ./codex/logs/115.md

## References
- docs/planning/PORT_WINDOWS_LOCAL_LLM_ACTION_PLAN.md (Ticket 5 — Parser)
- docs/guides/gpt5-prompting-best-practices-guide.md
