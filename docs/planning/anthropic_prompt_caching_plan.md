# Anthropic Prompt Caching Enablement Plan

## Overview
- **Goal:** add first-class support for Anthropic’s prompt caching so that long, repeated system prompts (and optionally tools/context blocks) are cached once and reused across the 5 000-item production runs.
- **Why now:** Sonnet 4.5 charges $3/MTok for uncached input but only $0.30/MTok for cache hits. Our shared-control design reuses an identical ~1 400-token system prompt for every treatment call, so enabling caching immediately returns ~90 % of that spend. Without it, the pilot alone burns hundreds of dollars, and the full run overshoots the $524 budget in `final_full_run_plan.md`.
- **Definition of Done:** batch jobs coming from `scripts.alt_run_all` send Anthropic Messages payloads with structured `system`/`messages` arrays that include `cache_control` on the reusable blocks; runtime metrics (`cache_read_input_tokens` / `cache_creation_input_tokens`) confirm hits inside orchestration logs; regression tests and smoke tooling continue to pass.

## Current State & Constraints
1. `scripts.prompt_adapter.render_payload` collapses the system prompt and user prompt to plain strings. This is fine for OpenAI, but it drops Anthropic’s ability to accept structured text blocks with `cache_control`.
2. `backends/anthropic/build_requests._extract_system_and_messages` further flattens each message to a raw string, so any structural annotations would be discarded.
3. Configs (`config/eval_config.*.yaml`) only point to file paths; they do not yet expose a way to mark a system prompt block as cacheable within the generated requests.
4. Batch orchestrator (`scripts.alt_run_all`) currently assumes prompts are strings; unit tests in `tests/backends/anthropic/` operate on that assumption.

## Implementation Plan

### Phase 1 — Schema & Prompt Rendering
1. **Extend prompt adapter output**
   - Update `scripts/prompt_adapter.render_payload` to optionally emit Anthropic-style message blocks:
     ```python
     body = {
         "messages": [
             {"role": "system", "content": [{"type": "text", "text": system_text, "cache_control": {"type": "ephemeral"}}]},
             {"role": "user", "content": [{"type": "text", "text": user_text}]},
         ]
     }
     ```
   - Provide a toggle so OpenAI continues to see string content. Recommendation: gate on `cfg.get("provider", {}).get("name") == "anthropic"` when rendering.
   - Maintain backwards compatibility for existing configs/tests by defaulting to the current behaviour unless the provider requests structured content.
2. **Config affordance**
   - Reuse existing prompt files; no format change needed.
   - Add an optional config flag (e.g., `provider.request_overrides.cache_control.enable_system_cache: true`) if we want to make caching opt-in per config; otherwise, simply turn it on for all Anthropic runs.
   - Document the flag in `config/schema.py` so validation accepts it if introduced.

### Phase 2 — Anthropic Request Builder
1. **Preserve block structure**
   - Modify `_extract_system_and_messages` in `backends/anthropic/build_requests.py` to:
     - Detect list-of-blocks content and forward it unchanged (instead of calling `_coerce_text`).
     - Split the first system block(s) into the `system` field, preserving `cache_control`.
     - Ensure user/assistant turns remain lists of block dicts (`{"type": "text", ...}`) so the API sees the annotations.
2. **Cache annotations**
   - When the prompt adapter didn’t already add a `cache_control` block (future-proofing), inject one in `build_requests` by wrapping the system text into `[{"type": "text", "text": system_text, "cache_control": {"type": "ephemeral"}}]`.
   - Respect optional config for TTL: look up `cfg["provider"]["request_overrides"]["cache_control_ttl"]` (if we introduce it) and set `{"type": "ephemeral", "ttl": "1h"}` as required.
3. **Thinking compatibility**
   - Extended thinking requires `temperature == 1.0` (already enforced). Ensure adding structured content doesn’t interfere with the existing temperature guard.

### Phase 3 — Orchestrator & Build Pipeline
1. **Batch shard previews**
   - Update `scripts.build_batches` preview writer (if used) to cope with list-of-blocks so `delta` / debug prints remain readable.
2. **alt_run_all integration**
   - Verify `_load_build_manifest` and `TrialRun` logic pass through the new message format unchanged.
   - Add logging in `scripts.alt_run_all` (submit phase) to surface `cache_read_input_tokens`/`cache_creation_input_tokens` when the API returns them; this helps spot regressions.

### Phase 4 — Configuration & Documentation
1. **Pilot configs**
   - Set `provider.request_overrides.cache_control: system` (or equivalent toggle) inside `config/eval_config.pilot.anthropic.yaml`.
   - Mirror the change into future prod configs.
2. **Docs**
   - Update `docs/developer_onboarding_guide.md` or `AGENTS.md` with a short subsection on Anthropic caching: how to enable, expected savings, and troubleshooting (e.g., watch for cache misses if TTL expires mid-batch).
3. **Tests**
   - Extend `tests/backends/anthropic/test_message_batches.py` to cover:
     - Requests that carry a `cache_control` block.
     - Degenerate cases (short prompts < 1024 tokens – confirm we simply skip adding cache).
   - Add a unit test for the prompt adapter verifying the Anthropic path emits structured blocks.
   - Run `pytest`, `python -m scripts.dry_run_multi_treatments --config config/eval_config.pilot.anthropic.yaml --dry_run`, and `python -m scripts.smoke_orchestration ... --dry_run` to ensure tooling remains happy.

## Rollout Checklist
1. Code updates merged (`scripts/prompt_adapter.py`, `backends/anthropic/build_requests.py`, optional schema/config tweaks).
2. Updated config(s) committed and validated via `python -m scripts.prepare_data` + `python -m scripts.build_batches --config config/eval_config.pilot.anthropic.yaml --temps 1.0`.
3. Dry-run smoke (`scripts.alt_run_all ... --dry_run`) shows requests with `cache_control` in the preview JSONL.
4. Pilot Anthropic batch reports non-zero `cache_read_input_tokens` in job logs.
5. Post-run report documents observed savings for future tuning.

## Open Questions / Decisions
- **TTL choice:** default 5 min should suffice for batch jobs processed in a single wave. If we see misses due to scheduling gaps, upgrade to `ttl: "1h"` per the docs.
- **Selective caching:** we’re starting with the system prompt; consider adding additional breakpoints (tool definitions, RAG context) later if those sections grow.
- **Shared control compatibility:** ensure the control/treatment system prompts share identical cached prefixes; otherwise we may need per-prompt cache keys, which the API handles automatically as long as the content matches.

## Ready for Implementation
This plan gives a 360° outline: schema changes, adapter rewrites, config wiring, and testing. Another agent can follow these steps to implement the feature without digging through Anthropic’s docs again. Once complete, we should see immediate cost savings and can move ahead with the staged pilot confidently.
