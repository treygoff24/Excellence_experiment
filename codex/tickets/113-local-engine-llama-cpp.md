id: 113
slug: local-engine-llama-cpp
title: Ticket 113 — Local engine (alternative): llama.cpp (llama-cpp-python)
ticket_file: ./codex/tickets/113-local-engine-llama-cpp.md
log_file: ./codex/logs/113.md

## Objective
- Provide an in-process `llama-cpp-python` client with CUDA/cuBLAS wheels support; parity with Ollama interface.

## Scope
- New `backends/local/llama_cpp_client.py` exposing `generate(messages|prompt, params)` and returning `{text, finish_reason, usage?, request_id, latency_s}`.
- Configurable `model_path`, `n_ctx`, `n_gpu_layers`, and basic warnings for VRAM pressure (e.g., 13B Q6 on 16 GB).

## Out of Scope
- Batch execution — Ticket 114.
- Token estimation — Ticket 118.

## Acceptance
- Small prompt returns text with no exceptions; VRAM usage visible with `nvidia-smi` on Windows.
- Swapping between Ollama and llama.cpp is config-only (no code changes outside config selection).
- Determinism & safety: Wheel version constraints documented in `docs/windows.md`; graceful error if CUDA wheel mismatch.

## Deliverables
- Files:
  - New: `backends/local/llama_cpp_client.py`
- Log: ./codex/logs/113.md

## References
- docs/planning/PORT_WINDOWS_LOCAL_LLM_ACTION_PLAN.md (Ticket 3 — llama.cpp client)
- docs/guides/gpt5-prompting-best-practices-guide.md
