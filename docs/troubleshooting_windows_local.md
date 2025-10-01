# Troubleshooting — Windows Local Runs (Ollama / llama.cpp)

Purpose: quick fixes for the most common failures when running locally on Windows (PowerShell examples). Keep runs deterministic; prefer smoke tests before full jobs.

- Ollama connection refused
  - Symptom: HTTP 111/ECONNREFUSED from http://127.0.0.1:11434
  - Fix: start the server and verify models are visible
    - PowerShell: `ollama serve`
    - Verify tags: `Invoke-RestMethod http://127.0.0.1:11434/api/tags | ConvertTo-Json -Depth 3`
      - Expected: JSON with a `models` array. If empty/missing, pull a model: `ollama pull llama3.1:8b-instruct-q4_K_M`
    - Health check (curl alternative): `curl http://127.0.0.1:11434/api/tags`

- Out-of-memory (OOM) during generation
  - Symptom: GPU OOM, driver reset, or backend error on first tokens
  - Fixes (apply in order):
    - Reduce context: set a smaller prompt/context window and lower `max_new_tokens` (e.g., 512).
    - Quantize: choose a Q4_K_M or similar 4‑bit quantization for 7–8B models.
    - Concurrency: set `max_concurrent_requests = 1` for local backends. The orchestrator automatically clamps llama.cpp runs to one worker.
    - Close other GPU apps (browsers with GPU accel, game launchers, GPU monitors).

- llama-cpp-python fails to import (`DLL load failed`)
  - Symptom: running `python -m scripts.run_all --config config\eval_config.local.llamacpp.yaml` exits immediately with an import error.
  - Fixes:
    - Install the CUDA/cuBLAS wheels that match your driver/toolkit version:
      ```powershell
      pip install llama-cpp-python==0.2.90 `
        --extra-index-url https://jllllll.github.io/llama-cpp-python-cuBLAS-wheels/whl/cu121
      ```
    - Verify NVidia drivers with `nvidia-smi`. If `DLL load failed` persists, uninstall the wheel and reinstall using the correct `cu11x`/`cu12x` index.
    - Ensure the Microsoft Visual C++ redistributable is installed (required by CUDA DLLs).

- finish_reason = length too often
  - Symptom: Many rows stop due to length rather than a natural stop
  - Fixes:
    - Increase `max_new_tokens` in `config/eval_config.yaml`:
      - `max_new_tokens.closed_book` and/or `max_new_tokens.open_book` (e.g., 1024 → 1536).
    - Configure `stop` sequences in `config/eval_config.yaml` (array of strings) to terminate at task‑appropriate boundaries.

- Parser row mismatch (combined JSONL → predictions.csv)
  - Symptom: `fireworks.parse_results` reports row count mismatch or missing `custom_id`
  - Requirements: Each output line must echo `custom_id` and include `response_text`, `finish_reason`, `request_id` (and `usage` if available).
  - Fixes:
    - Inspect your combined results JSONL: ensure every object has `custom_id` matching the input line.
    - Re‑run local batch validation (see Ticket 114) and parsing:
      - `python -m fireworks.parse_results --results_jsonl results/results_combined.jsonl --out_csv results/predictions.csv`
    - Quick sanity: run a smoke slice end‑to‑end:
      - `python -m scripts.smoke_orchestration --n 3 --prompt_set operational_only`

- Slow scoring/statistics on Windows
  - Symptom: Scoring or bootstrap CIs are noticeably slow or CPU‑bound
  - Fixes (PowerShell, per‑session):
    - `$env:MKL_NUM_THREADS = 8; $env:OMP_NUM_THREADS = 8; $env:NUMEXPR_MAX_THREADS = 8`
    - Avoid running inference concurrently with scoring; run post‑processing separately:
      - `make parse && make score && make stats && make report`

Notes
- Keep seeds and temps fixed for determinism; prefer `make smoke` for quick checks before full runs.
- If you’re using the local backend, start with a single concurrent request and 7–8B Q4 models on 16 GB VRAM.
