# Windows Guide (Local Runs)

This guide covers a Windows‑native workflow for running the A/B evaluation pipeline locally using Ollama or llama.cpp. It keeps schemas and downstream scoring/stats identical to the Fireworks path.

## Prerequisites
- Windows 11, PowerShell 7+
- Python 3.11 (Python Launcher `py` recommended)
- NVIDIA GPU with recent drivers
- One of:
  - Ollama installed and running (`ollama serve`)
  - llama-cpp-python with CUDA/cuBLAS wheels and a local GGUF model

## Bootstrap and Basics

```powershell
powershell -ExecutionPolicy Bypass -File tools\bootstrap.ps1
.\.venv\Scripts\Activate.ps1
```

## Example Configs
- Ollama: `config/eval_config.local.yaml` (default model: `llama3.1:8b-instruct-q4_K_M`)
- llama.cpp: `config/eval_config.local.llamacpp.yaml` (edit `local_model` to your GGUF path)

Validate a config loads:
```powershell
python -c "from config.schema import load_config; load_config('config/eval_config.local.yaml')"
```

## Installing llama-cpp-python on Windows

GPU-enabled llama.cpp wheels are distributed separately from PyPI. Use the
cuBLAS builds to avoid compiling locally:

```powershell
# Inside the virtual environment created by tools\bootstrap.ps1
pip install llama-cpp-python==0.2.90 `
  --extra-index-url https://jllllll.github.io/llama-cpp-python-cuBLAS-wheels/whl/cu121

# Quick sanity check (loads bindings without running a model)
python -c "from llama_cpp import Llama; print('llama.cpp OK')"
```

Common issues:
- `DLL load failed` — CUDA toolkit mismatch. Install the wheel that matches
  your driver/toolkit version (e.g., `cu118`, `cu121`).
- `ImportError: cannot open shared object file` — ensure the Microsoft Visual
  C++ redistributable is installed.
- `CUDA_ERROR_NO_DEVICE` — verify an NVIDIA GPU is visible via `nvidia-smi` and
  that drivers are up to date.

## Run Recipes (Local Backend)

Ollama quick start (recommended):
```powershell
# In a separate terminal
ollama serve
ollama pull llama3.1:8b-instruct-q4_K_M

# Plan manifest and gating
python -m scripts.run_all --config config\eval_config.local.yaml --plan_only

# Small end-to-end run (archives under experiments/run_<ID>/)
python -m scripts.run_all --config config\eval_config.local.yaml --archive `
  --limit_items 200 --parts_per_dataset 3
```

llama.cpp quick start:
```powershell
# Edit config/eval_config.local.llamacpp.yaml: set local_model to your GGUF
python -m scripts.run_all --config config\eval_config.local.llamacpp.yaml --plan_only
python -m scripts.run_all --config config\eval_config.local.llamacpp.yaml --archive `
  --limit_items 100 --parts_per_dataset 2
```

## Concurrency and Tokens
- Concurrency: `max_concurrent_requests` controls worker count. Ollama supports
  up to 2 workers locally; llama.cpp is forced to 1 worker per process for
  stability. Requesting higher values will emit a warning and clamp to 1.
- Token limits: `max_new_tokens.closed_book|open_book` in the config.
- Stop sequences: configure `stop: [...]` in the config to trim answers cleanly.
- Token accounting: both local engines now emit `prompt_tokens`,
  `completion_tokens`, and `total_tokens` in `predictions.csv`. When the engine
  omits usage data, the orchestrator estimates counts via `scripts.estimate_tokens`.

## Telemetry (Optional)
- Set `enable_local_telemetry: true` in your config to record NVIDIA NVML
  samples (GPU memory, utilization, peak temperature) per part. Requires
  `pynvml` and recent NVIDIA drivers. Telemetry summaries land in
  `state.json` under each part directory and surface in the manifest.

## Troubleshooting and Performance
- Troubleshooting: `docs/troubleshooting_windows_local.md`.
- Performance tuning and VRAM budgets: `docs/performance.md`.

Notes
- Local path writes per-part `results.jsonl` then combines them per trial. Downstream `parse → score → stats → report` are unchanged.
- Token estimation is automatic; telemetry is opt-in via config.
