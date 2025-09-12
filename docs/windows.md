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
- Concurrency: controlled by `max_concurrent_requests` in the config (default 1; safe on 16GB VRAM).
- Token limits: `max_new_tokens.closed_book|open_book` in the config.
- Stop sequences: configure `stop: [...]` in the config to trim answers cleanly.

## Troubleshooting and Performance
- Troubleshooting: `docs/troubleshooting_windows_local.md`.
- Performance tuning and VRAM budgets: `docs/performance.md`.

Notes
- Local path writes per-part `results.jsonl` then combines them per trial. Downstream `parse → score → stats → report` are unchanged.
- Token estimation and GPU telemetry are deferred; engines that report usage will surface tokens in predictions; others leave them blank.

