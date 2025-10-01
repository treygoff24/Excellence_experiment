# Performance Tuning & VRAM Budget Guide

This guide provides hardware-aware tuning presets and troubleshooting guidance for local LLM inference on Windows with limited VRAM. Optimized for RTX 5080 16GB but applicable to similar hardware configurations.

## Quick Start: Hardware Presets for RTX 5080 (16GB VRAM)

### Throughput Preset (Recommended for batch processing)
- **Model**: 8B Q4_K_M quantization
- **Context**: 4k-8k tokens
- **Max tokens**: 512-1024
- **Concurrency**: Ollama up to 2 workers; llama.cpp fixed to 1
- **Use case**: Fast evaluation runs, high throughput needed

```yaml
# In your config (e.g., config/eval_config.yaml)
model_id: "llama3.1:8b-instruct-q4_K_M"
max_new_tokens:
  closed_book: 512
  open_book: 1024
# Concurrency is controlled via CLI: --max_concurrent_jobs 2
```

### Quality Preset (Best accuracy, slower)
- **Model**: 8B Q6_K quantization  
- **Context**: 4k tokens
- **Max tokens**: 512-1024
- **Concurrency**: Ollama 1 worker, llama.cpp 1 worker
- **Use case**: Quality-focused runs, research validation

```yaml
# In your config (e.g., config/eval_config.yaml)
model_id: "llama3.1:8b-instruct-q6_K"
max_new_tokens:
  closed_book: 512
  open_book: 1024
# Concurrency is controlled via CLI: --max_concurrent_jobs 1
```

### Stretch Preset (Largest model possible)
- **Model**: 13B Q4_K_M quantization
- **Context**: 4k tokens  
- **Max tokens**: 512
- **Concurrency**: Single worker (both engines)
- **Use case**: Testing larger models within VRAM limits

```yaml
# In your config (e.g., config/eval_config.yaml)
model_id: "llama3.1:13b-instruct-q4_K_M"
max_new_tokens:
  closed_book: 512
  open_book: 512
# Concurrency is controlled via CLI: --max_concurrent_jobs 1
```

## Quick Decision Tree

```
Performance Issue?
├─ Out of Memory (OOM)?
│  ├─ Reduce context length (8k→4k→2k)
│  ├─ Lower quantization (Q6_K→Q4_K_M→Q4_K_S)
│  ├─ Reduce concurrency (2→1)
│  └─ Switch to smaller model (13B→8B→7B)
├─ Slow token generation?
│  ├─ Lower top_k (50→20→10)
│  ├─ Increase temperature slightly (0.0→0.1)
│  ├─ Try different engine (Ollama→llama.cpp)
│  └─ Check GPU utilization with nvidia-smi
├─ Frequent length truncation?
│  ├─ Increase max_new_tokens
│  ├─ Add better stop sequences
│  └─ Review prompt length vs context budget
└─ Connection/parsing errors?
   ├─ Ensure ollama serve is running
   ├─ Verify custom_id echoing in outputs
   └─ Check Windows firewall/antivirus
```

## Common Symptoms & Fixes

### Out of Memory (OOM) Errors

**Symptoms:**
- CUDA out of memory errors
- System freezing during inference
- Ollama/llama.cpp crashes

**Fixes (in order of preference):**
1. **Reduce context length**: Start with 4k, go down to 2k if needed
2. **Lower quantization**: Q6_K → Q4_K_M → Q4_K_S → Q3_K_M
3. **Reduce concurrency**: run with `--max_concurrent_jobs 1`. The local
   orchestrator automatically clamps llama.cpp workloads to a single worker.
4. **Switch to smaller model**: 13B → 8B → 7B
5. **Close other GPU applications**: Gaming software, video editors, browsers with hardware acceleration

### Slow Token Generation

**Symptoms:**
- <5 tokens/second on RTX 5080
- High GPU memory but low utilization
- Long pauses between tokens

**Fixes:**
1. **Optimize sampling parameters**:
   ```yaml
   top_k: 20        # Down from 50
   top_p: 0.95      # Down from 1.0
   temperature: 0.1 # Up from 0.0 slightly
   ```
2. **Try different engine**: Ollama often faster than llama.cpp for Llama models
3. **Check GPU utilization**: Use `nvidia-smi` - should see >80% GPU utilization
4. **Optimize Windows environment** (see Environment Optimization section)

### Frequent Length Truncation

**Symptoms:**
- High `finish_reason=length` rates (>10%)
- Answers cut off mid-sentence
- Poor evaluation scores due to incomplete responses

**Fixes:**
1. **Increase max_new_tokens**:
   ```yaml
   max_new_tokens:
     closed_book: 1024  # Up from 512
     open_book: 1536    # Up from 1024
   ```
2. **Add stop sequences** to prevent overgeneration:
   ```yaml
   stop: ["\n\nQuestion:", "Q:", "A:", "</answer>"]
   ```
3. **Review context budget**: Ensure prompt + response fits in model's context window

### Connection & Service Issues

**Symptoms:**
- `Connection refused` errors
- `ollama serve` not responding
- Empty or malformed responses

**Fixes:**
1. **Start Ollama service**:
   ```powershell
   ollama serve
   ```
2. **Verify service health**:
   ```powershell
   Invoke-WebRequest http://127.0.0.1:11434/api/tags
   ```
3. **Check Windows firewall**: Allow ollama.exe through firewall
4. **Verify model download**:
   ```powershell
   ollama pull llama3.1:8b-instruct-q4_K_M
   ```

### Parser Row Mismatches

**Symptoms:**
- Prediction counts don't match input counts
- Missing custom_id fields in outputs
- Scoring failures

**Fixes:**
1. **Ensure custom_id echoing**: Models must echo the `custom_id` in responses
2. **Validate output format**: Check that JSONL structure matches expected schema
3. **Re-run with smaller batch**: Test with `--limit_items 10` to isolate issues
4. **Check prompt engineering**: Ensure prompts guide proper response formatting

## Telemetry Interpretation

### GPU Monitoring with nvidia-smi

Monitor your GPU during inference:

```powershell
# Continuous monitoring (update every 2 seconds)
nvidia-smi -l 2

# Single snapshot
nvidia-smi --query-gpu=utilization.gpu,memory.used,memory.total,temperature.gpu,power.draw --format=csv
```

Alternatively, enable in-run sampling by setting `enable_local_telemetry: true`
in your local config (requires `pynvml` and NVIDIA drivers). The orchestrator
records peak GPU memory, utilization, and temperature in each part's
`state.json` plus the per-trial manifest under `job_counts`.

**Healthy inference indicators:**
- GPU utilization: 80-95%
- Memory usage: 12-15GB (on 16GB card)
- Temperature: <80°C
- Power draw: Near TGP limits

**Problem indicators:**
- Low GPU utilization (<50%): CPU bottleneck, poor batching
- Memory spikes to 16GB: Risk of OOM, reduce batch size
- Temperature >85°C: Thermal throttling, check cooling
- Power draw oscillating: Inefficient workload scheduling

### Windows Performance Monitor

Track system resources:

```powershell
# CPU and memory monitoring
Get-Counter "\Processor(_Total)\% Processor Time", "\Memory\Available MBytes" -SampleInterval 2 -MaxSamples 30
```

**Watch for:**
- CPU utilization >90%: May need to reduce thread counts
- Available memory <4GB: System-level memory pressure
- High disk I/O: Model loading/swapping issues

## Environment Optimization

### Thread Control (Critical for Performance)

Set these environment variables before running evaluations:

```powershell
$env:MKL_NUM_THREADS = "8"
$env:OMP_NUM_THREADS = "8"
$env:NUMEXPR_MAX_THREADS = "8"
```

Or in your PowerShell profile:
```powershell
# Add to $PROFILE
[Environment]::SetEnvironmentVariable("MKL_NUM_THREADS", "8", "User")
[Environment]::SetEnvironmentVariable("OMP_NUM_THREADS", "8", "User")
[Environment]::SetEnvironmentVariable("NUMEXPR_MAX_THREADS", "8", "User")
```

### Process Isolation

**Avoid concurrent GPU workloads:**
- Don't run scoring/statistics while inference is active
- Close browsers with hardware acceleration during runs
- Pause Windows Update and antivirus scans

**Recommended workflow:**
```powershell
# 1. Run inference only (stop after poll)
python -m scripts.run_all --config config\eval_config.yaml --to_step poll

# 2. Parse and score after inference completes
python -m fireworks.parse_results --results_jsonl results\results_combined.jsonl --out_csv results\predictions.csv
python -m scoring.score_predictions --pred_csv results\predictions.csv --prepared_dir data\prepared --out_dir results
```

### Windows-Specific Optimizations

1. **Disable Windows Game Mode** during inference (can interfere with GPU scheduling)
2. **Set High Performance power plan**:
   ```powershell
   powercfg /setactive 8c5e7fda-e8bf-4a96-9a85-a6e23a8c635c
   ```
3. **Increase virtual memory**: Set to 32GB if you have the disk space
4. **Close unnecessary services**: Disable Windows Search, Superfetch during runs

## Model-Specific Considerations

### Quantization Trade-offs

| Quantization | Memory | Quality | Speed | Notes |
|-------------|--------|---------|-------|-------|
| FP16 | Highest | Best | Slow | Usually too large for 16GB |
| Q6_K | High | Excellent | Medium | Good balance for quality runs |
| Q4_K_M | Medium | Good | Fast | **Recommended default** |
| Q4_K_S | Medium | Good | Fastest | Slightly lower quality |
| Q3_K_M | Low | Fair | Very fast | Use only if necessary |

### Model Size Guidelines

**For 16GB VRAM:**
- 7B models: Any quantization, high concurrency possible
- 8B models: Q4_K_M recommended, Q6_K for quality
- 13B models: Q4_K_M only, single job
- 20B+ models: Generally too large, try Q3_K_M experimentally

### Variability Across Models

**Important**: Performance characteristics vary significantly between model families:
- Llama models: Generally well-optimized in Ollama
- Code models (CodeLlama, etc.): May need more context
- Chat-tuned models: Often more verbose, adjust max_tokens accordingly
- Base models: Less constrained output, need better stop sequences

**Always measure with your specific model:**
```powershell
# Quick single-item test
python -m scripts.run_all --config config\eval_config.yaml --limit_items 1 --dry_run
```

## Command References

### Quick Smoke Tests

```powershell
# Tiny dry-run (no model loading)
python -m scripts.run_all --config config\eval_config.yaml --dry_run --limit_items 10

# Small end-to-end test
python -m scripts.run_all --config config\eval_config.yaml --limit_items 50 --parts_per_dataset 2 --max_concurrent_jobs 1

# Orchestration smoke test
python -m scripts.smoke_orchestration --n 3 --prompt_set operational_only --keep
```

### Manual Pipeline Steps

```powershell
# Step-by-step execution (useful for debugging)
python -m scripts.prepare_data --config config\eval_config.yaml
python -m scripts.build_batches --config config\eval_config.yaml
python -m scripts.run_all --config config\eval_config.yaml --skip_prepare --skip_build
```

### Monitoring Commands

```powershell
# Watch GPU usage
nvidia-smi -l 2

# Monitor specific job
python -m scripts.list_runs
python -m scripts.compare_runs run_20240909_123456 run_20240909_234567

# Stop runaway job
python -m scripts.stop_run --run_id run_20240909_123456
```

## Conservative Defaults Recommendation

When in doubt, start with these safe settings:

```yaml
# Safe config snippet (e.g., config/eval_config.yaml)
model_id: "llama3.1:8b-instruct-q4_K_M"
temps: [0.0]
max_new_tokens:
  closed_book: 512
  open_book: 1024
top_k: 50
top_p: 1.0
# Concurrency is controlled via CLI: --max_concurrent_jobs 1
```

**Rationale:**
- Q4_K_M quantization: Best balance of quality/performance/memory
- Single job: Eliminates resource contention
- 8B model: Fits comfortably in 16GB with room for context
- Conservative token limits: Prevents runaway generation
- Standard sampling: Deterministic, comparable to cloud APIs

Gradually increase complexity (larger models, more concurrency, longer context) only after validating that basic functionality works correctly.

## Performance Expectations

### Realistic Benchmarks (RTX 5080 16GB)

**Llama 3.1 8B Q4_K_M:**
- Throughput: 15-25 tokens/second
- Context processing: ~500 tokens/second
- Memory usage: ~6-8GB VRAM
- Concurrent jobs: 2 comfortable, 3 possible

**Llama 3.1 13B Q4_K_M:**
- Throughput: 8-15 tokens/second  
- Context processing: ~200 tokens/second
- Memory usage: ~12-14GB VRAM
- Concurrent jobs: 1 only

**Evaluation timing estimates:**
- 1000 items, 8B model, 1 job: ~45-60 minutes
- 1000 items, 8B model, 2 jobs: ~25-35 minutes
- 1000 items, 13B model, 1 job: ~90-120 minutes

Use these as baselines to identify performance issues. Significantly slower performance indicates configuration problems or resource contention.
