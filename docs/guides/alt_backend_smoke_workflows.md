# Alternative Backend Smoke Workflows

This HOWTO documents the offline smoke coverage for the OpenAI and Anthropic batch adapters. Everything runs on recorded fixtures—no API keys required—so you can verify plumbing, manifests, and parsing before touching live providers.

## 1. Quick Start

| Goal | Command |
| --- | --- |
| Run dual-provider dry smokes | `make alt-smoke` |
| Dry smokes with custom workspace | `python -m scripts.alt_smoke --mode dry-run --workdir /tmp/alt_smoke` |
| Replay recorded goldens (parse path) | `python -m scripts.alt_smoke --mode replay` |
| Target a single provider | `python -m scripts.alt_run_all --config <cfg> --backend openai --replay_dir tests/fixtures/alt_backends/openai/replay_success --skip_prepare --skip_build` |

> The harness writes artifacts under `experiments/alt_smoke/<provider>/experiments/run_smoke_<provider>_<mode>/...` by default. Override `--workdir` to isolate runs during local development.

## 2. Fixtures & Outputs

Fixtures live in `tests/fixtures/alt_backends/`:

* `batch_inputs/`: minimal shard pairs (`t0_control.jsonl`, `t0_treatment.jsonl`) reused across providers.
* `<provider>/replay_success/`: normalized results + golden `predictions.csv` used for replay mode (`metadata.json` lists expected counts).
* `<provider>/replay_validation|replay_partial|replay_expired/`: failure scenarios that trigger validation errors, partial result detection, and expired batches.

Running replay mode with `--replay_dir` copies these JSONL files into the trial’s `results/` directory and reuses `fireworks.parse_results.process_results` to generate `predictions.csv`. Goldens live in the fixture `expected/` folders for diffing.

## 3. Failure Injection & Resume Drills

`pytest tests/smoke/test_alt_backends.py` covers:

1. Dry-run harness sanity (OpenAI + Anthropic) via the CLI wrapper.
2. Replay-mode parity against the fixture goldens (`results.jsonl`, `predictions.csv`).
3. Failure simulations:
   * `replay_validation` → raises a deterministic validation error.
   * `replay_partial` → raises a partial-results error; rerun with `--resume` and `replay_success` to confirm recovery.
   * `replay_expired` → simulates expired batches at poll time; rerun with `--resume` to verify retry.

You can reproduce the same flows manually, e.g.:

```bash
python -m scripts.alt_run_all \
  --config /tmp/openai_smoke.yaml \
  --backend openai \
  --replay_dir tests/fixtures/alt_backends/openai/replay_partial \
  --skip_prepare --skip_build --run_id local_partial

# After the failure, rerun with the success fixture
python -m scripts.alt_run_all \
  --config /tmp/openai_smoke.yaml \
  --backend openai \
  --replay_dir tests/fixtures/alt_backends/openai/replay_success \
  --skip_prepare --skip_build --run_id local_partial --resume
```

## 4. When to Run

* **Before touching adapters:** ensure `make alt-smoke` and the replay mode both pass so manifests, fixtures, and shared-control plumbing remain intact.
* **Before PRs:** include the pytest module output plus the `make alt-smoke` run (or `scripts.alt_smoke` command) in evidence for CI confirmation.
* **While debugging:** point `--replay_dir` at one of the failure fixtures to repro failure modes deterministically.

With these smokes in place you can iterate on adapter code confidently, knowing basic submit/poll/parse wiring, manifests, and resume logic are covered without hitting live endpoints.
