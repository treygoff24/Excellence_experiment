from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

import pytest

from scripts.alt_smoke import BATCH_INPUTS, FIXTURE_ROOT, build_smoke_config


def _run_alt(
    *,
    config_path: Path,
    provider: str,
    run_id: str,
    replay_dir: Path | None = None,
    resume: bool = False,
    check: bool = True,
) -> subprocess.CompletedProcess[str]:
    cmd = [
        sys.executable,
        "-m",
        "scripts.alt_run_all",
        "--config",
        str(config_path),
        "--backend",
        provider,
        "--skip_prepare",
        "--skip_build",
        "--condition",
        "both",
        "--run_id",
        run_id,
    ]
    if replay_dir is None:
        cmd.append("--dry_run")
    else:
        cmd.extend(["--replay_dir", str(replay_dir)])
    if resume:
        cmd.append("--resume")

    return subprocess.run(
        cmd,
        text=True,
        capture_output=True,
        check=check,
    )


@pytest.mark.parametrize("mode", ["dry-run", "replay"])
def test_alt_smoke_script_executes(mode: str, tmp_path: Path) -> None:
    workdir = tmp_path / "smoke"
    cmd = [
        sys.executable,
        "-m",
        "scripts.alt_smoke",
        "--mode",
        mode,
        "--workdir",
        str(workdir),
        "--batch_inputs",
        str(BATCH_INPUTS),
    ]
    subprocess.run(cmd, check=True)

    if mode == "dry-run":
        placeholders = list(workdir.glob("*/experiments/run_smoke_*_dry/*/results/*_t0_dry.jsonl"))
        assert placeholders, "Dry-run smoke should create placeholder result files"
    else:
        predictions = list(workdir.glob("*/experiments/run_smoke_*_replay/*/results/predictions.csv"))
        assert predictions, "Replay smoke should emit predictions.csv for each provider"


@pytest.mark.parametrize("provider", ["openai", "anthropic"])
def test_replay_outputs_match_golden(provider: str, tmp_path: Path) -> None:
    provider_root = tmp_path / provider
    smoke_cfg = build_smoke_config(provider, workdir=provider_root, batch_inputs_dir=BATCH_INPUTS)

    replay_dir = FIXTURE_ROOT / provider / "replay_success"
    _run_alt(
        config_path=smoke_cfg.config_path,
        provider=provider,
        run_id=f"test_{provider}_replay",
        replay_dir=replay_dir,
    )

    run_root = smoke_cfg.experiments_dir / f"run_test_{provider}_replay"
    trial_dirs = list(run_root.glob("*/results"))
    assert trial_dirs, f"Expected trial results directory under {run_root}"

    generated_results = trial_dirs[0] / "results.jsonl"
    generated_predictions = trial_dirs[0] / "predictions.csv"

    assert generated_results.read_text() == (replay_dir / "results" / "results.jsonl").read_text()
    assert generated_predictions.read_text() == (replay_dir / "expected" / "predictions.csv").read_text()


@pytest.mark.parametrize("provider", ["openai", "anthropic"])
def test_replay_validation_error(provider: str, tmp_path: Path) -> None:
    provider_root = tmp_path / provider
    smoke_cfg = build_smoke_config(provider, workdir=provider_root, batch_inputs_dir=BATCH_INPUTS)
    replay_dir = FIXTURE_ROOT / provider / "replay_validation"

    with pytest.raises(subprocess.CalledProcessError) as exc:
        _run_alt(
            config_path=smoke_cfg.config_path,
            provider=provider,
            run_id=f"test_{provider}_validation",
            replay_dir=replay_dir,
            check=True,
        )

    stderr = exc.value.stderr or ""
    assert "validation" in stderr.lower()


@pytest.mark.parametrize("provider", ["openai", "anthropic"])
def test_replay_partial_requires_resume(provider: str, tmp_path: Path) -> None:
    provider_root = tmp_path / provider
    smoke_cfg = build_smoke_config(provider, workdir=provider_root, batch_inputs_dir=BATCH_INPUTS)

    replay_partial = FIXTURE_ROOT / provider / "replay_partial"
    run_id = f"test_{provider}_partial"
    first = _run_alt(
        config_path=smoke_cfg.config_path,
        provider=provider,
        run_id=run_id,
        replay_dir=replay_partial,
        check=False,
    )
    assert first.returncode != 0
    assert "expected" in (first.stderr or first.stdout).lower()

    replay_success = FIXTURE_ROOT / provider / "replay_success"
    _run_alt(
        config_path=smoke_cfg.config_path,
        provider=provider,
        run_id=run_id,
        replay_dir=replay_success,
        resume=True,
    )

    run_root = smoke_cfg.experiments_dir / f"run_{run_id}"
    prediction_paths = list(run_root.glob("*/results/predictions.csv"))
    assert prediction_paths, "Resume run should emit predictions.csv after resolving partial results"


@pytest.mark.parametrize("provider", ["openai", "anthropic"])
def test_replay_expired_batch_then_resume(provider: str, tmp_path: Path) -> None:
    provider_root = tmp_path / provider
    smoke_cfg = build_smoke_config(provider, workdir=provider_root, batch_inputs_dir=BATCH_INPUTS)

    replay_expired = FIXTURE_ROOT / provider / "replay_expired"
    run_id = f"test_{provider}_expired"
    first = _run_alt(
        config_path=smoke_cfg.config_path,
        provider=provider,
        run_id=run_id,
        replay_dir=replay_expired,
        check=False,
    )
    assert first.returncode != 0
    assert "expired" in (first.stderr or first.stdout).lower()

    replay_success = FIXTURE_ROOT / provider / "replay_success"
    _run_alt(
        config_path=smoke_cfg.config_path,
        provider=provider,
        run_id=run_id,
        replay_dir=replay_success,
        resume=True,
    )

    run_root = smoke_cfg.experiments_dir / f"run_{run_id}"
    prediction_paths = list(run_root.glob("*/results/predictions.csv"))
    assert prediction_paths, "Resume run should succeed after expired batch replay"
