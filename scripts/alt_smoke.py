from __future__ import annotations

import argparse
import os
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import yaml

FIXTURE_ROOT = Path("tests/fixtures/alt_backends").resolve()
BATCH_INPUTS = FIXTURE_ROOT / "batch_inputs"


@dataclass
class SmokeConfig:
    provider: str
    config_path: Path
    experiments_dir: Path
    results_dir: Path
    reports_dir: Path


def _load_base_config() -> dict:
    config_path = Path("config/alt_eval_config.yaml")
    with config_path.open("r", encoding="utf-8") as fh:
        data = yaml.safe_load(fh)
    data["sweep"] = None
    data["temps"] = [0.0]
    data["samples_per_item"] = {"0.0": 1}
    return data


def build_smoke_config(provider: str, *, workdir: Path, batch_inputs_dir: Path | None = None) -> SmokeConfig:
    provider_norm = provider.strip().lower()
    if provider_norm not in {"openai", "anthropic"}:
        raise ValueError(f"Unsupported provider '{provider}'. Expected 'openai' or 'anthropic'.")

    batch_inputs = Path(batch_inputs_dir or BATCH_INPUTS).resolve()
    if not batch_inputs.exists():
        raise FileNotFoundError(f"Batch inputs directory missing for smoke run: {batch_inputs}")

    cfg = _load_base_config()
    cfg.setdefault("provider", {})["name"] = provider_norm
    cfg["model_id"] = "gpt-4o-mini" if provider_norm == "openai" else "claude-3-5-haiku-20241022"

    experiments_dir = workdir / "experiments"
    results_dir = workdir / "results"
    reports_dir = workdir / "reports"
    for path in (experiments_dir, results_dir, reports_dir):
        path.mkdir(parents=True, exist_ok=True)

    paths = cfg.get("paths") or {}
    paths["batch_inputs_dir"] = str(batch_inputs)
    paths["results_dir"] = str(results_dir)
    paths["reports_dir"] = str(reports_dir)
    paths["experiments_dir"] = str(experiments_dir)
    paths["run_manifest"] = str(results_dir / f"{provider_norm}_smoke_manifest.json")
    cfg["paths"] = paths

    config_path = workdir / f"{provider_norm}_smoke.yaml"
    with config_path.open("w", encoding="utf-8") as fh:
        yaml.safe_dump(cfg, fh, sort_keys=False)

    return SmokeConfig(
        provider=provider_norm,
        config_path=config_path,
        experiments_dir=experiments_dir,
        results_dir=results_dir,
        reports_dir=reports_dir,
    )


def _run_alt_backend(
    *,
    config: SmokeConfig,
    mode: str,
    replay_dir: Path | None = None,
    additional_args: Iterable[str] | None = None,
) -> None:
    run_id = f"smoke_{config.provider}_{mode}"
    cmd = [
        sys.executable,
        "-m",
        "scripts.alt_run_all",
        "--config",
        str(config.config_path),
        "--backend",
        config.provider,
        "--skip_prepare",
        "--skip_build",
        "--condition",
        "both",
        "--run_id",
        run_id,
    ]
    if mode == "dry":
        cmd.append("--dry_run")
    if replay_dir is not None:
        cmd.extend(["--replay_dir", str(replay_dir)])
    if additional_args:
        cmd.extend(list(additional_args))

    subprocess.run(cmd, check=True)


def main() -> None:
    parser = argparse.ArgumentParser(description="Smoke runner for alternative backends (OpenAI & Anthropic)")
    parser.add_argument("--mode", choices=["dry-run", "replay", "both"], default="both")
    parser.add_argument("--workdir", default=str(Path("experiments") / "alt_smoke"))
    parser.add_argument("--batch_inputs", default=str(BATCH_INPUTS))
    args = parser.parse_args()

    workdir = Path(args.workdir).resolve()
    workdir.mkdir(parents=True, exist_ok=True)
    batch_inputs = Path(args.batch_inputs).resolve()

    modes: list[str]
    if args.mode == "both":
        modes = ["dry", "replay"]
    elif args.mode == "dry-run":
        modes = ["dry"]
    else:
        modes = ["replay"]

    for provider in ("openai", "anthropic"):
        provider_root = workdir / provider
        provider_root.mkdir(parents=True, exist_ok=True)
        smoke_cfg = build_smoke_config(provider, workdir=provider_root, batch_inputs_dir=batch_inputs)

        if "dry" in modes:
            _run_alt_backend(config=smoke_cfg, mode="dry")

        if "replay" in modes:
            replay_root = FIXTURE_ROOT / provider / "replay_success"
            if not replay_root.exists():
                raise FileNotFoundError(f"Replay fixtures missing for provider {provider}: {replay_root}")
            _run_alt_backend(config=smoke_cfg, mode="replay", replay_dir=replay_root)


if __name__ == "__main__":
    main()
