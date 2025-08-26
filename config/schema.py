from __future__ import annotations

import os
from typing import Dict, List, Optional

import yaml
from pydantic import BaseModel, Field, ValidationError, field_validator


def _format_temp_key(t: float | str) -> str:
    try:
        return f"{float(t):.1f}"
    except Exception:
        return str(t)


class PathsModel(BaseModel):
    raw_dir: str = Field(default="data/raw")
    prepared_dir: str = Field(default="data/prepared")
    batch_inputs_dir: str = Field(default="data/batch_inputs")
    results_dir: str = Field(default="results")
    reports_dir: str = Field(default="reports")
    run_manifest: str = Field(default="results/run_manifest.json")
    # Optional experiments root directory used by orchestrators
    experiments_dir: str = Field(default="experiments")


class PricingModel(BaseModel):
    input_per_million: float = Field(default=0.15)
    output_per_million: float = Field(default=0.60)
    batch_discount: float = Field(default=0.5)


class SizesModel(BaseModel):
    closed_book_max_items: Optional[int] = None
    open_book_max_items: Optional[int] = None
    triviaqa_max_items: Optional[int] = None
    nq_open_max_items: Optional[int] = None
    squad_v2_max_items: Optional[int] = None


class MaxNewTokensModel(BaseModel):
    closed_book: int = Field(default=1024)
    open_book: int = Field(default=1024)


class PromptSetModel(BaseModel):
    control: str
    treatment: str


class SweepModel(BaseModel):
    models: Optional[List[str]] = None
    prompt_sets: Optional[List[str]] = None
    temps: Optional[List[float]] = None
    top_p: Optional[List[float]] = None
    top_k: Optional[List[int]] = None
    max_new_tokens: Optional[Dict[str, List[int]]] = None

    @field_validator("temps")
    @classmethod
    def _validate_sweep_temps(cls, v: Optional[List[float]]):  # type: ignore[override]
        return None if v is None else [float(t) for t in v]


class TrialModel(BaseModel):
    id: Optional[str] = None
    model: Optional[str] = None
    model_id: Optional[str] = None
    prompt_set: Optional[str] = None
    temps: Optional[List[float]] = None
    top_p: Optional[float] = None
    top_k: Optional[int] = None
    max_new_tokens: Optional[MaxNewTokensModel] = None

    @field_validator("temps")
    @classmethod
    def _validate_trial_temps(cls, v: Optional[List[float]]):  # type: ignore[override]
        return None if v is None else [float(t) for t in v]


class EvalConfigModel(BaseModel):
    model_id: str
    temps: List[float] = Field(default_factory=lambda: [0.0, 0.7])
    samples_per_item: Dict[str, int]
    max_new_tokens: MaxNewTokensModel = Field(default_factory=MaxNewTokensModel)
    top_p: Optional[float] = 1.0
    top_k: Optional[int] = 50
    stop: List[str] = Field(default_factory=list)
    sizes: SizesModel = Field(default_factory=SizesModel)
    paths: PathsModel = Field(default_factory=PathsModel)
    pricing: PricingModel = Field(default_factory=PricingModel)
    use_batch_api: bool = True
    unsupported_threshold: float = Field(default=0.5)
    # New optional fields for flexible experimentation
    model_aliases: Dict[str, str] = Field(default_factory=dict)
    models: Optional[List[float | str]] = None  # allow aliases or full ids
    prompt_sets: Optional[Dict[str, PromptSetModel]] = None
    default_prompt_set: Optional[str] = None
    sweep: Optional[SweepModel] = None
    trials: Optional[List[TrialModel]] = None

    @field_validator("samples_per_item")
    @classmethod
    def _normalize_samples(cls, v: Dict[str | float, int], info):  # type: ignore[override]
        if not isinstance(v, dict):
            raise ValueError("samples_per_item must be a mapping of temperature -> K")
        normalized: Dict[str, int] = {}
        for k, val in v.items():
            if not isinstance(val, int) or val <= 0:
                raise ValueError("samples_per_item values must be positive integers")
            normalized[_format_temp_key(k)] = val
        return normalized

    @field_validator("temps")
    @classmethod
    def _validate_temps(cls, v: List[float]):  # type: ignore[override]
        if not v:
            raise ValueError("temps must contain at least one temperature")
        return [float(t) for t in v]

    @field_validator("unsupported_threshold")
    @classmethod
    def _validate_threshold(cls, v: float):  # type: ignore[override]
        if not (0.0 <= v <= 1.0):
            raise ValueError("unsupported_threshold must be in [0,1]")
        return float(v)


def load_config(path: str) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        raw = yaml.safe_load(f) or {}
    try:
        model = EvalConfigModel(**raw)
    except ValidationError as e:
        # Pretty error message that points to the config path
        raise SystemExit(f"Invalid configuration in {path}:\n{e}")
    cfg = model.model_dump()

    # Ensure directories exist lazily; do not create here to keep load side-effect free
    # but normalize any env vars or user home symbols
    for key in ("raw_dir", "prepared_dir", "batch_inputs_dir", "results_dir", "reports_dir"):
        cfg["paths"][key] = os.path.expanduser(os.path.expandvars(cfg["paths"][key]))
    # Normalize optional experiments_dir for orchestrators
    if cfg.get("paths", {}).get("experiments_dir"):
        cfg["paths"]["experiments_dir"] = os.path.expanduser(os.path.expandvars(cfg["paths"]["experiments_dir"]))

    # Backward-compatible default prompt set when not defined
    if not cfg.get("prompt_sets"):
        cfg["prompt_sets"] = {
            "default": {
                "control": "config/prompts/control_system.txt",
                "treatment": "config/prompts/treatment_system.txt",
            }
        }
        cfg["default_prompt_set"] = cfg.get("default_prompt_set") or "default"
    else:
        if not cfg.get("default_prompt_set"):
            try:
                first_key = sorted(list(cfg["prompt_sets"].keys()))[0]
            except Exception:
                first_key = "default"
            cfg["default_prompt_set"] = first_key

    # Normalize temps to floats
    cfg["temps"] = [float(t) for t in (cfg.get("temps") or [])]
    return cfg


