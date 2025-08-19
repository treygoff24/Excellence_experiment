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
    closed_book: int = Field(default=512)
    open_book: int = Field(default=512)


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
    return cfg


