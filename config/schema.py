from __future__ import annotations

import os
from typing import Any, Dict, List, Optional

import yaml
from pydantic import BaseModel, Field, ValidationError, field_validator, ConfigDict, model_validator


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


class ProviderBatchModel(BaseModel):
    endpoint: Optional[str] = None
    completion_window: Optional[str] = None
    max_output_tokens: Optional[int] = None
    max_tokens: Optional[int] = None
    temperature: Optional[float] = None
    top_p: Optional[float] = None

    @field_validator("max_output_tokens", "max_tokens")
    @classmethod
    def _validate_positive_int(cls, v: Optional[int]):  # type: ignore[override]
        if v is None:
            return None
        iv = int(v)
        if iv <= 0:
            raise ValueError("batch.{max_tokens,max_output_tokens} must be positive when provided")
        return iv

    @field_validator("temperature")
    @classmethod
    def _validate_temperature(cls, v: Optional[float]):  # type: ignore[override]
        if v is None:
            return None
        fv = float(v)
        if not (0.0 <= fv <= 2.0):
            raise ValueError("batch.temperature must be within [0,2]")
        return fv

    @field_validator("top_p")
    @classmethod
    def _validate_top_p(cls, v: Optional[float]):  # type: ignore[override]
        if v is None:
            return None
        fv = float(v)
        if not (0.0 <= fv <= 1.0):
            raise ValueError("batch.top_p must be within [0,1]")
        return fv


class ProviderModel(BaseModel):
    name: str
    model: str
    account_id: Optional[str] = None
    region: Optional[str] = None
    pricing_key: Optional[str] = None
    batch: Optional[ProviderBatchModel] = None
    metadata: Dict[str, Any] = Field(default_factory=dict)

    @field_validator("name")
    @classmethod
    def _normalize_name(cls, v: str):  # type: ignore[override]
        vv = (v or "").strip().lower()
        if not vv:
            raise ValueError("provider.name must be a non-empty string")
        return vv

    @field_validator("model")
    @classmethod
    def _validate_model(cls, v: str):  # type: ignore[override]
        vv = (v or "").strip()
        if not vv:
            raise ValueError("provider.model must be a non-empty string")
        return vv

    @model_validator(mode="after")
    def _default_pricing_key(self) -> "ProviderModel":
        if not self.pricing_key:
            object.__setattr__(self, "pricing_key", self.model)
        return self


class ProviderPricingEntry(BaseModel):
    model_config = ConfigDict(extra="allow")
    input_per_mtok: Optional[float] = None
    output_per_mtok: Optional[float] = None
    input_per_mtok_batch: Optional[float] = None
    output_per_mtok_batch: Optional[float] = None
    batch_discount: Optional[float] = None

    @field_validator(
        "input_per_mtok",
        "output_per_mtok",
        "input_per_mtok_batch",
        "output_per_mtok_batch",
        "batch_discount",
    )
    @classmethod
    def _validate_non_negative(cls, v: Optional[float], info):  # type: ignore[override]
        if v is None:
            return None
        try:
            fv = float(v)
        except Exception as e:
            raise ValueError(f"{info.field_name} must be numeric") from e
        if fv < 0:
            raise ValueError(f"{info.field_name} must be non-negative")
        return fv

    @model_validator(mode="after")
    def _ensure_rate_pairs(self) -> "ProviderPricingEntry":
        has_std = self.input_per_mtok is not None and self.output_per_mtok is not None
        has_batch = self.input_per_mtok_batch is not None and self.output_per_mtok_batch is not None
        if not (has_std or has_batch):
            raise ValueError("provider pricing entries must include both input and output rates")
        return self


class PricingModel(BaseModel):
    model_config = ConfigDict(extra="allow")
    input_per_million: float = Field(default=0.15)
    output_per_million: float = Field(default=0.60)
    batch_discount: float = Field(default=0.5)
    providers: Dict[str, Dict[str, ProviderPricingEntry]] = Field(default_factory=dict)

    @model_validator(mode="before")
    @classmethod
    def _collect_providers(cls, data: Any):
        if data is None:
            return {}
        if not isinstance(data, dict):
            raise TypeError("pricing must be a mapping")
        known = {"input_per_million", "output_per_million", "batch_discount", "providers"}
        provider_tables: Dict[str, Dict[str, Any]] = {}
        for key in list(data.keys()):
            if key not in known:
                provider_tables[key] = data.pop(key)
        if provider_tables:
            existing = data.get("providers") or {}
            merged: Dict[str, Dict[str, Any]] = {}
            if isinstance(existing, dict):
                merged.update(existing)
            for prov, table in provider_tables.items():
                merged[prov] = table
            data["providers"] = merged
        return data

    @field_validator("input_per_million", "output_per_million", "batch_discount")
    @classmethod
    def _validate_pricing_defaults(cls, v: float):  # type: ignore[override]
        try:
            fv = float(v)
        except Exception as e:
            raise ValueError("pricing defaults must be numeric") from e
        if fv < 0:
            raise ValueError("pricing defaults must be non-negative")
        return fv

    @field_validator("providers", mode="before")
    @classmethod
    def _normalize_providers(cls, v: Any):  # type: ignore[override]
        if v is None:
            return {}
        if not isinstance(v, dict):
            raise TypeError("pricing.providers must be a mapping of provider -> model -> rates")
        normalized: Dict[str, Dict[str, ProviderPricingEntry]] = {}
        for provider_name, table in v.items():
            if not isinstance(table, dict):
                raise TypeError("provider pricing table must be a mapping of model -> rates")
            normalized[provider_name] = {}
            for model_name, rates in table.items():
                if not isinstance(model_name, str) or not model_name:
                    raise ValueError("pricing provider model keys must be non-empty strings")
                normalized[provider_name][model_name] = ProviderPricingEntry.model_validate(rates)
        return normalized


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
    model_config = ConfigDict(protected_namespaces=())
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
    # Allow fields starting with "model_" (e.g., model_id, model_aliases) without warnings
    model_config = ConfigDict(protected_namespaces=())
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
    provider: Optional[ProviderModel] = None
    unsupported_threshold: float = Field(default=0.5)
    class UnsupportedModel(BaseModel):
        strategy: str = Field(default="baseline")  # baseline|overlap|nli
        threshold: float = Field(default=0.5)
        min_token_overlap: float = Field(default=0.6)
        nli_model: str | None = None

        @field_validator("threshold", "min_token_overlap")
        @classmethod
        def _validate_ufloats(cls, v: float):  # type: ignore[override]
            v = float(v)
            if not (0.0 <= v <= 1.0):
                raise ValueError("unsupported.* values must be in [0,1]")
            return v

    unsupported: UnsupportedModel = Field(default_factory=UnsupportedModel)
    # Optional analyses feature flags
    class OptionalAnalysesModel(BaseModel):
        enable_unsupported_sensitivity: bool = Field(default=True)
        enable_mixed_effects: bool = Field(default=True)
        enable_power_analysis: bool = Field(default=True)
        enable_cost_effectiveness: bool = Field(default=True)

    optional: OptionalAnalysesModel = Field(default_factory=OptionalAnalysesModel)
    # Statistical settings
    class StatsModel(BaseModel):
        bootstrap_samples: int = Field(default=5000)
        permutation_samples: int = Field(default=5000)
        random_seed: int = Field(default=1337)
        enable_permutation: bool = Field(default=True)
        enable_fdr: bool = Field(default=True)
        risk_thresholds: List[float] = Field(default_factory=lambda: [0.0, 0.25, 0.5, 0.75, 1.0])
        tost_alpha: float = Field(default=0.05)
        tost_margins: Dict[str, float] = Field(default_factory=lambda: {"em": 0.01, "f1": 0.01})

        @field_validator("bootstrap_samples", "permutation_samples")
        @classmethod
        def _validate_positive(cls, v: int):  # type: ignore[override]
            if int(v) <= 0:
                raise ValueError("samples must be positive")
            return int(v)

        @field_validator("risk_thresholds")
        @classmethod
        def _validate_thresholds(cls, v: List[float]):  # type: ignore[override]
            arr = sorted(max(0.0, min(1.0, float(x))) for x in (v or []))
            if not arr:
                arr = [0.0, 0.5, 1.0]
            return arr

        @field_validator("tost_alpha")
        @classmethod
        def _validate_alpha(cls, v: float):  # type: ignore[override]
            v = float(v)
            if not (0 < v < 1):
                raise ValueError("tost_alpha must be in (0,1)")
            return v

    stats: StatsModel = Field(default_factory=StatsModel)
    # New optional fields for flexible experimentation
    model_aliases: Dict[str, str] = Field(default_factory=dict)
    models: Optional[List[float | str]] = None  # allow aliases or full ids
    prompt_sets: Optional[Dict[str, PromptSetModel]] = None
    default_prompt_set: Optional[str] = None
    sweep: Optional[SweepModel] = None
    trials: Optional[List[TrialModel]] = None

    # Backend selection and local settings (Ticket 116)
    backend: str = Field(default="fireworks", description="Backend to use: 'fireworks' or 'local'")
    local_engine: Optional[str] = Field(default=None, description="Local engine: 'ollama' or 'llama_cpp'")
    local_endpoint: Optional[str] = Field(default=None, description="Endpoint for local HTTP engines (e.g., Ollama)")
    local_model: Optional[str] = Field(default=None, description="Local model identifier (Ollama tag or GGUF path)")
    max_concurrent_requests: int = Field(default=1, description="Max concurrent local requests")
    tokenizer: Optional[str] = Field(default=None, description="Tokenizer hint for accounting, e.g., 'llama' or 'hf:<repo>'")
    enable_local_telemetry: bool = Field(default=False, description="Enable NVML telemetry sampling during local runs")
    # Optional llama.cpp tuning
    local_n_ctx: Optional[int] = Field(default=None, description="llama.cpp context window (tokens)")
    local_n_gpu_layers: Optional[int] = Field(default=None, description="llama.cpp number of GPU layers (-1 for all)")

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

    @field_validator("backend")
    @classmethod
    def _validate_backend(cls, v: str):  # type: ignore[override]
        allowed = {"fireworks", "local", "alt"}
        vv = (v or "").strip().lower()
        if vv not in allowed:
            raise ValueError(f"backend must be one of {sorted(allowed)}")
        return vv

    @field_validator("local_engine")
    @classmethod
    def _validate_local_engine(cls, v: Optional[str]):  # type: ignore[override]
        if v is None:
            return None
        allowed = {"ollama", "llama_cpp"}
        vv = (v or "").strip().lower()
        if vv not in allowed:
            raise ValueError(f"local_engine must be one of {sorted(allowed)}")
        return vv

    @field_validator("max_concurrent_requests")
    @classmethod
    def _validate_max_concurrent(cls, v: int):  # type: ignore[override]
        try:
            iv = int(v)
        except Exception:
            raise ValueError("max_concurrent_requests must be an integer")
        if iv <= 0:
            raise ValueError("max_concurrent_requests must be > 0")
        return iv


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
    # Backward compatibility: if top-level unsupported_threshold set and unsupported.threshold missing, propagate
    try:
        if cfg.get("unsupported_threshold") is not None and not cfg.get("unsupported", {}).get("threshold"):
            cfg["unsupported"]["threshold"] = float(cfg.get("unsupported_threshold"))
    except Exception:
        pass

    # Normalize/expand local endpoint if present
    if cfg.get("local_endpoint"):
        cfg["local_endpoint"] = os.path.expanduser(os.path.expandvars(cfg["local_endpoint"]))
    return cfg
