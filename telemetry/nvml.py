"""NVML-powered telemetry sampling for local inference runs.

The helpers in this module are designed to degrade gracefully when NVML or the
`pynvml` bindings are unavailable. They provide lightweight GPU utilization and
memory snapshots that can be attached to local run artifacts.
"""

from __future__ import annotations

import time
from contextlib import contextmanager
from dataclasses import dataclass, field
from typing import Any, Dict, Iterator, List, Optional

try:  # pragma: no cover - optional dependency
    import pynvml  # type: ignore
except Exception:  # pragma: no cover
    pynvml = None  # type: ignore


_NVML_INITIALIZED = False
_NVML_AVAILABLE = False


def _ensure_initialized() -> bool:
    global _NVML_INITIALIZED, _NVML_AVAILABLE
    if _NVML_INITIALIZED:
        return _NVML_AVAILABLE
    _NVML_INITIALIZED = True
    if pynvml is None:
        _NVML_AVAILABLE = False
        return False
    try:
        pynvml.nvmlInit()  # type: ignore[func-returns-value]
        _NVML_AVAILABLE = True
    except Exception:
        _NVML_AVAILABLE = False
    return _NVML_AVAILABLE


@dataclass
class NvmlSession:
    """Tracks NVML samples for the lifetime of a single inference call."""

    enabled: bool
    device_index: int = 0
    metrics: Optional[Dict[str, Any]] = None
    _handle: Any = field(init=False, default=None)
    _samples: List[Dict[str, float]] = field(init=False, default_factory=list)
    _start: float = field(init=False, default_factory=time.time)

    def __post_init__(self) -> None:
        self.enabled = bool(self.enabled) and _ensure_initialized()
        if not self.enabled:
            return
        try:
            self._handle = pynvml.nvmlDeviceGetHandleByIndex(int(self.device_index))  # type: ignore[attr-defined]
        except Exception:
            self.enabled = False
            self._handle = None
            return
        self.sample()

    def sample(self) -> Optional[Dict[str, float]]:
        if not self.enabled or self._handle is None:
            return None
        try:
            mem = pynvml.nvmlDeviceGetMemoryInfo(self._handle)  # type: ignore[attr-defined]
            util = pynvml.nvmlDeviceGetUtilizationRates(self._handle)  # type: ignore[attr-defined]
            temp = pynvml.nvmlDeviceGetTemperature(self._handle, pynvml.NVML_TEMPERATURE_GPU)  # type: ignore[attr-defined]
        except Exception:
            self.enabled = False
            return None
        sample = {
            "gpu_mem_mb": mem.used / (1024 * 1024),
            "gpu_mem_total_mb": mem.total / (1024 * 1024),
            "gpu_utilization": float(getattr(util, "gpu", 0.0)),
            "memory_utilization": float(getattr(util, "memory", 0.0)),
            "temperature_c": float(temp),
            "timestamp": time.time(),
        }
        self._samples.append(sample)
        return sample

    def finalize(self) -> None:
        if not self.enabled:
            self.metrics = None
            return
        # Take a trailing sample so metrics include post-call utilization
        self.sample()
        duration_ms = (time.time() - self._start) * 1000.0
        gpu_mem = [s.get("gpu_mem_mb", 0.0) for s in self._samples]
        gpu_util = [s.get("gpu_utilization", 0.0) for s in self._samples]
        temps = [s.get("temperature_c", 0.0) for s in self._samples]
        self.metrics = {
            "latency_ms": round(duration_ms, 3),
            "gpu_mem_mb": round(max(gpu_mem) if gpu_mem else 0.0, 3),
            "gpu_utilization": max(gpu_util) if gpu_util else 0.0,
        }
        if temps:
            self.metrics["temperature_c"] = max(temps)

    # Allow direct use with "with NvmlSession(...) as session"
    def __enter__(self) -> "NvmlSession":
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> bool:
        self.finalize()
        return False


@contextmanager
def nvml_monitor(*, enabled: bool = True, device_index: int = 0) -> Iterator[NvmlSession]:
    """Context manager that yields an NVML sampling session.

    When NVML is unavailable or disabled, the yielded session has
    ``enabled == False`` and ``metrics`` remains ``None``.
    """

    session = NvmlSession(enabled=enabled, device_index=device_index)
    try:
        yield session
    finally:
        session.finalize()


__all__ = ["nvml_monitor", "NvmlSession"]
