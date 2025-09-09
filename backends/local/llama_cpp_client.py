"""llama.cpp inference client using llama-cpp-python bindings.

Provides in-process local inference with CUDA/cuBLAS support and configurable
GPU offloading. Alternative to Ollama for scenarios requiring more control
or avoiding HTTP daemon overhead.
"""

from __future__ import annotations

import logging
import time
import uuid
import warnings
from pathlib import Path
from typing import Any, Optional

try:
    from llama_cpp import Llama
except ImportError:
    raise ImportError(
        "llama-cpp-python is required for LlamaCppClient. "
        "Install with: pip install llama-cpp-python"
    ) from None

logger = logging.getLogger(__name__)


class LlamaCppClient:
    """In-process llama.cpp inference client.
    
    Implements the InferenceClient protocol using llama-cpp-python bindings.
    Supports both message-based chat completion and raw prompt generation.
    """
    
    def __init__(
        self,
        model_path: str,
        n_ctx: int = 4096,
        n_gpu_layers: int = -1,
        n_threads: Optional[int] = None,
        verbose: bool = False,
    ):
        """Initialize the llama.cpp client.
        
        Args:
            model_path: Path to GGUF model file
            n_ctx: Context window size (default 4096)
            n_gpu_layers: Number of layers to offload to GPU (-1 for all)
            n_threads: Number of CPU threads (None for auto)
            verbose: Enable verbose logging from llama.cpp
        """
        self.model_path = Path(model_path)
        self.n_ctx = n_ctx
        self.n_gpu_layers = n_gpu_layers
        self.n_threads = n_threads
        
        # Validate model exists
        if not self.model_path.exists():
            raise FileNotFoundError(f"Model file not found: {model_path}")
            
        # Check for potential VRAM issues with large models
        self._check_vram_warnings()
        
        # Initialize the model
        logger.info(f"Loading model: {model_path}")
        logger.info(f"Context: {n_ctx}, GPU layers: {n_gpu_layers}")
        
        try:
            self.llm = Llama(
                model_path=str(self.model_path),
                n_ctx=n_ctx,
                n_gpu_layers=n_gpu_layers,
                n_threads=n_threads,
                verbose=verbose,
            )
        except Exception as e:
            if "CUDA" in str(e) or "cuBLAS" in str(e):
                raise RuntimeError(
                    f"CUDA/cuBLAS initialization failed: {e}\n"
                    "This may indicate a mismatch between llama-cpp-python wheels "
                    "and your CUDA installation. Try reinstalling with: "
                    "pip uninstall llama-cpp-python && pip install llama-cpp-python --no-cache-dir"
                ) from e
            raise
            
        logger.info("Model loaded successfully")
    
    def _check_vram_warnings(self) -> None:
        """Issue warnings for potentially problematic model/VRAM combinations."""
        model_size_mb = self.model_path.stat().st_size / (1024 * 1024)
        
        # Rough heuristics for common quantization levels
        if model_size_mb > 12000:  # ~13B Q6_K or larger
            warnings.warn(
                f"Large model detected ({model_size_mb:.1f}MB). "
                f"This may exceed 16GB VRAM limits. Consider using Q4_K_M quantization "
                f"or reducing n_gpu_layers if you encounter OOM errors.",
                UserWarning,
                stacklevel=2
            )
        elif model_size_mb > 8000:  # ~8B Q6_K
            logger.info(f"Medium model size ({model_size_mb:.1f}MB) - should fit in 16GB VRAM")
    
    def generate(
        self,
        *,
        messages: Optional[list[dict[str, str]]] = None,
        prompt: Optional[str] = None,
        model: str = "",
        params: Optional[dict[str, Any]] = None,
    ) -> dict[str, Any]:
        """Generate text using llama.cpp.
        
        Args:
            messages: Chat messages in OpenAI format (preferred)
            prompt: Raw text prompt (alternative to messages)
            model: Model name (ignored - using loaded model)
            params: Generation parameters
            
        Returns:
            Dict with keys: text, finish_reason, usage, request_id, latency_s
        """
        if not messages and not prompt:
            raise ValueError("Either messages or prompt must be provided")
            
        if messages and prompt:
            raise ValueError("Provide either messages or prompt, not both")
        
        # Extract and validate parameters
        generation_params = self._extract_generation_params(params or {})
        
        start_time = time.time()
        request_id = str(uuid.uuid4())
        
        try:
            if messages:
                # Chat completion mode
                result = self.llm.create_chat_completion(
                    messages=messages,
                    **generation_params
                )
                text = result["choices"][0]["message"]["content"]
                finish_reason = result["choices"][0].get("finish_reason", "stop")
                usage = result.get("usage")
            else:
                # Raw completion mode
                result = self.llm(
                    prompt,
                    **generation_params
                )
                text = result["choices"][0]["text"]
                finish_reason = result["choices"][0].get("finish_reason", "stop")
                usage = result.get("usage")
                
        except Exception as e:
            logger.error(f"Generation failed: {e}")
            # Return error response in expected format
            return {
                "text": "",
                "finish_reason": "error",
                "usage": None,
                "request_id": request_id,
                "latency_s": time.time() - start_time,
                "error": str(e),
            }
        
        latency = time.time() - start_time
        
        return {
            "text": text,
            "finish_reason": finish_reason,
            "usage": usage,
            "request_id": request_id,
            "latency_s": latency,
        }
    
    def _extract_generation_params(self, params: dict[str, Any]) -> dict[str, Any]:
        """Extract and map parameters for llama.cpp generation."""
        generation_params = {}
        
        # Temperature
        if "temperature" in params:
            generation_params["temperature"] = float(params["temperature"])
            
        # Top-p sampling
        if "top_p" in params:
            generation_params["top_p"] = float(params["top_p"])
            
        # Top-k sampling  
        if "top_k" in params:
            generation_params["top_k"] = int(params["top_k"])
            
        # Max tokens (map from max_new_tokens)
        if "max_new_tokens" in params:
            generation_params["max_tokens"] = int(params["max_new_tokens"])
        elif "max_tokens" in params:
            generation_params["max_tokens"] = int(params["max_tokens"])
        else:
            generation_params["max_tokens"] = 1024  # Default
            
        # Stop sequences
        if "stop" in params and params["stop"]:
            stop = params["stop"]
            if isinstance(stop, str):
                generation_params["stop"] = [stop]
            elif isinstance(stop, list):
                generation_params["stop"] = stop
                
        # Repetition penalty
        if "repeat_penalty" in params:
            generation_params["repeat_penalty"] = float(params["repeat_penalty"])
            
        return generation_params