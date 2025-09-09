from __future__ import annotations

import time
import uuid
from typing import Any, Dict, List, Optional

import httpx


class OllamaClient:
    """HTTP client for the local Ollama server.

    Exposes a minimal `generate` method compatible with the InferenceClient
    protocol defined in `backends.interfaces`.
    """

    def __init__(
        self,
        base_url: str = "http://127.0.0.1:11434",
        model: Optional[str] = None,
        *,
        request_timeout_s: float = 600.0,
        connect_timeout_s: float = 5.0,
    ) -> None:
        self.base: str = base_url.rstrip("/")
        self.model: Optional[str] = model
        # httpx supports a rich Timeout object; use connect/read/write limits.
        self._timeout = httpx.Timeout(
            connect=connect_timeout_s, read=request_timeout_s, write=30.0, pool=connect_timeout_s
        )

    # -----------------------------
    # Public API
    # -----------------------------
    def generate(
        self,
        *,
        messages: Optional[List[Dict[str, str]]] = None,
        prompt: Optional[str] = None,
        model: str = "",
        params: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """Run a single generation call against Ollama.

        Exactly one of `messages` or `prompt` should be provided. Returns a
        mapping with at least: {"text", "finish_reason"}. Optional fields
        include {"usage", "request_id", "latency_s"}.
        """
        if (messages is None and prompt is None) or (messages is not None and prompt is not None):
            raise ValueError("Provide exactly one of `messages` or `prompt` to generate().")

        effective_model = (model or self.model or "").strip()
        if not effective_model:
            raise ValueError(
                "No model specified. Pass `model` to generate() or set a default in OllamaClient(model=...)."
            )

        body: Dict[str, Any] = {
            "model": effective_model,
            "stream": False,
        }

        if messages is not None:
            body["messages"] = messages
        elif prompt is not None:
            body["prompt"] = prompt

        # Map common parameters to Ollama options.
        options = self._map_params_to_ollama_options(params or {})
        if options:
            body["options"] = options

        endpoint = "/api/chat" if messages is not None else "/api/generate"

        t0 = time.time()
        try:
            with httpx.Client(timeout=self._timeout) as client:
                resp = client.post(f"{self.base}{endpoint}", json=body)
                resp.raise_for_status()
        except httpx.ConnectError as e:
            raise ConnectionError(
                "Unable to connect to Ollama at {0}. Is the server running? Try: `ollama serve`.".format(self.base)
            ) from e
        except httpx.HTTPStatusError as e:
            # Add helpful hint for common model-not-found case.
            txt = e.response.text if e.response is not None else ""
            if "not found" in txt.lower() or e.response is not None and e.response.status_code == 404:
                raise httpx.HTTPStatusError(
                    f"Model '{effective_model}' not available in Ollama. Pull it first: `ollama pull {effective_model}`.\n"
                    f"Server response: {txt}",
                    request=e.request,
                    response=e.response,
                )
            raise
        except httpx.RequestError as e:
            raise RuntimeError(
                f"Request to Ollama failed: {e}. Verify the service is reachable at {self.base} and try again."
            ) from e

        latency_s = time.time() - t0
        data = resp.json()

        # Extract text depending on endpoint.
        if messages is not None:
            text = ((data.get("message") or {}) or {}).get("content", "")
        else:
            text = data.get("response", "")

        done_reason = data.get("done_reason") or ("stop" if data.get("done") else None)

        # Optional usage accounting if available from Ollama
        usage: Optional[Dict[str, Any]] = None
        if any(k in data for k in ("eval_count", "prompt_eval_count")):
            usage = {
                "prompt_tokens": data.get("prompt_eval_count"),
                "completion_tokens": data.get("eval_count"),
            }

        return {
            "text": text,
            "finish_reason": done_reason or "stop",
            "usage": usage,
            "request_id": str(uuid.uuid4()),
            "latency_s": latency_s,
        }

    def health_check(self, *, model: Optional[str] = None) -> Dict[str, Any]:
        """Check whether the Ollama service is reachable and (optionally) if a model is present.

        Returns a dict: {"ok": bool, "models": [names...], "has_model": bool | None}
        and raises a descriptive error if the server is not reachable.
        """
        url = f"{self.base}/api/tags"
        try:
            with httpx.Client(timeout=httpx.Timeout(connect=3.0, read=5.0, write=5.0, pool=3.0)) as client:
                resp = client.get(url)
                resp.raise_for_status()
        except httpx.ConnectError as e:
            raise ConnectionError(
                f"Cannot reach Ollama at {self.base}. Start it with `ollama serve` and retry."
            ) from e
        except httpx.RequestError as e:
            raise RuntimeError(f"Health check failed: {e}") from e

        payload = resp.json() or {}
        models_info = payload.get("models") or []
        names = [m.get("name") for m in models_info if isinstance(m, dict)]
        has_model = None
        if model:
            has_model = model in set(names)

        return {"ok": True, "models": names, "has_model": has_model}

    # -----------------------------
    # Internals
    # -----------------------------
    @staticmethod
    def _map_params_to_ollama_options(params: Dict[str, Any]) -> Dict[str, Any]:
        """Translate common generation params into Ollama's `options`.

        Recognized keys: temperature, top_p, top_k, max_new_tokens, stop, seed
        """
        options: Dict[str, Any] = {}
        if not params:
            return options

        if "temperature" in params and params["temperature"] is not None:
            options["temperature"] = float(params["temperature"])
        if "top_p" in params and params["top_p"] is not None:
            options["top_p"] = float(params["top_p"])
        if "top_k" in params and params["top_k"] is not None:
            options["top_k"] = int(params["top_k"])

        # Map "max_new_tokens" (our config) -> "num_predict" (Ollama)
        if "max_new_tokens" in params and params["max_new_tokens"] is not None:
            try:
                options["num_predict"] = int(params["max_new_tokens"])
            except (TypeError, ValueError):
                pass

        if "stop" in params and params["stop"] is not None:
            options["stop"] = params["stop"]

        if "seed" in params and params["seed"] is not None:
            try:
                options["seed"] = int(params["seed"])
            except (TypeError, ValueError):
                pass

        return options

