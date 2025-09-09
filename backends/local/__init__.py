"""Local inference backends.

This package contains clients for running inference against local engines
such as Ollama (HTTP API) or llama.cpp (in-process bindings).
"""

__all__ = [
    "ollama_client",
    "local_batch",
]
