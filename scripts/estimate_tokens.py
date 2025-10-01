"""Utility helpers to approximate token counts for local backends.

The local engines (Ollama, llama.cpp) do not consistently return usage metrics.
This module provides a tolerant estimator that prefers `tiktoken` when
available and falls back to heuristic counts otherwise.
"""

from __future__ import annotations

from typing import Iterable, List, Mapping, Optional, Sequence, Union

try:  # pragma: no cover - optional dependency
    import tiktoken  # type: ignore
except Exception:  # pragma: no cover
    tiktoken = None  # type: ignore

MessageLike = Union[str, Mapping[str, object]]
MessagesInput = Union[Sequence[MessageLike], str, None]


def _flatten_content(content: object) -> List[str]:
    texts: List[str] = []
    if content is None:
        return texts
    if isinstance(content, str):
        texts.append(content)
        return texts
    if isinstance(content, Sequence):
        for part in content:
            if isinstance(part, str):
                texts.append(part)
            elif isinstance(part, Mapping):
                text_val = part.get("text")
                if isinstance(text_val, str):
                    texts.append(text_val)
    elif isinstance(content, Mapping):
        text_val = content.get("text")
        if isinstance(text_val, str):
            texts.append(text_val)
    return texts


def _normalize_messages(messages: MessagesInput) -> List[str]:
    if messages is None:
        return []
    if isinstance(messages, str):
        return [messages]

    texts: List[str] = []
    for msg in messages:
        if isinstance(msg, str):
            texts.append(msg)
            continue
        if not isinstance(msg, Mapping):
            continue
        content = msg.get("content")
        texts.extend(_flatten_content(content))
    return texts


def _choose_encoding(tokenizer_hint: Optional[str]):
    if tiktoken is None:
        return None
    if tokenizer_hint:
        try:
            return tiktoken.encoding_for_model(tokenizer_hint)
        except Exception:
            try:
                return tiktoken.get_encoding(tokenizer_hint)
            except Exception:
                pass
    # Default to a widely available encoding
    for name in ("cl100k_base", "o200k_base", "p50k_base"):
        try:
            return tiktoken.get_encoding(name)
        except Exception:
            continue
    return None


def _heuristic_count(text: str) -> int:
    text = text.strip()
    if not text:
        return 0
    words = text.replace("\n", " ").split()
    if words:
        return max(1, len(words))
    # Fallback to character-based approximation (~4 chars/token)
    return max(1, (len(text) + 3) // 4)


def estimate_tokens(messages: MessagesInput, tokenizer_hint: Optional[str] = None) -> dict:
    """Estimate token counts for a message sequence.

    Returns a dictionary containing ``total_tokens``. Callers that need prompt
    vs. completion splits can invoke the function on the prompt-only messages
    and again after appending the assistant response.
    """

    normalized = _normalize_messages(messages)
    if not normalized:
        return {"total_tokens": 0}

    text = "\n".join(normalized)
    encoder = _choose_encoding(tokenizer_hint)
    if encoder is not None:
        try:
            return {"total_tokens": len(encoder.encode(text))}
        except Exception:
            pass
    return {"total_tokens": _heuristic_count(text)}


__all__ = ["estimate_tokens"]
