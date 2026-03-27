"""Core LLM helpers: backend factory and runtime controls."""

from .llm_interface import build_langchain_chat_model, normalize_backend_name
from .runtime_control import (
    current_request_source,
    ensure_gigachat_gap,
    gigachat_debug_enabled,
    gigachat_min_interval_seconds,
    is_gigachat_model,
    log_runtime_event,
    request_source,
)

__all__ = [
    "build_langchain_chat_model",
    "normalize_backend_name",
    "current_request_source",
    "ensure_gigachat_gap",
    "gigachat_debug_enabled",
    "gigachat_min_interval_seconds",
    "is_gigachat_model",
    "log_runtime_event",
    "request_source",
]
