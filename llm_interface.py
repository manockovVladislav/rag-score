"""Unified LangChain chat backend factory (GigaChat / KoboldCpp)."""

from __future__ import annotations

from typing import Any


def _normalize_timeout(timeout: float | None) -> float | None:
    if timeout is None:
        return None
    value = float(timeout)
    if value <= 0:
        return None
    return value


def normalize_backend_name(name: str) -> str:
    key = (name or "").strip().lower()
    aliases = {
        "gigachat": "gigachat",
        "giga": "gigachat",
        "koboldcpp": "koboldcpp",
        "kobold": "koboldcpp",
        "koboltcpp": "koboldcpp",
        "kobalt": "koboldcpp",
        "cobalt": "koboldcpp",
    }
    if key not in aliases:
        raise ValueError(f"Неподдерживаемый LLM backend: {name}")
    return aliases[key]


def build_langchain_chat_model(
    backend: str,
    timeout: float | None,
    *,
    gigachat_options: dict[str, Any] | None = None,
    koboldcpp_options: dict[str, Any] | None = None,
):
    backend_name = normalize_backend_name(backend)
    # timeout <= 0 трактуется как "без лимита". Для изолированного стенда задайте 5 секунд.
    # Обычно приходит из MODEL_TIMEOUT_SECONDS/JUDGE_TIMEOUT_SECONDS.
    timeout = _normalize_timeout(timeout)

    if backend_name == "gigachat":
        from langchain_gigachat.chat_models import GigaChat

        kwargs: dict[str, Any] = {}
        if timeout is not None:
            kwargs["timeout"] = timeout
        # gigachat_options может содержать model/credentials/scope/base_url и другие поддерживаемые поля SDK.
        for key, value in (gigachat_options or {}).items():
            if value is not None:
                kwargs[key] = value
        return GigaChat(**kwargs)

    try:
        from langchain_openai import ChatOpenAI
    except Exception as exc:
        raise RuntimeError(
            "Для backend='koboldcpp' установите зависимость langchain-openai"
        ) from exc

    opts = koboldcpp_options or {}
    base_url = opts.get("base_url") or "http://127.0.0.1:5001/v1"
    api_key = opts.get("api_key") or "koboldcpp"
    model = opts.get("model") or "koboldcpp"
    temperature = opts.get("temperature") if "temperature" in opts else 0.0
    max_retries = int(opts.get("max_retries", 0))

    kwargs = {
        "base_url": base_url,
        "api_key": api_key,
        "model": model,
        "temperature": temperature,
        "max_retries": max_retries,
    }
    if timeout is not None:
        kwargs["timeout"] = timeout
    if "max_tokens" in opts and opts["max_tokens"] is not None:
        kwargs["max_tokens"] = int(opts["max_tokens"])

    return ChatOpenAI(**kwargs)


__all__ = ["build_langchain_chat_model", "normalize_backend_name"]
