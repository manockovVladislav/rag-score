"""Единая фабрика LLM-клиентов для проекта (GigaChat/KoboldCpp).

Модуль нужен как одна точка входа для выбора backend по имени окружения.
Это позволяет не размазывать условную логику по всему коду RAG/RAGAS и
гарантирует единые настройки таймаутов, параметров модели и троттлинга.
"""

from __future__ import annotations

import functools
import time
from typing import Any

from .runtime_control import (
    acquire_gigachat_request_slot,
    current_request_source,
    gigachat_min_interval_seconds,
    log_runtime_event,
)


def _normalize_timeout(timeout: float | None) -> float | None:
    """Нормализует значение таймаута для LLM-клиента.

    Зачем:
    - в проекте таймауты могут приходить из env строками и с разными значениями;
    - принято правило: `None` и `<= 0` означают «без лимита».

    Почему так:
    - разные backend-и интерпретируют «нулевой» таймаут по-разному;
    - единая нормализация делает поведение предсказуемым во всех местах.
    """

    if timeout is None:
        return None
    value = float(timeout)
    if value <= 0:
        return None
    return value


def normalize_backend_name(name: str) -> str:
    """Приводит произвольное имя backend к каноническому ключу.

    Зачем:
    - пользователи и конфиги используют разные написания (`cobalt`, `kobold`,
      `giga` и т.д.);
    - остальной код должен работать только с ограниченным набором ключей.

    Почему так:
    - централизованная таблица алиасов исключает дублирование условных веток;
    - при неподдерживаемом значении ошибка возникает сразу и явно.
    """

    key = (name or "").strip().lower()
    aliases = {
        "gigachat": "gigachat",
        "giga": "gigachat",
        "koboldcpp": "koboldcpp",
        "kobold": "koboldcpp",
        "cobold": "koboldcpp",
        "cobolt": "koboldcpp",
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
    """Создает и настраивает экземпляр LangChain-чата для выбранного backend.

    Что делает:
    - нормализует имя backend;
    - для `gigachat` собирает параметры (`base_url`, `model_name`, токен и т.п.),
      создает клиент и вешает hook-и троттлинга/логирования;
    - для `koboldcpp` создает `ChatOpenAI`-совместимый клиент с локальным endpoint.

    Зачем:
    - использовать один и тот же конструктор в RAG и в judge-части RAGAS;
    - поддерживать «одну точку входа» при переключении окружения.

    Почему так:
    - проще контролировать совместимость старых/новых имен параметров;
    - проще отлаживать сеть/таймауты, потому что инициализация централизована.
    """

    backend_name = normalize_backend_name(backend)
    # timeout <= 0 трактуется как "без лимита". Для изолированного стенда задайте 5 секунд.
    # Обычно приходит из MODEL_TIMEOUT_SECONDS/JUDGE_TIMEOUT_SECONDS.
    timeout = _normalize_timeout(timeout)

    if backend_name == "gigachat":
        from langchain_gigachat.chat_models import GigaChat

        kwargs: dict[str, Any] = {}
        if timeout is not None:
            kwargs["timeout"] = timeout
        opts = dict(gigachat_options or {})

        model_name = opts.pop("model_name", None) or opts.pop("model", None)
        api_token = opts.pop("api_token", None)
        access_token = opts.pop("access_token", None)
        credentials = opts.pop("credentials", None)
        base_url = opts.pop("base_url", None)

        if model_name:
            kwargs["model"] = model_name
        if base_url:
            kwargs["base_url"] = base_url
        if credentials:
            kwargs["credentials"] = credentials
        elif access_token or api_token:
            kwargs["access_token"] = access_token or api_token

        # Preserve additional optional SDK fields (scope, ssl, etc).
        for key, value in opts.items():
            if value is not None:
                kwargs[key] = value

        model = GigaChat(**kwargs)
        _attach_gigachat_hooks(model)
        log_runtime_event(
            "gigachat",
            "client_initialized",
            model_name=kwargs.get("model"),
            base_url=kwargs.get("base_url"),
            timeout_sec=kwargs.get("timeout"),
            min_interval_sec=round(gigachat_min_interval_seconds(), 3),
        )
        return model

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


def _attach_gigachat_hooks(model: Any) -> None:
    """Подключает к экземпляру GigaChat обертки троттлинга и runtime-логов.

    Что делает:
    - патчит внутренние sync-методы запросов (`_generate`, `_stream`);
    - перед каждым запросом резервирует слот через rate limiter;
    - пишет события `request_started/request_finished/request_failed`.

    Зачем:
    - в изолированной среде требуется жесткий интервал между запросами;
    - нужна прозрачная диагностика того, кто и когда сделал вызов.

    Почему реализовано как monkey-patch:
    - это не требует менять внешний API вызова модели (`invoke` и т.д.);
    - логика остается в одном месте и действует для всех пользователей клиента.
    """

    if getattr(model, "_codex_gigachat_hooks_attached", False):
        return

    def wrap_sync(method_name: str):
        """Патчит один синхронный метод запроса у конкретного клиента."""

        if not hasattr(model, method_name):
            return
        original = getattr(model, method_name)

        @functools.wraps(original)
        def wrapped(*args, **kwargs):
            """Выполняет исходный метод под контролем троттлинга и логирования.

            Порядок важен:
            1) получить слот запроса (при необходимости подождать);
            2) записать событие старта;
            3) вызвать оригинальный метод;
            4) записать итог (успех/ошибка + длительность).
            """

            source = current_request_source()
            acquire_gigachat_request_slot(
                component="gigachat",
                reason=f"{source}:{method_name}",
            )
            log_runtime_event("gigachat", "request_started", source=source, method=method_name)
            started = time.perf_counter()
            try:
                result = original(*args, **kwargs)
            except Exception as exc:
                elapsed = time.perf_counter() - started
                log_runtime_event(
                    "gigachat",
                    "request_failed",
                    source=source,
                    method=method_name,
                    elapsed_sec=round(elapsed, 3),
                    error=f"{type(exc).__name__}: {exc}",
                )
                raise
            elapsed = time.perf_counter() - started
            log_runtime_event(
                "gigachat",
                "request_finished",
                source=source,
                method=method_name,
                elapsed_sec=round(elapsed, 3),
            )
            return result

        object.__setattr__(model, method_name, wrapped)

    for method in ("_generate", "_stream"):
        wrap_sync(method)

    object.__setattr__(model, "_codex_gigachat_hooks_attached", True)


__all__ = ["build_langchain_chat_model", "normalize_backend_name"]
