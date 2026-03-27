"""Общие runtime-инструменты для троттлинга и диагностики GigaChat.

Модуль отделен от бизнес-логики RAG, чтобы:
- контролировать интервал запросов из одного места;
- единообразно логировать этапы и вызовы;
- переиспользовать те же правила и в RAG, и в RAGAS judge.
"""

from __future__ import annotations

from contextlib import contextmanager
from contextvars import ContextVar
from datetime import datetime
import os
import threading
import time
import typing as t


_DEFAULT_MIN_INTERVAL_SECONDS = 6.1
_MIN_INTERVAL_ENV = "GIGACHAT_MIN_INTERVAL_SECONDS"
_DEBUG_LOG_ENV = "GIGACHAT_DEBUG_LOG"

_request_source: ContextVar[str] = ContextVar("request_source", default="unknown")
_request_lock = threading.RLock()
_last_request_started_monotonic: float | None = None


def _parse_bool(value: str | None, *, default: bool) -> bool:
    """Преобразует строку из env в bool со страховкой по умолчанию.

    Поддерживает типовые формы (`1/0`, `true/false`, `yes/no`, `on/off`).
    Если значение неизвестное, возвращает `default`, чтобы не падать
    из-за опечатки в переменной окружения.
    """

    if value is None:
        return default
    normalized = value.strip().lower()
    if normalized in {"1", "true", "yes", "on"}:
        return True
    if normalized in {"0", "false", "no", "off"}:
        return False
    return default


def gigachat_debug_enabled() -> bool:
    """Возвращает, включены ли отладочные runtime-логи GigaChat.

    Читает флаг из `GIGACHAT_DEBUG_LOG`. По умолчанию включено, потому что
    в изолированной среде логи обычно важны для поиска таймаутов/лимитов.
    """

    return _parse_bool(os.environ.get(_DEBUG_LOG_ENV), default=True)


def gigachat_min_interval_seconds() -> float:
    """Читает и валидирует минимальный интервал между запросами к GigaChat.

    Источник: `GIGACHAT_MIN_INTERVAL_SECONDS`.
    Если значение отсутствует/некорректно/<=0, используется безопасный
    дефолт `6.1`, чтобы гарантировать паузу больше 6 секунд.
    """

    raw = os.environ.get(_MIN_INTERVAL_ENV, str(_DEFAULT_MIN_INTERVAL_SECONDS))
    try:
        value = float(raw)
    except Exception:
        return _DEFAULT_MIN_INTERVAL_SECONDS
    if value <= 0:
        return _DEFAULT_MIN_INTERVAL_SECONDS
    return value


def _timestamp() -> str:
    """Формирует локальный timestamp для строк логов."""

    return datetime.now().astimezone().strftime("%Y-%m-%d %H:%M:%S")


def _format_field(value: t.Any) -> str:
    """Приводит значение поля лога к компактной строке.

    Для float использует короткий формат с 3 знаками после запятой, чтобы
    логи были читаемыми и не разрастались.
    """

    if isinstance(value, float):
        return f"{value:.3f}"
    return str(value)


def log_runtime_event(component: str, event: str, **fields: t.Any) -> None:
    """Печатает единообразное runtime-событие в stdout.

    Формат:
    `[YYYY-mm-dd HH:MM:SS] [component] event key=value ...`

    Зачем:
    - быстро понять, какой этап сейчас выполняется;
    - видеть паузы троттлинга и длительность запросов без внешнего логгера.
    """

    if not gigachat_debug_enabled():
        return
    base = f"[{_timestamp()}] [{component}] {event}"
    suffix = " ".join(f"{key}={_format_field(val)}" for key, val in fields.items() if val is not None)
    print(f"{base} {suffix}".rstrip(), flush=True)


def current_request_source() -> str:
    """Возвращает текущий логический источник запросов (`rag`, `ragas`, ...)."""

    return _request_source.get()


@contextmanager
def request_source(source: str):
    """Временный контекст для маркировки исходящих запросов по источнику.

    Использует `ContextVar`, а не глобальную переменную, чтобы корректно
    работать в разных потоках/асинхронных задачах и не смешивать метки.
    """

    normalized = (source or "").strip() or "unknown"
    token = _request_source.set(normalized)
    try:
        yield
    finally:
        _request_source.reset(token)


def ensure_gigachat_gap(
    *,
    component: str,
    reason: str,
    min_interval_seconds: float | None = None,
) -> float:
    """Выдерживает минимальную паузу *после* последнего запроса GigaChat.

    Это «мягкая» функция: она только дожидается нужного интервала, но не
    резервирует новый запрос. Подходит для явных пауз между этапами пайплайна
    (`RAG -> wait -> RAGAS -> wait -> next run`).

    Returns:
        Фактическое время сна в секундах.
    """

    interval = min_interval_seconds or gigachat_min_interval_seconds()
    with _request_lock:
        if _last_request_started_monotonic is None:
            return 0.0

        elapsed = time.monotonic() - _last_request_started_monotonic
        sleep_for = max(0.0, interval - elapsed)
        if sleep_for > 0:
            log_runtime_event(
                component,
                "throttle_sleep",
                reason=reason,
                sleep_sec=round(sleep_for, 3),
                min_interval_sec=round(interval, 3),
            )
            time.sleep(sleep_for)
        return sleep_for


def acquire_gigachat_request_slot(
    *,
    component: str,
    reason: str,
    min_interval_seconds: float | None = None,
) -> float:
    """Резервирует слот под новый запрос с учетом лимита интервала.

    В отличие от `ensure_gigachat_gap`, эта функция не только ждет, но и
    обновляет внутреннюю метку «последний старт запроса». Ее нужно вызывать
    непосредственно перед отправкой запроса в сеть.

    Почему разделено на две функции:
    - `ensure_*` удобна для межэтапных пауз;
    - `acquire_*` критична для защиты каждого реального API-вызова.

    Returns:
        Фактическое время сна в секундах.
    """

    global _last_request_started_monotonic

    interval = min_interval_seconds or gigachat_min_interval_seconds()
    with _request_lock:
        sleep_for = 0.0
        if _last_request_started_monotonic is not None:
            elapsed = time.monotonic() - _last_request_started_monotonic
            sleep_for = max(0.0, interval - elapsed)
            if sleep_for > 0:
                log_runtime_event(
                    component,
                    "throttle_sleep",
                    reason=reason,
                    sleep_sec=round(sleep_for, 3),
                    min_interval_sec=round(interval, 3),
                )
                time.sleep(sleep_for)
        _last_request_started_monotonic = time.monotonic()
        return sleep_for


def is_gigachat_model(model: t.Any) -> bool:
    """Эвристически определяет, относится ли объект модели к GigaChat.

    Нужна для условного включения межэтапного throttling: если backend не
    GigaChat, лишние паузы не добавляются.
    """

    if model is None:
        return False
    name = type(model).__name__.lower()
    module = type(model).__module__.lower()
    return "gigachat" in name or "gigachat" in module
