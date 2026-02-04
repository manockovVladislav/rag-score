"""Реестр RAG-систем и декоратор для подключения через точку входа.

Идея:
- Каждая RAG-система регистрируется через декоратор @rag_system("name").
- Декоратор оборачивает фабрику, которая возвращает объект с методом answer(question).
- Реестр хранит фабрики; точка входа может подгружать модули, где описаны системы.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Dict, List, Protocol, Any
import importlib


@dataclass
class RagResponse:
    """Унифицированный ответ RAG-системы.

    answer  : финальный ответ пользователю (строка)
    contexts: список текстовых контекстов/пассажей, использованных системой
    """

    answer: str
    contexts: List[str]


class RagSystem(Protocol):
    """Протокол для совместимости разных реализаций RAG.

    Минимальный контракт: метод answer(question) -> RagResponse.
    """

    def answer(self, question: str) -> RagResponse:
        ...


# Внутренний реестр: имя системы -> фабрика
_REGISTRY: Dict[str, Callable[[], RagSystem]] = {}


def rag_system(name: str) -> Callable[[Callable[[], RagSystem]], Callable[[], RagSystem]]:
    """Декоратор для регистрации RAG-системы.

    Пример:
        @rag_system("e5-large")
        def build_e5():
            return MyE5Rag()

    Важно: функция должна возвращать объект с методом answer(question).
    """

    def decorator(factory: Callable[[], RagSystem]) -> Callable[[], RagSystem]:
        if name in _REGISTRY:
            raise ValueError(f"RAG-система '{name}' уже зарегистрирована")
        _REGISTRY[name] = factory
        return factory

    return decorator


def get_rag_system(name: str) -> RagSystem:
    """Получить экземпляр RAG-системы по имени из реестра."""

    if name not in _REGISTRY:
        available = ", ".join(sorted(_REGISTRY.keys()))
        raise KeyError(f"RAG-система '{name}' не найдена. Доступные: {available}")
    return _REGISTRY[name]()


def load_registries(modules: List[str]) -> None:
    """Импортировать модули, которые регистрируют RAG-системы.

    Каждый модуль должен содержать декораторы @rag_system(...).
    Импорт выполняется ради сайд-эффекта регистрации.
    """

    for module in modules:
        importlib.import_module(module)


def adapt_response(raw: Any) -> RagResponse:
    """Привести ответ системы к RagResponse.

    Допускаем несколько форматов для удобства интеграции:
    - RagResponse
    - dict с ключами "answer" и "contexts"
    - tuple/list (answer, contexts)
    """

    if isinstance(raw, RagResponse):
        return raw
    if isinstance(raw, dict):
        return RagResponse(answer=str(raw.get("answer", "")), contexts=list(raw.get("contexts", [])))
    if isinstance(raw, (list, tuple)) and len(raw) == 2:
        return RagResponse(answer=str(raw[0]), contexts=list(raw[1]))
    raise TypeError("Неподдерживаемый формат ответа от RAG-системы")
