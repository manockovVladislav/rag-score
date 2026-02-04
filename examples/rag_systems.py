"""Пример регистрации нескольких RAG-систем через декоратор.

Здесь демонстрационные заглушки. В боевом варианте:
- замените классы на реальные пайплайны
- внутри answer() вызывайте вашу RAG-систему
"""

from typing import List

from rag_eval import rag_system
from rag_eval.registry import RagResponse


class _BaseMockRag:
    """Базовая заглушка RAG.

    Реальная система должна:
    - собрать контексты (пассажи)
    - сгенерировать ответ
    - вернуть RagResponse
    """

    def __init__(self, name: str):
        self._name = name

    def answer(self, question: str) -> RagResponse:
        # Заглушка: имитируем работу, возвращая один контекст и простое эхо-ответ
        contexts: List[str] = [f"[{self._name}] Контекст для вопроса: {question}"]
        answer = f"[{self._name}] Ответ на: {question}"
        return RagResponse(answer=answer, contexts=contexts)


@rag_system("e5-large")
def build_e5_large():
    """Пример подключения самописной RAG-системы на e5-large."""

    return _BaseMockRag("e5-large")


@rag_system("bge-3")
def build_bge_3():
    """Пример подключения самописной RAG-системы на bge-3."""

    return _BaseMockRag("bge-3")


@rag_system("langchain")
def build_langchain():
    """Пример подключения RAG-системы на LangChain."""

    return _BaseMockRag("langchain")
