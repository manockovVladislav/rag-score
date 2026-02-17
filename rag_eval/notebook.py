"""Утилиты для запуска оценки из Jupyter."""

from __future__ import annotations

from typing import List

from .run_eval import run_eval


def run_eval_notebook(
    gold_path: str,
    registry_modules: List[str],
    system_names: List[str],
    output_root: str = "outputs",
) -> None:
    """Запуск ragas-оценки из ноутбука.

    Пример:
        from rag_eval.notebook import run_eval_notebook
        run_eval_notebook(
            "data/gold_questions.xlsx",
            ["rag_systems"],
            ["e5-large", "bge-3", "langchain"],
        )
    """

    run_eval(gold_path, registry_modules, system_names, output_root)
