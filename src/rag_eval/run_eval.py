"""Ядро прогона ragas по золотым вопросам из Excel.

Сценарий:
- Читаем Excel с колонками question и ground_truth
- Поднимаем RAG-системы из реестра (через декоратор)
- Прогоняем ragas
- Сохраняем аккуратные отчеты: CSV/JSON + графики
"""

from __future__ import annotations
import json
import os
from datetime import datetime
from typing import List, Dict, Any

import pandas as pd
from datasets import Dataset

from ragas import evaluate
from ragas.metrics import (
    faithfulness,
    answer_relevancy,
    context_precision,
    context_recall,
)

from .registry import get_rag_system, load_registries, adapt_response


def _ensure_dir(path: str) -> None:
    """Создать папку, если ее нет."""

    os.makedirs(path, exist_ok=True)


def _load_gold(path: str) -> pd.DataFrame:
    """Загрузить Excel с золотыми вопросами.

    Ожидаемые колонки:
    - question: текст вопроса
    - ground_truth: эталонный ответ
    """

    df = pd.read_excel(path)
    missing = {"question", "ground_truth"} - set(df.columns)
    if missing:
        raise ValueError(f"В Excel не хватает колонок: {', '.join(sorted(missing))}")
    return df


def _run_system(system_name: str, questions: List[str], output_dir: str) -> pd.DataFrame:
    """Прогнать одну RAG-систему по списку вопросов.

    Возвращает DataFrame с полями: question, answer, contexts.
    """

    system = get_rag_system(system_name)

    answers: List[str] = []
    contexts: List[List[str]] = []

    for q in questions:
        # Ответ может быть RagResponse или другим допустимым форматом
        raw = system.answer(q)
        response = adapt_response(raw)
        answers.append(response.answer)
        contexts.append(response.contexts)

    df = pd.DataFrame({
        "question": questions,
        "answer": answers,
        "contexts": contexts,
    })

    # Для трассируемости сохраняем сырые ответы системы
    df.to_csv(os.path.join(output_dir, "raw_answers.csv"), index=False)
    return df


def _ragas_evaluate(dataset: Dataset) -> Any:
    """Выполнить ragas evaluation с базовым набором метрик."""

    metrics = [
        faithfulness,
        answer_relevancy,
        context_precision,
        context_recall,
    ]
    return evaluate(dataset, metrics=metrics)


def _result_to_frames(result: Any) -> Dict[str, Any]:
    """Унифицировать результат ragas в DataFrame и агрегатные метрики.

    В разных версиях ragas API может отличаться, поэтому используем
    максимально устойчивые пути доступа.
    """

    if hasattr(result, "to_pandas"):
        per_question = result.to_pandas()
    else:
        # Последний шанс: если result уже похож на DataFrame
        per_question = pd.DataFrame(result)

    if hasattr(result, "aggregate"):
        overall = result.aggregate()
        # aggregate может вернуть dict, поэтому нормализуем
        if isinstance(overall, dict):
            overall_dict = overall
        else:
            overall_dict = {k: float(v) for k, v in overall.items()}
    else:
        overall_dict = per_question.mean(numeric_only=True).to_dict()

    return {"per_question": per_question, "overall": overall_dict}


def _save_reports(system_name: str, result_data: Dict[str, Any], output_dir: str) -> None:
    """Сохранить метрики, CSV и графики по системе."""

    import matplotlib.pyplot as plt

    per_question = result_data["per_question"]
    overall = result_data["overall"]

    # Сохраняем полную таблицу per-question
    per_question_path = os.path.join(output_dir, "per_question_metrics.csv")
    per_question.to_csv(per_question_path, index=False)

    # Сохраняем общий отчет
    overall_path = os.path.join(output_dir, "overall_metrics.json")
    with open(overall_path, "w", encoding="utf-8") as f:
        json.dump(overall, f, ensure_ascii=False, indent=2)

    # График: бар-диаграмма агрегированных метрик
    plt.figure(figsize=(8, 4))
    plt.title(f"RAGAS metrics: {system_name}")
    plt.bar(list(overall.keys()), list(overall.values()))
    plt.xticks(rotation=30, ha="right")
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "overall_metrics.png"), dpi=200)
    plt.close()

    # График: метрики по вопросам (если есть числовые колонки)
    numeric_cols = per_question.select_dtypes(include=["number"]).columns
    if len(numeric_cols) > 0:
        plt.figure(figsize=(10, 5))
        for col in numeric_cols:
            plt.plot(per_question[col], label=col)
        plt.title(f"Per-question metrics: {system_name}")
        plt.xlabel("Question index")
        plt.ylabel("Score")
        plt.legend()
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, "per_question_metrics.png"), dpi=200)
        plt.close()

    # Markdown отчет для удобного просмотра
    report_md = os.path.join(output_dir, "report.md")
    with open(report_md, "w", encoding="utf-8") as f:
        f.write(f"# Отчет по системе: {system_name}\n\n")
        f.write("## Агрегированные метрики\n\n")
        for k, v in overall.items():
            f.write(f"- **{k}**: {v}\n")
        f.write("\n## Файлы\n\n")
        f.write("- raw_answers.csv\n")
        f.write("- per_question_metrics.csv\n")
        f.write("- overall_metrics.json\n")
        f.write("- overall_metrics.png\n")
        f.write("- per_question_metrics.png\n")


def run_eval(
    gold_path: str,
    registry_modules: List[str],
    system_names: List[str],
    output_root: str,
) -> None:
    """Запуск оценки для выбранных систем."""

    load_registries(registry_modules)

    gold_df = _load_gold(gold_path)

    run_stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_root = os.path.join(output_root, run_stamp)
    _ensure_dir(run_root)

    for system_name in system_names:
        system_dir = os.path.join(run_root, system_name)
        _ensure_dir(system_dir)

        # Снимаем ответы системы
        raw_df = _run_system(system_name, gold_df["question"].tolist(), system_dir)

        # Формируем датасет ragas
        dataset = Dataset.from_dict({
            "question": gold_df["question"].tolist(),
            "answer": raw_df["answer"].tolist(),
            "contexts": raw_df["contexts"].tolist(),
            "ground_truth": gold_df["ground_truth"].tolist(),
        })

        # Запускаем ragas и сохраняем результаты
        result = _ragas_evaluate(dataset)
        result_data = _result_to_frames(result)
        _save_reports(system_name, result_data, system_dir)


__all__ = ["run_eval"]
