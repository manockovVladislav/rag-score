"""Extended ragas evaluation pipeline for one imported RAG system.

Outputs:
- scores.csv (full row-level data, including ragas + diagnostics)
- scores_compact.csv (compact view for quick reading)
- summary.csv (numeric summary over all metrics)
- summary_ragas.csv
- summary_bge_m3.csv
- summary_runtime.csv
- config.csv (effective runtime parameters)
- parameter_guide.csv (how to read key parameters)
- run_meta.json
- report.html
"""

from __future__ import annotations

from datetime import datetime, timezone
from html import escape
from pathlib import Path
import json
import os
import re
import time
import typing as t

from datasets import Dataset
import numpy as np
import pandas as pd
from ragas import evaluate
from ragas.metrics import (
    answer_relevancy,
    faithfulness,
    context_precision,
    context_recall,
)
from sentence_transformers import SentenceTransformer

try:
    from ragas.llms import LangchainLLMWrapper
    from ragas.embeddings import LangchainEmbeddingsWrapper
except Exception:
    LangchainLLMWrapper = None
    LangchainEmbeddingsWrapper = None


TEXT_PREVIEW_LIMIT = 260
_TOKEN_RE = re.compile(r"\w+", flags=re.UNICODE)


class RagResponse:
    def __init__(self, answer: str, contexts: list[str]) -> None:
        self.answer = answer
        self.contexts = contexts


class SentenceTransformerEmbeddingsAdapter:
    """LangChain-compatible embeddings over a preloaded SentenceTransformer."""

    def __init__(self, embedder: SentenceTransformer, device: str) -> None:
        self._embedder = embedder
        self._device = device

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        if not texts:
            return []
        vectors = self._embedder.encode(
            texts,
            normalize_embeddings=True,
            convert_to_numpy=True,
            device=self._device,
        )
        return vectors.tolist()

    def embed_query(self, text: str) -> list[float]:
        vectors = self.embed_documents([text])
        return vectors[0] if vectors else []


def _default_bge_model_path() -> str:
    return os.environ.get("RAG_EVAL_BGE_M3_MODEL_PATH", "/home/vladislav/models/bge-m3")


def _default_bge_device() -> str:
    return os.environ.get("RAG_EVAL_BGE_M3_DEVICE", os.environ.get("EMBEDDING_DEVICE", "cpu")).strip().lower()


def adapt_response(raw) -> RagResponse:
    if isinstance(raw, RagResponse):
        return raw
    if isinstance(raw, dict):
        return RagResponse(answer=str(raw.get("answer", "")), contexts=list(raw.get("contexts", [])))
    if isinstance(raw, (list, tuple)) and len(raw) == 2:
        return RagResponse(answer=str(raw[0]), contexts=list(raw[1]))

    answer = getattr(raw, "answer", None)
    contexts = getattr(raw, "contexts", None)
    if answer is not None and contexts is not None:
        return RagResponse(answer=str(answer), contexts=list(contexts))

    raise TypeError("Unsupported RAG response format")


def _extract_shared_chat_model(rag_system):
    getter = getattr(rag_system, "get_chat_model", None)
    if callable(getter):
        return getter()
    return getattr(rag_system, "_llm", None)


def _extract_shared_sentence_embedder(rag_system):
    getter = getattr(rag_system, "get_sentence_embedder", None)
    if callable(getter):
        return getter()

    retriever = getattr(rag_system, "_retriever", None)
    if retriever is None:
        return None
    return getattr(retriever, "_embedder", None)


def _extract_shared_embedding_device(rag_system) -> str | None:
    getter = getattr(rag_system, "get_embedding_device", None)
    if callable(getter):
        value = getter()
        if value is not None:
            return str(value)

    retriever = getattr(rag_system, "_retriever", None)
    if retriever is None:
        return None
    value = getattr(retriever, "_embedding_device", None)
    if value is None:
        return None
    return str(value)


def _extract_shared_embedding_model_path(rag_system) -> str | None:
    getter = getattr(rag_system, "get_embedding_model_path", None)
    if callable(getter):
        value = getter()
        if value is not None:
            return str(value)
    return None


def _resolve_judge_llm(judge_llm, rag_system):
    if judge_llm is not None:
        return judge_llm

    shared_llm = _extract_shared_chat_model(rag_system)
    if shared_llm is None:
        raise ValueError("judge_llm не передан и в rag_system не найден LLM.")
    if LangchainLLMWrapper is None:
        return shared_llm
    return LangchainLLMWrapper(shared_llm)


def _resolve_judge_embeddings(judge_embeddings, rag_system):
    if judge_embeddings is not None:
        return judge_embeddings
    if LangchainEmbeddingsWrapper is None:
        return None

    embedder = _extract_shared_sentence_embedder(rag_system)
    if embedder is None:
        return None
    device = _extract_shared_embedding_device(rag_system) or _default_bge_device()
    return LangchainEmbeddingsWrapper(SentenceTransformerEmbeddingsAdapter(embedder, device))


def build_shared_judges_from_rag_system(rag_system) -> tuple[t.Any, t.Any]:
    judge_llm = _resolve_judge_llm(None, rag_system)
    judge_embeddings = _resolve_judge_embeddings(None, rag_system)
    return judge_llm, judge_embeddings


def _normalize_gold(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df.columns = [str(c).strip().lower().replace(" ", "_").replace("-", "_") for c in df.columns]

    rename_map = {}
    for col in df.columns:
        if col in {"question", "query", "q", "вопрос", "вопрос_пользователя"}:
            rename_map[col] = "question"
        if col in {"ground_truth", "reference", "answer_gt", "эталон", "эталонный_ответ", "правильный_ответ"}:
            rename_map[col] = "ground_truth"

    df = df.rename(columns=rename_map)
    if "question" not in df.columns:
        raise ValueError(f"Колонка question не найдена. Доступные колонки: {list(df.columns)}")

    df["question"] = df["question"].astype(str).str.strip()
    df = df[df["question"].ne("")].reset_index(drop=True)

    if "ground_truth" in df.columns:
        df["ground_truth"] = df["ground_truth"].astype(str).str.strip()

    return df


def load_xlsx(path: str | Path) -> pd.DataFrame:
    gold_path = Path(path)
    if not gold_path.exists():
        raise FileNotFoundError(f"Не найден файл: {gold_path}")
    return _normalize_gold(pd.read_excel(gold_path))


def latest_xlsx(data_dir: str | Path) -> Path:
    root = Path(data_dir)
    xlsx_files = sorted(root.glob("*.xlsx"), key=lambda p: p.stat().st_mtime, reverse=True)
    if not xlsx_files:
        raise FileNotFoundError(f"Нет .xlsx в папке {root.resolve()}")
    return xlsx_files[0]


def _word_count(text: str) -> int:
    return len(_TOKEN_RE.findall(text or ""))


def run_rag_over_questions(gold_df: pd.DataFrame, rag_system) -> pd.DataFrame:
    rows: list[dict] = []
    for question in gold_df["question"].tolist():
        started = time.perf_counter()
        response = adapt_response(rag_system.answer(question))
        elapsed = time.perf_counter() - started

        answer = (response.answer or "").strip()
        contexts = [str(item).strip() for item in (response.contexts or []) if str(item).strip()]
        rows.append(
            {
                "question": question,
                "answer": answer,
                "contexts": contexts,
                "rag_answer_latency_sec": float(round(elapsed, 6)),
                "retrieved_context_count": int(len(contexts)),
                "contexts_total_char_len": int(sum(len(ctx) for ctx in contexts)),
                "answer_char_len": int(len(answer)),
                "answer_word_count": int(_word_count(answer)),
            }
        )

    eval_df = pd.DataFrame(rows)
    if "ground_truth" in gold_df.columns:
        eval_df["ground_truth"] = gold_df["ground_truth"].tolist()
        eval_df["ground_truth_char_len"] = eval_df["ground_truth"].astype(str).map(len)
        eval_df["ground_truth_word_count"] = eval_df["ground_truth"].astype(str).map(_word_count)
    return eval_df


def choose_metrics(eval_df: pd.DataFrame, judge_embeddings=None):
    # answer_relevancy requires embeddings. If embeddings are not passed explicitly,
    # ragas may create a default provider (typically external API).
    metrics = [faithfulness]
    if judge_embeddings is not None:
        metrics.insert(0, answer_relevancy)
    if "ground_truth" in eval_df.columns:
        metrics.extend([context_precision, context_recall])
    return metrics


def evaluate_with_ragas(
    eval_df: pd.DataFrame,
    judge_llm,
    judge_embeddings=None,
    ragas_run_config=None,
) -> tuple[pd.DataFrame, float]:
    dataset = Dataset.from_pandas(eval_df, preserve_index=False)
    started = time.perf_counter()
    result = evaluate(
        dataset,
        metrics=choose_metrics(eval_df, judge_embeddings=judge_embeddings),
        llm=judge_llm,
        embeddings=judge_embeddings,
        run_config=ragas_run_config,
    )
    elapsed = time.perf_counter() - started
    scores_df = result.to_pandas()
    return scores_df, elapsed


def summarize(scores_df: pd.DataFrame) -> pd.DataFrame:
    numeric = scores_df.select_dtypes(include="number")
    if numeric.empty:
        return pd.DataFrame()
    return (
        pd.DataFrame(
            {
                "mean": numeric.mean(),
                "std": numeric.std(),
                "min": numeric.min(),
                "max": numeric.max(),
                "na": numeric.isna().sum(),
            }
        )
        .reset_index()
        .rename(columns={"index": "metric"})
    )


def _token_set(text: str) -> set[str]:
    return {token.lower() for token in _TOKEN_RE.findall(text or "")}


def _token_jaccard(a: str, b: str) -> float:
    left = _token_set(a)
    right = _token_set(b)
    if not left and not right:
        return 1.0
    if not left or not right:
        return 0.0
    return float(len(left & right) / len(left | right))


def _cosine(vec_a: np.ndarray | None, vec_b: np.ndarray | None) -> float | None:
    if vec_a is None or vec_b is None:
        return None
    return float(np.dot(vec_a, vec_b))


def _shorten_text(value: str, limit: int = TEXT_PREVIEW_LIMIT) -> str:
    text = (value or "").strip()
    if len(text) <= limit:
        return text
    return text[: limit - 1].rstrip() + "…"


def _to_float_or_none(value: float | None) -> float | None:
    if value is None:
        return None
    return float(value)


def _best(values: list[float]) -> float | None:
    if not values:
        return None
    return float(max(values))


def _mean(values: list[float]) -> float | None:
    if not values:
        return None
    return float(np.mean(values))


def compute_bge_m3_diagnostics(
    eval_df: pd.DataFrame,
    *,
    model_path: str | None = None,
    device: str | None = None,
    embedder: SentenceTransformer | None = None,
) -> tuple[pd.DataFrame, dict[str, t.Any]]:
    resolved_model_path = model_path or _default_bge_model_path()
    resolved_device = (device or _default_bge_device() or "cpu").strip().lower()
    diag_meta: dict[str, t.Any] = {
        "enabled": False,
        "model_path": resolved_model_path,
        "device": resolved_device,
        "error": None,
        "reused_embedder": embedder is not None,
    }
    if eval_df.empty:
        return pd.DataFrame(index=eval_df.index), diag_meta

    resolved_embedder = embedder
    if resolved_embedder is None:
        try:
            resolved_embedder = SentenceTransformer(resolved_model_path, device=resolved_device or "cpu")
        except Exception as exc:
            diag_meta["error"] = str(exc)
            return pd.DataFrame(index=eval_df.index), diag_meta

    unique_texts: list[str] = []
    text_to_idx: dict[str, int] = {}

    def register(text: str) -> str:
        cleaned = (text or "").strip()
        if cleaned not in text_to_idx:
            text_to_idx[cleaned] = len(unique_texts)
            unique_texts.append(cleaned)
        return cleaned

    rows_payload: list[dict[str, t.Any]] = []
    for row in eval_df.to_dict(orient="records"):
        question = register(str(row.get("question", "")))
        answer = register(str(row.get("answer", "")))
        ground_truth = register(str(row.get("ground_truth", "")))
        contexts = [register(str(ctx)) for ctx in (row.get("contexts") or []) if str(ctx).strip()]
        rows_payload.append(
            {
                "question": question,
                "answer": answer,
                "ground_truth": ground_truth,
                "contexts": contexts,
            }
        )

    embeddings = resolved_embedder.encode(
        unique_texts,
        normalize_embeddings=True,
        convert_to_numpy=True,
        device=resolved_device or "cpu",
    ).astype("float32")

    diag_meta["enabled"] = True
    diag_meta["unique_texts_embedded"] = int(len(unique_texts))

    def vector(text: str) -> np.ndarray | None:
        if not text:
            return None
        return embeddings[text_to_idx[text]]

    diag_rows: list[dict[str, t.Any]] = []
    for payload in rows_payload:
        question = payload["question"]
        answer = payload["answer"]
        ground_truth = payload["ground_truth"]
        contexts: list[str] = payload["contexts"]

        question_vec = vector(question)
        answer_vec = vector(answer)
        ground_truth_vec = vector(ground_truth)
        context_vecs = [vector(ctx) for ctx in contexts]

        q_ctx = [_to_float_or_none(_cosine(question_vec, ctx_vec)) for ctx_vec in context_vecs if ctx_vec is not None]
        a_ctx = [_to_float_or_none(_cosine(answer_vec, ctx_vec)) for ctx_vec in context_vecs if ctx_vec is not None]
        gt_ctx = [
            _to_float_or_none(_cosine(ground_truth_vec, ctx_vec))
            for ctx_vec in context_vecs
            if ctx_vec is not None and ground_truth_vec is not None
        ]
        q_ctx_clean = [value for value in q_ctx if value is not None]
        a_ctx_clean = [value for value in a_ctx if value is not None]
        gt_ctx_clean = [value for value in gt_ctx if value is not None]

        best_q_idx = int(np.argmax(q_ctx_clean)) if q_ctx_clean else -1
        best_a_idx = int(np.argmax(a_ctx_clean)) if a_ctx_clean else -1
        best_gt_idx = int(np.argmax(gt_ctx_clean)) if gt_ctx_clean else -1

        top_context_q = contexts[best_q_idx] if best_q_idx >= 0 else ""
        top_context_a = contexts[best_a_idx] if best_a_idx >= 0 else ""
        top_context_gt = contexts[best_gt_idx] if best_gt_idx >= 0 else ""

        answer_ctx_jaccard = [_token_jaccard(answer, ctx) for ctx in contexts]
        question_ctx_jaccard = [_token_jaccard(question, ctx) for ctx in contexts]
        gt_ctx_jaccard = [_token_jaccard(ground_truth, ctx) for ctx in contexts] if ground_truth else []

        diag_rows.append(
            {
                "bge_question_answer_cosine": _to_float_or_none(_cosine(question_vec, answer_vec)),
                "bge_answer_ground_truth_cosine": _to_float_or_none(_cosine(answer_vec, ground_truth_vec)),
                "bge_question_ground_truth_cosine": _to_float_or_none(_cosine(question_vec, ground_truth_vec)),
                "bge_context_question_max_cosine": _best(q_ctx_clean),
                "bge_context_question_mean_cosine": _mean(q_ctx_clean),
                "bge_context_answer_max_cosine": _best(a_ctx_clean),
                "bge_context_answer_mean_cosine": _mean(a_ctx_clean),
                "bge_context_ground_truth_max_cosine": _best(gt_ctx_clean),
                "bge_context_ground_truth_mean_cosine": _mean(gt_ctx_clean),
                "bge_context_best_question_rank": best_q_idx,
                "bge_context_best_answer_rank": best_a_idx,
                "bge_context_best_ground_truth_rank": best_gt_idx,
                "top_context_preview_question": _shorten_text(top_context_q),
                "top_context_preview_answer": _shorten_text(top_context_a),
                "top_context_preview_ground_truth": _shorten_text(top_context_gt),
                "token_jaccard_answer_ground_truth": _to_float_or_none(_token_jaccard(answer, ground_truth)),
                "token_jaccard_question_answer": _to_float_or_none(_token_jaccard(question, answer)),
                "token_jaccard_question_ground_truth": _to_float_or_none(_token_jaccard(question, ground_truth)),
                "token_jaccard_answer_context_max": _best(answer_ctx_jaccard),
                "token_jaccard_question_context_max": _best(question_ctx_jaccard),
                "token_jaccard_ground_truth_context_max": _best(gt_ctx_jaccard),
            }
        )

    return pd.DataFrame(diag_rows), diag_meta


def _extract_langchain_llm(raw_llm):
    return getattr(raw_llm, "langchain_llm", None) or getattr(raw_llm, "llm", None) or raw_llm


def _extract_llm_config(prefix: str, llm_obj) -> dict[str, t.Any]:
    config: dict[str, t.Any] = {}
    if llm_obj is None:
        return config
    for key in [
        "model_name",
        "temperature",
        "max_retries",
        "request_timeout",
        "openai_api_base",
    ]:
        value = getattr(llm_obj, key, None)
        if value is not None:
            config[f"{prefix}_{key}"] = value
    config[f"{prefix}_class"] = type(llm_obj).__name__
    return config


def _extract_rag_system_config(rag_system) -> dict[str, t.Any]:
    config: dict[str, t.Any] = {
        "rag_system_class": type(rag_system).__name__,
        "rag_system_module": type(rag_system).__module__,
    }

    getter = getattr(rag_system, "get_system_config", None)
    if callable(getter):
        try:
            raw = getter()
            if isinstance(raw, dict):
                for key, value in raw.items():
                    config[f"rag_system_{key}"] = value
        except Exception:
            pass

    backend = getattr(rag_system, "_llm_backend", None)
    if backend is not None:
        config["rag_llm_backend"] = backend

    rag_llm = getattr(rag_system, "_llm", None)
    config.update(_extract_llm_config("rag_llm", rag_llm))

    retriever = getattr(rag_system, "_retriever", None)
    if retriever is not None:
        config["retriever_class"] = type(retriever).__name__
        embedding_device = getattr(retriever, "_embedding_device", None)
        if embedding_device is not None:
            config["retriever_embedding_device"] = embedding_device
        texts = getattr(retriever, "_texts", None)
        if isinstance(texts, list):
            config["retriever_docs_count"] = len(texts)

    return config


def _json_safe(value: t.Any) -> t.Any:
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    if isinstance(value, dict):
        return {str(k): _json_safe(v) for k, v in value.items()}
    return str(value)


def _format_value(value: t.Any) -> str:
    if isinstance(value, float):
        return f"{value:.6f}"
    if isinstance(value, (list, tuple, dict)):
        return json.dumps(value, ensure_ascii=False)
    if value is None:
        return ""
    return str(value)


def _config_description_map() -> dict[str, str]:
    return {
        "run_name": "Имя запуска/системы в отчете.",
        "gold_path": "Путь к XLSX с золотыми вопросами.",
        "rows": "Количество вопросов, обработанных в прогоне.",
        "use_shared_rag_system_models": "Использовать один общий LLM/BGE из rag_system для RAG и RAGAS.",
        "ragas_run_timeout": "Таймаут одной задачи RAGAS в секундах.",
        "ragas_max_workers": "Параллелизм RAGAS (для последовательного режима держите 1).",
        "ragas_metrics": "Набор метрик RAGAS в этом запуске.",
        "judge_llm_model_name": "Модель judge-LLM для RAGAS.",
        "judge_llm_openai_api_base": "Endpoint judge-LLM (для локального режима должен быть localhost).",
        "rag_llm_model_name": "Модель LLM внутри проверяемого RAG.",
        "rag_llm_openai_api_base": "Endpoint LLM проверяемого RAG.",
        "retriever_docs_count": "Количество документов/чанков в FAISS-хранилище.",
        "rag_system_retriever_docs_count": "Количество документов/чанков в FAISS-хранилище.",
        "rag_system_retriever_top_k": "Сколько контекстов возвращает ретривер.",
        "rag_system_embedding_model_path": "Путь к модели эмбеддингов внутри RAG-системы.",
        "rag_system_embedding_device": "Устройство эмбеддингов внутри RAG-системы.",
        "rag_system_llm_backend": "Backend LLM внутри RAG-системы (gigachat/koboldcpp).",
        "bge_m3_enabled": "Удалось ли включить bge-m3 диагностику.",
        "bge_m3_model_path": "Путь к модели bge-m3 для сравнений.",
        "bge_m3_device": "Устройство для bge-m3 (cpu/cuda).",
        "bge_m3_unique_texts_embedded": "Сколько уникальных строк было закодировано bge-m3.",
        "bge_m3_reused_embedder": "Диагностика использовала уже загруженный embedder без повторной загрузки.",
        "rag_total_generation_sec": "Суммарное время генерации ответов RAG (сек).",
        "rag_avg_generation_sec": "Среднее время генерации одного ответа RAG (сек).",
        "ragas_total_eval_sec": "Суммарное время вычисления метрик RAGAS (сек).",
    }


def _build_config_df(config_dict: dict[str, t.Any]) -> pd.DataFrame:
    descriptions = _config_description_map()
    rows = []
    for key in sorted(config_dict):
        rows.append(
            {
                "parameter": key,
                "value": _format_value(config_dict[key]),
                "description": descriptions.get(key, ""),
            }
        )
    return pd.DataFrame(rows)


def _build_parameter_guide_df() -> pd.DataFrame:
    rows = [
        {
            "metric_or_param": "faithfulness",
            "type": "ragas",
            "how_to_read": "Доля утверждений ответа, подтвержденных контекстами. Выше = лучше.",
        },
        {
            "metric_or_param": "context_precision",
            "type": "ragas",
            "how_to_read": "Насколько retrieved_contexts релевантны эталону. Выше = лучше.",
        },
        {
            "metric_or_param": "context_recall",
            "type": "ragas",
            "how_to_read": "Насколько retrieved_contexts покрывают эталон. Выше = лучше.",
        },
        {
            "metric_or_param": "answer_relevancy",
            "type": "ragas",
            "how_to_read": "Насколько ответ релевантен вопросу (требует embeddings). Выше = лучше.",
        },
        {
            "metric_or_param": "bge_question_answer_cosine",
            "type": "bge-m3",
            "how_to_read": "Семантическая близость вопроса и ответа по bge-m3. Выше = обычно лучше.",
        },
        {
            "metric_or_param": "bge_answer_ground_truth_cosine",
            "type": "bge-m3",
            "how_to_read": "Семантическая близость ответа и ground_truth. Выше = лучше.",
        },
        {
            "metric_or_param": "bge_context_question_max_cosine",
            "type": "bge-m3",
            "how_to_read": "Лучший retrieved context к вопросу. Низкое значение указывает на слабый retrieval.",
        },
        {
            "metric_or_param": "token_jaccard_answer_ground_truth",
            "type": "lexical",
            "how_to_read": "Лексическое пересечение answer и ground_truth. Быстрый ориентир, не заменяет семантику.",
        },
        {
            "metric_or_param": "rag_answer_latency_sec",
            "type": "runtime",
            "how_to_read": "Время ответа RAG на один вопрос в секундах.",
        },
        {
            "metric_or_param": "retrieved_context_count",
            "type": "retrieval",
            "how_to_read": "Сколько контекстов реально было возвращено ретривером.",
        },
    ]
    return pd.DataFrame(rows)


def _metric_description_map() -> dict[str, str]:
    guide_df = _build_parameter_guide_df()
    mapping = {
        str(row["metric_or_param"]): str(row["how_to_read"])
        for row in guide_df.to_dict(orient="records")
        if row.get("metric_or_param")
    }
    mapping.update(
        {
            "contexts_total_char_len": "Суммарная длина всех retrieved contexts в символах.",
            "answer_char_len": "Длина сгенерированного ответа в символах.",
            "answer_word_count": "Количество слов в сгенерированном ответе.",
            "ground_truth_char_len": "Длина эталонного ответа в символах.",
            "ground_truth_word_count": "Количество слов в эталонном ответе.",
        }
    )
    return mapping


def _format_numeric_for_html(df: pd.DataFrame) -> pd.DataFrame:
    view = df.copy()
    for col in view.columns:
        if pd.api.types.is_numeric_dtype(view[col]):
            view[col] = view[col].map(lambda x: "" if pd.isna(x) else f"{float(x):.4f}")
    return view


def _to_html_table(df: pd.DataFrame, *, max_rows: int | None = None) -> str:
    if df.empty:
        return "<p>Нет данных</p>"
    view = _format_numeric_for_html(df)
    if max_rows is not None:
        view = view.head(max_rows)
    return view.to_html(index=False, escape=True, classes="table table-sm table-striped", border=0)


def _metric_legend_html(df: pd.DataFrame) -> str:
    if df.empty:
        return ""

    descriptions = _metric_description_map()
    metric_names: list[str] = []

    if "metric" in df.columns:
        metric_names.extend(str(value) for value in df["metric"].dropna().tolist())

    metric_names.extend(str(column) for column in df.columns if str(column) in descriptions)

    unique_metrics: list[str] = []
    seen: set[str] = set()
    for metric in metric_names:
        if metric in descriptions and metric not in seen:
            seen.add(metric)
            unique_metrics.append(metric)

    if not unique_metrics:
        return ""

    items = "".join(
        f'<li><span class="code">{escape(metric)}</span> - {escape(descriptions[metric])}</li>'
        for metric in unique_metrics
    )
    return (
        '<div class="metric-help">'
        '<div class="metric-help-title">Описание метрик в этом блоке</div>'
        f"<ul>{items}</ul>"
        "</div>"
    )


def _compute_health_score(scores_df: pd.DataFrame) -> dict[str, t.Any]:
    # Aggregate quality-only metrics into one number (0..100) for quick status checks.
    metric_weights: list[tuple[str, float]] = [
        ("faithfulness", 1.0),
        ("answer_relevancy", 1.0),
        ("context_precision", 1.0),
        ("context_recall", 1.0),
        ("bge_answer_ground_truth_cosine", 0.7),
        ("bge_context_question_max_cosine", 0.5),
        ("token_jaccard_answer_ground_truth", 0.3),
    ]

    used: list[dict[str, t.Any]] = []
    weighted_sum = 0.0
    total_weight = 0.0
    for metric, weight in metric_weights:
        if metric not in scores_df.columns:
            continue
        series = pd.to_numeric(scores_df[metric], errors="coerce").dropna()
        if series.empty:
            continue
        mean_value = float(series.mean())
        # Keep score in interpretable range even if metric has noisy/out-of-range values.
        normalized = max(0.0, min(1.0, mean_value))
        weighted_sum += normalized * weight
        total_weight += weight
        used.append({"metric": metric, "mean": mean_value, "weight": weight})

    if total_weight == 0:
        return {"score_100": None, "status": "n/a", "status_ru": "Недостаточно данных", "used": used}

    score_100 = round((weighted_sum / total_weight) * 100.0, 1)
    if score_100 >= 85:
        status = ("excellent", "Отлично")
    elif score_100 >= 70:
        status = ("good", "Хорошо")
    elif score_100 >= 50:
        status = ("warning", "Нужно улучшить")
    else:
        status = ("critical", "Критично")

    return {"score_100": score_100, "status": status[0], "status_ru": status[1], "used": used}


def _health_details_html(payload: dict[str, t.Any]) -> str:
    used = payload.get("used") or []
    if not used:
        return '<p class="section-note">Нет доступных quality-метрик для вычисления интегрального показателя.</p>'

    items = "".join(
        (
            f"<li><span class=\"code\">{escape(str(item['metric']))}</span>: "
            f"mean={float(item['mean']):.3f}, weight={float(item['weight']):.1f}</li>"
        )
        for item in used
    )
    return (
        '<div class="metric-help">'
        '<div class="metric-help-title">Из чего собран Health Score</div>'
        f"<ul>{items}</ul>"
        "</div>"
    )


def _build_display_table(scores_df: pd.DataFrame) -> pd.DataFrame:
    df = scores_df.copy()

    if "question" in df.columns:
        df["question_preview"] = df["question"].astype(str).map(lambda x: _shorten_text(x, 180))
    if "answer" in df.columns:
        df["answer_preview"] = df["answer"].astype(str).map(lambda x: _shorten_text(x, 220))
    if "ground_truth" in df.columns:
        df["ground_truth_preview"] = df["ground_truth"].astype(str).map(lambda x: _shorten_text(x, 220))

    if "contexts" in df.columns:
        df["first_context_preview"] = df["contexts"].map(
            lambda ctxs: _shorten_text(str((ctxs or [""])[0]) if ctxs else "", TEXT_PREVIEW_LIMIT)
        )

    preferred = [
        "question_preview",
        "answer_preview",
        "ground_truth_preview",
        "top_context_preview_question",
        "first_context_preview",
        "answer_relevancy",
        "faithfulness",
        "context_precision",
        "context_recall",
        "bge_question_answer_cosine",
        "bge_answer_ground_truth_cosine",
        "bge_context_question_max_cosine",
        "token_jaccard_answer_ground_truth",
        "rag_answer_latency_sec",
        "retrieved_context_count",
    ]
    selected = [col for col in preferred if col in df.columns]
    return df[selected] if selected else df


def build_html_report(
    *,
    run_dir: Path,
    run_name: str,
    config_df: pd.DataFrame,
    guide_df: pd.DataFrame,
    summary_df: pd.DataFrame,
    summary_ragas_df: pd.DataFrame,
    summary_bge_df: pd.DataFrame,
    summary_runtime_df: pd.DataFrame,
    scores_df: pd.DataFrame,
    artifacts: list[str],
) -> Path:
    created_at = datetime.now(timezone.utc).isoformat()
    health = _compute_health_score(scores_df)
    health_score = health["score_100"]
    health_status = str(health["status"])
    health_status_ru = str(health["status_ru"])
    health_score_text = f"{health_score:.1f}" if health_score is not None else "n/a"

    dashboard_df = _build_display_table(scores_df)
    ranking_source = dashboard_df.copy()
    worst_col = next(
        (col for col in ["faithfulness", "context_precision", "answer_relevancy", "bge_answer_ground_truth_cosine"] if col in ranking_source.columns),
        None,
    )
    worst_df = ranking_source.sort_values(by=worst_col, ascending=True, na_position="last").head(10) if worst_col else ranking_source.head(10)
    slow_df = (
        ranking_source.sort_values(by="rag_answer_latency_sec", ascending=False, na_position="last").head(10)
        if "rag_answer_latency_sec" in ranking_source.columns
        else pd.DataFrame()
    )

    html = f"""<!doctype html>
<html lang="ru">
<head>
  <meta charset="utf-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1" />
  <title>RAG Eval Report - {escape(run_name)}</title>
  <style>
    body {{
      margin: 0;
      padding: 24px;
      background: #f7f9fc;
      color: #14213d;
      font-family: "Segoe UI", "DejaVu Sans", sans-serif;
    }}
    h1, h2 {{ margin: 0 0 12px; }}
    .muted {{ color: #5c677d; margin-bottom: 20px; }}
    .card {{
      background: #ffffff;
      border: 1px solid #e5e9f2;
      border-radius: 10px;
      padding: 16px;
      margin-bottom: 16px;
      box-shadow: 0 1px 2px rgba(9, 30, 66, 0.05);
    }}
    .table {{
      width: 100%;
      border-collapse: collapse;
      font-size: 13px;
    }}
    .table th, .table td {{
      border: 1px solid #dbe2ef;
      padding: 6px 8px;
      vertical-align: top;
      text-align: left;
    }}
    .table th {{
      background: #edf2fb;
    }}
    .files li {{ line-height: 1.5; }}
    .section-note {{
      color: #31456a;
      margin: 0 0 10px;
      font-size: 14px;
      line-height: 1.45;
    }}
    .metric-help {{
      margin-top: 10px;
      padding: 10px 12px;
      border: 1px solid #dbe2ef;
      border-radius: 8px;
      background: #f8fbff;
      font-size: 13px;
      color: #23395d;
    }}
    .metric-help-title {{
      font-weight: 600;
      margin-bottom: 6px;
    }}
    .metric-help ul {{
      margin: 0;
      padding-left: 20px;
    }}
    .metric-help li {{
      line-height: 1.4;
      margin-bottom: 4px;
    }}
    .health {{
      display: flex;
      align-items: center;
      gap: 16px;
      flex-wrap: wrap;
    }}
    .health-score {{
      font-size: 42px;
      font-weight: 700;
      line-height: 1;
      color: #0b3d91;
    }}
    .health-scale {{
      font-size: 16px;
      color: #51617f;
      margin-left: 6px;
    }}
    .health-badge {{
      display: inline-block;
      border-radius: 999px;
      padding: 4px 10px;
      font-size: 13px;
      font-weight: 600;
      border: 1px solid #cfd8ea;
      background: #eef3ff;
      color: #1f3f7f;
    }}
    .health-badge.excellent {{
      background: #eaf9ee;
      border-color: #b8e3c4;
      color: #1d6b38;
    }}
    .health-badge.good {{
      background: #eef7ff;
      border-color: #c5dff7;
      color: #1f4f8a;
    }}
    .health-badge.warning {{
      background: #fff8e8;
      border-color: #f1ddab;
      color: #8a6200;
    }}
    .health-badge.critical {{
      background: #fff0f0;
      border-color: #efc2c2;
      color: #8f1d1d;
    }}
    .code {{
      font-family: Consolas, "Liberation Mono", monospace;
      background: #f1f5f9;
      border-radius: 4px;
      padding: 2px 6px;
    }}
  </style>
</head>
<body>
  <h1>RAG Evaluation Report</h1>
  <div class="muted">Run: <span class="code">{escape(run_name)}</span> | Created (UTC): {escape(created_at)}</div>

  <div class="card">
    <h2>System Health Score</h2>
    <p class="section-note">Единый индикатор состояния RAG по ключевым quality-метрикам. 100 = максимально хорошо, 0 = критически плохо.</p>
    <div class="health">
      <div class="health-score">{escape(health_score_text)}<span class="health-scale">/100</span></div>
      <div class="health-badge {escape(health_status)}">{escape(health_status_ru)}</div>
    </div>
    {_health_details_html(health)}
  </div>

  <div class="card">
    <h2>Summary (RAGAS)</h2>
    <p class="section-note">Сводка только по основным метрикам RAGAS для оценки качества ответа и релевантности контекстов.</p>
    {_to_html_table(summary_ragas_df)}
    {_metric_legend_html(summary_ragas_df)}
  </div>

  <div class="card">
    <h2>Worst 10 Samples</h2>
    <p class="section-note">10 самых проблемных примеров по ключевой метрике качества (обычно `faithfulness` или `context_precision`).</p>
    {_to_html_table(worst_df, max_rows=10)}
    {_metric_legend_html(worst_df)}
  </div>

  <div class="card">
    <h2>Per-Question Diagnostics</h2>
    <p class="section-note">Построчная диагностика по каждому вопросу: превью ответа/контекста и ключевые метрики качества.</p>
    {_to_html_table(dashboard_df)}
    {_metric_legend_html(dashboard_df)}
  </div>

  <div class="card">
    <h2>Summary (Runtime / Retrieval)</h2>
    <p class="section-note">Метрики производительности и retrieval: задержка ответа, объем и количество извлеченного контекста.</p>
    {_to_html_table(summary_runtime_df)}
    {_metric_legend_html(summary_runtime_df)}
  </div>

  <div class="card">
    <h2>Slowest 10 Samples</h2>
    <p class="section-note">10 самых медленных ответов RAG по времени генерации, чтобы найти узкие места по latency.</p>
    {_to_html_table(slow_df, max_rows=10)}
    {_metric_legend_html(slow_df)}
  </div>

  <div class="card">
    <h2>Summary (BGE-M3 Diagnostics)</h2>
    <p class="section-note">Сводка по дополнительным диагностическим метрикам на эмбеддингах и лексическом пересечении.</p>
    {_to_html_table(summary_bge_df)}
    {_metric_legend_html(summary_bge_df)}
  </div>

  <div class="card">
    <h2>Summary (All Numeric)</h2>
    <p class="section-note">Общая статистика по всем числовым колонкам: среднее, разброс, минимумы/максимумы и пропуски.</p>
    {_to_html_table(summary_df)}
    {_metric_legend_html(summary_df)}
  </div>

  <div class="card">
    <h2>Config</h2>
    <p class="section-note">Параметры текущего запуска: модели, endpoint-ы, лимиты и служебные настройки.</p>
    {_to_html_table(config_df)}
  </div>

  <div class="card">
    <h2>Parameter Guide</h2>
    <p class="section-note">Справочник интерпретации параметров и метрик в отчете.</p>
    {_to_html_table(guide_df)}
  </div>

  <div class="card">
    <h2>Artifacts</h2>
    <p class="section-note">Список файлов, которые были сохранены для этого прогона.</p>
    <ul class="files">
      {''.join(f'<li>{escape(name)}</li>' for name in artifacts)}
    </ul>
  </div>
</body>
</html>
"""

    path = run_dir / "report.html"
    path.write_text(html, encoding="utf-8")
    return path


def run_single_rag_eval(
    gold_path: str | Path,
    rag_system,
    judge_llm=None,
    judge_embeddings=None,
    ragas_run_config=None,
    reports_dir: str | Path = "reports",
    run_name: str = "rag",
    use_shared_rag_system_models: bool = True,
) -> Path:
    run_stamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    run_dir = Path(reports_dir) / f"{run_stamp}_{run_name}"
    run_dir.mkdir(parents=True, exist_ok=True)

    gold_df = load_xlsx(gold_path)
    eval_df = run_rag_over_questions(gold_df, rag_system)
    if use_shared_rag_system_models:
        resolved_judge_llm = _resolve_judge_llm(None, rag_system)
        resolved_judge_embeddings = _resolve_judge_embeddings(None, rag_system)
    else:
        resolved_judge_llm = _resolve_judge_llm(judge_llm, rag_system)
        resolved_judge_embeddings = _resolve_judge_embeddings(judge_embeddings, rag_system)
    scores_df, ragas_elapsed = evaluate_with_ragas(
        eval_df,
        judge_llm=resolved_judge_llm,
        judge_embeddings=resolved_judge_embeddings,
        ragas_run_config=ragas_run_config,
    )

    alias_pairs = [
        ("question", "user_input"),
        ("answer", "response"),
        ("ground_truth", "reference"),
        ("contexts", "retrieved_contexts"),
    ]
    for canonical, alias in alias_pairs:
        if canonical not in scores_df.columns and alias in scores_df.columns:
            scores_df[canonical] = scores_df[alias]

    for col in eval_df.columns:
        if col not in scores_df.columns:
            scores_df[col] = eval_df[col]

    shared_embedder = _extract_shared_sentence_embedder(rag_system)
    diag_df, diag_meta = compute_bge_m3_diagnostics(
        eval_df,
        model_path=_extract_shared_embedding_model_path(rag_system),
        device=_extract_shared_embedding_device(rag_system),
        embedder=shared_embedder,
    )
    if not diag_df.empty:
        scores_df = scores_df.join(diag_df)

    summary_df = summarize(scores_df)
    ragas_cols = [col for col in ["answer_relevancy", "faithfulness", "context_precision", "context_recall"] if col in scores_df.columns]
    bge_cols = [col for col in scores_df.columns if col.startswith("bge_") or col.startswith("token_jaccard_")]
    runtime_cols = [
        col
        for col in [
            "rag_answer_latency_sec",
            "retrieved_context_count",
            "contexts_total_char_len",
            "answer_char_len",
            "answer_word_count",
            "ground_truth_char_len",
            "ground_truth_word_count",
        ]
        if col in scores_df.columns
    ]
    summary_ragas_df = summarize(scores_df[ragas_cols]) if ragas_cols else pd.DataFrame()
    summary_bge_df = summarize(scores_df[bge_cols]) if bge_cols else pd.DataFrame()
    summary_runtime_df = summarize(scores_df[runtime_cols]) if runtime_cols else pd.DataFrame()

    compact_df = _build_display_table(scores_df)

    scores_df.to_csv(run_dir / "scores.csv", index=False)
    compact_df.to_csv(run_dir / "scores_compact.csv", index=False)
    summary_df.to_csv(run_dir / "summary.csv", index=False)
    summary_ragas_df.to_csv(run_dir / "summary_ragas.csv", index=False)
    summary_bge_df.to_csv(run_dir / "summary_bge_m3.csv", index=False)
    summary_runtime_df.to_csv(run_dir / "summary_runtime.csv", index=False)

    judge_inner_llm = _extract_langchain_llm(resolved_judge_llm)
    config = {
        "run_name": run_name,
        "gold_path": str(Path(gold_path).resolve()),
        "rows": int(len(scores_df)),
        "ragas_metrics": ragas_cols,
        "ragas_run_timeout": getattr(ragas_run_config, "timeout", None),
        "ragas_max_workers": getattr(ragas_run_config, "max_workers", None),
        "rag_total_generation_sec": float(eval_df["rag_answer_latency_sec"].sum()) if "rag_answer_latency_sec" in eval_df else None,
        "rag_avg_generation_sec": float(eval_df["rag_answer_latency_sec"].mean()) if "rag_answer_latency_sec" in eval_df else None,
        "ragas_total_eval_sec": float(ragas_elapsed),
        "use_shared_rag_system_models": bool(use_shared_rag_system_models),
        "bge_m3_enabled": diag_meta.get("enabled", False),
        "bge_m3_model_path": diag_meta.get("model_path"),
        "bge_m3_device": diag_meta.get("device"),
        "bge_m3_unique_texts_embedded": diag_meta.get("unique_texts_embedded"),
        "bge_m3_error": diag_meta.get("error"),
        "bge_m3_reused_embedder": diag_meta.get("reused_embedder"),
    }
    config.update(_extract_llm_config("judge_llm", judge_inner_llm))
    config.update(_extract_rag_system_config(rag_system))

    config_df = _build_config_df(config)
    config_df.to_csv(run_dir / "config.csv", index=False)

    guide_df = _build_parameter_guide_df()
    guide_df.to_csv(run_dir / "parameter_guide.csv", index=False)

    artifacts = [
        "scores.csv",
        "scores_compact.csv",
        "summary.csv",
        "summary_ragas.csv",
        "summary_bge_m3.csv",
        "summary_runtime.csv",
        "config.csv",
        "parameter_guide.csv",
        "run_meta.json",
        "report.html",
    ]

    html_path = build_html_report(
        run_dir=run_dir,
        run_name=run_name,
        config_df=config_df,
        guide_df=guide_df,
        summary_df=summary_df,
        summary_ragas_df=summary_ragas_df,
        summary_bge_df=summary_bge_df,
        summary_runtime_df=summary_runtime_df,
        scores_df=scores_df,
        artifacts=artifacts,
    )

    meta = {
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "run_name": run_name,
        "gold_path": str(Path(gold_path).resolve()),
        "rows": int(len(scores_df)),
        "metrics": [col for col in scores_df.columns if pd.api.types.is_numeric_dtype(scores_df[col])],
        "ragas_metrics": ragas_cols,
        "bge_m3": {
            "enabled": bool(diag_meta.get("enabled", False)),
            "model_path": diag_meta.get("model_path"),
            "device": diag_meta.get("device"),
            "error": diag_meta.get("error"),
            "unique_texts_embedded": diag_meta.get("unique_texts_embedded"),
            "reused_embedder": bool(diag_meta.get("reused_embedder", False)),
        },
        "timings_sec": {
            "rag_total_generation": config.get("rag_total_generation_sec"),
            "rag_avg_generation": config.get("rag_avg_generation_sec"),
            "ragas_total_eval": float(ragas_elapsed),
        },
        "config": {key: _json_safe(value) for key, value in config.items()},
        "artifacts": artifacts,
        "html_report": html_path.name,
    }
    (run_dir / "run_meta.json").write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")

    return run_dir


__all__ = [
    "LangchainEmbeddingsWrapper",
    "LangchainLLMWrapper",
    "build_shared_judges_from_rag_system",
    "latest_xlsx",
    "run_single_rag_eval",
]
