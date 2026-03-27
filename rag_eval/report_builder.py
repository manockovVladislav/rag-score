"""HTML report generation helpers for RAG evaluation runs."""

from __future__ import annotations

from datetime import datetime, timezone
from html import escape
from pathlib import Path
import typing as t

import pandas as pd


TEXT_PREVIEW_LIMIT = 260


def _shorten_text(value: str, limit: int = TEXT_PREVIEW_LIMIT) -> str:
    text = (value or "").strip()
    if len(text) <= limit:
        return text
    return text[: limit - 1].rstrip() + "…"


def _display_metric_name(name: str) -> str:
    value = str(name)
    if "bge_m3" in value:
        value = value.replace("bge_m3", "vectorizer")
    if value.startswith("vectorizer_"):
        return "Векторизатор_" + value[len("vectorizer_") :]
    if value.startswith("bge_m3_"):
        return "Векторизатор_" + value[len("bge_m3_") :]
    if value.startswith("bge_"):
        return "Векторизатор_" + value[len("bge_") :]
    return value


def _metric_description_map() -> dict[str, str]:
    return {
        "faithfulness": "Доля утверждений ответа, подтвержденных контекстами. Выше = лучше.",
        "context_precision": "Насколько retrieved_contexts релевантны эталону. Выше = лучше.",
        "context_recall": "Насколько retrieved_contexts покрывают эталон. Выше = лучше.",
        "answer_relevancy": "Насколько ответ релевантен вопросу (требует embeddings). Выше = лучше.",
        "Векторизатор_question_answer_cosine": "Семантическая близость вопроса и ответа по векторизатору. Выше = обычно лучше.",
        "Векторизатор_answer_ground_truth_cosine": "Семантическая близость ответа и ground_truth. Выше = лучше.",
        "Векторизатор_context_question_max_cosine": "Лучший retrieved context к вопросу. Низкое значение указывает на слабый retrieval.",
        "token_jaccard_answer_ground_truth": "Лексическое пересечение answer и ground_truth. Быстрый ориентир, не заменяет семантику.",
        "rag_answer_latency_sec": "Время ответа RAG на один вопрос в секундах.",
        "retrieved_context_count": "Сколько контекстов реально было возвращено ретривером.",
        "contexts_total_char_len": "Суммарная длина всех retrieved contexts в символах.",
        "answer_char_len": "Длина сгенерированного ответа в символах.",
        "answer_word_count": "Количество слов в сгенерированном ответе.",
        "ground_truth_char_len": "Длина эталонного ответа в символах.",
        "ground_truth_word_count": "Количество слов в эталонном ответе.",
    }


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
    for metric_col in ("metric", "parameter", "metric_or_param"):
        if metric_col in view.columns:
            view[metric_col] = view[metric_col].map(lambda x: _display_metric_name(str(x)))
    view = view.rename(columns={col: _display_metric_name(str(col)) for col in view.columns})
    if max_rows is not None:
        view = view.head(max_rows)
    return view.to_html(index=False, escape=True, classes="table table-sm table-striped", border=0)


def _metric_legend_html(df: pd.DataFrame) -> str:
    if df.empty:
        return ""

    descriptions = _metric_description_map()
    metric_names: list[str] = []

    if "metric" in df.columns:
        metric_names.extend(_display_metric_name(str(value)) for value in df["metric"].dropna().tolist())

    metric_names.extend(
        _display_metric_name(str(column))
        for column in df.columns
        if _display_metric_name(str(column)) in descriptions
    )

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


def _ragas_alerts_html(scores_df: pd.DataFrame) -> str:
    alerts: list[str] = []

    if "faithfulness" in scores_df.columns:
        faith = pd.to_numeric(scores_df["faithfulness"], errors="coerce")
        if faith.notna().sum() == 0:
            alerts.append(
                "faithfulness не рассчиталась (все значения NaN). Обычно это значит, что judge-модель не дала стабильный структурированный ответ для этой метрики."
            )

    if "context_precision" in scores_df.columns:
        precision = pd.to_numeric(scores_df["context_precision"], errors="coerce").dropna()
        if not precision.empty and float(precision.max()) == 0.0:
            alerts.append(
                "context_precision = 0 по всем строкам: retrieved_contexts нерелевантны эталону или корпус не покрывает вопросы."
            )

    if "context_recall" in scores_df.columns:
        recall = pd.to_numeric(scores_df["context_recall"], errors="coerce").dropna()
        if not recall.empty and float(recall.max()) == 0.0:
            alerts.append("context_recall = 0 по всем строкам: retrieved_contexts не содержат информацию из ground_truth.")

    if not alerts:
        return ""

    items = "".join(f"<li>{escape(item)}</li>" for item in alerts)
    return (
        '<div class="card">'
        "<h2>Metric Alerts</h2>"
        '<p class="section-note">Автоматические предупреждения по метрикам текущего запуска.</p>'
        f"<ul>{items}</ul>"
        "</div>"
    )


def _health_details_html(payload: dict[str, t.Any]) -> str:
    used = payload.get("used") or []
    if not used:
        return '<p class="section-note">Нет доступных quality-метрик для вычисления интегрального показателя.</p>'

    items = "".join(
        (
            f"<li><span class=\"code\">{escape(_display_metric_name(str(item['metric'])))}</span>: "
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


def build_display_table(scores_df: pd.DataFrame) -> pd.DataFrame:
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
    ragas_alerts_html = _ragas_alerts_html(scores_df)
    health_score = health["score_100"]
    health_score_text = f"{health_score:.1f}" if health_score is not None else "n/a"

    dashboard_df = build_display_table(scores_df)
    ranking_source = dashboard_df.copy()
    worst_col = next(
        (
            col
            for col in ["faithfulness", "context_precision", "answer_relevancy", "bge_answer_ground_truth_cosine"]
            if col in ranking_source.columns
        ),
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
    </div>
    {_health_details_html(health)}
  </div>

  {ragas_alerts_html}

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
    <h2>Сводка (Диагностика векторизатора)</h2>
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
      {''.join(f'<li>{escape(_display_metric_name(name))}</li>' for name in artifacts)}
    </ul>
  </div>
</body>
</html>
"""

    path = run_dir / "report.html"
    path.write_text(html, encoding="utf-8")
    return path


__all__ = ["build_display_table", "build_html_report"]
