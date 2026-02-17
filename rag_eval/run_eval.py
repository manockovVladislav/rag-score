"""Ядро прогона ragas по золотым вопросам из Excel.

Сценарий:
- Читаем Excel с колонками question и ground_truth (или только question)
- Поднимаем RAG-системы из реестра (через декоратор)
- Прогоняем ragas
- Сохраняем HTML-отчет + CSV/JSON
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from datetime import datetime
from zoneinfo import ZoneInfo
import json
import base64

import pandas as pd
from tqdm.auto import tqdm
from datasets import Dataset
import matplotlib.pyplot as plt
from jinja2 import Template

from ragas import evaluate
from ragas.metrics import (
    answer_relevancy,
    faithfulness,
    context_precision,
    context_recall,
    context_relevancy,
    answer_correctness,
    answer_similarity,
)

try:
    from ragas.llms import LangchainLLMWrapper
    from ragas.embeddings import LangchainEmbeddingsWrapper
except Exception:
    LangchainLLMWrapper = None
    LangchainEmbeddingsWrapper = None

try:
    from sentence_transformers import SentenceTransformer
except Exception:
    SentenceTransformer = None

from .registry import get_rag_system, load_registries, adapt_response


TZ = ZoneInfo("Asia/Kolkata")


@dataclass
class EvalConfig:
    data_dir: Path = Path("./data")
    reports_dir: Path = Path("./reports")
    run_id: str | None = None


def _run_id() -> str:
    return datetime.now(TZ).strftime("%Y-%m-%d_%H-%M-%S")


def _normalize_gold(df: pd.DataFrame, source_path: str) -> pd.DataFrame:
    df = df.copy()
    df.columns = [str(c).strip().lower().replace(" ", "_").replace("-", "_") for c in df.columns]

    rename = {}
    for c in df.columns:
        if c in {"question", "query", "q", "вопрос", "вопрос_пользователя"}:
            rename[c] = "question"
        if c in {"ground_truth", "reference", "answer_gt", "эталон", "эталонный_ответ", "правильный_ответ"}:
            rename[c] = "ground_truth"
    df = df.rename(columns=rename)

    if "question" not in df.columns:
        raise ValueError(f"Не найдена колонка вопроса. Колонки: {list(df.columns)}")

    df["question"] = df["question"].astype(str).str.strip()
    df = df[df["question"].ne("")].reset_index(drop=True)

    if "ground_truth" in df.columns:
        df["ground_truth"] = df["ground_truth"].astype(str).str.strip()

    df.attrs["source_xlsx"] = source_path
    return df


def load_latest_xlsx(data_dir: Path) -> pd.DataFrame:
    xlsx_files = sorted(data_dir.glob("*.xlsx"), key=lambda p: p.stat().st_mtime, reverse=True)
    if not xlsx_files:
        raise FileNotFoundError(f"Нет .xlsx в {data_dir.resolve()}")
    path = xlsx_files[0]

    df = pd.read_excel(path)
    return _normalize_gold(df, str(path))


def load_xlsx(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Файл не найден: {path}")
    df = pd.read_excel(path)
    return _normalize_gold(df, str(path))


def run_rag_over_questions(gold_df: pd.DataFrame, rag_callable) -> pd.DataFrame:
    rows = []
    for q in tqdm(gold_df["question"].tolist(), desc="RAG answering"):
        raw = rag_callable(q)
        response = adapt_response(raw)
        ans = (response.answer or "").strip()
        ctx = response.contexts or []
        ctx = [str(x).strip() for x in ctx if str(x).strip()]
        rows.append({"question": q, "answer": ans, "contexts": ctx})

    df = pd.DataFrame(rows)
    if "ground_truth" in gold_df.columns:
        df["ground_truth"] = gold_df["ground_truth"].tolist()
    return df


def choose_metrics(df: pd.DataFrame):
    metrics = [answer_relevancy, faithfulness]
    if "ground_truth" in df.columns:
        metrics += [context_precision, context_recall, context_relevancy]
    return metrics


def _normalize_text(text: str) -> str:
    return " ".join(str(text).lower().split())


def _tokenize(text: str) -> list[str]:
    text = _normalize_text(text)
    return [t for t in text.split() if t]


def _token_prf(answer: str, truth: str) -> tuple[float, float, float]:
    a = _tokenize(answer)
    t = _tokenize(truth)
    if not a and not t:
        return 1.0, 1.0, 1.0
    if not a or not t:
        return 0.0, 0.0, 0.0
    a_set = set(a)
    t_set = set(t)
    inter = a_set & t_set
    precision = len(inter) / max(1, len(a_set))
    recall = len(inter) / max(1, len(t_set))
    f1 = 0.0 if precision + recall == 0 else 2 * precision * recall / (precision + recall)
    return precision, recall, f1


def _char_ngrams(text: str, n: int = 3) -> list[str]:
    text = _normalize_text(text).replace(" ", "")
    if len(text) < n:
        return [text] if text else []
    return [text[i:i + n] for i in range(len(text) - n + 1)]


def _char_prf(answer: str, truth: str, n: int = 3) -> tuple[float, float, float]:
    a = _char_ngrams(answer, n=n)
    t = _char_ngrams(truth, n=n)
    if not a and not t:
        return 1.0, 1.0, 1.0
    if not a or not t:
        return 0.0, 0.0, 0.0
    a_set = set(a)
    t_set = set(t)
    inter = a_set & t_set
    precision = len(inter) / max(1, len(a_set))
    recall = len(inter) / max(1, len(t_set))
    f1 = 0.0 if precision + recall == 0 else 2 * precision * recall / (precision + recall)
    return precision, recall, f1


def _cosine(u: list[float], v: list[float]) -> float:
    num = sum(x * y for x, y in zip(u, v))
    du = sum(x * x for x in u) ** 0.5
    dv = sum(y * y for y in v) ** 0.5
    if du == 0 or dv == 0:
        return 0.0
    return num / (du * dv)


def compute_autonomous_answer_metrics(eval_df: pd.DataFrame, embedder=None) -> pd.DataFrame:
    if "ground_truth" not in eval_df.columns:
        return pd.DataFrame(index=eval_df.index)

    rows = []
    for _, row in eval_df.iterrows():
        ans = row.get("answer", "")
        gt = row.get("ground_truth", "")
        tok_p, tok_r, tok_f1 = _token_prf(ans, gt)
        chr_p, chr_r, chr_f1 = _char_prf(ans, gt, n=3)
        rec = {
            "answer_precision_tok": tok_p,
            "answer_recall_tok": tok_r,
            "answer_f1_tok": tok_f1,
            "answer_precision_chr3": chr_p,
            "answer_recall_chr3": chr_r,
            "answer_f1_chr3": chr_f1,
        }
        rows.append(rec)

    df = pd.DataFrame(rows, index=eval_df.index)

    if embedder is not None:
        a_texts = eval_df["answer"].astype(str).tolist()
        t_texts = eval_df["ground_truth"].astype(str).tolist()
        a_emb = embedder.encode(a_texts, normalize_embeddings=False, show_progress_bar=False)
        t_emb = embedder.encode(t_texts, normalize_embeddings=False, show_progress_bar=False)
        cos = [_cosine(a.tolist(), b.tolist()) for a, b in zip(a_emb, t_emb)]
        df["answer_cosine_emb"] = cos

    return df


def fig_to_b64(fig) -> str:
    import io
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    return base64.b64encode(buf.getvalue()).decode("utf-8")


def summarize(scores_df: pd.DataFrame) -> pd.DataFrame:
    num = scores_df.select_dtypes(include="number")
    if num.empty:
        return pd.DataFrame()
    agg = pd.DataFrame({
        "mean": num.mean(),
        "std": num.std(),
        "min": num.min(),
        "p10": num.quantile(0.10),
        "p50": num.quantile(0.50),
        "p90": num.quantile(0.90),
        "max": num.max(),
        "na": num.isna().sum(),
    }).reset_index().rename(columns={"index": "metric"})
    return agg


def histograms(scores_df: pd.DataFrame) -> dict[str, str]:
    plots = {}
    for col in scores_df.select_dtypes(include="number").columns:
        fig = plt.figure()
        plt.hist(scores_df[col].dropna(), bins=20)
        plt.title(col)
        plt.xlabel("score")
        plt.ylabel("count")
        plots[col] = fig_to_b64(fig)
    return plots


def worst_cases(scores_df: pd.DataFrame, metric_cols: list[str], topn: int = 15) -> dict[str, pd.DataFrame]:
    def trunc(s: str, n: int) -> str:
        s = str(s)
        return s if len(s) <= n else s[:n] + " …"

    out = {}
    for m in metric_cols:
        if m not in scores_df.columns:
            continue
        tmp = scores_df.sort_values(m, ascending=True).head(topn).copy()
        tmp["answer"] = tmp["answer"].map(lambda x: trunc(x, 400))
        tmp["contexts"] = tmp["contexts"].map(
            lambda xs: trunc("\n\n".join(xs if isinstance(xs, list) else [str(xs)]), 700)
        )
        if "ground_truth" in tmp.columns:
            tmp["ground_truth"] = tmp["ground_truth"].map(lambda x: trunc(x, 350))
        out[m] = tmp
    return out


def save_report(
    run_dir: Path,
    source_xlsx: str,
    scores_df: pd.DataFrame,
    run_config: dict | None = None,
):
    run_dir.mkdir(parents=True, exist_ok=True)

    summary_df = summarize(scores_df)
    plots = histograms(scores_df)

    base_cols = {"question", "answer", "contexts", "ground_truth"}
    metric_cols = [
        c for c in scores_df.columns
        if c not in base_cols and pd.api.types.is_numeric_dtype(scores_df[c])
    ]
    worst = worst_cases(scores_df, metric_cols, topn=15)

    scores_df.to_csv(run_dir / "scores.csv", index=False)
    summary_df.to_csv(run_dir / "summary.csv", index=False)

    meta = {
        "source_xlsx": source_xlsx,
        "rows": int(len(scores_df)),
        "metrics": metric_cols,
        "config": run_config or {},
    }
    with open(run_dir / "run_meta.json", "w", encoding="utf-8") as f:
        json.dump(
            meta,
            f,
            ensure_ascii=False,
            indent=2,
        )

    metric_descriptions = {
        "faithfulness": "Доля утверждений в ответе, подтверждённых контекстом.",
        "answer_relevancy": "Насколько ответ соответствует вопросу.",
        "context_precision": "Доля релевантных контекстов среди всех контекстов.",
        "context_recall": "Доля релевантного контекста, покрытого выбранными контекстами.",
        "context_relevancy": "Насколько контексты релевантны вопросу.",
        "answer_precision_tok": "Пересечение токенов ответа и эталона (precision).",
        "answer_recall_tok": "Пересечение токенов ответа и эталона (recall).",
        "answer_f1_tok": "F1 по токенам ответа и эталона.",
        "answer_precision_chr3": "Пересечение символьных 3-грамм (precision).",
        "answer_recall_chr3": "Пересечение символьных 3-грамм (recall).",
        "answer_f1_chr3": "F1 по символьным 3-граммам.",
        "answer_cosine_emb": "Косинусная близость эмбеддингов ответа и эталона.",
    }

    used_descriptions = [
        {"metric": m, "description": metric_descriptions.get(m, "—")}
        for m in metric_cols
    ]

    tpl = Template(r"""
<!doctype html><html lang="ru"><head>
<meta charset="utf-8">
<title>RAGAS Report {{ run_id }}</title>
<style>
body { font-family: Arial, sans-serif; margin: 24px; }
.meta { padding: 12px; background: #f5f5f5; border-radius: 8px; }
table { border-collapse: collapse; width: 100%; margin: 10px 0 18px; }
th, td { border: 1px solid #ddd; padding: 8px; vertical-align: top; }
th { background: #fafafa; }
img { max-width: 950px; width: 100%; height: auto; border: 1px solid #eee; border-radius: 8px; }
details { margin: 10px 0 18px; }
code { background: #f2f2f2; padding: 2px 4px; border-radius: 4px; }
</style></head><body>
<h1>RAGAS отчёт — {{ run_id }}</h1>
<div class="meta">
  <div><b>XLSX:</b> <code>{{ xlsx }}</code></div>
  <div><b>Строк:</b> {{ rows }}</div>
  <div><b>Метрики:</b> {{ metrics|join(", ") }}</div>
</div>

<h2>Сводка</h2>
{{ summary_html|safe }}

<h2>Config</h2>
{{ config_html|safe }}

<h2>Расшифровка метрик</h2>
{{ metric_desc_html|safe }}

<h2>Распределения</h2>
{% for name, b64 in plots.items() %}
  <h3>{{ name }}</h3>
  <img src="data:image/png;base64,{{ b64 }}">
{% endfor %}

<h2>Худшие кейсы</h2>
{% for name, df_html in worst_html.items() %}
  <details>
    <summary><b>{{ name }}</b> — top worst</summary>
    {{ df_html|safe }}
  </details>
{% endfor %}
</body></html>
""")

    summary_html = summary_df.to_html(index=False) if not summary_df.empty else "<p>Нет сводки.</p>"
    config_html = (
        pd.DataFrame(
            [{"key": k, "value": v} for k, v in (run_config or {}).items()]
        ).to_html(index=False)
        if run_config
        else "<p>Нет данных.</p>"
    )
    metric_desc_html = pd.DataFrame(used_descriptions).to_html(index=False) if used_descriptions else "<p>Нет метрик.</p>"
    worst_html = {m: df.to_html(index=False) for m, df in worst.items()}

    html = tpl.render(
        run_id=run_dir.name,
        xlsx=source_xlsx,
        rows=len(scores_df),
        metrics=metric_cols,
        summary_html=summary_html,
        config_html=config_html,
        metric_desc_html=metric_desc_html,
        plots=plots,
        worst_html=worst_html,
    )
    (run_dir / f"ragas_report_{run_dir.name}.html").write_text(html, encoding="utf-8")


def evaluate_with_ragas(eval_df: pd.DataFrame, judge_llm, judge_embeddings=None):
    hf_ds = Dataset.from_pandas(eval_df, preserve_index=False)
    metrics = choose_metrics(eval_df)

    res = evaluate(
        hf_ds,
        metrics=metrics,
        llm=judge_llm,
        embeddings=judge_embeddings,
    )
    return res


def run_full_pipeline(
    cfg: EvalConfig,
    rag_callable,
    judge_llm,
    judge_embeddings=None,
    gold_path: Path | None = None,
    local_embedder=None,
    run_config: dict | None = None,
) -> Path:
    cfg.reports_dir.mkdir(parents=True, exist_ok=True)
    run_id = cfg.run_id or _run_id()
    run_dir = cfg.reports_dir / run_id

    if gold_path is not None:
        gold = load_xlsx(gold_path)
    else:
        gold = load_latest_xlsx(cfg.data_dir)
    eval_df = run_rag_over_questions(gold, rag_callable)

    res = evaluate_with_ragas(eval_df, judge_llm, judge_embeddings)
    scores_df = res.to_pandas()

    auto_metrics = compute_autonomous_answer_metrics(eval_df, embedder=local_embedder)
    if not auto_metrics.empty:
        scores_df = pd.concat([scores_df.reset_index(drop=True), auto_metrics.reset_index(drop=True)], axis=1)

    save_report(run_dir, gold.attrs.get("source_xlsx", "unknown.xlsx"), scores_df, run_config=run_config)
    return run_dir


def run_eval(
    gold_path: str,
    registry_modules: list[str],
    system_names: list[str],
    output_root: str,
    judge_llm=None,
    judge_embeddings=None,
    local_embedder=None,
    run_config: dict | None = None,
) -> None:
    """Запуск оценки для выбранных систем."""

    load_registries(registry_modules)

    run_stamp = _run_id()
    run_root = Path(output_root) / run_stamp
    run_root.mkdir(parents=True, exist_ok=True)

    for system_name in system_names:
        system = get_rag_system(system_name)

        def _call(q: str):
            return system.answer(q)

        cfg = EvalConfig(
            data_dir=Path(gold_path).parent,
            reports_dir=run_root,
            run_id=system_name,
        )

        run_full_pipeline(
            cfg=cfg,
            rag_callable=_call,
            judge_llm=judge_llm,
            judge_embeddings=judge_embeddings,
            gold_path=Path(gold_path),
            local_embedder=local_embedder,
            run_config=run_config,
        )


__all__ = [
    "EvalConfig",
    "load_latest_xlsx",
    "run_full_pipeline",
    "run_eval",
    "LangchainLLMWrapper",
    "LangchainEmbeddingsWrapper",
    "SentenceTransformer",
    "compute_autonomous_answer_metrics",
]
