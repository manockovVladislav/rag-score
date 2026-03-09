"""Notebook helper for running a single RAG system evaluation."""

from __future__ import annotations

from .run_eval import run_single_rag_eval


def run_eval_notebook(
    gold_path: str,
    rag_system,
    judge_llm,
    judge_embeddings=None,
    ragas_run_config=None,
    reports_dir: str = "reports",
    run_name: str = "rag",
):
    return run_single_rag_eval(
        gold_path=gold_path,
        rag_system=rag_system,
        judge_llm=judge_llm,
        judge_embeddings=judge_embeddings,
        ragas_run_config=ragas_run_config,
        reports_dir=reports_dir,
        run_name=run_name,
    )


__all__ = ["run_eval_notebook"]
