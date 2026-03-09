"""RAG systems used for evaluation."""

from .gigachat_bge_m3 import RAG_SYSTEM_NAME, build_rag_system

__all__ = ["RAG_SYSTEM_NAME", "build_rag_system"]
