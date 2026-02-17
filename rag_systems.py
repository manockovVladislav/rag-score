"""Регистрация RAG-систем и FAISS-ретривера."""

from typing import List, Protocol
from pathlib import Path
import json
import os

try:
    import faiss  # type: ignore
except Exception:
    faiss = None

try:
    from sentence_transformers import SentenceTransformer
except Exception:
    SentenceTransformer = None

from rag_eval import rag_system
from rag_eval.registry import RagResponse


class Retriever(Protocol):
    def retrieve(self, query: str) -> List[str]:
        ...


VECTOR_DB_DIR = Path(os.environ.get("VECTOR_DB_DIR", "vector_db"))
RETRIEVER_TOP_K = int(os.environ.get("RETRIEVER_TOP_K", "5"))
E5_MODEL_PATH = os.environ.get("E5_MODEL_PATH")
BGE3_MODEL_PATH = os.environ.get("BGE3_MODEL_PATH")


class FaissRetriever:
    def __init__(self, index, texts: List[str], embedder, top_k: int):
        self._index = index
        self._texts = texts
        self._embedder = embedder
        self._top_k = top_k

    def retrieve(self, query: str) -> List[str]:
        vec = self._embedder.encode([query], normalize_embeddings=True)
        _, idx = self._index.search(vec, self._top_k)
        ids = idx[0].tolist() if len(idx) > 0 else []
        return [self._texts[i] for i in ids if 0 <= i < len(self._texts)]


def _load_texts(db_path: Path) -> List[str]:
    json_path = db_path / "docs.json"
    jsonl_path = db_path / "docs.jsonl"

    if json_path.exists():
        data = json.loads(json_path.read_text(encoding="utf-8"))
        return [str(d.get("text", "")) for d in data]
    if jsonl_path.exists():
        texts = []
        for line in jsonl_path.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            obj = json.loads(line)
            texts.append(str(obj.get("text", "")))
        return texts
    raise FileNotFoundError("Не найден docs.json или docs.jsonl в vector_db/")


def build_retriever(model_name: str | None = None, db_path: str | Path = VECTOR_DB_DIR) -> Retriever:
    """Единая точка подключения ретривера (FAISS).

    Нужно:
    - FAISS индекс: vector_db/index.faiss
    - Документы: vector_db/docs.json или vector_db/docs.jsonl (поле text)
    - Локальные модели эмбеддингов: E5_MODEL_PATH / BGE3_MODEL_PATH
    """

    if faiss is None:
        raise RuntimeError("faiss не установлен. Установите faiss-cpu.")
    if SentenceTransformer is None:
        raise RuntimeError("sentence-transformers не установлен.")

    model_key = (model_name or "").lower()
    if model_key in {"e5-large", "e5_large"}:
        model_path = E5_MODEL_PATH
    elif model_key in {"bge-3", "bge_3", "bge3"}:
        model_path = BGE3_MODEL_PATH
    else:
        model_path = None

    if not model_path:
        raise RuntimeError("Не задан путь к локальной модели эмбеддингов.")

    db_path = Path(db_path)
    index_path = db_path / "index.faiss"
    if not index_path.exists():
        raise FileNotFoundError(f"Не найден FAISS индекс: {index_path}")

    index = faiss.read_index(str(index_path))
    texts = _load_texts(db_path)
    embedder = SentenceTransformer(model_path)
    return FaissRetriever(index=index, texts=texts, embedder=embedder, top_k=RETRIEVER_TOP_K)


class _BaseMockRag:
    def __init__(self, name: str, retriever: Retriever):
        self._name = name
        self._retriever = retriever

    def answer(self, question: str) -> RagResponse:
        contexts = self._retriever.retrieve(question)
        answer = f"[{self._name}] Ответ на: {question}"
        return RagResponse(answer=answer, contexts=contexts)


@rag_system("e5-large")
def build_e5_large():
    retriever = build_retriever("e5-large")
    return _BaseMockRag("e5-large", retriever)


@rag_system("bge-3")
def build_bge_3():
    retriever = build_retriever("bge-3")
    return _BaseMockRag("bge-3", retriever)


@rag_system("langchain")
def build_langchain():
    retriever = build_retriever("langchain")
    return _BaseMockRag("langchain", retriever)
