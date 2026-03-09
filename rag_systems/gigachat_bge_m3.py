"""Single RAG system: GigaChat API + BGE-M3 retriever over local FAISS."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import json
import os

import faiss  # type: ignore
from sentence_transformers import SentenceTransformer

from llm_interface import build_langchain_chat_model, normalize_backend_name


# System identity
RAG_SYSTEM_NAME = "gigachat_bge_m3"

# System config (kept here intentionally)
PROJECT_ROOT = Path(__file__).resolve().parents[1]
# TODO(ISOLATED_ENV): укажите путь к директории с FAISS-артефактами (index.faiss + docs.json/docs.jsonl).
# Пример: export VECTOR_DB_DIR=/mnt/data/vector_db
VECTOR_DB_DIR = Path(os.environ.get("VECTOR_DB_DIR", str(PROJECT_ROOT / "vector_db"))).expanduser()
INDEX_PATH = VECTOR_DB_DIR / "index.faiss"
DOCS_JSON_PATH = VECTOR_DB_DIR / "docs.json"
DOCS_JSONL_PATH = VECTOR_DB_DIR / "docs.jsonl"

# TODO(ISOLATED_ENV): укажите путь к локальной модели эмбеддингов (bge-m3).
# Пример: export EMBEDDING_MODEL_PATH=/models/bge-m3
EMBEDDING_MODEL_PATH = os.environ.get("EMBEDDING_MODEL_PATH", "/home/vladislav/models/bge-m3")
EMBEDDING_DEVICE = os.environ.get("EMBEDDING_DEVICE", "cpu").strip().lower()
# TODO(ISOLATED_ENV): регулирует, сколько ближайших контекстов возвращает ретривер.
# Для "вернуть 5 ближайших вариантов" оставьте 5:
# export RETRIEVER_TOP_K=5
RETRIEVER_TOP_K = int(os.environ.get("RETRIEVER_TOP_K", "5"))
# TODO(ISOLATED_ENV): таймаут ответа LLM в секундах (для GigaChat на вашем стенде = 5).
# Пример: export MODEL_TIMEOUT_SECONDS=5
MODEL_TIMEOUT_SECONDS = float(os.environ.get("MODEL_TIMEOUT_SECONDS", "5"))

# LLM backend config
RAG_LLM_BACKEND = os.environ.get("RAG_LLM_BACKEND", "gigachat")

# Optional GigaChat settings
# TODO(ISOLATED_ENV): укажите параметры доступа к GigaChat API.
# Обязательное: GIGACHAT_CREDENTIALS. Опционально: model/scope/base_url.
GIGACHAT_MODEL = os.environ.get("GIGACHAT_MODEL")
GIGACHAT_CREDENTIALS = os.environ.get("GIGACHAT_CREDENTIALS")
GIGACHAT_SCOPE = os.environ.get("GIGACHAT_SCOPE")
GIGACHAT_BASE_URL = os.environ.get("GIGACHAT_BASE_URL")

# Optional KoboldCpp settings (OpenAI-compatible endpoint)
KOBOLDCPP_BASE_URL = os.environ.get("KOBOLDCPP_BASE_URL", "http://127.0.0.1:5001/v1")
KOBOLDCPP_API_KEY = os.environ.get("KOBOLDCPP_API_KEY", "koboldcpp")
KOBOLDCPP_MODEL = os.environ.get("KOBOLDCPP_MODEL", "koboldcpp")
KOBOLDCPP_TEMPERATURE = float(os.environ.get("KOBOLDCPP_TEMPERATURE", "0.0"))
KOBOLDCPP_MAX_TOKENS = os.environ.get("KOBOLDCPP_MAX_TOKENS")
KOBOLDCPP_MAX_RETRIES = int(os.environ.get("KOBOLDCPP_MAX_RETRIES", "0"))

PROMPT_TEMPLATE = """Ты помощник, отвечающий только по предоставленным контекстам.
Если в контекстах нет ответа, честно скажи, что данных недостаточно.

Контексты:
{contexts}

Вопрос:
{question}

Дай короткий и точный ответ на русском языке.
"""


@dataclass
class RagAnswer:
    answer: str
    contexts: list[str]


def _load_texts() -> list[str]:
    if DOCS_JSON_PATH.exists():
        data = json.loads(DOCS_JSON_PATH.read_text(encoding="utf-8"))
        return [str(item.get("text", "")).strip() for item in data if str(item.get("text", "")).strip()]

    if DOCS_JSONL_PATH.exists():
        texts: list[str] = []
        for line in DOCS_JSONL_PATH.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            item = json.loads(line)
            text = str(item.get("text", "")).strip()
            if text:
                texts.append(text)
        return texts

    raise FileNotFoundError(f"Не найден docs.json или docs.jsonl в {VECTOR_DB_DIR}")


class BgeM3FaissRetriever:
    def __init__(self) -> None:
        if not INDEX_PATH.exists():
            raise FileNotFoundError(f"Не найден индекс FAISS: {INDEX_PATH}")

        self._index = faiss.read_index(str(INDEX_PATH))
        self._texts = _load_texts()
        self._embedding_device = EMBEDDING_DEVICE
        if self._embedding_device not in {"cpu", "cuda"}:
            raise ValueError("EMBEDDING_DEVICE должен быть 'cpu' или 'cuda'")

        try:
            self._embedder = SentenceTransformer(EMBEDDING_MODEL_PATH, device=self._embedding_device)
        except Exception:
            # Fallback keeps local runs alive on incompatible CUDA setups.
            self._embedding_device = "cpu"
            self._embedder = SentenceTransformer(EMBEDDING_MODEL_PATH, device=self._embedding_device)

    def retrieve(self, query: str, top_k: int = RETRIEVER_TOP_K) -> list[str]:
        # top_k задает число ближайших чанков FAISS для одного запроса.
        # Можно менять глобально через RETRIEVER_TOP_K или точечно через аргумент top_k.
        vectors = self._embedder.encode(
            [query],
            normalize_embeddings=True,
            convert_to_numpy=True,
            device=self._embedding_device,
        ).astype("float32")
        _, indexes = self._index.search(vectors, top_k)
        found = indexes[0].tolist() if len(indexes) else []
        return [self._texts[i] for i in found if 0 <= i < len(self._texts)]


class BgeM3RagSystem:
    def __init__(self, retriever: BgeM3FaissRetriever | None = None, llm_backend: str = RAG_LLM_BACKEND) -> None:
        self._retriever = retriever or BgeM3FaissRetriever()
        self._llm_backend = normalize_backend_name(llm_backend)
        self._llm = build_langchain_chat_model(
            backend=self._llm_backend,
            timeout=MODEL_TIMEOUT_SECONDS,
            gigachat_options={
                "model": GIGACHAT_MODEL,
                "credentials": GIGACHAT_CREDENTIALS,
                "scope": GIGACHAT_SCOPE,
                "base_url": GIGACHAT_BASE_URL,
            },
            koboldcpp_options={
                "base_url": KOBOLDCPP_BASE_URL,
                "api_key": KOBOLDCPP_API_KEY,
                "model": KOBOLDCPP_MODEL,
                "temperature": KOBOLDCPP_TEMPERATURE,
                "max_tokens": KOBOLDCPP_MAX_TOKENS,
                "max_retries": KOBOLDCPP_MAX_RETRIES,
            },
        )

    def answer(self, question: str) -> RagAnswer:
        contexts = self._retriever.retrieve(question)
        context_text = "\n\n".join(f"{i + 1}. {ctx}" for i, ctx in enumerate(contexts)) or "Контексты не найдены."
        prompt = PROMPT_TEMPLATE.format(contexts=context_text, question=question)

        raw = self._llm.invoke(prompt)
        text = getattr(raw, "content", raw)
        if isinstance(text, list):
            text = " ".join(str(part) for part in text)

        return RagAnswer(answer=str(text).strip(), contexts=contexts)


def build_rag_system(llm_backend: str | None = None) -> BgeM3RagSystem:
    return BgeM3RagSystem(llm_backend=llm_backend or RAG_LLM_BACKEND)
