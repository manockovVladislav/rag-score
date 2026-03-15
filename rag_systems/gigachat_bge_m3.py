"""Single RAG system: GigaChat API + BGE-M3 retriever over local FAISS."""

from __future__ import annotations

from pathlib import Path
import json
import os

import faiss  # type: ignore
from sentence_transformers import SentenceTransformer

from llm_interface import build_langchain_chat_model, normalize_backend_name


RAG_SYSTEM_NAME = "gigachat_bge_m3"


class RagSystemConfig:
    def __init__(
        self,
        *,
        vector_db_dir: str | Path,
        embedding_model_path: str,
        embedding_device: str,
        retriever_top_k: int,
        model_timeout_seconds: float,
        llm_backend: str,
        gigachat_model: str | None,
        gigachat_credentials: str | None,
        gigachat_scope: str | None,
        gigachat_base_url: str | None,
        koboldcpp_base_url: str,
        koboldcpp_api_key: str,
        koboldcpp_model: str,
        koboldcpp_temperature: float,
        koboldcpp_max_tokens: str | None,
        koboldcpp_max_retries: int,
        prompt_template: str,
    ) -> None:
        self.vector_db_dir = Path(vector_db_dir).expanduser()
        self.embedding_model_path = embedding_model_path
        self.embedding_device = embedding_device.strip().lower()
        self.retriever_top_k = int(retriever_top_k)
        self.model_timeout_seconds = float(model_timeout_seconds)
        self.llm_backend = normalize_backend_name(llm_backend)

        self.gigachat_model = gigachat_model
        self.gigachat_credentials = gigachat_credentials
        self.gigachat_scope = gigachat_scope
        self.gigachat_base_url = gigachat_base_url

        self.koboldcpp_base_url = koboldcpp_base_url
        self.koboldcpp_api_key = koboldcpp_api_key
        self.koboldcpp_model = koboldcpp_model
        self.koboldcpp_temperature = float(koboldcpp_temperature)
        self.koboldcpp_max_tokens = koboldcpp_max_tokens
        self.koboldcpp_max_retries = int(koboldcpp_max_retries)

        self.prompt_template = prompt_template

    @property
    def index_path(self) -> Path:
        return self.vector_db_dir / "index.faiss"

    @property
    def docs_json_path(self) -> Path:
        return self.vector_db_dir / "docs.json"

    @property
    def docs_jsonl_path(self) -> Path:
        return self.vector_db_dir / "docs.jsonl"

    def to_dict(self) -> dict[str, str | int | float | None]:
        return {
            "vector_db_dir": str(self.vector_db_dir),
            "embedding_model_path": self.embedding_model_path,
            "embedding_device": self.embedding_device,
            "retriever_top_k": self.retriever_top_k,
            "model_timeout_seconds": self.model_timeout_seconds,
            "llm_backend": self.llm_backend,
            "gigachat_model": self.gigachat_model,
            "gigachat_base_url": self.gigachat_base_url,
            "koboldcpp_base_url": self.koboldcpp_base_url,
            "koboldcpp_model": self.koboldcpp_model,
            "koboldcpp_temperature": self.koboldcpp_temperature,
            "koboldcpp_max_tokens": self.koboldcpp_max_tokens,
            "koboldcpp_max_retries": self.koboldcpp_max_retries,
        }


class RagAnswer:
    def __init__(self, answer: str, contexts: list[str]) -> None:
        self.answer = answer
        self.contexts = contexts


def default_prompt_template() -> str:
    return """Ты помощник, отвечающий только по предоставленным контекстам.
Если в контекстах нет ответа, честно скажи, что данных недостаточно.

Контексты:
{contexts}

Вопрос:
{question}

Дай короткий и точный ответ на русском языке.
"""


def load_config_from_env(env: dict[str, str] | None = None) -> RagSystemConfig:
    source = env or os.environ
    project_root = Path(__file__).resolve().parents[1]

    return RagSystemConfig(
        vector_db_dir=source.get("VECTOR_DB_DIR", str(project_root / "vector_db")),
        embedding_model_path=source.get("EMBEDDING_MODEL_PATH", "/home/vladislav/models/bge-m3"),
        embedding_device=source.get("EMBEDDING_DEVICE", "cpu"),
        retriever_top_k=int(source.get("RETRIEVER_TOP_K", "5")),
        model_timeout_seconds=float(source.get("MODEL_TIMEOUT_SECONDS", "5")),
        llm_backend=source.get("RAG_LLM_BACKEND", "gigachat"),
        gigachat_model=source.get("GIGACHAT_MODEL"),
        gigachat_credentials=source.get("GIGACHAT_CREDENTIALS"),
        gigachat_scope=source.get("GIGACHAT_SCOPE"),
        gigachat_base_url=source.get("GIGACHAT_BASE_URL"),
        koboldcpp_base_url=source.get("KOBOLDCPP_BASE_URL", "http://127.0.0.1:5001/v1"),
        koboldcpp_api_key=source.get("KOBOLDCPP_API_KEY", "koboldcpp"),
        koboldcpp_model=source.get("KOBOLDCPP_MODEL", "koboldcpp"),
        koboldcpp_temperature=float(source.get("KOBOLDCPP_TEMPERATURE", "0.0")),
        koboldcpp_max_tokens=source.get("KOBOLDCPP_MAX_TOKENS"),
        koboldcpp_max_retries=int(source.get("KOBOLDCPP_MAX_RETRIES", "0")),
        prompt_template=source.get("RAG_PROMPT_TEMPLATE", default_prompt_template()),
    )


def _load_texts(config: RagSystemConfig) -> list[str]:
    if config.docs_json_path.exists():
        data = json.loads(config.docs_json_path.read_text(encoding="utf-8"))
        return [str(item.get("text", "")).strip() for item in data if str(item.get("text", "")).strip()]

    if config.docs_jsonl_path.exists():
        texts: list[str] = []
        for line in config.docs_jsonl_path.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            item = json.loads(line)
            text = str(item.get("text", "")).strip()
            if text:
                texts.append(text)
        return texts

    raise FileNotFoundError(f"Не найден docs.json или docs.jsonl в {config.vector_db_dir}")


def build_sentence_embedder(model_path: str, device: str) -> tuple[SentenceTransformer, str]:
    normalized_device = (device or "cpu").strip().lower()
    if normalized_device not in {"cpu", "cuda"}:
        raise ValueError("EMBEDDING_DEVICE должен быть 'cpu' или 'cuda'")

    try:
        return SentenceTransformer(model_path, device=normalized_device), normalized_device
    except Exception:
        # Fallback keeps local runs alive on incompatible CUDA setups.
        return SentenceTransformer(model_path, device="cpu"), "cpu"


class BgeM3FaissRetriever:
    def __init__(
        self,
        *,
        index,
        texts: list[str],
        embedder: SentenceTransformer,
        embedding_device: str,
        top_k: int,
    ) -> None:
        self._index = index
        self._texts = texts
        self._embedder = embedder
        self._embedding_device = embedding_device
        self._top_k = int(top_k)

    @classmethod
    def from_config(
        cls,
        config: RagSystemConfig,
        *,
        embedder: SentenceTransformer | None = None,
        embedding_device: str | None = None,
    ) -> "BgeM3FaissRetriever":
        if not config.index_path.exists():
            raise FileNotFoundError(f"Не найден индекс FAISS: {config.index_path}")

        index = faiss.read_index(str(config.index_path))
        texts = _load_texts(config)

        loaded_embedder = embedder
        loaded_device = embedding_device or config.embedding_device
        if loaded_embedder is None:
            loaded_embedder, loaded_device = build_sentence_embedder(
                config.embedding_model_path,
                config.embedding_device,
            )
        elif embedding_device is None:
            detected = str(getattr(loaded_embedder, "device", "")).lower()
            if detected.startswith("cuda"):
                loaded_device = "cuda"
            elif detected.startswith("cpu"):
                loaded_device = "cpu"

        return cls(
            index=index,
            texts=texts,
            embedder=loaded_embedder,
            embedding_device=loaded_device,
            top_k=config.retriever_top_k,
        )

    def retrieve(self, query: str, top_k: int | None = None) -> list[str]:
        target_top_k = int(top_k or self._top_k)
        vectors = self._embedder.encode(
            [query],
            normalize_embeddings=True,
            convert_to_numpy=True,
            device=self._embedding_device,
        ).astype("float32")
        _, indexes = self._index.search(vectors, target_top_k)
        found = indexes[0].tolist() if len(indexes) else []
        return [self._texts[i] for i in found if 0 <= i < len(self._texts)]

    @property
    def embedder(self) -> SentenceTransformer:
        return self._embedder

    @property
    def embedding_device(self) -> str:
        return self._embedding_device

    @property
    def texts(self) -> list[str]:
        return self._texts

    @property
    def top_k(self) -> int:
        return self._top_k


class BgeM3RagSystem:
    def __init__(
        self,
        *,
        retriever: BgeM3FaissRetriever,
        llm,
        llm_backend: str,
        prompt_template: str,
        embedding_model_path: str,
    ) -> None:
        self._retriever = retriever
        self._llm = llm
        self._llm_backend = llm_backend
        self._prompt_template = prompt_template
        self._embedding_model_path = embedding_model_path

    def answer(self, question: str) -> RagAnswer:
        contexts = self._retriever.retrieve(question)
        context_text = "\n\n".join(f"{i + 1}. {ctx}" for i, ctx in enumerate(contexts)) or "Контексты не найдены."
        prompt = self._prompt_template.format(contexts=context_text, question=question)

        raw = self._llm.invoke(prompt)
        text = getattr(raw, "content", raw)
        if isinstance(text, list):
            text = " ".join(str(part) for part in text)

        return RagAnswer(answer=str(text).strip(), contexts=contexts)

    def get_chat_model(self):
        return self._llm

    def get_sentence_embedder(self) -> SentenceTransformer:
        return self._retriever.embedder

    def get_embedding_device(self) -> str:
        return self._retriever.embedding_device

    def get_embedding_model_path(self) -> str:
        return self._embedding_model_path

    def get_system_config(self) -> dict[str, str | int | float | None]:
        return {
            "llm_backend": self._llm_backend,
            "embedding_model_path": self._embedding_model_path,
            "embedding_device": self._retriever.embedding_device,
            "retriever_docs_count": len(self._retriever.texts),
            "retriever_top_k": self._retriever.top_k,
        }


def build_rag_system(
    llm_backend: str | None = None,
    *,
    config: RagSystemConfig | None = None,
    shared_llm=None,
    shared_embedder: SentenceTransformer | None = None,
) -> BgeM3RagSystem:
    cfg = config or load_config_from_env()

    backend = llm_backend or cfg.llm_backend
    backend = normalize_backend_name(backend)

    retriever = BgeM3FaissRetriever.from_config(cfg, embedder=shared_embedder)

    llm = shared_llm
    if llm is None:
        llm = build_langchain_chat_model(
            backend=backend,
            timeout=cfg.model_timeout_seconds,
            gigachat_options={
                "model": cfg.gigachat_model,
                "credentials": cfg.gigachat_credentials,
                "scope": cfg.gigachat_scope,
                "base_url": cfg.gigachat_base_url,
            },
            koboldcpp_options={
                "base_url": cfg.koboldcpp_base_url,
                "api_key": cfg.koboldcpp_api_key,
                "model": cfg.koboldcpp_model,
                "temperature": cfg.koboldcpp_temperature,
                "max_tokens": cfg.koboldcpp_max_tokens,
                "max_retries": cfg.koboldcpp_max_retries,
            },
        )

    return BgeM3RagSystem(
        retriever=retriever,
        llm=llm,
        llm_backend=backend,
        prompt_template=cfg.prompt_template,
        embedding_model_path=cfg.embedding_model_path,
    )


__all__ = [
    "RAG_SYSTEM_NAME",
    "RagSystemConfig",
    "RagAnswer",
    "BgeM3FaissRetriever",
    "BgeM3RagSystem",
    "build_sentence_embedder",
    "load_config_from_env",
    "build_rag_system",
]
