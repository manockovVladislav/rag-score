# RAGAS оценка для RAG

Проект для оценки RAG-систем через ragas. Запуск только из Jupyter.

## Что делаем

1. Кладём векторную базу в `vector_db/`.
2. Выбираем модель ретривера (`e5-large` или `bge-3`).
3. Запускаем `eval_rag.ipynb`.
4. Получаем отчёт по золотым вопросам.

## Структура

- `data/` — Excel с вопросами
- `rag_eval/` — код оценки и отчёта
- `rag_systems.py` — реестр RAG и FAISS‑ретривер
- `eval_rag.ipynb` — запуск
- `vector_db/` — FAISS индекс и документы
- `reports/` — отчёты (создаётся автоматически)

## Запуск

Откройте `eval_rag.ipynb` и выполните ячейки сверху вниз.

## Форматы и типы данных

### 1) Золотые вопросы (Excel)

Файл в `data/` (например, `gold_questions.xlsx`).
Колонки:
- `question` (str, обяз.) — текст вопроса
- `ground_truth` (str, опц.) — эталонный ответ

Допускаются синонимы колонок, они будут нормализованы:
- `question`: `query`, `q`, `вопрос`, `вопрос_пользователя`
- `ground_truth`: `reference`, `answer_gt`, `эталон`, `эталонный_ответ`, `правильный_ответ`

### 2) Векторная база (FAISS)

Папка `vector_db/` рядом с проектом.
Должны быть:
- `index.faiss` — FAISS индекс
- `docs.json` **или** `docs.jsonl` — тексты документов

Структура папки:
```
vector_db/
  index.faiss
  docs.json            # или docs.jsonl
```

Формат `docs.json`:
```json
[
  {
    "id": "doc-001",
    "text": "Полный текст документа или чанка",
    "metadata": {
      "source": "file.pdf",
      "page": 12,
      "title": "Документ"
    }
  }
]
```

Формат `docs.jsonl`:
```jsonl
{"id": "doc-001", "text": "Полный текст документа или чанка", "metadata": {"source": "file.pdf", "page": 12}}
{"id": "doc-002", "text": "Ещё один чанк", "metadata": {"source": "site", "url": "https://..."}}
```

Важно: порядок текстов в `docs.*` должен совпадать с порядком вектора в `index.faiss`.

### 3) Ретривер и модели

В `eval_rag.ipynb` задаются пути к локальным моделям:
- `E5_MODEL_PATH`
- `BGE3_MODEL_PATH`

В `rag_systems.py` ретривер использует:
- `VECTOR_DB_DIR` (по умолчанию `vector_db/`)
- `RETRIEVER_TOP_K`

Формат данных для ретривера:
- вход: `query: str` (текст вопроса)
- выход: `list[str]` (список текстов/чанков, которые пойдут в `contexts`)

Как формировать индекс:
- каждый элемент в `docs.*` соответствует одному вектору в `index.faiss`
- обязательное поле: `text`
- опциональные поля: `id`, `metadata` (любые ключи)

Структура кода ретривера:
```
build_retriever(model_name, db_path)
  -> faiss.read_index(...)
  -> load docs.json / docs.jsonl
  -> SentenceTransformer(model_path)
  -> retrieve(query) -> list[str]
```

## Отчёт

HTML‑отчёт лежит в `reports/<run_id>/`.
Секция **Config** показывает параметры ретривера из ноутбука.
