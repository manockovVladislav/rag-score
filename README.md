# RAGAS оценка одной RAG-системы

Минимальный проект для оценки одной проверяемой RAG-системы через `ragas` в Jupyter.

## Быстрый старт

1. Установите зависимости:
   - `pip install -r requirements.txt`
2. Запустите KoboldCpp (если используете локальный backend):
   - `./koboldcpp --model ~/models/gguf/Phi-3.5-mini-instruct-IQ4_XS.gguf --port 5001`
3. Выберите backend в окружении:
   - `export LLM_BACKEND=koboldcpp` или `export LLM_BACKEND=gigachat`
4. Откройте `eval_rag.ipynb` и выполните ячейки сверху вниз.

## Архитектура

- `rag_systems/gigachat_bge_m3.py` — проверяемая система:
  - генерация через переключаемый backend (`gigachat` или `koboldcpp`);
  - ретривер на локальной `bge-m3` (`/home/vladislav/models/bge-m3`);
  - чтение FAISS базы из `vector_db/` в корне проекта;
  - таймаут вызова модели: через `MODEL_TIMEOUT_SECONDS` (по умолчанию `0`, без таймаута);
  - конфиг собирается через `RagSystemConfig` (из env через `load_config_from_env`).
- `llm_interface.py` — единый интерфейс подключения LLM backend для RAG и RAGAS judge.
- `eval_rag.ipynb` — только запуск проверки через RAGAS по золотым вопросам.
- `rag_eval/run_eval.py` — пайплайн оценки одной системы с расширенной диагностикой и HTML-отчётом.

## Структура

- `data/` — золотые вопросы (xlsx)
- `vector_db/` — векторная база (`index.faiss` + `docs.json` или `docs.jsonl`)
- `rag_systems/` — файл проверяемой RAG-системы
- `rag_eval/` — код запуска оценки
- `eval_rag.ipynb` — точка запуска
- `reports/` — результаты (создаются автоматически)

Важно: файл `rag_systems.py` больше не используется, логика системы перенесена в `rag_systems/gigachat_bge_m3.py`.

## Запуск

1. (Опционально) запустите KoboldCpp:
   - `./koboldcpp --model ~/models/gguf/Phi-3.5-mini-instruct-IQ4_XS.gguf --port 5001`
2. Выберите backend:
   - `LLM_BACKEND=koboldcpp` или `LLM_BACKEND=gigachat`
3. Для `koboldcpp` при необходимости задайте:
   - `KOBOLDCPP_BASE_URL` (по умолчанию `http://127.0.0.1:5001/v1`)
   - `KOBOLDCPP_API_KEY` (по умолчанию `koboldcpp`)
   - `KOBOLDCPP_MODEL` (по умолчанию `koboldcpp`)
4. Убедитесь, что в `vector_db/` есть:
   - `index.faiss`
   - `docs.json` или `docs.jsonl` (поле `text`)
5. Убедитесь, что `data/*.xlsx` содержит колонку `question` (и опционально `ground_truth`).
6. Откройте `eval_rag.ipynb` и выполните ячейки сверху вниз.

## Отчёт

После каждого запуска в `reports/<timestamp>_<run_name>/` сохраняются:

- `scores.csv` — полный построчный результат: RAGAS-метрики + runtime + retrieval + bge-m3 диагностика.
- `scores_compact.csv` — сокращённый построчный вид для быстрого просмотра.
- `summary.csv` — общий агрегированный summary по всем числовым колонкам.
- `summary_ragas.csv` — агрегаты только по RAGAS-метрикам.
- `summary_bge_m3.csv` — агрегаты только по дополнительным bge-m3 метрикам.
- `summary_runtime.csv` — агрегаты по времени/размерам ответа/количеству контекстов.
- `config.csv` — все фактические параметры запуска (модель, endpoint, timeout, top_k и т.д.).
- `parameter_guide.csv` — краткая расшифровка метрик и параметров.
- `run_meta.json` — машинно-читаемые метаданные запуска.
- `report.html` — итоговый человеко-читаемый дашборд по запуску.

## Как читать параметры

- Сначала смотрите `report.html`: там сводка, худшие кейсы и медленные кейсы.
- Для воспроизводимости запускайте `config.csv` и `run_meta.json`: в них полный конфиг RAG/LLM/judge и тайминги.
- Для сравнения качества:
  - `summary_ragas.csv` — judge-оценка ответа;
  - `summary_bge_m3.csv` — независимое семантическое сравнение на локальном `bge-m3`.
- Для анализа по вопросам используйте `scores_compact.csv`, а если нужен полный контекст — `scores.csv`.

## Примечания

- Выполнение сделано последовательно (один поток через `RunConfig(max_workers=1)`).
- По умолчанию включен экономный режим: один shared `LLM` и один shared `bge-m3` для RAG и RAGAS (`run_single_rag_eval(..., use_shared_rag_system_models=True)`).
- Если нужен отдельный judge, отключите shared-режим: `run_single_rag_eval(..., use_shared_rag_system_models=False, judge_llm=..., judge_embeddings=...)`.
- Для изолированного локального запуска:
  - `ISOLATED_LOCAL_ONLY=1` (по умолчанию) разрешает только `LLM_BACKEND=koboldcpp`
  - `GOLD_LIMIT=3` (по умолчанию) — быстрый smoke-прогон по первым 3 вопросам; `GOLD_LIMIT=0` — полный gold
  - `MODEL_TIMEOUT_SECONDS=0` (по умолчанию) отключает таймаут генерации в RAG-системе
  - `RAGAS_TIMEOUT_SECONDS=1800` (по умолчанию) — таймаут задач метрик RAGAS
- `answer_relevancy` считается при наличии `judge_embeddings`; в экономном режиме они берутся из shared `bge-m3` ретривера автоматически.
- Один `LLM_BACKEND` используется сразу для двух частей (через один объект LLM):
  - генерация ответов в проверяемой RAG-системе;
  - judge-модель для расчёта метрик RAGAS.
