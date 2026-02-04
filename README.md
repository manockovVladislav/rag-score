# RAGAS тестовая система для RAG

Этот репозиторий — отдельная система оценки качества RAG на базе **ragas**.
Золотые вопросы хранятся в Excel и могут пополняться аналитиком.
Подключение разных RAG-систем делается через декоратор `@rag_system(...)`.

## Быстрый старт через Jupyter Notebook

Основной рекомендуемый запуск — из ноутбука. Готовый пример: `examples/run_eval.ipynb`.

Минимальный код:

```python
import sys, os
root = os.path.abspath('..')  # если ноутбук лежит в examples/
src = os.path.join(root, 'src')
if src not in sys.path:
    sys.path.insert(0, src)

from rag_eval import run_eval_notebook

run_eval_notebook(
    gold_path=os.path.join(root, 'data', 'gold_questions.xlsx'),
    registry_modules=['examples.rag_systems'],
    system_names=['e5-large', 'bge-3', 'langchain'],
)
```

Результаты будут сохранены в `outputs/<timestamp>/<system_name>/`.

## Формат Excel с золотыми вопросами

Ожидаются колонки:
- `question` — текст вопроса
- `ground_truth` — эталонный ответ

Пример находится в `data/gold_questions.xlsx`.

## Подключение своих RAG-систем

1) Создайте модуль, например `examples/rag_systems.py`.
2) Зарегистрируйте системы через декоратор.
3) В `answer(question)` верните `RagResponse`.

Минимальный шаблон:

```python
from rag_eval import rag_system
from rag_eval.registry import RagResponse

class MyRag:
    def answer(self, question: str) -> RagResponse:
        # 1) собрать контексты
        # 2) сгенерировать ответ
        return RagResponse(answer="...", contexts=["..."])

@rag_system("my-rag")
def build_my_rag():
    return MyRag()
```

Запуск:

```bash
python -m rag_eval.run_eval \
  --gold data/gold_questions.xlsx \
  --registries examples.rag_systems,my_custom_module \
  --systems my-rag
```

## Что сохраняется

Для каждой системы создается папка с результатами:
- `raw_answers.csv` — сырые ответы + контексты
- `per_question_metrics.csv` — метрики по каждому вопросу
- `overall_metrics.json` — агрегированные метрики
- `overall_metrics.png` — график агрегированных метрик
- `per_question_metrics.png` — график метрик по вопросам (если есть)
- `report.md` — короткий markdown отчет

## Где менять параметры

- Список метрик ragas: `src/rag_eval/run_eval.py`
- Формат данных: `src/rag_eval/run_eval.py`
- Реестр и адаптер: `src/rag_eval/registry.py`

## Запуск через Jupyter Notebook

Готовый пример ноутбука: `examples/run_eval.ipynb`.

Минимальный код внутри ноутбука:

```python
import sys, os
root = os.path.abspath('..')  # если ноутбук лежит в examples/
src = os.path.join(root, 'src')
if src not in sys.path:
    sys.path.insert(0, src)

from rag_eval.notebook import run_eval_notebook

run_eval_notebook(
    gold_path=os.path.join(root, 'data', 'gold_questions.xlsx'),
    registry_modules=['examples.rag_systems'],
    system_names=['e5-large', 'bge-3', 'langchain'],
)
```
