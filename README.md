# Фичи из новостей для прогноза котировок (FORECAST)

Проект строит признаки по схеме «тикер × день» из новостной ленты для последующего объединения с дневными свечами.

## Установка

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

## Запуск (лексиконный базовый пайплайн)

```bash
# Обучение (сохранение артефактов конфигурации)
python3 news_features.py train \
  --news task_1_news.csv \
  --out features_train.parquet \
  --artifacts artifacts/

# Инференс (генерация признаков на тесте)
python3 news_features.py predict \
  --news task_1_news_test.csv \
  --out features_test.parquet \
  --artifacts artifacts/
```

## Нейросетевой пайплайн (BiGRU+attention)

### Обучение модели сопоставления новости→тикеры
```bash
python3 train_news_ticker.py \
  --news task_1_news.csv \
  --artifacts artifacts/ \
  --epochs 5 --batch_size 64 --max_len 256
```
Артефакты: `artifacts/model.pt`, `artifacts/vocab.json`, `artifacts/tickers.json`.

### Инференс и агрегация влияния на свечи (с временным затуханием)
```bash
python3 infer_news_to_candles.py \
  --news task_1_news.csv \
  --candles task_1_candles.csv \
  --artifacts artifacts/ \
  --out nn_features.parquet
```
Выход: признаки `nn_news_sum`, `nn_news_mean`, `nn_news_count` по ключу (`ticker`, `date`).

## Выходные признаки (по тикеру и дню)
- count_news: число новостей
- sentiment_mean/sum: средняя и сумма полярности
- share_pos/share_neg: доли позитивных/негативных новостей
- dup_count: число дубликатов в этот день
- keyword_*: бинарные индикаторы ключевых событий (дивиденды, санкции и др.)
- nn_news_sum/mean/count: от нейросети (релевантность новости тикеру с экспон. затуханием)
- Скользящие окна 1/3/5/10/20 дней для метрик выше (mean/sum)

Признаки сохраняются в Parquet/CSV и готовы к джойну со свечами по ключу (`ticker`, `date`).
