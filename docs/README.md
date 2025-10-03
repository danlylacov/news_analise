# News Analysis & Forecasting Project

## 📁 Структура проекта

```
news_analize/
├── src/                          # Исходный код
│   ├── core/                     # Основные модули
│   │   ├── news_nlp.py           # NLP обработка новостей
│   │   ├── news_features.py      # Генерация фич из новостей
│   │   ├── auto_label_tickers.py # Автоматическая разметка тикеров
│   │   └── infer_news_to_candles.py # Инференс новостей к свечам
│   ├── ml/                       # ML компоненты
│   │   ├── nn_model.py           # Нейронная сеть
│   │   ├── nn_data.py            # Подготовка данных для ML
│   │   └── train_news_ticker.py  # Обучение модели
│   ├── api/                      # API компоненты
│   │   └── app.py                # FastAPI приложение
│   └── utils/                    # Утилиты
│       ├── create_full_json.py   # Создание полного JSON
│       ├── create_synchronized_json.py # Синхронизированный JSON
│       └── fix_parameters.py     # Исправление параметров
├── datasets/                     # Данные
│   ├── raw/                      # Исходные данные
│   │   ├── test_news.csv         # Тестовые новости
│   │   ├── public_test_candles.csv # Тестовые свечи
│   │   ├── train_news.csv        # Обучающие новости
│   │   └── train_candles.csv     # Обучающие свечи
│   ├── processed/                # Обработанные данные
│   │   ├── nn_features.parquet   # Новостные фичи
│   │   └── candles_with_news.csv # Свечи с новостями
│   └── output/                   # Результаты
├── notebooks/                    # Jupyter ноутбуки
│   ├── colab/                    # Google Colab
│   │   ├── colab_news_forecast.ipynb
│   │   └── colab_github_news_forecast.ipynb
│   └── kaggle/                   # Kaggle
│       └── kaggle_news_forecast.ipynb
├── tests/                        # Тесты
│   ├── unit/                     # Модульные тесты
│   └── integration/              # Интеграционные тесты
│       ├── test_full_response.py # Тест полного ответа
│       ├── test_json_api.py      # Тест JSON API
│       ├── test_json_local.py    # Локальный тест
│       ├── test_fixed_files.py   # Тест исправленных файлов
│       └── demo_response.py      # Демо ответа
├── scripts/                      # Скрипты
│   ├── data_processing/          # Обработка данных
│   └── testing/                  # Тестирование
│       └── test_api_curl.sh      # Curl тест
├── examples/                     # Примеры
│   ├── json_samples/             # JSON образцы
│   │   ├── swagger_test_data.json
│   │   ├── full_swagger_data.json
│   │   ├── full_swagger_data_fixed.json
│   │   └── synchronized_swagger_data.json
│   └── api_responses/            # Ответы API
│       ├── example_api_response.json
│       └── api_response_*.json
├── config/                       # Конфигурация
│   ├── requirements.txt          # Зависимости Python
│   ├── docker-compose.yml        # Docker Compose
│   └── Dockerfile                # Docker образ
├── docs/                         # Документация
│   └── README.md                 # Основная документация
├── artifacts/                    # Артефакты модели
│   ├── model.pt                  # Обученная модель
│   ├── vocab.json                # Словарь
│   ├── tickers.json              # Тикеры
│   └── lexicon.json              # Лексикон
└── data/                         # Данные (Docker volume)
    └── artifacts/                # Артефакты для Docker
```

## 🚀 Быстрый старт

### 1. Установка зависимостей
```bash
pip install -r config/requirements.txt
```

### 2. Обучение модели
```bash
python src/ml/train_news_ticker.py --news datasets/raw/train_news.csv --artifacts artifacts/ --epochs 3
```

### 3. Запуск API
```bash
python src/api/app.py
# или через Docker
docker-compose -f config/docker-compose.yml up
```

### 4. Тестирование
```bash
# Python тесты
python tests/integration/test_full_response.py

# Curl тест
bash scripts/testing/test_api_curl.sh
```

## 📊 Основные компоненты

### Core модули
- **news_nlp.py**: Обработка текста, лемматизация, анализ тональности
- **news_features.py**: Генерация фич из новостей
- **auto_label_tickers.py**: Автоматическая привязка новостей к тикерам
- **infer_news_to_candles.py**: Связывание новостей со свечами

### ML компоненты
- **nn_model.py**: BiGRU модель с attention для классификации
- **nn_data.py**: Подготовка данных для нейронной сети
- **train_news_ticker.py**: Обучение модели новости→тикеры

### API
- **app.py**: FastAPI сервер для инференса новостей

## 🔧 Использование

### JSON API
```bash
curl -X POST "http://localhost:8000/infer" \
  -H "Content-Type: application/json" \
  -d @examples/json_samples/full_swagger_data_fixed.json
```

### Параметры
- `p_threshold`: Порог релевантности новостей (0.1-0.5)
- `half_life_days`: Период полураспада влияния (0.5-30 дней)
- `max_days`: Максимальный возраст новостей (5-2000 дней)

## 📈 Результаты

API возвращает:
- `nn_news_sum`: Суммарное влияние новостей
- `nn_news_mean`: Среднее влияние
- `nn_news_max`: Максимальное влияние
- `nn_news_count`: Количество релевантных новостей

## 🐳 Docker

```bash
# Сборка и запуск
docker-compose -f config/docker-compose.yml up --build

# Тестирование
curl -X POST "http://localhost:8000/infer" \
  -H "Content-Type: application/json" \
  -d @examples/json_samples/swagger_test_data.json
```