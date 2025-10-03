# 📰 News Analysis & Forecasting Project

Система анализа новостей и прогнозирования движения цен на финансовых рынках с использованием нейронных сетей.

## 🎯 Возможности

- **Автоматическая разметка новостей** по тикерам компаний
- **Нейронная сеть** для определения релевантности новостей
- **Агрегация новостных фич** с учетом временного затухания
- **REST API** для интеграции с внешними системами
- **Docker контейнеризация** для легкого развертывания

## 🚀 Быстрый старт

### 1. Клонирование и установка
```bash
git clone <repository-url>
cd news_analize
pip install -r config/requirements.txt
```

### 2. Обучение модели
```bash
python src/ml/train_news_ticker.py \
  --news datasets/raw/train_news.csv \
  --artifacts artifacts/ \
  --epochs 3
```

### 3. Запуск API
```bash
# Локально
python src/api/app.py

# Или через Docker
docker-compose -f config/docker-compose.yml up
```

### 4. Тестирование
```bash
# Тест с исправленными параметрами
python tests/integration/test_full_response.py

# Curl тест
bash scripts/testing/test_api_curl.sh
```

## 📁 Структура проекта

```
news_analize/
├── src/                    # Исходный код
│   ├── core/              # Основные модули
│   ├── ml/                # ML компоненты  
│   ├── api/               # API сервер
│   └── utils/             # Утилиты
├── datasets/              # Данные
│   ├── raw/               # Исходные данные
│   ├── processed/         # Обработанные данные
│   └── output/            # Результаты
├── notebooks/             # Jupyter ноутбуки
├── tests/                 # Тесты
├── scripts/               # Скрипты
├── examples/              # Примеры
├── config/                # Конфигурация
├── docs/                  # Документация
└── artifacts/             # Артефакты модели
```

## 🔧 API Использование

### JSON запрос
```bash
curl -X POST "http://localhost:8000/infer" \
  -H "Content-Type: application/json" \
  -d @examples/json_samples/full_swagger_data_fixed.json
```

### Параметры
- `p_threshold`: Порог релевантности (0.1-0.5)
- `half_life_days`: Период полураспада (0.5-30 дней)  
- `max_days`: Максимальный возраст новостей (5-2000 дней)

### Ответ
```json
{
  "status": "success",
  "rows_features": 358,
  "rows_joined": 378,
  "features_preview": [...],
  "joined_preview": [...]
}
```

## 📊 Новостные фичи

- **nn_news_sum**: Суммарное влияние новостей
- **nn_news_mean**: Среднее влияние
- **nn_news_max**: Максимальное влияние одной новости
- **nn_news_count**: Количество релевантных новостей

## 🐳 Docker

```bash
# Сборка и запуск
docker-compose -f config/docker-compose.yml up --build

# Проверка статуса
curl http://localhost:8000/health
```

## 📚 Документация

- **[Основная документация](docs/README.md)** - Обзор проекта и структура
- **[API документация](docs/API_DOCUMENTATION.md)** - Подробное описание API, входных и выходных данных
- **[Технические спецификации](docs/MODEL_SPECIFICATIONS.md)** - Архитектура модели и технические детали
- **[Примеры использования](docs/USAGE_EXAMPLES.md)** - Практические примеры и код
- **[API Schema](docs/API_SCHEMA.md)** - OpenAPI спецификация и JSON схемы
- **[Swagger UI](http://localhost:8000/docs)** - Интерактивная документация API

## 🤝 Вклад в проект

1. Форкните репозиторий
2. Создайте ветку для новой функции
3. Внесите изменения
4. Создайте Pull Request

## 📄 Лицензия

MIT License - см. файл LICENSE для деталей.
