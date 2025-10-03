# 🔌 API Schema Reference

## 📋 OpenAPI Specification

```yaml
openapi: 3.0.0
info:
  title: News Analysis & Forecasting API
  version: 1.0.0
  description: API для анализа новостей и прогнозирования движения цен

servers:
  - url: http://localhost:8000
    description: Локальный сервер разработки

paths:
  /infer:
    post:
      summary: Анализ влияния новостей на цены
      requestBody:
        required: true
        content:
          application/json:
            schema:
              $ref: '#/components/schemas/InferRequest'
      responses:
        '200':
          description: Успешный анализ
          content:
            application/json:
              schema:
                $ref: '#/components/schemas/InferResponse'
        '422':
          description: Ошибка валидации
        '500':
          description: Внутренняя ошибка сервера

  /health:
    post:
      summary: Проверка состояния API
      responses:
        '200':
          description: API работает
          content:
            application/json:
              schema:
                $ref: '#/components/schemas/HealthResponse'

components:
  schemas:
    NewsItem:
      type: object
      required:
        - publish_date
        - title
        - publication
      properties:
        publish_date:
          type: string
          format: date-time
          example: "2025-04-15T10:00:00"
        title:
          type: string
          example: "Сбербанк повышает ставки по депозитам"
        publication:
          type: string
          example: "РБК"

    CandleItem:
      type: object
      required:
        - begin
        - ticker
        - open
        - high
        - low
        - close
        - volume
      properties:
        begin:
          type: string
          format: date-time
          example: "2025-04-16T00:00:00"
        ticker:
          type: string
          example: "SBER"
        open:
          type: number
          example: 250.5
        high:
          type: number
          example: 255.8
        low:
          type: number
          example: 248.2
        close:
          type: number
          example: 253.1
        volume:
          type: integer
          example: 15000000

    InferRequest:
      type: object
      required:
        - news
        - candles
      properties:
        news:
          type: array
          items:
            $ref: '#/components/schemas/NewsItem'
        candles:
          type: array
          items:
            $ref: '#/components/schemas/CandleItem'
        artifacts_dir:
          type: string
          default: "artifacts"
        p_threshold:
          type: number
          minimum: 0.0
          maximum: 1.0
          default: 0.5
        half_life_days:
          type: number
          minimum: 0.1
          maximum: 100.0
          default: 0.5
        max_days:
          type: number
          minimum: 1
          maximum: 2000
          default: 5

    FeatureItem:
      type: object
      properties:
        ticker:
          type: string
          example: "SBER"
        date:
          type: string
          format: date
          example: "2025-04-16"
        nn_news_sum:
          type: number
          example: 0.2345
        nn_news_mean:
          type: number
          example: 0.1567
        nn_news_max:
          type: number
          example: 0.3456
        nn_news_count:
          type: integer
          example: 3

    JoinedItem:
      allOf:
        - $ref: '#/components/schemas/CandleItem'
        - type: object
          properties:
            date:
              type: string
              format: date
              example: "2025-04-16"
            nn_news_sum:
              type: number
              example: 0.2345
            nn_news_mean:
              type: number
              example: 0.1567
            nn_news_max:
              type: number
              example: 0.3456
            nn_news_count:
              type: number
              example: 3

    InferResponse:
      type: object
      properties:
        status:
          type: string
          example: "success"
        rows_features:
          type: integer
          example: 358
        rows_joined:
          type: integer
          example: 378
        features_preview:
          type: array
          items:
            $ref: '#/components/schemas/FeatureItem'
        joined_preview:
          type: array
          items:
            $ref: '#/components/schemas/JoinedItem'
        message:
          type: string
          nullable: true

    HealthResponse:
      type: object
      properties:
        status:
          type: string
          example: "healthy"
        message:
          type: string
          example: "Все артефакты найдены"
        missing_files:
          type: array
          items:
            type: string
```

## 🔧 JSON Schema для валидации

### **InferRequest Schema**
```json
{
  "$schema": "http://json-schema.org/draft-07/schema#",
  "type": "object",
  "required": ["news", "candles"],
  "properties": {
    "news": {
      "type": "array",
      "items": {
        "type": "object",
        "required": ["publish_date", "title", "publication"],
        "properties": {
          "publish_date": {"type": "string", "format": "date-time"},
          "title": {"type": "string", "minLength": 1},
          "publication": {"type": "string", "minLength": 1}
        }
      },
      "maxItems": 10000
    },
    "candles": {
      "type": "array",
      "items": {
        "type": "object",
        "required": ["begin", "ticker", "open", "high", "low", "close", "volume"],
        "properties": {
          "begin": {"type": "string", "format": "date-time"},
          "ticker": {"type": "string", "pattern": "^[A-Z]{2,6}$"},
          "open": {"type": "number", "minimum": 0},
          "high": {"type": "number", "minimum": 0},
          "low": {"type": "number", "minimum": 0},
          "close": {"type": "number", "minimum": 0},
          "volume": {"type": "integer", "minimum": 0}
        }
      },
      "maxItems": 1000
    },
    "artifacts_dir": {"type": "string", "default": "artifacts"},
    "p_threshold": {"type": "number", "minimum": 0.0, "maximum": 1.0, "default": 0.5},
    "half_life_days": {"type": "number", "minimum": 0.1, "maximum": 100.0, "default": 0.5},
    "max_days": {"type": "number", "minimum": 1, "maximum": 2000, "default": 5}
  }
}
```

## 📊 Типы данных

### **Поддерживаемые тикеры**
```json
{
  "supported_tickers": [
    "AFLT", "ALRS", "CHMF", "GAZP", "GMKN",
    "LKOH", "MAGN", "MGNT", "MOEX", "MTSS",
    "NVTK", "PHOR", "PLZL", "ROSN", "RUAL",
    "SBER", "SIBN", "T", "VTBR"
  ]
}
```

### **Форматы дат**
- **ISO 8601**: `2025-04-16T00:00:00`
- **Дата**: `2025-04-16`
- **Время**: `10:00:00`

### **Числовые типы**
- **Цены**: `number` (float64)
- **Объемы**: `integer` (int64)
- **Вероятности**: `number` (0.0-1.0)

## 🚨 Коды ошибок

| Код | Описание | Причина |
|-----|----------|---------|
| 200 | OK | Успешный запрос |
| 400 | Bad Request | Неверный формат JSON |
| 422 | Unprocessable Entity | Ошибка валидации данных |
| 500 | Internal Server Error | Ошибка сервера |
| 503 | Service Unavailable | Сервис недоступен |

## 🔍 Примеры запросов

### **Минимальный запрос**
```json
{
  "news": [
    {
      "publish_date": "2025-04-15T10:00:00",
      "title": "Тест",
      "publication": "Тест"
    }
  ],
  "candles": [
    {
      "begin": "2025-04-16T00:00:00",
      "ticker": "SBER",
      "open": 100.0,
      "high": 110.0,
      "low": 90.0,
      "close": 105.0,
      "volume": 1000
    }
  ]
}
```

### **Полный запрос**
```json
{
  "news": [...],
  "candles": [...],
  "artifacts_dir": "artifacts",
  "p_threshold": 0.3,
  "half_life_days": 7.0,
  "max_days": 30
}
```

## 📈 Rate Limits

- **Запросов в минуту**: 60
- **Новостей за запрос**: 10,000
- **Свечей за запрос**: 1,000
- **Размер запроса**: 100MB

## 🔐 Аутентификация

В текущей версии аутентификация не требуется. В будущих версиях планируется:
- API ключи
- JWT токены
- OAuth 2.0
