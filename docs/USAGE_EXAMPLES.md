# 🚀 Примеры использования API

## 📋 Быстрые примеры

### 1. **Проверка здоровья API**
```bash
curl -X POST "http://localhost:8000/health"
```

### 2. **Простой тест с одной новостью**
```bash
curl -X POST "http://localhost:8000/infer" \
  -H "Content-Type: application/json" \
  -d '{
    "news": [
      {
        "publish_date": "2025-04-15T10:00:00",
        "title": "Сбербанк объявляет о новых кредитных продуктах",
        "publication": "РБК"
      }
    ],
    "candles": [
      {
        "begin": "2025-04-16T00:00:00",
        "ticker": "SBER",
        "open": 250.0,
        "high": 255.0,
        "low": 248.0,
        "close": 253.0,
        "volume": 1000000
      }
    ],
    "p_threshold": 0.3,
    "half_life_days": 2.0,
    "max_days": 7
  }'
```

### 3. **Тест с полными данными**
```bash
curl -X POST "http://localhost:8000/infer" \
  -H "Content-Type: application/json" \
  -d @examples/json_samples/full_swagger_data_fixed.json
```

### 4. **Сохранение результата в файл**
```bash
curl -X POST "http://localhost:8000/infer" \
  -H "Content-Type: application/json" \
  -d @examples/json_samples/swagger_test_data.json \
  -o result.json
```

## 🐍 Python примеры

### **Базовый запрос**
```python
import requests
import json

# Подготовка данных
data = {
    "news": [
        {
            "publish_date": "2025-04-15T10:00:00",
            "title": "Газпром увеличивает экспорт газа",
            "publication": "Коммерсант"
        }
    ],
    "candles": [
        {
            "begin": "2025-04-16T00:00:00",
            "ticker": "GAZP",
            "open": 180.0,
            "high": 185.0,
            "low": 178.0,
            "close": 183.0,
            "volume": 2000000
        }
    ],
    "p_threshold": 0.5,
    "half_life_days": 1.0,
    "max_days": 5
}

# Отправка запроса
response = requests.post(
    "http://localhost:8000/infer",
    json=data,
    timeout=30
)

# Обработка ответа
if response.status_code == 200:
    result = response.json()
    print(f"Статус: {result['status']}")
    print(f"Фич: {result['rows_features']}")
    print(f"Объединенных строк: {result['rows_joined']}")
else:
    print(f"Ошибка: {response.status_code}")
```

### **Обработка больших данных**
```python
import requests
import pandas as pd
from datetime import datetime, timedelta

def analyze_news_impact(news_df, candles_df, params):
    """Анализ влияния новостей на цены"""
    
    # Конвертация в JSON формат
    news_json = []
    for _, row in news_df.iterrows():
        news_json.append({
            "publish_date": row['publish_date'],
            "title": row['title'],
            "publication": row['publication']
        })
    
    candles_json = []
    for _, row in candles_df.iterrows():
        candles_json.append({
            "begin": row['begin'],
            "ticker": row['ticker'],
            "open": float(row['open']),
            "high": float(row['high']),
            "low": float(row['low']),
            "close": float(row['close']),
            "volume": int(row['volume'])
        })
    
    # Запрос к API
    payload = {
        "news": news_json,
        "candles": candles_json,
        **params
    }
    
    response = requests.post(
        "http://localhost:8000/infer",
        json=payload,
        timeout=60
    )
    
    if response.status_code == 200:
        return response.json()
    else:
        raise Exception(f"API Error: {response.status_code}")

# Использование
news_df = pd.read_csv('datasets/raw/test_news.csv')
candles_df = pd.read_csv('datasets/raw/public_test_candles.csv')

params = {
    "p_threshold": 0.3,
    "half_life_days": 7.0,
    "max_days": 30
}

result = analyze_news_impact(news_df, candles_df, params)
```

## 📊 Анализ результатов

### **Интерпретация фич**
```python
def interpret_features(features):
    """Интерпретация новостных фич"""
    
    for feature in features:
        ticker = feature['ticker']
        date = feature['date']
        
        # Анализ влияния
        if feature['nn_news_sum'] > 0.5:
            impact_level = "ВЫСОКОЕ"
        elif feature['nn_news_sum'] > 0.2:
            impact_level = "СРЕДНЕЕ"
        else:
            impact_level = "НИЗКОЕ"
        
        # Анализ активности
        if feature['nn_news_count'] > 5:
            activity_level = "ВЫСОКАЯ"
        elif feature['nn_news_count'] > 2:
            activity_level = "СРЕДНЯЯ"
        else:
            activity_level = "НИЗКАЯ"
        
        print(f"{ticker} ({date}):")
        print(f"  Влияние: {impact_level} ({feature['nn_news_sum']:.3f})")
        print(f"  Активность: {activity_level} ({feature['nn_news_count']} новостей)")
        print(f"  Макс. влияние: {feature['nn_news_max']:.3f}")
        print()

# Использование
if 'features_preview' in result:
    interpret_features(result['features_preview'])
```

### **Поиск аномалий**
```python
def find_news_anomalies(joined_data):
    """Поиск аномальных новостных влияний"""
    
    anomalies = []
    
    for row in joined_data:
        # Высокое влияние при низкой активности
        if row['nn_news_sum'] > 0.8 and row['nn_news_count'] < 3:
            anomalies.append({
                'type': 'HIGH_IMPACT_LOW_COUNT',
                'ticker': row['ticker'],
                'date': row['date'],
                'impact': row['nn_news_sum'],
                'count': row['nn_news_count']
            })
        
        # Высокая активность при низком влиянии
        elif row['nn_news_count'] > 10 and row['nn_news_sum'] < 0.1:
            anomalies.append({
                'type': 'HIGH_COUNT_LOW_IMPACT',
                'ticker': row['ticker'],
                'date': row['date'],
                'impact': row['nn_news_sum'],
                'count': row['nn_news_count']
            })
    
    return anomalies

# Использование
if 'joined_preview' in result:
    anomalies = find_news_anomalies(result['joined_preview'])
    for anomaly in anomalies:
        print(f"Аномалия: {anomaly['type']}")
        print(f"Тикер: {anomaly['ticker']}, Дата: {anomaly['date']}")
        print(f"Влияние: {anomaly['impact']}, Количество: {anomaly['count']}")
        print()
```

## 🔧 Настройка параметров

### **Для разных стратегий**

#### **Скальпинг (внутридневная торговля)**
```python
scalping_params = {
    "p_threshold": 0.7,      # Только очень релевантные новости
    "half_life_days": 0.5,   # Быстрое затухание
    "max_days": 1           # Только сегодняшние новости
}
```

#### **Свинг-трейдинг (недельные позиции)**
```python
swing_params = {
    "p_threshold": 0.4,      # Умеренная фильтрация
    "half_life_days": 3.0,   # Среднее затухание
    "max_days": 7           # Недельный горизонт
}
```

#### **Позиционная торговля (месячные позиции)**
```python
position_params = {
    "p_threshold": 0.2,      # Низкая фильтрация
    "half_life_days": 14.0,  # Медленное затухание
    "max_days": 30          # Месячный горизонт
}
```

## 🚨 Обработка ошибок

### **Типичные ошибки и решения**

```python
def safe_api_call(data):
    """Безопасный вызов API с обработкой ошибок"""
    
    try:
        response = requests.post(
            "http://localhost:8000/infer",
            json=data,
            timeout=60
        )
        
        if response.status_code == 200:
            return response.json()
        elif response.status_code == 422:
            print("Ошибка валидации данных")
            print(response.json())
            return None
        elif response.status_code == 500:
            print("Внутренняя ошибка сервера")
            return None
        else:
            print(f"Неизвестная ошибка: {response.status_code}")
            return None
            
    except requests.exceptions.ConnectionError:
        print("Ошибка соединения - проверьте, что API запущен")
        return None
    except requests.exceptions.Timeout:
        print("Таймаут запроса - попробуйте уменьшить объем данных")
        return None
    except Exception as e:
        print(f"Неожиданная ошибка: {e}")
        return None
```

## 📈 Мониторинг производительности

### **Измерение времени выполнения**
```python
import time

def benchmark_api_call(data):
    """Бенчмарк API вызова"""
    
    start_time = time.time()
    
    response = requests.post(
        "http://localhost:8000/infer",
        json=data,
        timeout=60
    )
    
    end_time = time.time()
    execution_time = end_time - start_time
    
    if response.status_code == 200:
        result = response.json()
        print(f"Время выполнения: {execution_time:.2f} сек")
        print(f"Обработано новостей: {len(data['news'])}")
        print(f"Обработано свечей: {len(data['candles'])}")
        print(f"Получено фич: {result['rows_features']}")
        
        # Производительность
        news_per_sec = len(data['news']) / execution_time
        candles_per_sec = len(data['candles']) / execution_time
        
        print(f"Производительность: {news_per_sec:.0f} новостей/сек, {candles_per_sec:.0f} свечей/сек")
    
    return response
```
