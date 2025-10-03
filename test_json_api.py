#!/usr/bin/env python3
"""
Тест нового JSON API для инференса новостей
"""
import requests
import json
from datetime import datetime, timedelta

def test_inference_api():
    url = "http://176.57.217.27:8000/infer"
    
    # Подготавливаем тестовые данные
    news_data = [
        {
            "publish_date": "2023-01-15T08:00:00",
            "title": "Сбербанк повышает ставки по депозитам",
            "publication": "РБК"
        },
        {
            "publish_date": "2023-01-15T10:30:00", 
            "title": "Газпром увеличивает экспорт газа в Европу",
            "publication": "Коммерсант"
        },
        {
            "publish_date": "2023-01-16T09:00:00",
            "title": "Лукойл планирует новые инвестиции в нефтепереработку",
            "publication": "Ведомости"
        }
    ]
    
    candles_data = [
        {
            "begin": "2023-01-16T09:00:00",
            "ticker": "SBER",
            "open": 200.5,
            "high": 205.8,
            "low": 198.2,
            "close": 203.1,
            "volume": 1500000
        },
        {
            "begin": "2023-01-16T09:00:00",
            "ticker": "GAZP", 
            "open": 180.3,
            "high": 185.5,
            "low": 178.8,
            "close": 183.2,
            "volume": 2300000
        },
        {
            "begin": "2023-01-17T09:00:00",
            "ticker": "LKOH",
            "open": 6500.0,
            "high": 6650.0,
            "low": 6450.0,
            "close": 6620.0,
            "volume": 980000
        }
    ]
    
    payload = {
        "news": news_data,
        "candles": candles_data,
        "artifacts_dir": "artifacts",
        "p_threshold": 0.5,
        "half_life_days": 0.5,
        "max_days": 5.0
    }
    
    print("Отправляем запрос к API...")
    print(f"URL: {url}")
    print(f"Новостей: {len(news_data)}")
    print(f"Свечей: {len(candles_data)}")
    
    try:
        response = requests.post(url, json=payload)
        print(f"Статус ответа: {response.status_code}")
        
        if response.status_code == 200:
            result = response.json()
            print(f"Ответ получен:")
            print(f"- Статус: {result['status']}")
            print(f"- Строк с фичами: {result['rows_features']}")
            print(f"- Строк в объединенном: {result['rows_joined']}")
            
            if result.get('features_preview'):
                print("\nПример фич:")
                for i, feat in enumerate(result['features_preview'][:3]):
                    print(f"  {i+1}. {feat}")
            
            if result.get('joined_preview'):
                print("\nПример объединенных данных:")
                for i, row in enumerate(result['joined_preview'][:3]):
                    print(f"  {i+1}. {row}")
                    
        else:
            print(f"Ошибка: {response.status_code}")
            print(f"Ответ: {response.text}")
            
    except Exception as e:
        print(f"Исключение: {e}")

def test_health_check():
    url = "http://176.57.217.27:8000/health"
    
    print("\nПроверка состояния API...")
    try:
        response = requests.post(url)
        print(f"Статус: {response.status_code}")
        print(f"Ответ: {response.json()}")
    except Exception as e:
        print(f"Ошибка: {e}")

if __name__ == "__main__":
    test_health_check()
    test_inference_api()
