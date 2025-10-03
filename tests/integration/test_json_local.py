#!/usr/bin/env python3
"""
Простой тест JSON API без Docker
"""
import json
import requests
import pandas as pd
from datetime import datetime

def test_json_api_local():
    """Тестируем JSON API локально"""
    
    # Читаем тестовые данные
    with open('swagger_test_data.json', 'r', encoding='utf-8') as f:
        test_data = json.load(f)
    
    print("=== ТЕСТ JSON API ===")
    print(f"Новостей: {len(test_data['news'])}")
    print(f"Свечей: {len(test_data['candles'])}")
    print(f"Тикеры: {set(c['ticker'] for c in test_data['candles'])}")
    
    # Тестируем разные URL
    urls_to_test = [
        "http://localhost:8000/infer",
        "http://127.0.0.1:8000/infer", 
        "http://176.57.217.27:8000/infer"
    ]
    
    for url in urls_to_test:
        print(f"\n--- Тестируем {url} ---")
        try:
            response = requests.post(url, json=test_data, timeout=30)
            print(f"Статус: {response.status_code}")
            
            if response.status_code == 200:
                result = response.json()
                print(f"✅ Успех!")
                print(f"Статус ответа: {result.get('status', 'unknown')}")
                print(f"Строк с фичами: {result.get('rows_features', 0)}")
                print(f"Строк в объединенном: {result.get('rows_joined', 0)}")
                
                if result.get('features_preview'):
                    print(f"\nПример фич (первые 3):")
                    for i, feat in enumerate(result['features_preview'][:3]):
                        print(f"  {i+1}. {feat}")
                
                if result.get('joined_preview'):
                    print(f"\nПример объединенных данных (первые 3):")
                    for i, row in enumerate(result['joined_preview'][:3]):
                        print(f"  {i+1}. {row}")
                        
                return True  # Успешный тест
                
            else:
                print(f"❌ Ошибка: {response.status_code}")
                print(f"Ответ: {response.text[:200]}...")
                
        except requests.exceptions.ConnectionError:
            print("❌ Соединение отклонено - сервер не запущен")
        except requests.exceptions.Timeout:
            print("❌ Таймаут - сервер не отвечает")
        except Exception as e:
            print(f"❌ Исключение: {e}")
    
    return False

def test_health_endpoints():
    """Тестируем health endpoints"""
    urls_to_test = [
        "http://localhost:8000/health",
        "http://127.0.0.1:8000/health",
        "http://176.57.217.27:8000/health"
    ]
    
    print("\n=== ТЕСТ HEALTH ENDPOINTS ===")
    for url in urls_to_test:
        print(f"\n--- Тестируем {url} ---")
        try:
            response = requests.post(url, timeout=10)
            print(f"Статус: {response.status_code}")
            if response.status_code == 200:
                result = response.json()
                print(f"✅ Health check: {result}")
            else:
                print(f"❌ Ошибка: {response.text}")
        except Exception as e:
            print(f"❌ Исключение: {e}")

if __name__ == "__main__":
    test_health_endpoints()
    success = test_json_api_local()
    
    if success:
        print("\n🎉 JSON API работает корректно!")
    else:
        print("\n⚠️ JSON API недоступен. Проверьте:")
        print("1. Запущен ли сервер (uvicorn app:app --host 0.0.0.0 --port 8000)")
        print("2. Доступны ли артефакты в папке artifacts/")
        print("3. Установлены ли все зависимости")
