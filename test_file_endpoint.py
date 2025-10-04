#!/usr/bin/env python3
"""
Тестовый скрипт для проверки нового эндпоинта /process-news-file
"""

import requests
import json
import pandas as pd
from datetime import datetime
import io

# Конфигурация
API_URL = "http://localhost:8000"
CALLBACK_URL = "http://localhost:8080/news"  # URL вашего callback эндпоинта
SESSION_ID = f"test_session_{datetime.now().strftime('%Y%m%d_%H%M%S')}"

def create_test_news_file():
    """Создает тестовый CSV файл с новостями"""
    test_data = [
        {
            "publish_date": "2025-01-01 10:00:00",
            "title": "Сбербанк объявил о росте прибыли",
            "publication": "РБК"
        },
        {
            "publish_date": "2025-01-01 11:00:00", 
            "title": "Газпром увеличил добычу газа",
            "publication": "Коммерсант"
        },
        {
            "publish_date": "2025-01-01 12:00:00",
            "title": "Лукойл планирует новые проекты",
            "publication": "Ведомости"
        },
        {
            "publish_date": "2025-01-01 13:00:00",
            "title": "Российские акции выросли на фоне новостей",
            "publication": "Интерфакс"
        },
        {
            "publish_date": "2025-01-01 14:00:00",
            "title": "ЦБ РФ сохранил ключевую ставку",
            "publication": "ТАСС"
        }
    ]
    
    df = pd.DataFrame(test_data)
    
    # Создаем CSV в памяти
    csv_buffer = io.StringIO()
    df.to_csv(csv_buffer, index=False, encoding='utf-8')
    csv_content = csv_buffer.getvalue()
    
    return csv_content.encode('utf-8')

def test_file_endpoint():
    """Тестирует эндпоинт /process-news-file"""
    
    print(f"Тестирование эндпоинта: {API_URL}/process-news-file")
    print(f"Session ID: {SESSION_ID}")
    print(f"Callback URL: {CALLBACK_URL}")
    print("-" * 50)
    
    # Создаем тестовый файл
    file_content = create_test_news_file()
    
    # Подготавливаем данные для multipart/form-data
    files = {
        'file': ('test_news.csv', file_content, 'text/csv')
    }
    
    data = {
        'callbackUrl': CALLBACK_URL,
        'sessionId': SESSION_ID,
        'artifacts_dir': 'artifacts',
        'p_threshold': '0.5',
        'half_life_days': '0.5',
        'max_days': '5.0',
        'add_sentiment': 'true'
    }
    
    try:
        # Отправляем запрос
        response = requests.post(
            f"{API_URL}/process-news-file",
            files=files,
            data=data,
            timeout=30
        )
        
        print(f"Статус ответа: {response.status_code}")
        print(f"Заголовки ответа: {dict(response.headers)}")
        
        if response.status_code == 200:
            result = response.json()
            print(f"Результат: {json.dumps(result, indent=2, ensure_ascii=False)}")
            print("\n✅ Файл успешно принят к обработке!")
            print(f"Session ID для отслеживания: {result['sessionId']}")
        else:
            print(f"❌ Ошибка: {response.status_code}")
            print(f"Текст ответа: {response.text}")
            
    except requests.exceptions.RequestException as e:
        print(f"❌ Ошибка при отправке запроса: {e}")
    except Exception as e:
        print(f"❌ Неожиданная ошибка: {e}")

def test_health_endpoint():
    """Тестирует эндпоинт /health"""
    
    print(f"\nТестирование эндпоинта: {API_URL}/health")
    print("-" * 50)
    
    try:
        response = requests.post(f"{API_URL}/health", timeout=10)
        
        print(f"Статус ответа: {response.status_code}")
        
        if response.status_code == 200:
            result = response.json()
            print(f"Результат: {json.dumps(result, indent=2, ensure_ascii=False)}")
            
            if result.get('status') == 'healthy':
                print("✅ API готов к работе!")
            else:
                print(f"⚠️ API не готов: {result.get('message', 'Неизвестная ошибка')}")
        else:
            print(f"❌ Ошибка: {response.status_code}")
            print(f"Текст ответа: {response.text}")
            
    except requests.exceptions.RequestException as e:
        print(f"❌ Ошибка при отправке запроса: {e}")
    except Exception as e:
        print(f"❌ Неожиданная ошибка: {e}")

def test_root_endpoint():
    """Тестирует корневой эндпоинт"""
    
    print(f"\nТестирование корневого эндпоинта: {API_URL}/")
    print("-" * 50)
    
    try:
        response = requests.get(f"{API_URL}/", timeout=10)
        
        print(f"Статус ответа: {response.status_code}")
        
        if response.status_code == 200:
            result = response.json()
            print(f"Результат: {json.dumps(result, indent=2, ensure_ascii=False)}")
            print("✅ API доступен!")
        else:
            print(f"❌ Ошибка: {response.status_code}")
            print(f"Текст ответа: {response.text}")
            
    except requests.exceptions.RequestException as e:
        print(f"❌ Ошибка при отправке запроса: {e}")
    except Exception as e:
        print(f"❌ Неожиданная ошибка: {e}")

if __name__ == "__main__":
    print("🚀 Запуск тестов нового эндпоинта /process-news-file")
    print("=" * 60)
    
    # Тестируем корневой эндпоинт
    test_root_endpoint()
    
    # Тестируем health эндпоинт
    test_health_endpoint()
    
    # Тестируем новый эндпоинт
    test_file_endpoint()
    
    print("\n" + "=" * 60)
    print("🏁 Тестирование завершено!")
    print("\nПримечания:")
    print("- Убедитесь, что API сервер запущен на localhost:8000")
    print("- Убедитесь, что callback сервер доступен на localhost:8080")
    print("- Проверьте наличие файлов артефактов в папке 'artifacts'")
    print("- Результат обработки будет отправлен на callback URL")
