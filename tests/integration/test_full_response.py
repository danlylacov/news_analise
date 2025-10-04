#!/usr/bin/env python3
"""
Тестирование того, что возвращает full_swagger_data.json
"""
import json
import requests
import pandas as pd
from datetime import datetime

def test_full_json_response():
    """Тестируем полный JSON файл и показываем что возвращает API"""
    
    print("=== ТЕСТ ПОЛНОГО JSON ФАЙЛА ===")
    
    # Читаем полный JSON файл
    with open('examples/json_samples/full_swagger_data.json', 'r', encoding='utf-8') as f:
        full_data = json.load(f)
    
    print(f"📊 Входные данные:")
    print(f"  Новостей: {len(full_data['news'])}")
    print(f"  Свечей: {len(full_data['candles'])}")
    print(f"  Тикеров: {len(set(c['ticker'] for c in full_data['candles']))}")
    print(f"  Параметры: p_threshold={full_data['p_threshold']}, half_life_days={full_data['half_life_days']}, max_days={full_data['max_days']}")
    
    # Показываем примеры входных данных
    print(f"\n📰 Пример новости:")
    news_example = full_data['news'][0]
    print(f"  Дата: {news_example['publish_date']}")
    print(f"  Заголовок: {news_example['title'][:100]}...")
    print(f"  Издание: {news_example['publication'][:50]}...")
    
    print(f"\n📈 Пример свечи:")
    candle_example = full_data['candles'][0]
    print(f"  Дата: {candle_example['begin']}")
    print(f"  Тикер: {candle_example['ticker']}")
    print(f"  OHLCV: {candle_example['open']}/{candle_example['high']}/{candle_example['low']}/{candle_example['close']}/{candle_example['volume']}")
    
    # Тестируем API
    print(f"\n🚀 Отправляем запрос в API...")
    
    # Попробуем разные URL
    urls_to_test = [
        "http://188.225.74.138:8000/infer"
    ]
    
    for url in urls_to_test:
        print(f"\n--- Тестируем {url} ---")
        try:
            print("Отправляем запрос...")
            response = requests.post(url, json=full_data)
            print(f"Статус ответа: {response.status_code}")
            
            if response.status_code == 200:
                result = response.json()
                print(f"✅ УСПЕХ!")
                print(f"\n📋 Что возвращает API:")
                print(f"  Статус: {result.get('status', 'unknown')}")
                print(f"  Строк с фичами: {result.get('rows_features', 0)}")
                print(f"  Строк в объединенном: {result.get('rows_joined', 0)}")
                
                # Сохраняем результат в файл
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                output_filename = f"api_response_{timestamp}.json"
                
                with open(output_filename, 'w', encoding='utf-8') as f:
                    json.dump(result, f, ensure_ascii=False, indent=2)
                
                print(f"💾 Результат сохранен в файл: {output_filename}")
                
                if result.get('features_preview'):
                    print(f"\n🔍 Пример фич (первые 5):")
                    for i, feat in enumerate(result['features_preview'][:5]):
                        print(f"  {i+1}. Тикер: {feat.get('ticker', 'N/A')}, Дата: {feat.get('date', 'N/A')}")
                        print(f"     nn_news_sum: {feat.get('nn_news_sum', 0):.4f}")
                        print(f"     nn_news_mean: {feat.get('nn_news_mean', 0):.4f}")
                        print(f"     nn_news_max: {feat.get('nn_news_max', 0):.4f}")
                        print(f"     nn_news_count: {feat.get('nn_news_count', 0)}")
                        print()
                
                if result.get('joined_preview'):
                    print(f"\n🔗 Пример объединенных данных (первые 3):")
                    for i, row in enumerate(result['joined_preview'][:3]):
                        print(f"  {i+1}. Тикер: {row.get('ticker', 'N/A')}, Дата: {row.get('begin', 'N/A')}")
                        print(f"     Цена закрытия: {row.get('close', 'N/A')}")
                        print(f"     Объем: {row.get('volume', 'N/A')}")
                        print(f"     nn_news_sum: {row.get('nn_news_sum', 0):.4f}")
                        print(f"     nn_news_count: {row.get('nn_news_count', 0)}")
                        print()
                
                # Анализируем результаты
                print(f"\n📊 Анализ результатов:")
                if result.get('features_preview'):
                    features = result['features_preview']
                    non_zero_features = [f for f in features if f.get('nn_news_sum', 0) > 0]
                    print(f"  Фич с ненулевыми значениями: {len(non_zero_features)} из {len(features)}")
                    
                    if non_zero_features:
                        max_sum = max(f.get('nn_news_sum', 0) for f in non_zero_features)
                        max_count = max(f.get('nn_news_count', 0) for f in non_zero_features)
                        print(f"  Максимальная сумма влияния: {max_sum:.4f}")
                        print(f"  Максимальное количество новостей: {max_count}")
                
                return True
                
            else:
                print(f"❌ Ошибка: {response.status_code}")
                print(f"Ответ: {response.text[:500]}...")
                
                # Сохраняем ошибку в файл
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                error_filename = f"api_error_{timestamp}.json"
                
                error_data = {
                    "timestamp": timestamp,
                    "url": url,
                    "status_code": response.status_code,
                    "error_text": response.text,
                    "request_data_summary": {
                        "news_count": len(full_data['news']),
                        "candles_count": len(full_data['candles']),
                        "parameters": {
                            "p_threshold": full_data['p_threshold'],
                            "half_life_days": full_data['half_life_days'],
                            "max_days": full_data['max_days']
                        }
                    }
                }
                
                with open(error_filename, 'w', encoding='utf-8') as f:
                    json.dump(error_data, f, ensure_ascii=False, indent=2)
                
                print(f"💾 Ошибка сохранена в файл: {error_filename}")
                
        except requests.exceptions.ConnectionError:
            print("❌ Соединение отклонено - сервер не запущен")
            
            # Сохраняем ошибку соединения
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            error_filename = f"api_connection_error_{timestamp}.json"
            
            error_data = {
                "timestamp": timestamp,
                "url": url,
                "error_type": "ConnectionError",
                "error_message": "Соединение отклонено - сервер не запущен",
                "request_data_summary": {
                    "news_count": len(full_data['news']),
                    "candles_count": len(full_data['candles']),
                    "parameters": {
                        "p_threshold": full_data['p_threshold'],
                        "half_life_days": full_data['half_life_days'],
                        "max_days": full_data['max_days']
                    }
                }
            }
            
            with open(error_filename, 'w', encoding='utf-8') as f:
                json.dump(error_data, f, ensure_ascii=False, indent=2)
            
            print(f"💾 Ошибка соединения сохранена в файл: {error_filename}")
            
        except requests.exceptions.Timeout:
            print("❌ Таймаут - сервер не отвечает")
            
            # Сохраняем ошибку таймаута
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            error_filename = f"api_timeout_error_{timestamp}.json"
            
            error_data = {
                "timestamp": timestamp,
                "url": url,
                "error_type": "Timeout",
                "error_message": "Таймаут - сервер не отвечает",
                "request_data_summary": {
                    "news_count": len(full_data['news']),
                    "candles_count": len(full_data['candles']),
                    "parameters": {
                        "p_threshold": full_data['p_threshold'],
                        "half_life_days": full_data['half_life_days'],
                        "max_days": full_data['max_days']
                    }
                }
            }
            
            with open(error_filename, 'w', encoding='utf-8') as f:
                json.dump(error_data, f, ensure_ascii=False, indent=2)
            
            print(f"💾 Ошибка таймаута сохранена в файл: {error_filename}")
            
        except Exception as e:
            print(f"❌ Исключение: {e}")
            
            # Сохраняем общее исключение
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            error_filename = f"api_exception_{timestamp}.json"
            
            error_data = {
                "timestamp": timestamp,
                "url": url,
                "error_type": "Exception",
                "error_message": str(e),
                "request_data_summary": {
                    "news_count": len(full_data['news']),
                    "candles_count": len(full_data['candles']),
                    "parameters": {
                        "p_threshold": full_data['p_threshold'],
                        "half_life_days": full_data['half_life_days'],
                        "max_days": full_data['max_days']
                    }
                }
            }
            
            with open(error_filename, 'w', encoding='utf-8') as f:
                json.dump(error_data, f, ensure_ascii=False, indent=2)
            
            print(f"💾 Исключение сохранено в файл: {error_filename}")
    
    return False

def show_json_structure():
    """Показываем структуру JSON файла"""
    print("=== СТРУКТУРА JSON ФАЙЛА ===")
    
    with open('examples/json_samples/full_swagger_data.json', 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    print("📋 Структура:")
    print("  {")
    print("    \"news\": [")
    print("      {")
    print("        \"publish_date\": \"2020-01-01 14:00:00\",")
    print("        \"title\": \"Заголовок новости\",")
    print("        \"publication\": \"Название издания\"")
    print("      },")
    print("      ...")
    print("    ],")
    print("    \"candles\": [")
    print("      {")
    print("        \"begin\": \"2025-04-16\",")
    print("        \"ticker\": \"SBER\",")
    print("        \"open\": 250.5,")
    print("        \"high\": 255.8,")
    print("        \"low\": 248.2,")
    print("        \"close\": 253.1,")
    print("        \"volume\": 15000000")
    print("      },")
    print("      ...")
    print("    ],")
    print("    \"artifacts_dir\": \"artifacts\",")
    print("    \"p_threshold\": 0.5,")
    print("    \"half_life_days\": 0.5,")
    print("    \"max_days\": 5")
    print("  }")

if __name__ == "__main__":
    show_json_structure()
    print("\n" + "="*50 + "\n")
    success = test_full_json_response()
    
    if success:
        print("\n🎉 JSON файл успешно обработан API!")
        print("\n💡 Что происходит:")
        print("1. API принимает новости и свечи в JSON формате")
        print("2. Автоматически размечает новости по тикерам")
        print("3. Вычисляет релевантность новостей для каждого тикера")
        print("4. Агрегирует новостные фичи по датам и тикерам")
        print("5. Объединяет свечи с новостными фичами")
        print("6. Возвращает результат в JSON формате")
    else:
        print("\n⚠️ API недоступен. Запустите сервер:")
        print("uvicorn app:app --host 0.0.0.0 --port 8000")
