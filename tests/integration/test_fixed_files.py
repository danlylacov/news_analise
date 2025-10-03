#!/usr/bin/env python3
"""
Тестирование исправленных JSON файлов
"""
import json
import requests

def test_fixed_json_files():
    """Тестируем исправленные JSON файлы"""
    
    files_to_test = [
        ("examples/json_samples/synchronized_swagger_data.json", "Синхронизированные данные"),
        ("examples/json_samples/full_swagger_data_fixed.json", "Исправленные параметры")
    ]
    
    url = "http://176.57.217.27:8000/infer"
    
    for filename, description in files_to_test:
        print(f"\n{'='*60}")
        print(f"ТЕСТ: {description}")
        print(f"Файл: {filename}")
        print(f"{'='*60}")
        
        try:
            # Читаем JSON файл
            with open(filename, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            print(f"📊 Данные:")
            print(f"  Новостей: {len(data['news'])}")
            print(f"  Свечей: {len(data['candles'])}")
            print(f"  Параметры:")
            print(f"    - p_threshold: {data['p_threshold']}")
            print(f"    - half_life_days: {data['half_life_days']}")
            print(f"    - max_days: {data['max_days']}")
            
            # Показываем временные диапазоны
            if data['news']:
                news_dates = [item['publish_date'] for item in data['news']]
                print(f"  Новости: {min(news_dates)} - {max(news_dates)}")
            
            if data['candles']:
                candle_dates = [item['begin'] for item in data['candles']]
                print(f"  Свечи: {min(candle_dates)} - {max(candle_dates)}")
            
            print(f"\n🚀 Отправляем запрос...")
            response = requests.post(url, json=data, timeout=120)
            
            print(f"Статус ответа: {response.status_code}")
            
            if response.status_code == 200:
                result = response.json()
                print(f"✅ УСПЕХ!")
                print(f"  Статус: {result.get('status', 'unknown')}")
                print(f"  Строк с фичами: {result.get('rows_features', 0)}")
                print(f"  Строк в объединенном: {result.get('rows_joined', 0)}")
                
                # Анализируем результаты
                if result.get('features_preview'):
                    features = result['features_preview']
                    non_zero_features = [f for f in features if f.get('nn_news_sum', 0) > 0]
                    print(f"  Фич с ненулевыми значениями: {len(non_zero_features)} из {len(features)}")
                    
                    if non_zero_features:
                        max_sum = max(f.get('nn_news_sum', 0) for f in non_zero_features)
                        max_count = max(f.get('nn_news_count', 0) for f in non_zero_features)
                        print(f"  Максимальная сумма влияния: {max_sum:.4f}")
                        print(f"  Максимальное количество новостей: {max_count}")
                        
                        # Показываем примеры ненулевых фич
                        print(f"\n🔍 Примеры ненулевых фич:")
                        for i, feat in enumerate(non_zero_features[:3]):
                            print(f"  {i+1}. {feat['ticker']} ({feat['date']}):")
                            print(f"     nn_news_sum: {feat['nn_news_sum']:.4f}")
                            print(f"     nn_news_count: {feat['nn_news_count']}")
                    else:
                        print(f"  ❌ Все фичи равны нулю!")
                
            else:
                print(f"❌ Ошибка: {response.status_code}")
                print(f"Ответ: {response.text[:500]}...")
                
        except FileNotFoundError:
            print(f"❌ Файл {filename} не найден")
        except requests.exceptions.ConnectionError:
            print(f"❌ Соединение отклонено - сервер не запущен")
        except requests.exceptions.Timeout:
            print(f"❌ Таймаут - сервер не отвечает")
        except Exception as e:
            print(f"❌ Исключение: {e}")

if __name__ == "__main__":
    test_fixed_json_files()
