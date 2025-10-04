#!/usr/bin/env python3
"""
Итоговый тест сентимент-анализа на всех данных
"""

import requests
import pandas as pd
import json
import time
from datetime import datetime

def test_all_data_with_sentiment():
    """Тестирование сентимент-анализа на всех данных"""
    
    url = "http://localhost:8000"
    
    # Проверяем доступность API
    try:
        response = requests.post(f"{url}/health", json={})
        if response.status_code == 200:
            print("✅ API доступен")
        else:
            print("❌ API недоступен")
            return
    except Exception as e:
        print(f"❌ Ошибка подключения к API: {e}")
        return
    
    # Загружаем все данные
    print("📂 Загружаем все данные...")
    try:
        # Загружаем новости
        train_news = pd.read_csv('datasets/raw/train_news.csv', nrows=1000)  # Ограничиваем для теста
        train_candles = pd.read_csv('datasets/raw/train_candles.csv', nrows=2000)  # Ограничиваем для теста
        
        print(f"✅ Загружено {len(train_news)} новостей и {len(train_candles)} свечей")
        
    except Exception as e:
        print(f"❌ Ошибка загрузки данных: {e}")
        return
    
    # Подготавливаем данные для API
    print("🔄 Подготавливаем данные для API...")
    
    news_data = [
        {
            "title": str(row['title']),
            "publication": str(row['publication']),
            "publish_date": str(row['publish_date'])
        }
        for _, row in train_news.iterrows()
    ]
    
    candles_data = [
        {
            "ticker": str(row['ticker']),
            "begin": str(row['begin']),
            "open": float(row['open']),
            "high": float(row['high']),
            "low": float(row['low']),
            "close": float(row['close']),
            "volume": int(row['volume'])
        }
        for _, row in train_candles.iterrows()
    ]
    
    # Тестируем с включенным сентимент-анализом
    print("\n🔍 Тестируем с включенным сентимент-анализом...")
    
    request_data = {
        "news": news_data,
        "candles": candles_data,
        "artifacts_dir": "artifacts",
        "p_threshold": 0.3,
        "half_life_days": 0.5,
        "max_days": 2.0,
        "add_sentiment": True
    }
    
    start_time = time.time()
    
    try:
        response = requests.post(f"{url}/infer", json=request_data, timeout=300)
        
        end_time = time.time()
        processing_time = end_time - start_time
        
        if response.status_code == 200:
            result = response.json()
            
            print(f"✅ Анализ завершен за {processing_time:.2f} сек")
            print(f"📊 Статус: {result.get('status', 'unknown')}")
            print(f"📊 Строк в фичах: {result.get('rows_features', 0)}")
            print(f"📊 Строк в объединенных данных: {result.get('rows_joined', 0)}")
            
            # Анализируем результат
            if result.get('features_preview'):
                features_df = result['features_preview']
                
                # Проверяем наличие сентимент-колонок
                sentiment_cols = [col for col in features_df[0].keys() if 'sentiment' in col]
                print(f"\n📈 Сентимент-колонки: {sentiment_cols}")
                
                # Подсчитываем статистику
                total_news = sum(f.get('nn_news_count', 0) for f in features_df)
                records_with_sentiment = sum(1 for f in features_df if f.get('sentiment_count', 0) > 0)
                
                print(f"   Всего новостей обработано: {total_news}")
                print(f"   Записей с сентиментом: {records_with_sentiment}")
                
                if records_with_sentiment > 0:
                    # Показываем топ-5 записей с наибольшим количеством новостей
                    print(f"\n🏆 Топ-5 записей с новостями:")
                    sorted_features = sorted(features_df, key=lambda x: x.get('nn_news_count', 0), reverse=True)
                    
                    for i, feature in enumerate(sorted_features[:5], 1):
                        print(f"   {i}. Тикер: {feature.get('ticker', 'N/A')}, Дата: {feature.get('date', 'N/A')}")
                        print(f"      Новостей: {feature.get('nn_news_count', 0)}")
                        print(f"      Сентимент (средний): {feature.get('sentiment_mean', 0):.3f}")
                        print(f"      Позитивных: {feature.get('sentiment_positive_count', 0)}")
                        print(f"      Негативных: {feature.get('sentiment_negative_count', 0)}")
                        print(f"      Нейтральных: {feature.get('sentiment_neutral_count', 0)}")
                
                # Статистика по сентименту
                sentiment_means = [f.get('sentiment_mean', 1.0) for f in features_df if f.get('sentiment_count', 0) > 0]
                if sentiment_means:
                    avg_sentiment = sum(sentiment_means) / len(sentiment_means)
                    print(f"\n📊 Статистика сентимента:")
                    print(f"   Средний сентимент: {avg_sentiment:.3f}")
                    print(f"   Минимальный: {min(sentiment_means):.3f}")
                    print(f"   Максимальный: {max(sentiment_means):.3f}")
            
            # Сохраняем результат
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            output_file = f"api_response_all_data_sentiment_{timestamp}.json"
            
            with open(output_file, 'w', encoding='utf-8') as f:
                json.dump(result, f, ensure_ascii=False, indent=2)
            
            print(f"\n💾 Результат сохранен в файл: {output_file}")
            print(f"📈 Время обработки: {processing_time:.2f} сек")
            print(f"📊 Скорость: {len(news_data)/processing_time:.1f} новостей/сек")
            
        else:
            print(f"❌ Ошибка API: {response.status_code}")
            print(f"Ответ: {response.text}")
            
    except Exception as e:
        print(f"❌ Ошибка: {e}")

if __name__ == "__main__":
    print("🚀 Итоговый тест сентимент-анализа на всех данных")
    print("=" * 60)
    test_all_data_with_sentiment()
