#!/usr/bin/env python3
"""
Создание полного JSON файла со всеми данными из CSV
"""
import pandas as pd
import json
from datetime import datetime

def create_full_json_data():
    """Создаем JSON со всеми данными из CSV файлов"""
    
    print("Читаем новости...")
    # Читаем все новости
    df_news = pd.read_csv('test_news.csv')
    print(f"Загружено новостей: {len(df_news)}")
    
    print("Читаем свечи...")
    # Читаем все свечи
    df_candles = pd.read_csv('public_test_candles.csv')
    print(f"Загружено свечей: {len(df_candles)}")
    
    # Конвертируем новости в JSON формат
    news_list = []
    for _, row in df_news.iterrows():
        news_item = {
            "publish_date": str(row['publish_date']),
            "title": str(row['title']),
            "publication": str(row['publication'])
        }
        news_list.append(news_item)
    
    # Конвертируем свечи в JSON формат
    candles_list = []
    for _, row in df_candles.iterrows():
        candle_item = {
            "begin": str(row['begin']),
            "ticker": str(row['ticker']),
            "open": float(row['open']),
            "high": float(row['high']),
            "low": float(row['low']),
            "close": float(row['close']),
            "volume": int(row['volume'])
        }
        candles_list.append(candle_item)
    
    # Создаем полный JSON объект
    full_data = {
        "news": news_list,
        "candles": candles_list,
        "artifacts_dir": "artifacts",
        "p_threshold": 0.5,
        "half_life_days": 0.5,
        "max_days": 5
    }
    
    print(f"Создан JSON с {len(news_list)} новостями и {len(candles_list)} свечами")
    
    # Сохраняем в файл
    with open('full_swagger_data.json', 'w', encoding='utf-8') as f:
        json.dump(full_data, f, ensure_ascii=False, indent=2)
    
    print("✅ Файл full_swagger_data.json создан!")
    
    # Показываем статистику
    print(f"\n📊 Статистика:")
    print(f"Новостей: {len(news_list)}")
    print(f"Свечей: {len(candles_list)}")
    print(f"Уникальных тикеров: {len(set(c['ticker'] for c in candles_list))}")
    print(f"Тикеры: {sorted(set(c['ticker'] for c in candles_list))}")
    
    # Показываем примеры данных
    print(f"\n📰 Пример новости:")
    print(f"Дата: {news_list[0]['publish_date']}")
    print(f"Заголовок: {news_list[0]['title'][:100]}...")
    print(f"Издание: {news_list[0]['publication']}")
    
    print(f"\n📈 Пример свечи:")
    print(f"Дата: {candles_list[0]['begin']}")
    print(f"Тикер: {candles_list[0]['ticker']}")
    print(f"OHLCV: {candles_list[0]['open']}/{candles_list[0]['high']}/{candles_list[0]['low']}/{candles_list[0]['close']}/{candles_list[0]['volume']}")

if __name__ == "__main__":
    create_full_json_data()
