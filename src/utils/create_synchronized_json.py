#!/usr/bin/env python3
"""
Создание JSON с синхронизированными по времени новостями и свечами
"""
import pandas as pd
import json
from datetime import datetime, timedelta

def create_synchronized_json():
    """Создаем JSON где новости и свечи из одного временного периода"""
    
    print("Читаем данные...")
    df_news = pd.read_csv('test_news.csv')
    df_candles = pd.read_csv('public_test_candles.csv')
    
    print(f"Исходные данные:")
    print(f"  Новостей: {len(df_news)}")
    print(f"  Свечей: {len(df_candles)}")
    
    # Конвертируем даты
    df_news['publish_date'] = pd.to_datetime(df_news['publish_date'])
    df_candles['begin'] = pd.to_datetime(df_candles['begin'])
    
    print(f"\nВременные диапазоны:")
    print(f"  Новости: {df_news['publish_date'].min()} - {df_news['publish_date'].max()}")
    print(f"  Свечи: {df_candles['begin'].min()} - {df_candles['begin'].max()}")
    
    # Вариант 1: Используем свечи из 2020 года (если есть)
    print(f"\n=== ВАРИАНТ 1: Ищем свечи из 2020 года ===")
    candles_2020 = df_candles[df_candles['begin'].dt.year == 2020]
    print(f"Свечей в 2020 году: {len(candles_2020)}")
    
    if len(candles_2020) > 0:
        print("✅ Найдены свечи из 2020 года!")
        selected_candles = candles_2020.head(50)  # Берем первые 50
        selected_news = df_news.head(100)  # Берем первые 100 новостей
    else:
        print("❌ Свечей из 2020 года нет")
        
        # Вариант 2: Сдвигаем новости к датам свечей
        print(f"\n=== ВАРИАНТ 2: Сдвигаем новости к датам свечей ===")
        
        # Берем первые 50 свечей
        selected_candles = df_candles.head(50).copy()
        
        # Сдвигаем новости к периоду свечей
        news_start = selected_candles['begin'].min()
        news_end = selected_candles['begin'].max()
        
        # Берем первые 100 новостей и сдвигаем их даты
        selected_news = df_news.head(100).copy()
        
        # Создаем равномерное распределение новостей по периоду свечей
        news_dates = pd.date_range(start=news_start, end=news_end, periods=len(selected_news))
        selected_news['publish_date'] = news_dates
    
    print(f"\nВыбранные данные:")
    print(f"  Новостей: {len(selected_news)}")
    print(f"  Свечей: {len(selected_candles)}")
    print(f"  Новости: {selected_news['publish_date'].min()} - {selected_news['publish_date'].max()}")
    print(f"  Свечи: {selected_candles['begin'].min()} - {selected_candles['begin'].max()}")
    
    # Конвертируем в JSON формат
    news_list = []
    for _, row in selected_news.iterrows():
        news_item = {
            "publish_date": row['publish_date'].strftime('%Y-%m-%d %H:%M:%S'),
            "title": str(row['title']),
            "publication": str(row['publication'])
        }
        news_list.append(news_item)
    
    candles_list = []
    for _, row in selected_candles.iterrows():
        candle_item = {
            "begin": row['begin'].strftime('%Y-%m-%d'),
            "ticker": str(row['ticker']),
            "open": float(row['open']),
            "high": float(row['high']),
            "low": float(row['low']),
            "close": float(row['close']),
            "volume": int(row['volume'])
        }
        candles_list.append(candle_item)
    
    # Создаем JSON с увеличенными параметрами
    synchronized_data = {
        "news": news_list,
        "candles": candles_list,
        "artifacts_dir": "artifacts",
        "p_threshold": 0.3,  # Снижаем порог
        "half_life_days": 2.0,  # Увеличиваем период полураспада
        "max_days": 30  # Увеличиваем окно поиска
    }
    
    # Сохраняем файл
    with open('synchronized_swagger_data.json', 'w', encoding='utf-8') as f:
        json.dump(synchronized_data, f, ensure_ascii=False, indent=2)
    
    print(f"\n✅ Создан файл synchronized_swagger_data.json")
    print(f"📊 Статистика:")
    print(f"  Новостей: {len(news_list)}")
    print(f"  Свечей: {len(candles_list)}")
    print(f"  Тикеров: {len(set(c['ticker'] for c in candles_list))}")
    print(f"  Параметры:")
    print(f"    - p_threshold: {synchronized_data['p_threshold']}")
    print(f"    - half_life_days: {synchronized_data['half_life_days']}")
    print(f"    - max_days: {synchronized_data['max_days']}")
    
    return synchronized_data

if __name__ == "__main__":
    create_synchronized_json()
