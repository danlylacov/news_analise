#!/usr/bin/env python3
"""
Скрипт для тестирования производительности оптимизированной модели
"""
import time
import pandas as pd
import numpy as np
import sys
from pathlib import Path

# Добавляем src в путь
sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))

from src.core.infer_news_to_candles import infer_news_to_candles_df
from src.core.infer_news_to_candles_optimized import infer_news_to_candles_df_optimized


def create_test_data(num_news: int = 1000, num_candles: int = 500):
    """Создает тестовые данные для бенчмарка"""
    np.random.seed(42)
    
    # Создаем тестовые новости
    news_data = []
    publications = ['РБК', 'Коммерсант', 'Ведомости', 'Интерфакс', 'ТАСС']
    titles = [
        'Сбербанк объявил о росте прибыли',
        'Газпром увеличил добычу газа',
        'Лукойл планирует новые проекты',
        'НЛМК повысил цены на металл',
        'МТС расширяет сеть 5G'
    ]
    
    for i in range(num_news):
        news_data.append({
            'publish_date': f'2024-01-{(i % 30) + 1:02d}',
            'title': np.random.choice(titles),
            'publication': np.random.choice(publications)
        })
    
    # Создаем тестовые свечи
    candles_data = []
    tickers = ['SBER', 'GAZP', 'LKOH', 'NLMK', 'MTSS']
    
    for i in range(num_candles):
        candles_data.append({
            'begin': f'2024-01-{(i % 30) + 1:02d} 10:00:00',
            'ticker': np.random.choice(tickers),
            'open': 100 + np.random.randn() * 10,
            'high': 110 + np.random.randn() * 10,
            'low': 90 + np.random.randn() * 10,
            'close': 105 + np.random.randn() * 10,
            'volume': np.random.randint(1000, 10000)
        })
    
    return pd.DataFrame(news_data), pd.DataFrame(candles_data)


def benchmark_function(func, df_news, df_candles, artifacts_dir, **kwargs):
    """Бенчмарк функции"""
    start_time = time.time()
    try:
        result = func(df_news, df_candles, artifacts_dir, **kwargs)
        end_time = time.time()
        return end_time - start_time, result, None
    except Exception as e:
        end_time = time.time()
        return end_time - start_time, None, str(e)


def main():
    print("🚀 Тестирование производительности модели...")
    
    # Создаем тестовые данные
    print("📊 Создание тестовых данных...")
    df_news, df_candles = create_test_data(num_news=1000, num_candles=500)
    
    artifacts_dir = "artifacts"
    
    # Проверяем наличие артефактов
    import os
    if not os.path.exists(artifacts_dir):
        print(f"❌ Артефакты не найдены в {artifacts_dir}")
        return
    
    print(f"📈 Тестовые данные: {len(df_news)} новостей, {len(df_candles)} свечей")
    
    # Параметры для тестирования
    params = {
        'p_threshold': 0.5,
        'half_life_days': 2.0,
        'max_days': 10.0
    }
    
    # Тестируем оригинальную версию
    print("\n🔄 Тестирование оригинальной версии...")
    original_time, original_result, original_error = benchmark_function(
        infer_news_to_candles_df, df_news, df_candles, artifacts_dir, **params
    )
    
    if original_error:
        print(f"❌ Ошибка в оригинальной версии: {original_error}")
        return
    
    # Тестируем оптимизированную версию
    print("⚡ Тестирование оптимизированной версии...")
    optimized_time, optimized_result, optimized_error = benchmark_function(
        infer_news_to_candles_df_optimized, df_news, df_candles, artifacts_dir, **params
    )
    
    if optimized_error:
        print(f"❌ Ошибка в оптимизированной версии: {optimized_error}")
        return
    
    # Сравниваем результаты
    print("\n📊 Результаты тестирования:")
    print(f"Оригинальная версия: {original_time:.2f} секунд")
    print(f"Оптимизированная версия: {optimized_time:.2f} секунд")
    
    if original_time > 0:
        speedup = original_time / optimized_time
        print(f"⚡ Ускорение: {speedup:.2f}x")
        print(f"💾 Экономия времени: {((original_time - optimized_time) / original_time * 100):.1f}%")
    
    # Проверяем корректность результатов
    if original_result and optimized_result:
        orig_features, orig_joined = original_result
        opt_features, opt_joined = optimized_result
        
        print(f"\n🔍 Проверка корректности:")
        print(f"Оригинальные фичи: {len(orig_features)} строк")
        print(f"Оптимизированные фичи: {len(opt_features)} строк")
        print(f"Оригинальные объединенные: {len(orig_joined)} строк")
        print(f"Оптимизированные объединенные: {len(opt_joined)} строк")
        
        # Проверяем числовую точность
        if len(orig_features) == len(opt_features) and len(orig_features) > 0:
            feature_cols = ['nn_news_sum', 'nn_news_mean', 'nn_news_max', 'nn_news_count']
            max_diff = 0
            for col in feature_cols:
                if col in orig_features.columns and col in opt_features.columns:
                    diff = np.abs(orig_features[col] - opt_features[col]).max()
                    max_diff = max(max_diff, diff)
            
            print(f"📐 Максимальная разность в фичах: {max_diff:.6f}")
            
            if max_diff < 1e-5:
                print("✅ Результаты численно идентичны!")
            else:
                print("⚠️  Есть небольшие различия в результатах")
    
    print("\n🎉 Тестирование завершено!")


if __name__ == "__main__":
    main()
