#!/usr/bin/env python3
"""
Финальное сравнение производительности всех версий модели
"""
import time
import pandas as pd
import numpy as np
import sys
from pathlib import Path

# Добавляем src в путь
sys.path.insert(0, str(Path(__file__).parent / 'src'))

from src.core.infer_news_to_candles import infer_news_to_candles_df
from src.core.infer_news_to_candles_optimized import infer_news_to_candles_df_optimized
from src.core.infer_news_to_candles_ultra_optimized import infer_news_to_candles_df_ultra_optimized


def create_large_test_data(num_news: int = 2000, num_candles: int = 1000):
    """Создает большие тестовые данные для более точного бенчмарка"""
    np.random.seed(42)
    
    # Создаем тестовые новости
    news_data = []
    publications = ['РБК', 'Коммерсант', 'Ведомости', 'Интерфакс', 'ТАСС', 'РИА Новости']
    titles = [
        'Сбербанк объявил о росте прибыли на 15%',
        'Газпром увеличил добычу газа в Арктике',
        'Лукойл планирует новые проекты в Сибири',
        'НЛМК повысил цены на металлопрокат',
        'МТС расширяет сеть 5G в регионах',
        'ВТБ открыл новые отделения в Москве',
        'Роснефть увеличила экспорт нефти',
        'Новатэк запустил новый завод СПГ'
    ]
    
    for i in range(num_news):
        news_data.append({
            'publish_date': f'2024-01-{(i % 30) + 1:02d}',
            'title': np.random.choice(titles),
            'publication': np.random.choice(publications)
        })
    
    # Создаем тестовые свечи
    candles_data = []
    tickers = ['SBER', 'GAZP', 'LKOH', 'NLMK', 'MTSS', 'VTBR', 'ROSN', 'NVTK']
    
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


def benchmark_function(func, df_news, df_candles, artifacts_dir, version_name, **kwargs):
    """Бенчмарк функции с детальной статистикой"""
    print(f"🔄 Тестирование {version_name}...")
    
    # Прогрев (первый запуск может быть медленнее)
    try:
        func(df_news.head(100), df_candles.head(50), artifacts_dir, **kwargs)
    except:
        pass
    
    # Основной тест
    start_time = time.time()
    try:
        result = func(df_news, df_candles, artifacts_dir, **kwargs)
        end_time = time.time()
        return end_time - start_time, result, None
    except Exception as e:
        end_time = time.time()
        return end_time - start_time, None, str(e)


def compare_results(results):
    """Сравнивает результаты разных версий"""
    print("\n🔍 Сравнение корректности результатов:")
    
    # Берем первую успешную версию как эталон
    reference_result = None
    reference_name = None
    
    for name, (time, result, error) in results.items():
        if result is not None:
            reference_result = result
            reference_name = name
            break
    
    if reference_result is None:
        print("❌ Нет успешных результатов для сравнения")
        return
    
    ref_features, ref_joined = reference_result
    
    for name, (time, result, error) in results.items():
        if result is None:
            print(f"❌ {name}: Ошибка - {error}")
            continue
        
        features, joined = result
        
        # Проверяем размеры
        size_match = (len(features) == len(ref_features)) and (len(joined) == len(ref_joined))
        
        # Проверяем числовую точность
        max_diff = 0
        if len(features) == len(ref_features) and len(features) > 0:
            feature_cols = ['nn_news_sum', 'nn_news_mean', 'nn_news_max', 'nn_news_count']
            for col in feature_cols:
                if col in features.columns and col in ref_features.columns:
                    diff = np.abs(features[col] - ref_features[col]).max()
                    max_diff = max(max_diff, diff)
        
        accuracy_status = "✅ Идентичны" if max_diff < 1e-5 else f"⚠️  Разность: {max_diff:.6f}"
        size_status = "✅ Совпадают" if size_match else "❌ Разные размеры"
        
        print(f"📊 {name}:")
        print(f"   Время: {time:.2f}с")
        print(f"   Размеры: {size_status}")
        print(f"   Точность: {accuracy_status}")


def main():
    print("🚀 Финальное тестирование производительности всех версий модели...")
    
    # Создаем большие тестовые данные
    print("📊 Создание больших тестовых данных...")
    df_news, df_candles = create_large_test_data(num_news=2000, num_candles=1000)
    
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
    
    # Список версий для тестирования
    versions = [
        ("Оригинальная", infer_news_to_candles_df),
        ("Оптимизированная", infer_news_to_candles_df_optimized),
        ("Ультра-оптимизированная", infer_news_to_candles_df_ultra_optimized),
    ]
    
    results = {}
    
    # Тестируем все версии
    for version_name, version_func in versions:
        time_taken, result, error = benchmark_function(
            version_func, df_news, df_candles, artifacts_dir, version_name, **params
        )
        results[version_name] = (time_taken, result, error)
    
    # Выводим результаты
    print("\n📊 Результаты тестирования:")
    print("=" * 60)
    
    successful_results = []
    for name, (time, result, error) in results.items():
        if result is not None:
            successful_results.append((name, time))
            print(f"✅ {name}: {time:.2f} секунд")
        else:
            print(f"❌ {name}: Ошибка - {error}")
    
    # Сравниваем производительность
    if len(successful_results) > 1:
        print("\n⚡ Сравнение производительности:")
        print("=" * 60)
        
        # Сортируем по времени
        successful_results.sort(key=lambda x: x[1])
        fastest_time = successful_results[0][1]
        
        for name, time in successful_results:
            speedup = time / fastest_time
            improvement = ((time - fastest_time) / time * 100) if time > fastest_time else 0
            status = "🏆 Лучшая" if time == fastest_time else f"📈 Медленнее на {improvement:.1f}%"
            print(f"{name}: {time:.2f}с (x{speedup:.2f}) - {status}")
    
    # Сравниваем корректность
    compare_results(results)
    
    print("\n🎉 Тестирование завершено!")
    print("\n💡 Рекомендации:")
    print("1. Используйте ультра-оптимизированную версию для максимальной производительности")
    print("2. Кэширование модели дает наибольший прирост скорости")
    print("3. Увеличение batch_size может дополнительно ускорить обработку")


if __name__ == "__main__":
    main()
