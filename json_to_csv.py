#!/usr/bin/env python3
"""
Скрипт для конвертации JSON результатов API в CSV формат
"""

import json
import pandas as pd
import argparse
import os
from datetime import datetime

def json_to_csv(json_file_path, output_dir=None):
    """
    Конвертирует JSON файл с результатами API в CSV файлы
    
    Args:
        json_file_path: Путь к JSON файлу
        output_dir: Директория для сохранения CSV файлов (по умолчанию - та же директория)
    """
    
    # Проверяем существование файла
    if not os.path.exists(json_file_path):
        print(f"❌ Файл {json_file_path} не найден")
        return
    
    # Определяем директорию для сохранения
    if output_dir is None:
        output_dir = os.path.dirname(json_file_path) or '.'
    
    # Создаем директорию если не существует
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
    
    print(f"📂 Загружаем JSON файл: {json_file_path}")
    
    try:
        # Загружаем JSON данные
        with open(json_file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        print(f"✅ JSON файл загружен успешно")
        print(f"📊 Статус: {data.get('status', 'unknown')}")
        print(f"📊 Строк в фичах: {data.get('rows_features', 0)}")
        print(f"📊 Строк в объединенных данных: {data.get('rows_joined', 0)}")
        
        # Создаем базовое имя файла
        base_name = os.path.splitext(os.path.basename(json_file_path))[0]
        
        # Конвертируем features_preview в CSV
        if data.get('features_preview'):
            print(f"\n🔄 Конвертируем features_preview в CSV...")
            
            features_df = pd.DataFrame(data['features_preview'])
            
            # Добавляем метаданные
            features_df['source_file'] = os.path.basename(json_file_path)
            features_df['export_timestamp'] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            
            features_csv_path = os.path.join(output_dir, f"{base_name}_features.csv")
            features_df.to_csv(features_csv_path, index=False, encoding='utf-8')
            
            print(f"✅ Features сохранены в: {features_csv_path}")
            print(f"   Колонки: {list(features_df.columns)}")
            print(f"   Строк: {len(features_df)}")
            
            # Показываем статистику по сентимент-колонкам
            sentiment_cols = [col for col in features_df.columns if 'sentiment' in col]
            if sentiment_cols:
                print(f"   Сентимент-колонки: {sentiment_cols}")
                
                # Статистика по сентименту
                if 'sentiment_mean' in features_df.columns:
                    sentiment_stats = features_df['sentiment_mean'].describe()
                    print(f"   Статистика сентимента:")
                    print(f"     Среднее: {sentiment_stats['mean']:.3f}")
                    print(f"     Мин: {sentiment_stats['min']:.3f}")
                    print(f"     Макс: {sentiment_stats['max']:.3f}")
        
        # Конвертируем joined_preview в CSV
        if data.get('joined_preview'):
            print(f"\n🔄 Конвертируем joined_preview в CSV...")
            
            joined_df = pd.DataFrame(data['joined_preview'])
            
            # Добавляем метаданные
            joined_df['source_file'] = os.path.basename(json_file_path)
            joined_df['export_timestamp'] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            
            joined_csv_path = os.path.join(output_dir, f"{base_name}_joined.csv")
            joined_df.to_csv(joined_csv_path, index=False, encoding='utf-8')
            
            print(f"✅ Joined данные сохранены в: {joined_csv_path}")
            print(f"   Колонки: {list(joined_df.columns)}")
            print(f"   Строк: {len(joined_df)}")
            
            # Показываем статистику по тикерам
            if 'ticker' in joined_df.columns:
                ticker_counts = joined_df['ticker'].value_counts()
                print(f"   Тикеры: {dict(ticker_counts)}")
        
        # Создаем сводный отчет
        print(f"\n📋 Создаем сводный отчет...")
        
        summary_data = {
            'source_file': [os.path.basename(json_file_path)],
            'export_timestamp': [datetime.now().strftime("%Y-%m-%d %H:%M:%S")],
            'api_status': [data.get('status', 'unknown')],
            'rows_features': [data.get('rows_features', 0)],
            'rows_joined': [data.get('rows_joined', 0)],
            'has_features': [bool(data.get('features_preview'))],
            'has_joined': [bool(data.get('joined_preview'))],
            'features_columns': [len(data['features_preview'][0]) if data.get('features_preview') else 0],
            'joined_columns': [len(data['joined_preview'][0]) if data.get('joined_preview') else 0]
        }
        
        # Добавляем информацию о сентимент-анализе
        if data.get('features_preview'):
            features_df = pd.DataFrame(data['features_preview'])
            sentiment_cols = [col for col in features_df.columns if 'sentiment' in col]
            summary_data['sentiment_columns'] = [len(sentiment_cols)]
            summary_data['has_sentiment'] = [len(sentiment_cols) > 0]
            
            if sentiment_cols:
                summary_data['avg_sentiment'] = [features_df['sentiment_mean'].mean() if 'sentiment_mean' in features_df.columns else 0]
        else:
            summary_data['sentiment_columns'] = [0]
            summary_data['has_sentiment'] = [False]
            summary_data['avg_sentiment'] = [0]
        
        summary_df = pd.DataFrame(summary_data)
        summary_csv_path = os.path.join(output_dir, f"{base_name}_summary.csv")
        summary_df.to_csv(summary_csv_path, index=False, encoding='utf-8')
        
        print(f"✅ Сводный отчет сохранен в: {summary_csv_path}")
        
        print(f"\n🎉 Конвертация завершена успешно!")
        print(f"📁 Результаты сохранены в директории: {output_dir}")
        
    except Exception as e:
        print(f"❌ Ошибка при конвертации: {e}")

def main():
    """Основная функция"""
    parser = argparse.ArgumentParser(description='Конвертация JSON результатов API в CSV')
    parser.add_argument('json_file', help='Путь к JSON файлу для конвертации')
    parser.add_argument('--output-dir', '-o', help='Директория для сохранения CSV файлов')
    parser.add_argument('--batch', '-b', action='store_true', help='Обработать все JSON файлы в директории')
    
    args = parser.parse_args()
    
    if args.batch:
        # Обрабатываем все JSON файлы в директории
        json_dir = os.path.dirname(args.json_file) if os.path.isfile(args.json_file) else args.json_file
        json_files = [f for f in os.listdir(json_dir) if f.endswith('.json')]
        
        print(f"🔄 Найдено {len(json_files)} JSON файлов для обработки")
        
        for json_file in json_files:
            json_path = os.path.join(json_dir, json_file)
            print(f"\n{'='*60}")
            print(f"Обрабатываем: {json_file}")
            print(f"{'='*60}")
            json_to_csv(json_path, args.output_dir)
    else:
        # Обрабатываем один файл
        json_to_csv(args.json_file, args.output_dir)

if __name__ == "__main__":
    main()
