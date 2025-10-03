#!/usr/bin/env python3
"""
Скрипт для обновления путей в коде после реорганизации проекта
"""
import os
import re
from pathlib import Path

def update_imports_in_file(file_path):
    """Обновляет импорты в файле"""
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()
        
        original_content = content
        
        # Обновляем импорты для перемещенных модулей
        replacements = [
            # Core модули
            (r'from news_nlp import', 'from src.core.news_nlp import'),
            (r'from news_features import', 'from src.core.news_features import'),
            (r'from auto_label_tickers import', 'from src.core.auto_label_tickers import'),
            (r'from infer_news_to_candles import', 'from src.core.infer_news_to_candles import'),
            
            # ML модули
            (r'from nn_model import', 'from src.ml.nn_model import'),
            (r'from nn_data import', 'from src.ml.nn_data import'),
            
            # API модули
            (r'from app import', 'from src.api.app import'),
        ]
        
        for pattern, replacement in replacements:
            content = re.sub(pattern, replacement, content)
        
        # Обновляем пути к файлам
        file_replacements = [
            (r'"test_news\.csv"', '"datasets/raw/test_news.csv"'),
            (r'"public_test_candles\.csv"', '"datasets/raw/public_test_candles.csv"'),
            (r'"train_news\.csv"', '"datasets/raw/train_news.csv"'),
            (r'"train_candles\.csv"', '"datasets/raw/train_candles.csv"'),
            (r'"artifacts/"', '"artifacts/"'),  # Оставляем как есть
            (r'"full_swagger_data\.json"', '"examples/json_samples/full_swagger_data.json"'),
            (r'"full_swagger_data_fixed\.json"', '"examples/json_samples/full_swagger_data_fixed.json"'),
            (r'"synchronized_swagger_data\.json"', '"examples/json_samples/synchronized_swagger_data.json"'),
        ]
        
        for pattern, replacement in file_replacements:
            content = re.sub(pattern, replacement, content)
        
        if content != original_content:
            with open(file_path, 'w', encoding='utf-8') as f:
                f.write(content)
            print(f"✅ Обновлен: {file_path}")
            return True
        else:
            print(f"⏭️ Без изменений: {file_path}")
            return False
            
    except Exception as e:
        print(f"❌ Ошибка в {file_path}: {e}")
        return False

def update_all_files():
    """Обновляет все Python файлы в проекте"""
    updated_files = []
    
    # Обновляем файлы в src/
    for root, dirs, files in os.walk('src'):
        for file in files:
            if file.endswith('.py'):
                file_path = os.path.join(root, file)
                if update_imports_in_file(file_path):
                    updated_files.append(file_path)
    
    # Обновляем тесты
    for root, dirs, files in os.walk('tests'):
        for file in files:
            if file.endswith('.py'):
                file_path = os.path.join(root, file)
                if update_imports_in_file(file_path):
                    updated_files.append(file_path)
    
    # Обновляем утилиты
    for root, dirs, files in os.walk('src/utils'):
        for file in files:
            if file.endswith('.py'):
                file_path = os.path.join(root, file)
                if update_imports_in_file(file_path):
                    updated_files.append(file_path)
    
    print(f"\n📊 Обновлено файлов: {len(updated_files)}")
    for file in updated_files:
        print(f"  - {file}")

if __name__ == "__main__":
    print("🔄 Обновление путей в коде...")
    update_all_files()
    print("✅ Готово!")
