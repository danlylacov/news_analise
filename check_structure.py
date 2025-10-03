#!/usr/bin/env python3
"""
Скрипт для проверки структуры проекта
"""
import os
from pathlib import Path

def check_project_structure():
    """Проверяет структуру проекта"""
    print("🔍 Проверка структуры проекта...")
    
    required_dirs = [
        'src/core',
        'src/ml', 
        'src/api',
        'src/utils',
        'datasets/raw',
        'datasets/processed',
        'datasets/output',
        'notebooks/colab',
        'notebooks/kaggle',
        'tests/integration',
        'tests/unit',
        'scripts/data_processing',
        'scripts/testing',
        'examples/json_samples',
        'examples/api_responses',
        'config',
        'docs',
        'artifacts'
    ]
    
    missing_dirs = []
    for dir_path in required_dirs:
        if not os.path.exists(dir_path):
            missing_dirs.append(dir_path)
    
    if missing_dirs:
        print("❌ Отсутствующие папки:")
        for dir_path in missing_dirs:
            print(f"  - {dir_path}")
    else:
        print("✅ Все необходимые папки присутствуют")
    
    # Проверяем ключевые файлы
    required_files = [
        'src/core/news_nlp.py',
        'src/core/news_features.py',
        'src/core/auto_label_tickers.py',
        'src/core/infer_news_to_candles.py',
        'src/ml/nn_model.py',
        'src/ml/nn_data.py',
        'src/ml/train_news_ticker.py',
        'src/api/app.py',
        'config/requirements.txt',
        'config/docker-compose.yml',
        'config/Dockerfile',
        'README.md',
        'docs/README.md',
        'run.py'
    ]
    
    missing_files = []
    for file_path in required_files:
        if not os.path.exists(file_path):
            missing_files.append(file_path)
    
    if missing_files:
        print("❌ Отсутствующие файлы:")
        for file_path in missing_files:
            print(f"  - {file_path}")
    else:
        print("✅ Все необходимые файлы присутствуют")
    
    # Проверяем данные
    data_files = [
        'datasets/raw/test_news.csv',
        'datasets/raw/public_test_candles.csv',
        'examples/json_samples/full_swagger_data_fixed.json'
    ]
    
    missing_data = []
    for file_path in data_files:
        if not os.path.exists(file_path):
            missing_data.append(file_path)
    
    if missing_data:
        print("⚠️ Отсутствующие данные:")
        for file_path in missing_data:
            print(f"  - {file_path}")
    else:
        print("✅ Основные данные присутствуют")
    
    # Проверяем артефакты
    artifacts = [
        'artifacts/model.pt',
        'artifacts/vocab.json',
        'artifacts/tickers.json',
        'artifacts/lexicon.json'
    ]
    
    missing_artifacts = []
    for artifact in artifacts:
        if not os.path.exists(artifact):
            missing_artifacts.append(artifact)
    
    if missing_artifacts:
        print("⚠️ Отсутствующие артефакты:")
        for artifact in missing_artifacts:
            print(f"  - {artifact}")
        print("💡 Запустите обучение модели для создания артефактов")
    else:
        print("✅ Все артефакты присутствуют")
    
    print("\n📊 Статистика:")
    print(f"  Папок: {len([d for d in os.listdir('.') if os.path.isdir(d)])}")
    
    total_files = 0
    for root, dirs, files in os.walk('.'):
        total_files += len(files)
    print(f"  Файлов: {total_files}")
    
    print("\n🎯 Следующие шаги:")
    print("1. Обновите пути в коде: python scripts/data_processing/update_paths.py")
    print("2. Запустите обучение: python run.py train")
    print("3. Запустите API: python run.py api")
    print("4. Запустите тесты: python run.py test")

if __name__ == "__main__":
    check_project_structure()
