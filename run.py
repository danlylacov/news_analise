#!/usr/bin/env python3
"""
Основной скрипт для запуска компонентов системы
"""
import sys
import os
import argparse
from pathlib import Path

# Добавляем src в путь
sys.path.insert(0, str(Path(__file__).parent.parent.parent / 'src'))

def train_model():
    """Обучение модели"""
    print("🚀 Запуск обучения модели...")
    os.system("python src/ml/train_news_ticker.py --news datasets/raw/train_news.csv --artifacts artifacts/ --epochs 3")

def run_api():
    """Запуск API сервера"""
    print("🚀 Запуск API сервера...")
    os.system("python src/api/app.py")

def run_tests():
    """Запуск тестов"""
    print("🚀 Запуск тестов...")
    os.system("python tests/integration/test_full_response.py")

def create_json_samples():
    """Создание JSON образцов"""
    print("🚀 Создание JSON образцов...")
    os.system("python src/utils/create_full_json.py")
    os.system("python src/utils/fix_parameters.py")

def main():
    parser = argparse.ArgumentParser(description='Управление системой анализа новостей')
    parser.add_argument('command', choices=['train', 'api', 'test', 'json', 'all'], 
                       help='Команда для выполнения')
    
    args = parser.parse_args()
    
    if args.command == 'train':
        train_model()
    elif args.command == 'api':
        run_api()
    elif args.command == 'test':
        run_tests()
    elif args.command == 'json':
        create_json_samples()
    elif args.command == 'all':
        print("🚀 Запуск всех компонентов...")
        create_json_samples()
        train_model()
        run_tests()

if __name__ == "__main__":
    main()
