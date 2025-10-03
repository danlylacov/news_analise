#!/usr/bin/env python3
"""
Создание версии full_swagger_data.json с исправленными параметрами
"""
import json

def fix_parameters_in_full_json():
    """Исправляем параметры в полном JSON файле"""
    
    print("Читаем полный JSON файл...")
    with open('full_swagger_data.json', 'r', encoding='utf-8') as f:
        full_data = json.load(f)
    
    print(f"Исходные параметры:")
    print(f"  p_threshold: {full_data['p_threshold']}")
    print(f"  half_life_days: {full_data['half_life_days']}")
    print(f"  max_days: {full_data['max_days']}")
    
    # Исправляем параметры для работы с разными временными периодами
    full_data['p_threshold'] = 0.1  # Очень низкий порог
    full_data['half_life_days'] = 30.0  # Большой период полураспада
    full_data['max_days'] = 2000  # Очень большое окно (5+ лет)
    
    print(f"\nНовые параметры:")
    print(f"  p_threshold: {full_data['p_threshold']} (снижен для большей чувствительности)")
    print(f"  half_life_days: {full_data['half_life_days']} (увеличен для учета старых новостей)")
    print(f"  max_days: {full_data['max_days']} (увеличен для покрытия временного разрыва)")
    
    # Сохраняем исправленную версию
    with open('full_swagger_data_fixed.json', 'w', encoding='utf-8') as f:
        json.dump(full_data, f, ensure_ascii=False, indent=2)
    
    print(f"\n✅ Создан файл full_swagger_data_fixed.json")
    print(f"📊 Статистика:")
    print(f"  Новостей: {len(full_data['news'])}")
    print(f"  Свечей: {len(full_data['candles'])}")
    print(f"  Тикеров: {len(set(c['ticker'] for c in full_data['candles']))}")
    
    return full_data

if __name__ == "__main__":
    fix_parameters_in_full_json()
