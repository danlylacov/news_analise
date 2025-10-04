#!/usr/bin/env python3
"""
Mock callback сервер для тестирования нового эндпоинта
"""

from flask import Flask, request, jsonify
import json
from datetime import datetime

app = Flask(__name__)

@app.route('/news', methods=['POST'])
def handle_news_callback():
    """Обработка callback от news API"""
    try:
        data = request.get_json()
        
        print(f"\n🎯 Получен callback:")
        print(f"   Время: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"   Session ID: {data.get('sessionId', 'N/A')}")
        print(f"   Статус: {data.get('status', 'N/A')}")
        
        if data.get('status') == 'success':
            print(f"   ✅ Успешная обработка!")
            if 'data' in data:
                try:
                    parsed_data = json.loads(data['data'])
                    print(f"   📊 Статистика:")
                    print(f"      - Новостей обработано: {parsed_data.get('summary', {}).get('news_count', 'N/A')}")
                    print(f"      - Строк features: {parsed_data.get('summary', {}).get('rows_features', 'N/A')}")
                    print(f"      - Строк joined: {parsed_data.get('summary', {}).get('rows_joined', 'N/A')}")
                    
                    # Показываем пример features
                    features = parsed_data.get('features', [])
                    if features:
                        print(f"   📈 Пример features (первые 2 записи):")
                        for i, feature in enumerate(features[:2]):
                            print(f"      {i+1}. Тикер: {feature.get('ticker', 'N/A')}, "
                                  f"Дата: {feature.get('date', 'N/A')}, "
                                  f"News count: {feature.get('nn_news_count', 'N/A')}")
                except json.JSONDecodeError as e:
                    print(f"   ⚠️ Ошибка парсинга данных: {e}")
        else:
            print(f"   ❌ Ошибка обработки!")
            print(f"   Сообщение: {data.get('errorMessage', 'N/A')}")
        
        print(f"   📝 Полные данные: {json.dumps(data, indent=2, ensure_ascii=False)}")
        
        return jsonify({"message": "Callback processed successfully"})
        
    except Exception as e:
        print(f"❌ Ошибка в callback сервере: {e}")
        return jsonify({"error": str(e)}), 500

@app.route('/health', methods=['GET'])
def health():
    """Проверка состояния callback сервера"""
    return jsonify({"status": "healthy", "message": "Callback server is running"})

if __name__ == '__main__':
    print("🚀 Запуск mock callback сервера на порту 8081")
    print("📡 Ожидание callback от news API...")
    print("=" * 50)
    app.run(host='0.0.0.0', port=8081, debug=True)
