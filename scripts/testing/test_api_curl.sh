#!/bin/bash
# Скрипт для тестирования API через curl с сохранением результата

echo "=== ТЕСТ API ЧЕРЕЗ CURL ==="
echo "Дата: $(date)"
echo ""

# Проверяем наличие файлов
if [ ! -f "full_swagger_data_fixed.json" ]; then
    echo "❌ Файл full_swagger_data_fixed.json не найден!"
    exit 1
fi

echo "📊 Используем файл: full_swagger_data_fixed.json"
echo "📊 Размер файла: $(du -h full_swagger_data_fixed.json | cut -f1)"
echo ""

# Создаем имя файла с timestamp
TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
OUTPUT_FILE="curl_response_${TIMESTAMP}.json"
ERROR_FILE="curl_error_${TIMESTAMP}.json"

echo "🚀 Отправляем запрос..."
echo "💾 Результат будет сохранен в: $OUTPUT_FILE"
echo ""

# Выполняем curl запрос
curl -X POST "http://127.0.0.1:8000/infer" \
  -H "Content-Type: application/json" \
  -d @full_swagger_data_fixed.json \
  --max-time 120 \
  --connect-timeout 30 \
  --write-out "HTTP Status: %{http_code}\nTotal time: %{time_total}s\n" \
  --output "$OUTPUT_FILE" \
  --silent --show-error

CURL_EXIT_CODE=$?

echo ""
echo "📋 Результат curl:"
echo "Exit code: $CURL_EXIT_CODE"

if [ $CURL_EXIT_CODE -eq 0 ]; then
    echo "✅ Запрос выполнен успешно"
    
    if [ -f "$OUTPUT_FILE" ]; then
        echo "📄 Размер ответа: $(du -h $OUTPUT_FILE | cut -f1)"
        echo "📄 Первые строки ответа:"
        head -20 "$OUTPUT_FILE"
        
        # Проверяем, является ли ответ валидным JSON
        if jq empty "$OUTPUT_FILE" 2>/dev/null; then
            echo "✅ Ответ является валидным JSON"
            
            # Извлекаем ключевые поля
            echo ""
            echo "📊 Ключевые поля ответа:"
            echo "  Статус: $(jq -r '.status // "N/A"' "$OUTPUT_FILE")"
            echo "  Строк с фичами: $(jq -r '.rows_features // "N/A"' "$OUTPUT_FILE")"
            echo "  Строк в объединенном: $(jq -r '.rows_joined // "N/A"' "$OUTPUT_FILE")"
            
            # Считаем ненулевые фичи
            NON_ZERO_COUNT=$(jq '[.features_preview[]? | select(.nn_news_sum > 0)] | length' "$OUTPUT_FILE")
            TOTAL_FEATURES=$(jq '.features_preview | length' "$OUTPUT_FILE")
            echo "  Фич с ненулевыми значениями: $NON_ZERO_COUNT из $TOTAL_FEATURES"
            
        else
            echo "❌ Ответ не является валидным JSON"
            echo "📄 Содержимое файла:"
            cat "$OUTPUT_FILE"
        fi
    else
        echo "❌ Файл ответа не создан"
    fi
    
else
    echo "❌ Ошибка выполнения curl"
    
    # Сохраняем информацию об ошибке
    cat > "$ERROR_FILE" << EOF
{
  "timestamp": "$(date -Iseconds)",
  "curl_exit_code": $CURL_EXIT_CODE,
  "error_type": "curl_error",
  "input_file": "full_swagger_data_fixed.json",
  "url": "http://127.0.0.1:8000/infer"
}
EOF
    
    echo "💾 Информация об ошибке сохранена в: $ERROR_FILE"
fi

echo ""
echo "🏁 Тест завершен"
