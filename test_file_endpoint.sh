#!/bin/bash

# Тестовый скрипт для проверки нового эндпоинта /process-news-file
# Использует curl для отправки multipart/form-data запроса

API_URL="http://localhost:8000"
CALLBACK_URL="http://localhost:8080/news"
SESSION_ID="test_session_$(date +%Y%m%d_%H%M%S)"

echo "🚀 Тестирование эндпоинта /process-news-file"
echo "API URL: $API_URL"
echo "Callback URL: $CALLBACK_URL"
echo "Session ID: $SESSION_ID"
echo "=============================================="

# Создаем тестовый CSV файл
cat > test_news.csv << EOF
publish_date,title,publication
2025-01-01 10:00:00,"Сбербанк объявил о росте прибыли",РБК
2025-01-01 11:00:00,"Газпром увеличил добычу газа",Коммерсант
2025-01-01 12:00:00,"Лукойл планирует новые проекты",Ведомости
2025-01-01 13:00:00,"Российские акции выросли на фоне новостей",Интерфакс
2025-01-01 14:00:00,"ЦБ РФ сохранил ключевую ставку",ТАСС
EOF

echo "📁 Создан тестовый файл test_news.csv"

# Отправляем запрос
echo "📤 Отправка запроса..."
curl -X POST \
  -F "file=@test_news.csv" \
  -F "callbackUrl=$CALLBACK_URL" \
  -F "sessionId=$SESSION_ID" \
  -F "artifacts_dir=artifacts" \
  -F "p_threshold=0.5" \
  -F "half_life_days=0.5" \
  -F "max_days=5.0" \
  -F "add_sentiment=true" \
  -H "Accept: application/json" \
  "$API_URL/process-news-file" \
  -w "\n\nHTTP Status: %{http_code}\nTotal Time: %{time_total}s\n" \
  -v

echo ""
echo "=============================================="
echo "🏁 Тест завершен!"
echo ""
echo "Примечания:"
echo "- Убедитесь, что API сервер запущен на localhost:8000"
echo "- Убедитесь, что callback сервер доступен на localhost:8080"
echo "- Проверьте наличие файлов артефактов в папке 'artifacts'"
echo "- Результат обработки будет отправлен на callback URL"
echo ""
echo "Для проверки callback можно использовать:"
echo "curl -X POST http://localhost:8080/news \\"
echo "  -H 'Content-Type: application/json' \\"
echo "  -d '{\"sessionId\":\"$SESSION_ID\",\"status\":\"success\",\"data\":\"test data\"}'"

# Удаляем тестовый файл
rm -f test_news.csv
echo "🧹 Тестовый файл удален"
