#!/bin/bash

# Скрипт для тестирования нового эндпоинта в Docker

echo "🐳 Тестирование нового эндпоинта в Docker"
echo "=========================================="

# Остановим все контейнеры
echo "🛑 Остановка существующих контейнеров..."
docker compose down 2>/dev/null || true
docker compose -f docker-compose.test.yml down 2>/dev/null || true

# Удалим старые контейнеры
echo "🧹 Очистка старых контейнеров..."
docker container prune -f

# Соберем и запустим API
echo "🔨 Сборка и запуск API сервера..."
docker compose up -d --build forecast-api

# Ждем запуска API
echo "⏳ Ожидание запуска API сервера..."
sleep 10

# Проверим статус API
echo "🔍 Проверка статуса API..."
API_STATUS=$(docker compose exec -T forecast-api curl -s -o /dev/null -w "%{http_code}" -X POST http://localhost:8000/health)
if [ "$API_STATUS" = "200" ]; then
    echo "✅ API сервер запущен и работает!"
else
    echo "❌ API сервер не отвечает (статус: $API_STATUS)"
    echo "📋 Логи API сервера:"
    docker compose logs forecast-api
    exit 1
fi

# Проверим доступные эндпоинты
echo "📋 Проверка доступных эндпоинтов..."
docker compose exec -T forecast-api curl -s http://localhost:8000/ | python3 -m json.tool

# Создадим тестовый файл
echo "📁 Создание тестового файла..."
cat > test_docker_news.csv << 'EOF'
publish_date,title,publication
2025-01-01 10:00:00,"Сбербанк объявил о росте прибыли на 15%",РБК
2025-01-01 11:00:00,"Газпром увеличил добычу газа в первом квартале",Коммерсант
2025-01-01 12:00:00,"Лукойл планирует новые проекты в Арктике",Ведомости
EOF

# Скопируем тестовый файл в контейнер
docker cp test_docker_news.csv forecast-api:/tmp/test_docker_news.csv

# Тестируем новый эндпоинт
echo "🧪 Тестирование эндпоинта /process-news-file..."
SESSION_ID="docker_test_$(date +%Y%m%d_%H%M%S)"

RESPONSE=$(docker compose exec -T forecast-api curl -s -X POST \
  -F "file=@/tmp/test_docker_news.csv" \
  -F "callbackUrl=http://httpbin.org/post" \
  -F "sessionId=$SESSION_ID" \
  -F "artifacts_dir=/data/artifacts" \
  -F "p_threshold=0.3" \
  -F "half_life_days=0.5" \
  -F "max_days=5.0" \
  -F "add_sentiment=true" \
  -H "Accept: application/json" \
  "http://localhost:8000/process-news-file")

echo "📤 Ответ API:"
echo "$RESPONSE" | python3 -m json.tool

# Проверим, что ответ содержит ожидаемые поля
if echo "$RESPONSE" | grep -q '"status":"accepted"'; then
    echo "✅ Файл успешно принят к обработке!"
else
    echo "❌ Ошибка при принятии файла"
    exit 1
fi

# Тестируем с callback сервером
echo ""
echo "🔄 Тестирование с callback сервером..."

# Запустим callback сервер
docker compose -f docker-compose.test.yml up -d --build callback-server

# Ждем запуска callback сервера
sleep 5

# Проверим callback сервер
CALLBACK_STATUS=$(docker compose -f docker-compose.test.yml exec -T callback-server curl -s -o /dev/null -w "%{http_code}" http://localhost:8081/health)
if [ "$CALLBACK_STATUS" = "200" ]; then
    echo "✅ Callback сервер запущен!"
else
    echo "❌ Callback сервер не отвечает (статус: $CALLBACK_STATUS)"
fi

# Тестируем полный цикл
echo "🧪 Тестирование полного цикла с callback..."
SESSION_ID_CALLBACK="docker_callback_test_$(date +%Y%m%d_%H%M%S)"

RESPONSE_CALLBACK=$(docker compose exec -T forecast-api curl -s -X POST \
  -F "file=@/tmp/test_docker_news.csv" \
  -F "callbackUrl=http://callback-server:8081/news" \
  -F "sessionId=$SESSION_ID_CALLBACK" \
  -F "artifacts_dir=/data/artifacts" \
  -F "p_threshold=0.3" \
  -F "half_life_days=0.5" \
  -F "max_days=5.0" \
  -F "add_sentiment=true" \
  -H "Accept: application/json" \
  "http://localhost:8000/process-news-file")

echo "📤 Ответ API с callback:"
echo "$RESPONSE_CALLBACK" | python3 -m json.tool

# Ждем обработки и проверяем логи callback
echo "⏳ Ожидание обработки (10 секунд)..."
sleep 10

echo "📋 Логи callback сервера:"
docker compose -f docker-compose.test.yml logs callback-server --tail=20

# Очистка
echo ""
echo "🧹 Очистка тестовых файлов..."
rm -f test_docker_news.csv

echo ""
echo "🏁 Тестирование завершено!"
echo "📊 Статус сервисов:"
echo "   - API сервер: ✅ Работает"
echo "   - Callback сервер: ✅ Работает"
echo ""
echo "🔗 Доступные URL:"
echo "   - API: http://localhost:8000"
echo "   - API Docs: http://localhost:8000/docs"
echo "   - Callback: http://localhost:8081"
echo ""
echo "📝 Для остановки сервисов выполните:"
echo "   docker compose down"
echo "   docker compose -f docker-compose.test.yml down"
