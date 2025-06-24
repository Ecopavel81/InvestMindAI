#!/bin/bash

# Скрипт для быстрой настройки проекта

echo "🚀 Настройка Django проекта с email аутентификацией..."

# Создание необходимых директорий
echo "📁 Создание директорий..."
mkdir -p static
mkdir -p media
mkdir -p staticfiles
mkdir -p templates/accounts

# Копирование .env файла
if [ ! -f .env ]; then
    echo "📝 Создание .env файла..."
    cp .env.example .env
    echo "⚠️  Отредактируйте .env файл с вашими настройками!"
fi

# Проверка Docker
if ! command -v docker &> /dev/null; then
    echo "❌ Docker не установлен. Установите Docker для продолжения."
    exit 1
fi

if ! command -v docker-compose &> /dev/null; then
    echo "❌ Docker Compose не установлен. Установите Docker Compose для продолжения."
    exit 1
fi

# Остановка существующих контейнеров
echo "🛑 Остановка существующих контейнеров..."
docker-compose down

# Удаление старых volumes (опционально)
read -p "🗑️  Удалить существующие данные БД? (y/N): " -n 1 -r
echo
if [[ $REPLY =~ ^[Yy]$ ]]; then
    docker-compose down -v
    echo "✅ Данные БД удалены"
fi

# Сборка и запуск контейнеров
echo "🏗️  Сборка и запуск контейнеров..."
docker-compose up --build -d

# Ожидание запуска БД
echo "⏳ Ожидание запуска PostgreSQL..."
sleep 10

# Применение миграций
echo "🔄 Применение миграций..."
docker-compose exec web python manage.py migrate

# Создание суперпользователя
echo "👤 Создание суперпользователя..."
echo "Введите данные для администратора:"
docker-compose exec web python manage.py createsuperuser

# Сбор статических файлов
echo "📦 Сбор статических файлов..."
docker-compose exec web python manage.py collectstatic --noinput

echo "✅ Настройка завершена!"
echo "🌐 Проект доступен по адресу: http://localhost:8000"
echo "🔧 Админка доступна по адресу: http://localhost:8000/admin"
echo ""
echo "📋 Полезные команды:"
echo "  docker-compose logs web    # Просмотр логов Django"
echo "  docker-compose logs db     # Просмотр логов PostgreSQL"
echo "  docker-compose down        # Остановка проекта"
echo "  docker-compose up -d       # Запуск проекта"