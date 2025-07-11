# Django Email Authentication Project

Django проект с регистрацией по email, использующий Poetry для управления зависимостями, uv для Python окружения и Docker для контейнеризации.

## Особенности

- ✅ Регистрация пользователей по email
- ✅ Подтверждение email через ссылку
- ✅ Кастомная модель пользователя
- ✅ Bootstrap UI
- ✅ Docker контейнеризация
- ✅ PostgreSQL база данных
- ✅ Poetry для управления зависимостями
- ✅ uv для Python окружения

## Требования

- Docker и Docker Compose
- Python 3.11+
- Poetry
- uv

## Быстрый старт с Docker

1. Клонируйте проект и перейдите в директорию:
```bash
git clone <your-repo>
cd django-email-auth
```

2. Создайте файл `.env` на основе `.env.example`:
```bash
cp .env.example .env
```

3. Отредактируйте `.env` файл с вашими настройками.

4. Запустите проект с Docker Compose:
```bash
docker-compose up --build
```

5. Создайте суперпользователя:
```bash
docker-compose exec web python manage.py createsuperuser
```

Проект будет доступен по адресу: http://localhost:8000

## Локальная разработка

### Установка зависимостей

1. Установите uv:
```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

2. Установите Poetry:
```bash
curl -sSL https://install.python-poetry.org | python3.11 -
```

3. Установите зависимости:
```bash
poetry install
```

4. Активируйте виртуальное окружение:
```bash
poetry shell

source /home/eco/.cache/pypoetry/virtualenvs/django-email-auth--JxTsaVp-py3.11/bin/activate
```

### Настройка базы данных

1. Установите PostgreSQL или используйте Docker:
```bash
docker run --name postgres-dev -e POSTGRES_PASSWORD=postgres -p 5432:5432 -d postgres:15-alpine

1.1. Подключение к Supabase
poetry add psycopg2-binary dj-database-url django-environ
poetry add supabase
poetry add --group dev python-dotenv
```

2. Создайте и примените миграции:
```bash
python manage.py makemigrations
python manage.py migrate
```

3. Создайте суперпользователя:
```bash
python manage.py createsuperuser
```

### Запуск сервера разработки

```bash
python manage.py runserver
```

## Структура проекта

```
django_email_auth/
├── config/              # Настройки Django
│   ├── __init__.py
│   ├── settings.py
│   ├── urls.py
│   └── wsgi.py
├── accounts/            # Приложение аутентификации
│   ├── models.py        # Кастомная модель User
│   ├── views.py         # Представления для регистрации/входа
│   ├── forms.py         # Формы аутентификации
│   ├── urls.py          # URL маршруты
│   └── admin.py         # Админка
├── templates/           # HTML шаблоны
│   ├── base.html
│   ├── home.html
│   └── accounts/
├── static/              # Статические файлы
├── media/               # Медиа файлы
├── pyproject.toml       # Poetry конфигурация
├── Dockerfile           # Docker образ
├── docker-compose.yml   # Docker Compose
├── .env.example         # Пример переменных окружения
└── manage.py           # Django manage script
```

## Настройка email

Для продакшна настройте реальный SMTP сервер в `.env`:

```env
EMAIL_BACKEND=django.core.mail.backends.smtp.EmailBackend
EMAIL_HOST=smtp.gmail.com
EMAIL_PORT=587
EMAIL_USE_TLS=True
EMAIL_HOST_USER=your-email@gmail.com
EMAIL_HOST_PASSWORD=your-app-password
DEFAULT_FROM_EMAIL=noreply@yoursite.com
```

## Развертывание

### Heroku

1. Установите Heroku CLI
2. Создайте приложение:
```bash
heroku create your-app-name
```

3. Добавьте переменные окружения:
```bash
heroku config:set SECRET_KEY=your-secret-key
heroku config:set DEBUG=False
heroku config:set ALLOWED_HOSTS=your-app-name.herokuapp.com
```

4. Добавьте PostgreSQL:
```bash
heroku addons:create heroku-postgresql:mini
```

5. Деплой:
```bash
git push heroku main
heroku run python manage.py migrate
heroku run python manage.py createsuperuser
```

### VPS с Docker

1. Скопируйте файлы на сервер
2. Настройте `.env` для продакшна
3. Запустите:
```bash
docker-compose -f docker-compose.prod.yml up -d
```

## Безопасность

В продакшне обязательно:

1. Измените `SECRET_KEY`
2. Установите `DEBUG=False`
3. Настройте правильные `ALLOWED_HOSTS`
4. Используйте HTTPS
5. Настройте firewall
6. Регулярно обновляйте зависимости

## Лицензия

MIT