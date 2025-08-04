# Weather Data ETL Pipeline

![Python](https://img.shields.io/badge/python-3.9+-blue.svg)
![TimescaleDB](https://img.shields.io/badge/TimescaleDB-2.0+-blue.svg)
![Docker](https://img.shields.io/badge/docker-%230db7ed.svg?logo=docker&logoColor=white)

ETL-пайплайн для обработки метеорологических данных с поддержкой:

- **Источники данных**:
  - JSON-файлы
  - API Open-Meteo

- **Приёмники данных**:
  - CSV-файлы
  - TimescaleDB

> Разработан в рамках тестового задания для отбора в проект **ШИФТ (Data Engineering)**

## 🚀 Ключевые особенности

- **Поддержка нескольких источников данных**: JSON-файлы и Open-Meteo API
- **Гибкие варианты выгрузки**: CSV или TimescaleDB
- **Оптимизация для временных рядов**: специальная структура для метеорологических данных
- **Защита от дубликатов**: автоматическое пропускание повторяющихся записей
- **Контейнеризация**: полная поддержка Docker

## 🏗️ Архитектурные решения

### TimescaleDB для временных рядов

Данные о погоде по своей природе являются временными рядами, поэтому я использовала TimescaleDB с:

- Гипертаблицами с интервалом партиционирования 7 дней (для недельных прогнозов)
- Уникальным индексом по полю `datetime` для быстрого поиска по временным диапазонам и гарантии уникальности записей
- Оптимизированными запросами для временных рядов

Я выбрала стратегию обработки дубликатов: ON CONFLICT DO NOTHING по следующим причинам:
- Сохраняется историческая целостность (вряд ли нам понадобится перезаписывать прогноз погоды, скорее мы захотим узнать сбылся прогноз или нет)
- ON CONFLICT делает 1 запрос вместо 2 — БД сама обрабатывает конфликты на уровне индекса
- При конфликте TimescaleDB не проверяет всю таблицу, а ищет дубликаты по индексу datetime
- Процесс завершается успешно даже при дублирующих данных

### Обработка данных

- Валидация входных параметров
- Трансформация в единый формат
- Комплексная обработка ошибок

## 📂 Структура проекта

```
weather-etl/
├── src/
│   ├── data/
│   │   ├── input/          # Входные JSON-файлы
│   │   └── output/         # Выходные CSV-файлы
│   ├── constants.py        # Константы проекта
│   ├── validators.py       # Валидация данных
│   ├── api_client.py       # Клиент Open-Meteo API
│   ├── transformations.py  # Трансформация данных
│   ├── storage/
│   │   ├── csv_writer.py   # Запись в CSV
│   │   └── db_writer.py   # Запись в TimescaleDB
│   └── main.py            # Основной скрипт
├── tests/                  # Юнит-тесты
├── run_tests.py            # Запуск тестов
├── Dockerfile
├── docker-compose.yml
└── .gitignore
```

## 🛠️ Настройка окружения

1. Создайте .env файл

2. Отредактируйте `.env` (обязательные параметры):
```ini
# TimescaleDB
POSTGRES_USER=admin
POSTGRES_PASSWORD=password
POSTGRES_HOST=timescaledb
POSTGRES_PORT=5432
POSTGRES_DB=mydb
PGADMIN_DEFAULT_EMAIL=admin@example.com
PGADMIN_DEFAULT_PASSWORD=your_password

## 🏃 Запуск проекта

1. Соберите и запустите контейнеры:
```bash
docker-compose up -d --build
```

2. Проверьте имя контейнера:
```bash
docker ps
```

## 💻 Использование

### Запуск тестов
```bash
docker exec -it <CONTAINER_NAME> python /app/run_tests.py
```

### Обработка JSON-файлов
```bash
# Загрузка в TimescaleDB
docker exec <CONTAINER_NAME> python main.py json_to_db

# Экспорт в CSV
docker exec <CONTAINER_NAME> python main.py json_to_csv
```

### Запрос данных из API
```bash
# Загрузка в TimescaleDB (с указанием дат)
docker exec <CONTAINER_NAME> python main.py api_to_db \
  --start-date <YYYY-MM-DD> --end-date <YYYY-MM-DD>

# Экспорт в CSV (с указанием дат)
docker exec <CONTAINER_NAME> python main.py api_to_csv \
  --start-date <YYYY-MM-DD> --end-date <YYYY-MM-DD>
```



