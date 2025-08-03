# Базовый образ Python
FROM python:3.11-slim-bookworm

# Рабочая директория в контейнере
WORKDIR /app

# Установка зависимостей
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Копирование кода (volume перезапишет при запуске)
COPY src/ .

# Точка входа
#ENTRYPOINT ["python", "main.py"]
#CMD ["--help"]  # Покажет help если команда не указана