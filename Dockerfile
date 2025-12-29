# Dockerfile
FROM python:3.10-slim

WORKDIR /app

# Установка системных зависимостей
RUN apt-get update && apt-get install -y \
    gcc \
    g++ \
    git \
    && rm -rf /var/lib/apt/lists/*

# Копирование requirements
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Копирование исходного кода
COPY . .

# Создание директорий для данных
RUN mkdir -p data/raw data/processed data/embeddings

# Экспорт порта
EXPOSE 8501

# Запуск приложения
CMD ["streamlit", "run", "app/streamlit_app_hf.py", "--server.port=8501", "--server.address=0.0.0.0"]