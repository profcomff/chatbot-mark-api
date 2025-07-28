FROM python:3.11.13-slim

# Установка переменных окружения
ARG APP_VERSION=dev
ENV APP_VERSION=${APP_VERSION} \
    APP_NAME=answer \
    APP_MODULE=${APP_NAME}.routes.base:app \
    PYTHONPATH=/app \
    NLTK_DATA=/app/nltk_data

# Установка системных зависимостей и загрузка данных NLTK
WORKDIR /app
RUN apt-get update && \
    apt-get install -y --no-install-recommends wget && \
    mkdir -p /app/nltk_data && \
    pip install nltk && \
    python -c "import nltk; \
              nltk.download('stopwords', download_dir='/app/nltk_data'); \
              nltk.download('punkt_tab', download_dir='/app/nltk_data'); \
              nltk.download('punkt', download_dir='/app/nltk_data')" && \
    apt-get remove -y wget && \
    rm -rf /var/lib/apt/lists/*

# Копирование зависимостей и установка
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Копирование исходного кода
COPY . .

# Проверка структуры проекта (после COPY)
RUN ls -lR /app

CMD ["uvicorn", "answer.routes.base:app", "--host", "0.0.0.0", "--port", "8000"]