FROM python:3.11.13-slim

ARG APP_VERSION=dev
ENV APP_VERSION=${APP_VERSION} \
    APP_NAME=answer \
    APP_MODULE=${APP_NAME}.routes.base:app \
    PYTHONPATH=/app \
    NLTK_DATA=/app/nltk_data \
    CA_BUNDLE_URL=https://gu-st.ru/content/Other/doc/russiantrustedca.pem

WORKDIR /app

# Установка системных зависимостей (включая curl для runtime)
RUN apt-get update && \
    apt-get install -y --no-install-recommends wget curl ca-certificates && \
    mkdir -p /app/nltk_data && \
    pip install nltk && \
    python -c "import nltk; \
              nltk.download('stopwords', download_dir='/app/nltk_data'); \
              nltk.download('punkt_tab', download_dir='/app/nltk_data'); \
              nltk.download('punkt', download_dir='/app/nltk_data')" && \
    apt-get remove -y wget && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

# Entrypoint для загрузки сертификата
COPY entrypoint.sh /entrypoint.sh
RUN chmod +x /entrypoint.sh

ENTRYPOINT ["/entrypoint.sh"]
CMD ["uvicorn", "answer.routes.base:app", "--host", "0.0.0.0", "--port", "8000"]
