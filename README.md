# Ассистент

## Запуск

1. Перейдите в папку проекта

2. Установите сертификаты Минцифры (для работы Гигачат API)
    ```console
    cd llm
    curl -o russian_trusted_root_ca.crt "https://gu-st.ru/content/Other/doc/russiantrustedca.pem"
    ```

3. Создайте виртуальное окружение и активируйте его:
    ```console
    python3 -m venv venv
    source ./venv/bin/activate  # На MacOS и Linux
    venv\Scripts\activate     # На Windows
    ```

4. Установите зависимости
    ```console
    pip install -r requirements.txt
    ```
    ```console
    python -m nltk.downloader punkt_tab
    ```

5. Установите переменные окружения
    ```console
    # Ключ для доступа к бд
    export QDRANT_API_KEY="qdrant_api_key"

    # Ключ для гигачат API
    export GIGA_KEY_PATH="gigakey.txt"
    ```

6. Запустите приложение
    ```console
    python -m answer
    ```
    
## Запуск через Docker
```console
# Установка сертификатов
cd llm
curl -o russian_trusted_root_ca.crt "https://gu-st.ru/content/Other/doc/russiantrustedca.pem"

# Сборка образа
docker build -t my-fastapi-langchain .

# Поднятие контейнера
docker run -d \
  -p 127.0.0.1:8000:8000 \
  --name my-fastapi-langchain \
  -v "/Локальный/путь/до/chroma_db:/app/chroma_db" \
  -v "/Локальный/путь/до/gigakey.txt:/app/gigakey.txt:ro" \
  -e CHROMA_DIR="/app/chroma_db" \
  -e GIGA_KEY_PATH="/app/gigakey.txt" \
  -e APP_MODULE="answer.routes.base:app" \
  -e PYTHONPATH="/app" \
  my-fastapi-langchain  
```


## ENV-file description
- `DB_DSN=postgresql://postgres@localhost:5432/postgres` – Данные для подключения к БД
