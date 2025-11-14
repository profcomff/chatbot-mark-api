# Ассистент

## Создание Векторной БД с нуля 
1. Установите зависимости
    ```console
    pip install -r requirements.txt
    ```

2. Установите переменные окружения
    ```console
    export QDRANT_API_KEY="qdrant_api_key"
    
    export COLLECTION_NAME='collection_name' # Используйте не занятое
    ```
3. Запустите скрипт
    ```console
    cd scripts
    
    python create_qdrant_db.py 
    ```

## Парсинг документов из bookstack в новую коллекцию в qdrant

```bash
export BOOKSTACK_TOKEN_ID = "BOOKSTACK_TOKEN_ID"
export BOOKSTACK_TOKEN_SECRET = "BOOKSTACK_TOKEN_SECRET"

export COLLECTION_NAME="name_for_new_qdrant_collection"
```
Запуск скрипта
```bash
python scripts/bookstack2qdrant.py
```
## Запуск API

1. Перейдите в папку проекта

2. Создайте виртуальное окружение и активируйте его:
    ```console
    python3 -m venv venv
    source ./venv/bin/activate  # На MacOS и Linux
    venv\Scripts\activate     # На Windows
    ```

3. Установите зависимости
    ```console
    pip install -r requirements.txt
    ```
    ```console
    python -m nltk.downloader punkt_tab
    ```

4. Установите переменные окружения
    ```console
    # Ключ для доступа к бд
    export QDRANT_API_KEY="qdrant_api_key"

    # Токен ТГ Бота
    export BOT_TOKEN="BOT_TOKEN"

    # Токен YandexGPT
    export SERVICE_ACCOUNT_ID="FROM YAGPT"
    
    export KEY_ID="FROM YAGPT"
    
    export PRIVATE_KEY="FROM YAGPT"
    ```

5. Запустите приложение
    ```console
    python -m answer
    ```

## ENV-file description
- `DB_DSN=postgresql://postgres@localhost:5432/postgres` – Данные для подключения к БД
