# Ассистент

-

## Запуск

1. Перейдите в папку проекта

2. Установите сертификаты Минцифры (для работы Гигачат Api)
    ```console
    foo@bar:~$ cd llm
    foo@bar:~$ curl -o russian_trusted_root_ca.crt "https://gu-st.ru/content/Other/doc/russiantrustedca.pem"
    ```


2. Создайте виртуальное окружение командой и активируйте его:
    ```console
    foo@bar:~$ python3 -m venv venv
    foo@bar:~$ source ./venv/bin/activate  # На MacOS и Linux
    foo@bar:~$ venv\Scripts\activate  # На Windows
    ```

3. Установите библиотеки
    ```console
    foo@bar:~$ pip install -r requirements.txt


4. Установите переменные окружения
    ```console
    # Путь до хрома бд (по дефолту лежит в корневой папке проекта)
    foo@bar:~$ export CHROMA_DIR="/путь/до/папки/chroma_db"

    # Ключ для гигачат апи
    foo@bar:~$ export GIGA_KEY_PATH="/путь/до/ключа/gigakey.txt"

5. Запускайте приложение!
    ```console
    foo@bar:~$ python -m answer
    ```

## ENV-file description
- `DB_DSN=postgresql://postgres@localhost:5432/postgres` – Данные для подключения к БД
