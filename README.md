# Ассистент

-

## Запуск

1. Перейдите в папку проекта

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
    # path to the data with answer
    foo@bar:~$ export ANSWER_DATA="/your/path"
    # path to the answer embeddings
    foo@bar:~$ export EMB_DATA="/your/path"
    # Model from HF
    foo@bar:~$ export EMB_MODEL="d0rj/e5-base-en-ru"
    
    
    ```

4. Запускайте приложение!
    ```console
    foo@bar:~$ python -m answer
    ```

## ENV-file description
- `DB_DSN=postgresql://postgres@localhost:5432/postgres` – Данные для подключения к БД
