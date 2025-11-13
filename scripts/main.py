# bookstack_importer.py
import requests
import json
import csv
import os

# --- НАСТРОЙКИ ---
# Замените на URL вашего локального инстанса BookStack
BOOKSTACK_URL = "https://bookstack.profcomff.com" 
# Замените на ваши учетные данные API (Token ID и Token Secret)
# Их можно сгенерировать в профиле вашего пользователя в BookStack
TOKEN_ID = "cpLcxwHfJdVmZp0V7qYmXbH2AEsEwAKS"
TOKEN_SECRET = "YMLSWtuqqW2VLlW5JLSeHfHQ6hUe8DAO"

# Название книги, в которую будут добавляться главы и страницы
# Если книга с таким названием не существует, она будет создана
BOOK_NAME = "Справочник студента"
BOOK_DESCRIPTION = ""

# Имя файла с данными (должен лежать в той же папке, что и скрипт)
# Формат: Глава (Tab) Название страницы (Tab) Содержимое страницы
DATA_FILE = "data.csv"

# --- ИДЕНТИФИКАЦИЯ КНИГИ ---
# Укажите ID книги, если он вам известен (например, 5).
# Если оставить None, скрипт будет искать книгу по имени ниже.
BOOK_ID = 1  # <-- ЗАМЕНИТЕ НА ID ВАШЕЙ КНИГИ
# Имя книги (используется, если BOOK_ID = None)
BOOK_NAME = "База знаний Профкома"

# --- РЕЖИМ РАБОТЫ ---
# Если True, скрипт будет ОБНОВЛЯТЬ существующие страницы.
# Если False, он будет ПРОПУСКАТЬ уже созданные страницы.
# УСТАНОВИТЕ True, чтобы исправить ваши страницы.
UPDATE_EXISTING_PAGES = True

# Если True, скрипт ничего не будет отправлять в BookStack, только покажет в консоли, что собирается сделать.
DRY_RUN = False

# --- КОНЕЦ НАСТРОЕК ---


def convert_text_to_html(text):
    """
    Преобразует обычный текст с переносами строк в простой HTML.
    """
    if not text:
        return ""
    # Заменяем каждый перенос строки на HTML-тег <br>
    # и экранируем кавычки на всякий случай
    processed_text = text.strip()
    return processed_text.replace('\n', '<br />')


def main():
    if TOKEN_ID == "ВАШ_TOKEN_ID" or TOKEN_SECRET == "ВАШ_TOKEN_SECRET":
        print("!!! ОШИБКА: Пожалуйста, укажите ваши TOKEN_ID и TOKEN_SECRET в настройках скрипта.")
        return

    print(f"--- Запуск импорта в BookStack ({'СУХОЙ ЗАПУСК' if DRY_RUN else 'РАБОЧИЙ РЕЖИМ'}) ---")
    print(f"Режим обновления существующих страниц: {'Включен' if UPDATE_EXISTING_PAGES else 'Отключен'}")

    api_url = f"{BOOKSTACK_URL}/api"
    headers = {
        "Authorization": f"Token {TOKEN_ID}:{TOKEN_SECRET}",
        "Content-Type": "application/json",
        "Accept": "application/json"
    }
    
    # --- Шаг 1: Определить ID книги ---
    book_id = None
    if BOOK_ID:
        print(f"\n[1] Используется заданный ID книги: {BOOK_ID}")
        book_id = BOOK_ID
    else:
        # Логика поиска книги по имени (если ID не задан)
        # ... (можно оставить код из предыдущей версии, если нужно)
        print(f"!!! ОШИБКА: BOOK_ID не указан. Пожалуйста, укажите ID книги для точности.")
        return

    # --- Шаг 2: Обработка CSV файла ---
    if not os.path.exists(DATA_FILE):
        print(f"\n!!! ОШИБКА: Файл с данными '{DATA_FILE}' не найден.")
        return
        
    print(f"\n[2] Чтение данных из файла '{DATA_FILE}' и создание/обновление глав/страниц...")
    chapter_cache = {}

    with open(DATA_FILE, mode='r', encoding='utf-8-sig') as csvfile:
        reader = csv.reader(csvfile, delimiter=',')
        try:
            next(reader)  # Пропуск строки заголовка
        except StopIteration:
            return # Файл пуст

        for i, row in enumerate(reader):
            if len(row) != 3:
                print(f"    ! Предупреждение: Пропуск строки {i+2}, т.к. в ней не 3 столбца.")
                continue
            
            chapter_name, page_name, page_raw_content = row
            page_html_content = convert_text_to_html(page_raw_content)
            
            print(f"\n  - Обработка: Глава '{chapter_name}' -> Страница '{page_name}'")

            # --- Шаг 2.1: Найти или создать Главу ---
            chapter_id = chapter_cache.get(chapter_name)
            if not chapter_id:
                try:
                    params = {'filter[name]': chapter_name, 'filter[book_id]': book_id}
                    response = requests.get(f"{api_url}/chapters", headers=headers, params=params)
                    response.raise_for_status()
                    chapters_data = response.json()

                    if chapters_data['total'] > 0:
                        chapter_id = chapters_data['data'][0]['id']
                        print(f"    > Найдена существующая глава. ID: {chapter_id}")
                    else:
                        print("    > Глава не найдена. Создание новой главы...")
                        chapter_payload = {"book_id": book_id, "name": chapter_name}
                        if DRY_RUN:
                            chapter_id = -1
                            print("    > [DRY RUN] Глава была бы создана.")
                        else:
                            response = requests.post(f"{api_url}/chapters", headers=headers, json=chapter_payload)
                            response.raise_for_status()
                            chapter_id = response.json()['id']
                            print(f"    > Глава успешно создана. ID: {chapter_id}")
                    
                    chapter_cache[chapter_name] = chapter_id
                
                except requests.exceptions.RequestException as e:
                    print(f"    !!! ОШИБКА при работе с главой '{chapter_name}': {e}")
                    continue
            else:
                print(f"    > Глава найдена в кэше. ID: {chapter_id}")

            # --- Шаг 2.2: Найти, а затем ОБНОВИТЬ или СОЗДАТЬ Страницу ---
            try:
                params = {'filter[name]': page_name, 'filter[chapter_id]': chapter_id}
                response = requests.get(f"{api_url}/pages", headers=headers, params=params)
                response.raise_for_status()
                pages_data = response.json()
                
                # ЛОГИКА ОБНОВЛЕНИЯ ИЛИ СОЗДАНИЯ
                if pages_data['total'] > 0:
                    page_id = pages_data['data'][0]['id']
                    if UPDATE_EXISTING_PAGES:
                        print(f"    > Найдена страница ID:{page_id}. Обновление содержимого...")
                        page_payload = {
                            "book_id": book_id,
                            "chapter_id": chapter_id,
                            "name": page_name,
                            "html": page_html_content
                        }
                        if DRY_RUN:
                            print("    > [DRY RUN] Страница была бы обновлена.")
                        else:
                            # Используем PUT для обновления
                            response = requests.put(f"{api_url}/pages/{page_id}", headers=headers, json=page_payload)
                            response.raise_for_status()
                            print("    > Страница успешно обновлена.")
                    else:
                        print(f"    > Найдена страница ID:{page_id}. Пропускаем (обновление отключено).")
                else:
                    print(f"    > Страница не найдена. Создание новой...")
                    page_payload = {
                        "book_id": book_id,
                        "chapter_id": chapter_id,
                        "name": page_name,
                        "html": page_html_content
                    }
                    if DRY_RUN:
                        print("    > [DRY RUN] Страница была бы создана.")
                    else:
                        # Используем POST для создания
                        response = requests.post(f"{api_url}/pages", headers=headers, json=page_payload)
                        response.raise_for_status()
                        print("    > Страница успешно создана.")

            except requests.exceptions.RequestException as e:
                print(f"    !!! ОШИБКА при работе со страницей '{page_name}': {e}")
                if hasattr(e, 'response') and e.response is not None:
                    print(f"        Ответ сервера: {e.response.text}")
                continue
    
    print("\n--- Импорт завершен! ---")

if __name__ == "__main__":
    main()