# parse_bookstack.py
import requests
import csv
import os
import time
from urllib.parse import urljoin

# --- НАСТРОЙКИ ---
BOOKSTACK_URL = "https://bookstack.profcomff.com" 
TOKEN_ID = "cpLcxwHfJdVmZp0V7qYmXbH2AEsEwAKS"
TOKEN_SECRET = "YMLSWtuqqW2VLlW5JLSeHfHQ6hUe8DAO"

# ID книги для экспорта (из вашего импортера)
BOOK_ID = 1

# Имя выходного файла
OUTPUT_CSV = "exported_data.csv"

# Таймаут для запросов (в секундах)
TIMEOUT = 30

# --- КОНЕЦ НАСТРОЕК ---

headers = {
    "Authorization": f"Token {TOKEN_ID}:{TOKEN_SECRET}",
    "Content-Type": "application/json",
    "Accept": "application/json"
}

def test_connection():
    """Проверка подключения к API"""
    print("🔌 Проверка подключения к BookStack API...")
    try:
        response = requests.get(f"{BOOKSTACK_URL}/api/books", headers=headers, timeout=TIMEOUT)
        print(f"   Status Code: {response.status_code}")
        if response.status_code == 200:
            data = response.json()
            print(f"   ✅ Успешно! Доступно книг: {data['total']}")
            return True
        else:
            print(f"   ❌ Ошибка: {response.text}")
            return False
    except Exception as e:
        print(f"   ❌ Ошибка подключения: {e}")
        return False

def html_to_plain_text(html_content):
    """
    Преобразует HTML в простой текст, аналогичный тому, что был в импорте.
    Заменяет <br> и другие HTML теги на переносы строк.
    """
    if not html_content:
        return ""
    
    # Простое преобразование HTML в текст
    import re
    import html
    # Заменяем <br> на переносы строк
    text = re.sub(r'<br\s*/?>', '\n', html_content, flags=re.IGNORECASE)
    # Заменяем параграфы на двойные переносы
    text = re.sub(r'</p>\s*<p>', '\n\n', text)
    text = re.sub(r'<p>|</p>', '', text)
    # Удаляем другие HTML теги
    text = re.sub(r'<[^>]+>', '', text)
    # Декодируем HTML сущности
    text = html.unescape(text)
    # Убираем лишние пробелы
    text = re.sub(r'\n\s+\n', '\n\n', text)
    return text.strip()

def get_all_chapters(book_id):
    """Получить все главы книги"""
    print("📖 Получение глав...")
    chapters = []
    page = 1
    while True:
        print(f"   Страница {page}...")
        params = {'filter[book_id]': book_id, 'page': page, 'count': 100}
        try:
            response = requests.get(f"{BOOKSTACK_URL}/api/chapters", headers=headers, params=params, timeout=TIMEOUT)
            response.raise_for_status()
            data = response.json()
            chapters.extend(data['data'])
            print(f"   Получено глав: {len(data['data'])}")
            
            if len(data['data']) < 100:
                break
            page += 1
        except requests.exceptions.RequestException as e:
            print(f"   ❌ Ошибка при получении глав: {e}")
            break
        except Exception as e:
            print(f"   ❌ Неожиданная ошибка: {e}")
            break
    
    print(f"✅ Всего глав получено: {len(chapters)}")
    return chapters

def get_all_pages(book_id):
    """Получить все страницы книги - альтернативный метод"""
    print("📄 Получение страниц...")
    all_pages = []
    page_num = 1
    max_pages_to_check = 1  # Ограничиваем количество страниц для проверки
    
    while page_num <= max_pages_to_check:
        print(f"   📋 Запрос страницы {page_num}...")
        params = {'page': page_num, 'count': 139}
        
        try:
            response = requests.get(f"{BOOKSTACK_URL}/api/pages", headers=headers, params=params, timeout=TIMEOUT)
            response.raise_for_status()
            data = response.json()
            
            # Фильтруем только страницы из нужной книги
            book_pages = [page for page in data['data'] if page.get('book_id') == book_id]
            all_pages.extend(book_pages)
            
            print(f"   ✅ Из {len(data['data'])} страниц, {len(book_pages)} из книги {book_id}")
            print(f"   📊 Всего собрано: {len(all_pages)} страниц")
            
            # Останавливаемся, если на этой странице нет страниц из нужной книги
            if len(book_pages) == 0 and page_num > 1:
                print("   🏁 Больше нет страниц из указанной книги")
                break
                
            page_num += 1
            time.sleep(0.3)
            
        except Exception as e:
            print(f"   ❌ Ошибка: {e}")
            break
    
    print(f"✅ ФИНАЛЬНЫЙ РЕЗУЛЬТАТ: Всего страниц из книги {book_id}: {len(all_pages)}")
    return all_pages

def get_page_content(page_id, retries=3):
    """Получить содержимое страницы с повторными попытками"""
    for attempt in range(retries):
        try:
            response = requests.get(f"{BOOKSTACK_URL}/api/pages/{page_id}", headers=headers, timeout=TIMEOUT)
            response.raise_for_status()
            return response.json()
        except requests.exceptions.Timeout:
            print(f"   ⏰ Таймаут при получении страницы {page_id}, попытка {attempt + 1}/{retries}")
            if attempt < retries - 1:
                time.sleep(2)
            else:
                return None
        except Exception as e:
            print(f"   ❌ Ошибка при получении страницы {page_id}: {e}")
            return None
    return None

def main():
    print("🚀 BookStack Exporter")
    print(f"📚 Экспорт книги ID: {BOOK_ID}")
    print(f"💾 Выходной файл: {OUTPUT_CSV}")
    
    # Проверка подключения
    if not test_connection():
        print("❌ Не удалось подключиться к BookStack API")
        return
    
    # Получаем все главы и страницы
    chapters = get_all_chapters(BOOK_ID)
    pages = get_all_pages(BOOK_ID)
    
    if not pages:
        print("❌ Не удалось получить страницы")
        return
    
    # Создаем словарь для быстрого доступа к главам по ID
    chapter_dict = {chapter['id']: chapter for chapter in chapters}
    
    # Экспортируем в CSV
    print("💾 Сохранение в CSV...")
    with open(OUTPUT_CSV, 'w', encoding='utf-8', newline='') as csvfile:
        writer = csv.writer(csvfile)
        # Заголовок как в импортере
        writer.writerow(['Chapter', 'Page', 'Content'])
        
        exported_count = 0
        error_count = 0
        
        for i, page in enumerate(pages):
            print(f"📝 Обработка {i+1}/{len(pages)}: {page['name']}...")
            
            try:
                # Получаем содержимое страницы
                page_details = get_page_content(page['id'])
                
                if not page_details:
                    error_count += 1
                    continue
                
                # Определяем название главы
                chapter_name = ""
                if page.get('chapter_id') and page['chapter_id'] in chapter_dict:
                    chapter_name = chapter_dict[page['chapter_id']]['name']
                
                # Преобразуем HTML в простой текст
                plain_text = html_to_plain_text(page_details.get('html', ''))
                
                # Записываем в CSV
                writer.writerow([
                    chapter_name,
                    page['name'],
                    plain_text
                ])
                
                exported_count += 1
                
                # Небольшая задержка чтобы не перегружать сервер
                if i % 10 == 0:
                    time.sleep(0.5)
                    
            except Exception as e:
                print(f"❌ Ошибка при обработке страницы '{page.get('name', 'Unknown')}': {e}")
                error_count += 1
                continue
    
    print(f"\n🎉 Экспорт завершен!")
    print(f"📊 Статистика:")
    print(f"   Успешно экспортировано: {exported_count}")
    print(f"   Ошибок: {error_count}")
    print(f"   Всего страниц: {len(pages)}")
    print(f"💾 Файл: {OUTPUT_CSV}")


if __name__ == "__main__":
    main()