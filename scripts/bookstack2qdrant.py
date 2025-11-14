import sys
import os
import time
from tqdm import tqdm
import torch
import re
import html
import requests
from langchain_qdrant import QdrantVectorStore
from qdrant_client import QdrantClient
from qdrant_client.http.models import Distance, VectorParams
from langchain.schema import Document

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from search.nn import init_embedder

BOOKSTACK_URL = "https://bookstack.profcomff.com" 
BOOKSTACK_TOKEN_ID = os.environ["BOOKSTACK_TOKEN_ID"]
BOOKSTACK_TOKEN_SECRET = os.environ["BOOKSTACK_TOKEN_SECRET"]

BOOK_ID = 1 # ID книги для экспорта (из вашего импортера)
TIMEOUT = 30

headers = {
    "Authorization": f"Token {BOOKSTACK_TOKEN_ID}:{BOOKSTACK_TOKEN_SECRET}",
    "Content-Type": "application/json",
    "Accept": "application/json"
}


def test_connection():
    print("Проверка подключения к BookStack API...")
    try:
        response = requests.get(f"{BOOKSTACK_URL}/api/books", headers=headers, timeout=TIMEOUT)
        print(f"   Status Code: {response.status_code}")
        if response.status_code == 200:
            data = response.json()
            print(f"Успешно! Доступно книг: {data['total']}")
            return True
        else:
            print(f"Ошибка: {response.text}")
            return False
    except Exception as e:
        print(f"Ошибка подключения: {e}")
        return False


def html_to_plain_text(html_content):
    if not html_content:
        return ""
    
    # Простое преобразование HTML в текст

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


def get_book_info(book_id):
    """Получить информацию о книге, включая slug"""
    print(f"Получение информации о книге {book_id}...")
    try:
        response = requests.get(f"{BOOKSTACK_URL}/api/books/{book_id}", headers=headers, timeout=TIMEOUT)
        response.raise_for_status()
        book_data = response.json()
        print(f"   Книга: {book_data['name']}, slug: {book_data['slug']}")
        return book_data
    except Exception as e:
        print(f"Ошибка при получении информации о книге: {e}")
        return None
    

def construct_page_url(book_slug, page_slug, page_details):
    """Сконструировать правильный URL для страницы"""
    # Если страница находится в главе
    if page_details.get('chapter_id'):
        chapter_slug = page_details.get('chapter_slug', '')
        if chapter_slug:
            return f"{BOOKSTACK_URL}/books/{book_slug}/chapter/{chapter_slug}#{page_slug}"
    
    # Обычная страница в книге
    return f"{BOOKSTACK_URL}/books/{book_slug}/page/{page_slug}"


def get_all_chapters(book_id):
    """Получить все главы книги"""
    print("Получение глав...")
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
            print(f"Ошибка при получении глав: {e}")
            break
        except Exception as e:
            print(f"Неожиданная ошибка: {e}")
            break
    
    print(f"Всего глав получено: {len(chapters)}")
    return chapters


def get_all_pages(book_id):
    """Получить все страницы книги"""
    print("📄 Получение страниц...")
    all_pages = []
    page_num = 1
    max_pages_to_check = 1  # Ограничиваем количество страниц для проверки - не знаю почему так
    
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
                print("Больше нет страниц из указанной книги")
                break
                
            page_num += 1
            time.sleep(0.3)
            
        except Exception as e:
            print(f"Ошибка: {e}")
            break
    
    print(f"ФИНАЛЬНЫЙ РЕЗУЛЬТАТ: Всего страниц из книги {book_id}: {len(all_pages)}")
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
            print(f"Ошибка при получении страницы {page_id}: {e}")
            return None
    return None


def safe_add_documents(vector_store, chunks, batch_size=1000):
    with tqdm(total=len(chunks), desc="Добавление в БД", unit="doc") as pbar:
        for i in range(0, len(chunks), batch_size):
            try:
                batch = chunks[i:i+batch_size]
                vector_store.add_documents(batch)
                pbar.update(len(batch))
            except Exception as e:
                if "Batch size" in str(e) and "greater than max" in str(e):
                    new_size = batch_size // 2
                    print(f"Ошибка: {e}. Уменьшаю размер батча до {new_size}")
                    return safe_add_documents(vector_store, chunks[i:], new_size)
                raise
            finally:
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
    print("Все документы успешно добавлены!")


def main():
    if not test_connection():
        print("Не удалось подключиться к BookStack API")
        return
    
    embedder = init_embedder()
    qdrant_api_key = os.getenv("QDRANT_API_KEY")

    qdrant_client = QdrantClient(
        url="http://qdrant.profcomff.com:6333",
        api_key=qdrant_api_key
    )

    collection_name = os.getenv("COLLECTION_NAME")

    qdrant_client.create_collection(
        collection_name=collection_name,
        vectors_config=VectorParams(size=768, distance=Distance.COSINE),
    )

    vector_store = QdrantVectorStore(
        client=qdrant_client,
        collection_name=collection_name,
        embedding=embedder,
    )

    book_data = get_book_info(BOOK_ID)
    chapters = get_all_chapters(BOOK_ID)
    pages = get_all_pages(BOOK_ID)

    chapter_dict = {chapter['id']: chapter for chapter in chapters}
        
    error_count = 0
    exported_count = 0

    all_chunks = []
    for i, page in enumerate(pages):
        print(f"📝 Обработка {i+1}/{len(pages)}: {page['name']}...")

        page_details = get_page_content(page['id'])
        
        if not page_details:
            error_count += 1
            continue
            
        topic_name = ""
        if page.get('chapter_id') and page['chapter_id'] in chapter_dict:
            topic_name = chapter_dict[page['chapter_id']]['name']
            
        answer = html_to_plain_text(page_details.get('html', ''))

        kw = None #TBA
        page_slug = page_details.get('slug', '')
        if page_slug:
            page_url = construct_page_url(book_data['slug'], page_slug, page_details)
        else:
            # Fallback: используем старый метод если slug недоступен
            page_url = f"{BOOKSTACK_URL}/books/{book_data['slug']}/page/{page['id']}"

        all_chunks.append(Document(
            page_content=answer,
            metadata={
                "source": topic_name.strip(),
                "key_words": kw,
                "number_id": i,
                "url": page_url,
            }
        ))
            
        exported_count += 1
            
        if i % 10 == 0:         # Небольшая задержка чтобы не перегружать сервер
            time.sleep(0.5)

    safe_add_documents(vector_store, all_chunks)

    
if __name__ == "__main__":
    main()