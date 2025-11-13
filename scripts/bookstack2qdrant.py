import sys
import os
import time
from tqdm import tqdm
from pathlib import Path
import torch
from langchain_qdrant import QdrantVectorStore
from qdrant_client import QdrantClient
from qdrant_client.http.models import Distance, VectorParams
from langchain.schema import Document

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from search.nn import init_embedder
from parse_bookstack import test_connection, html_to_plain_text
from parse_bookstack import get_all_chapters, get_all_pages, get_page_content

BOOKSTACK_URL = "https://bookstack.profcomff.com" 
TOKEN_ID = ""
TOKEN_SECRET = ""

BOOK_ID = 1 # ID книги для экспорта (из вашего импортера)
TIMEOUT = 30

headers = {
    "Authorization": f"Token {TOKEN_ID}:{TOKEN_SECRET}",
    "Content-Type": "application/json",
    "Accept": "application/json"
}


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
        print("❌ Не удалось подключиться к BookStack API")
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
    
    chapters = get_all_chapters(BOOK_ID)
    pages = get_all_pages(BOOK_ID)

    chapter_dict = {chapter['id']: chapter for chapter in chapters}

    # writer.writerow(['Chapter', 'Page', 'Content'])
        
    error_count = 0
    exported_count = 0

    all_chunks = []
    for i, page in enumerate(pages):
        print(f"📝 Обработка {i+1}/{len(pages)}: {page['name']}...")

        page_details = get_page_content(page['id'])
        
        if not page_details:
            error_count += 1
            continue
            
        # Определяем название главы
        topic_name = ""
        if page.get('chapter_id') and page['chapter_id'] in chapter_dict:
            topic_name = chapter_dict[page['chapter_id']]['name']
            
        # Преобразуем HTML в простой текст
        answer = html_to_plain_text(page_details.get('html', ''))

        kw = None

        all_chunks.append(Document(
            page_content=answer,
            metadata={
                "source": topic_name.strip(),
                "key_words": kw,
                'number_id': i
            }
        ))
            
        exported_count += 1
            
        # Небольшая задержка чтобы не перегружать сервер
        if i % 10 == 0:
            time.sleep(0.5)

    safe_add_documents(vector_store, all_chunks)

    
if __name__ == "__main__":
    main()