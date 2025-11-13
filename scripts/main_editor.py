# bookstack_importer.py
import requests
import json
import csv
import os
import re
from tqdm import tqdm


# --- НАСТРОЙКИ ---
# Замените на URL вашего локального инстанса BookStack
BOOKSTACK_URL = "https://bookstack.profcomff.com" 
TOKEN_ID = "cpLcxwHfJdVmZp0V7qYmXbH2AEsEwAKS"
TOKEN_SECRET = "YMLSWtuqqW2VLlW5JLSeHfHQ6hUe8DAO"
BOOK_ID = 1  # <-- ЗАМЕНИТЕ НА ID ВАШЕЙ КНИГИ

# --- КОНЕЦ НАСТРОЕК ---


def fix_html_content(html_content):
    processed_text = html_content.strip()
    processed_text = re.sub(r'<br[^>]*>[\s]*<br[^>]*>', '</p><p>', processed_text)
    processed_text = re.sub(r'<br[^>]*>', ' ', processed_text)
    if not processed_text.startswith('<p>'):
        processed_text = f"<p>{processed_text}</p>"
    return processed_text

headers = {
    "Authorization": f"Token {TOKEN_ID}:{TOKEN_SECRET}",
    "Content-Type": "application/json"
}

def get_all_pages(book_id):
    pages = []
    page = 1
    while True:
        params = {'filter[book_id]': book_id, 'page': page, 'count': 100}
        try:
            response = requests.get(f"{BOOKSTACK_URL}/api/pages", headers=headers, params=params, timeout=30)
            response.raise_for_status()
        except requests.RequestException as e:
            print(f"Ошибка при обработке страницы {page_id}: {e}")
            continue
        response = requests.get(f"{BOOKSTACK_URL}/api/pages", headers=headers, params=params, timeout=30)
        data = response.json()
        pages.extend(data['data'])
        if len(pages) >= data['total']:
            break
        page += 1
    return pages

pages = get_all_pages(BOOK_ID)
for page in tqdm(pages, desc="Обновление страниц"):
    page_id = page['id']
    response = requests.get(f"{BOOKSTACK_URL}/api/pages/{page_id}", headers=headers)
    page_details = response.json()
    fixed_html = fix_html_content(page_details.get('html', ''))
    update_data = {
        'book_id': page_details['book_id'],
        'name': page_details['name'],
        'html': fixed_html
    }
    if 'chapter_id' in page_details:
        update_data['chapter_id'] = page_details['chapter_id']
    requests.put(f"{BOOKSTACK_URL}/api/pages/{page_id}", headers=headers, json=update_data)