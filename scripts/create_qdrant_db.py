import sys
import os
from pathlib import Path

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from search.nn import init_embedder

import pandas as pd
from langchain_qdrant import QdrantVectorStore
from qdrant_client import QdrantClient
from qdrant_client.http.models import Distance, VectorParams
from langchain.schema import Document
from tqdm import tqdm
import torch
import getpass
from pathlib import Path

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

embedder = init_embedder()
qdrant_api_key = getpass.getpass("Enter QDRANT_API_KEY: ")

qdrant_client = QdrantClient(
    url="http://qdrant.profcomff.com:6333",
    api_key=qdrant_api_key
)

collection_name = "demo_collection_test"
qdrant_client.create_collection(
    collection_name=collection_name,
    vectors_config=VectorParams(size=768, distance=Distance.COSINE),
)

vector_store = QdrantVectorStore(
    client=qdrant_client,
    collection_name=collection_name,
    embedding=embedder,
)

table_path = Path(__file__).parent.parent / "file" / "database_v4_key_words.xlsx"
answers = pd.read_excel(table_path)

all_chunks = []
for answer, topic_name, kw in zip(answers['answer'], answers['topic_name'], answers['Key words']):
    all_chunks.append(Document(
        page_content=answer,
        metadata={
            "source": topic_name.strip(),
            "key_words": kw,
        }
    ))

safe_add_documents(vector_store, all_chunks)