import json
import math
from langchain_core.documents import Document
from .preprocess import preprocess_lemma


def get_documents_from_qdrant(client, collection_name, page_content_field="page_content", metadata_field="metadata"):
    documents = []
    points, next_page = client.scroll(
        collection_name=collection_name,
        with_payload=True
    )

    while points:
        for point in points:
            doc_text = point.payload.get(page_content_field, "")
            metadata = point.payload.get(metadata_field, {})
            documents.append(Document(page_content=doc_text, metadata=metadata))
        
        if next_page is None:
            break

        points, next_page = client.scroll(
            collection_name=collection_name,
            with_payload=True,
            offset=next_page
        )
    
    return documents


def generate_keywords_dict(vector_store, output_json_path=None):
    keywords_dict = {}

    points, next_page = vector_store.client.scroll(
        collection_name=vector_store.collection_name,
        with_payload=True
    )

    while points:
        for point in points:
            doc_id = str(point.id)
            payload = point.payload or {}
            metadata = payload.get("metadata", payload) if isinstance(payload, dict) else {}

            key_words_val = metadata.get("key_words")
            if not key_words_val:
                continue
            
            if isinstance(key_words_val, float) and math.isnan(key_words_val): 
                continue

            for kw in key_words_val.split(","):
                kw = kw.strip().lower()
                if not kw:
                    continue

                lemmas = preprocess_lemma(
                    kw,
                    filter_stopwords=True,
                    filter_lemmatized_banned_words=True
                )
                if not lemmas:
                    continue

                processed_kw = " ".join(lemmas)
                keywords_dict.setdefault(processed_kw, []).append(doc_id)

        if next_page is None:
            break

        points, next_page = vector_store.client.scroll(
            collection_name=vector_store.collection_name,
            with_payload=True,
            offset=next_page
        )

    if output_json_path:
        with open(output_json_path, "w", encoding="utf-8") as f:
            json.dump(keywords_dict, f, ensure_ascii=False, indent=2)

    return keywords_dict


def key_words_search(words, key_words_dict, vector_store, verbose=False):
    query_text = " ".join(words).lower()
    matching_ids = set()

    for key_word, id_list in key_words_dict.items():
        if key_word in query_text:
            matching_ids.update(str(i) for i in id_list)

    if verbose:
        print(f"Key words search: Found {len(matching_ids)} matching documents")

    if not matching_ids:
        return [], ""

    docs = vector_store.get_by_ids(list(matching_ids))

    formatted_results = []
    docs_for_combination = []
    for doc in docs:
        meta = doc.metadata or {}
        if "source" in meta:
            formatted_results.append({"topic": meta["source"], "full_text": doc.page_content})
        docs_for_combination.append(doc.page_content)

    combined_text = "\n".join(docs_for_combination) if docs_for_combination else ""
    return formatted_results, combined_text


def semantic_search(query, ensemble_retriever, ensemble_k, verbose=False):
    if verbose:
        print("Semantic search: Using hybrid retrieval (BM25 + vector search)")
    
    rankings = ensemble_retriever.invoke(query)[:ensemble_k]
    
    results = [{"topic": r.metadata['source'], "full_text": r.page_content} for r in rankings]
    combined_text = "\n".join(r.page_content for r in rankings)
    return results, combined_text


def get_context(query, key_words_dict, ensemble_retriever, vector_store, ensemble_k, verbose=True):
    words = preprocess_lemma(query, filter_stopwords=False, filter_lemmatized_banned_words=False)
    query_key = " ".join(words)
    if len(words) < 3 and any(kw in query_key for kw in key_words_dict):
        if verbose:
            print("→ Using KEY WORDS SEARCH")
        return key_words_search(words, key_words_dict, vector_store, verbose)
    else:
        if verbose:
            print("→ Using SEMANTIC SEARCH")
        return semantic_search(query, ensemble_retriever, ensemble_k, verbose)
