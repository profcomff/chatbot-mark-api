import json
from langchain_community.retrievers import BM25Retriever
from langchain.retrievers import EnsembleRetriever
from .preprocess import preprocess_lemma

def generate_keywords_dict(vector_store, output_json_path=None):
    all_docs = vector_store.get(include=["metadatas"])
    keywords_dict = {}

    for doc_id, metadata in zip(all_docs["ids"], all_docs["metadatas"]):
        if "key_words" not in metadata or not metadata["key_words"]:
            continue

        for kw in metadata["key_words"].split(","):
            kw = kw.strip().lower()
            if not kw:
                continue

            lemmas = preprocess_lemma(kw, filter_stopwords=True, filter_lemmatized_banned_words=True)
            if not lemmas:
                continue
            processed_kw = " ".join(lemmas)

            keywords_dict.setdefault(processed_kw, []).append(doc_id)

    if output_json_path:
        with open(output_json_path, "w", encoding="utf-8") as f:
            json.dump(keywords_dict, f, ensure_ascii=False, indent=2)

    return keywords_dict


def key_words_search(words, key_words_dict, vector_store, verbose=False):
    query_text = " ".join(words)
    matching_ids = set()

    for kw in key_words_dict:
        if kw in query_text:
            matching_ids.update(key_words_dict[kw])
    if verbose:
        print(f"Key words search: Found {len(matching_ids)} matching documents")
    if not matching_ids:
        return [], ""
    
    results = vector_store.get(ids=list(matching_ids), include=["metadatas", "documents"])
    formatted_results = [{"topic": results["metadatas"][i]["source"], "full_text": results["documents"][i]}
                         for i in range(len(results["documents"])) if "source" in results["metadatas"][i]]
    
    combined_text = "\n".join(results["documents"]) if results["documents"] else ""
    return formatted_results, combined_text

def semantic_search(query, bm_25, vector_store, ensemble_k=5, retriever_k=10, verbose=False):
    if verbose:
        print("Semantic search: Using hybrid retrieval (BM25 + vector search)")
    bm_25.k = retriever_k
    vector_retriever = vector_store.as_retriever(search_kwargs={"k": retriever_k})
    ensemble_retriever = EnsembleRetriever(retrievers=[bm_25, vector_retriever], weights=[0.5, 0.5])
    rankings = ensemble_retriever.invoke(query)[:ensemble_k]
    results = [{"topic": r.metadata['source'], "full_text": r.page_content} for r in rankings]
    combined_text = "\n".join(r.page_content for r in rankings)
    return results, combined_text

def get_context(query, key_words_dict, bm_25, vector_store, ensemble_k=5, retriever_k=10, verbose=True):
    words = preprocess_lemma(query, filter_stopwords=False, filter_lemmatized_banned_words=False)
    query_key = " ".join(words)
    if len(words) < 3 and any(kw in query_key for kw in key_words_dict):
        if verbose:
            print("→ Using KEY WORDS SEARCH")
        return key_words_search(words, key_words_dict, vector_store, verbose)
    if verbose:
        print("→ Using SEMANTIC SEARCH")
    return semantic_search(query, bm_25, vector_store, ensemble_k, retriever_k, verbose)
