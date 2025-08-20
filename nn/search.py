import torch
from langchain_core.embeddings import Embeddings
from langchain_community.retrievers import BM25Retriever
from langchain.retrievers import EnsembleRetriever
from pymystem3 import Mystem #added
from nltk.stem.snowball import SnowballStemmer
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
import re
from tqdm import tqdm
import json

_MYSTEM = Mystem() #added 
_STEMMER = SnowballStemmer("russian")
_PREPROCESS_REGEX = re.compile(r'[^а-яё\s]')  
_STOP_WORDS = set(stopwords.words('russian'))
_BANNED_WORDS = {'мгу', 'физфак', 'физический', 'университет'}
_STEMMED_BANNED_WORDS = {_STEMMER.stem(w) for w in _BANNED_WORDS}
_LEMMATIZED_BANNED_WORDS = {lemma.strip() for w in _BANNED_WORDS for lemma in _MYSTEM.lemmatize(w)}

def preprocess(text, filter_stopwords=False, filter_stemmed_banned_words=False):
    cleaned = _PREPROCESS_REGEX.sub('', text.lower())
    words = word_tokenize(cleaned, language="russian")

    if filter_stopwords:
        filtered_tokens = [word for word in words if word.strip() and word not in _STOP_WORDS]
    else:
        filtered_tokens = [word for word in words if word.strip()]

    stemmed_words = [_STEMMER.stem(word) for word in filtered_tokens]
    
    if filter_stemmed_banned_words:
        return [word for word in stemmed_words if word not in _STEMMED_BANNED_WORDS]
    else:
        return stemmed_words

def preprocess_lemma(text, filter_stopwords=False, filter_lemmatized_banned_words=False):
    cleaned = _PREPROCESS_REGEX.sub('', text.lower())
    words = word_tokenize(cleaned, language="russian")

    if filter_stopwords:
        words = [w for w in words if w.strip() and w not in _STOP_WORDS]
    else:
        words = [w for w in words if w.strip()]

    lemmas = [_MYSTEM.lemmatize(w)[0].strip() for w in words]

    if filter_lemmatized_banned_words:
        return [w for w in lemmas if w not in _LEMMATIZED_BANNED_WORDS]
    else:
        return lemmas


class E5LangChainEmbedder(Embeddings):
    def __init__(
        self,
        tokenizer,
        model,
        device: str = 'cpu',
        embed_batch_size: int = 8,
        add_prefix: bool = False,
        disable_tqdm: bool = False
    ):
        self.tokenizer = tokenizer
        self.model = model.to(device)
        self.device = device
        self.embed_batch_size = embed_batch_size
        self.add_prefix = add_prefix
        self.disable_tqdm = disable_tqdm
        self.model.eval()

    def _average_pool(self, last_hidden_states, attention_mask):
        last_hidden = last_hidden_states.masked_fill(~attention_mask[..., None].bool(), 0.0)
        return last_hidden.sum(dim=1) / attention_mask.sum(dim=1)[..., None]

    def embed_documents(self, texts):
        if self.add_prefix:
            texts = ["passage: " + t for t in texts]

        all_embeddings = []
        for i in tqdm(range(0, len(texts), self.embed_batch_size),
                     desc="Вычисление эмбеддингов", unit="batch",
                     disable=self.disable_tqdm):
            batch_texts = texts[i:i + self.embed_batch_size]
            batch_dict = self.tokenizer(
                batch_texts,
                max_length=512,
                padding=True,
                truncation=True,
                return_tensors='pt'
            ).to(self.device)

            with torch.no_grad():
                outputs = self.model(**batch_dict)
                embeddings = self._average_pool(
                    outputs.last_hidden_state,
                    batch_dict['attention_mask']
                )
                embeddings = torch.nn.functional.normalize(embeddings, p=2, dim=1)
                all_embeddings.extend(embeddings.cpu().tolist())

        return all_embeddings

    def embed_query(self, text):
        if self.add_prefix:
            text = "query: " + text
        
        batch_dict = self.tokenizer(
            [text],
            max_length=512,
            padding=True,
            truncation=True,
            return_tensors='pt'
        ).to(self.device)
        
        with torch.no_grad():
            outputs = self.model(**batch_dict)
            embeddings = self._average_pool(
                outputs.last_hidden_state,
                batch_dict['attention_mask']
            )
            embeddings = torch.nn.functional.normalize(embeddings, p=2, dim=1)
            return embeddings.cpu().tolist()[0]
    
def generate_keywords_dict(vector_store, output_json_path=None):
    
    all_docs = vector_store.get(include=["metadatas"])
    keywords_dict = {}
    
    for doc_id, metadata in zip(all_docs["ids"], all_docs["metadatas"]):
        if "key_words" not in metadata:
            continue
            
        key_words_str = metadata["key_words"]
        if not key_words_str or str(key_words_str).strip() == "":
            continue
        
        keywords = [
            kw.strip().lower() 
            for kw in str(key_words_str).split(",") 
            if kw.strip()
        ]
        
        for kw in keywords:
            if kw not in keywords_dict:
                keywords_dict[kw] = []
            keywords_dict[kw].append(doc_id)
    
    print(f"Create key_words_dict")
    
    if output_json_path:
        with open(output_json_path, 'w', encoding='utf-8') as f:
            json.dump(keywords_dict, f, ensure_ascii=False, indent=2)
            print(f"Saved key_words_dict: {output_json_path}")
    
    return keywords_dict
    
def key_words_search(words, key_words_dict, vector_store, verbose=False):
    query_text = " ".join(words)  # склеиваем леммы обратно в строку
    matching_ids = set()

    for kw in key_words_dict:
        if kw in query_text:
            matching_ids.update(key_words_dict[kw])

    if verbose:
        print(f"Key words search: Found {len(matching_ids)} matching documents")

    if not matching_ids:
        return [], ""

    results = vector_store.get(
        ids=list(matching_ids),
        include=["metadatas", "documents"]
    )

    formatted_results = []
    for i in range(len(results["documents"])):
        if i < len(results["metadatas"]) and "source" in results["metadatas"][i]:
            formatted_results.append({
                "topic": results["metadatas"][i]["source"],
                "full_text": results["documents"][i]
            })

    combined_text = "\n".join(results["documents"]) if results["documents"] else ""

    return formatted_results, combined_text


def semantic_search(query, bm_25, vector_store, ensemble_k=5, retriever_k=10, verbose=False):
    if verbose:
        print("Semantic search: Using hybrid retrieval (BM25 + vector search)")
    
    bm_25.k = retriever_k
    vector_retriever = vector_store.as_retriever(search_kwargs={"k": retriever_k})

    ensemble_retriever = EnsembleRetriever(
        retrievers=[bm_25, vector_retriever],
        weights=[0.5, 0.5]
    )

    rankings = ensemble_retriever.invoke(query)[:ensemble_k]

    results = []
    for res in rankings:
        results.append({
            "topic": res.metadata['source'],
            "full_text": res.page_content
        })

    combined_text = "\n".join(doc.page_content for doc in rankings)
    
    return results, combined_text


def get_context(query, key_words_dict, bm_25, vector_store, 
                ensemble_k=5, retriever_k=10, verbose=True):

    words = preprocess_lemma(query, filter_stopwords=False, filter_lemmatized_banned_words=False) #lemma!!!
    query_key = " ".join(words)
    if len(words) < 3 and any(kw in query_key for kw in key_words_dict):
        if verbose:
            print("→ Using KEY WORDS SEARCH")
        return key_words_search(words, key_words_dict, vector_store, verbose)
    
    if verbose:
        print("→ Using SEMANTIC SEARCH")
    return semantic_search(query, bm_25, vector_store, ensemble_k, retriever_k, verbose)