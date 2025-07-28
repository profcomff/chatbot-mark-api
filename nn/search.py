import torch
from langchain_core.embeddings import Embeddings
from langchain_community.retrievers import BM25Retriever
from langchain.retrievers import EnsembleRetriever
from nltk.stem.snowball import SnowballStemmer
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
import re
from tqdm import tqdm


_STEMMER = SnowballStemmer("russian")
_PREPROCESS_REGEX = re.compile(r'[^а-яё\s]')  
_STOP_WORDS = set(stopwords.words('russian'))
_BANNED_WORDS = {'мгу', 'физфак', 'физический', 'университет'}
_STEMMED_BANNED_WORDS = {_STEMMER.stem(w) for w in _BANNED_WORDS}

def preprocess(text):
    cleaned = _PREPROCESS_REGEX.sub('', text.lower())
    
    words = word_tokenize(cleaned, language="russian")
    
    filtered_tokens = [word for word in words if word.strip() and word not in _STOP_WORDS]
    
    stemmed_words = [_STEMMER.stem(word) for word in filtered_tokens]
    
    return [word for word in stemmed_words if word not in _STEMMED_BANNED_WORDS]


class E5LangChainEmbedder(Embeddings):
    def __init__(
        self,
        tokenizer,
        model,
        device='cpu',
        embed_batch_size=8,
        chroma_batch_size=1000,
        passage_prefix="",
        query_prefix=""  
    ):
        self.tokenizer = tokenizer
        self.model = model.to(device)
        self.device = device
        self.embed_batch_size = embed_batch_size
        self.chroma_batch_size = chroma_batch_size
        self.passage_prefix = passage_prefix
        self.query_prefix = query_prefix
        self.model.eval()

    def _average_pool(self, last_hidden_states: Tensor, attention_mask: Tensor) -> Tensor:
        last_hidden = last_hidden_states.masked_fill(~attention_mask[..., None].bool(), 0.0)
        return last_hidden.sum(dim=1) / attention_mask.sum(dim=1)[..., None]

    def embed_documents(self, texts):
        if self.passage_prefix:
            prefixed_texts = [self.passage_prefix + text for text in texts]
        else:
            prefixed_texts = texts

        all_embeddings = []
        for i in tqdm(range(0, len(prefixed_texts), self.embed_batch_size),
                     desc="Вычисление эмбеддингов", unit="batch"):
            batch_texts = prefixed_texts[i:i+self.embed_batch_size]
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
        if self.query_prefix:
            prefixed_text = self.query_prefix + text
        else:
            prefixed_text = text
        return self.embed_documents([prefixed_text])[0]
    
    
def get_context(query, tokenizer, model, bm_25, vector_store, ensemble_k=5, retrivier_k=10):
    
    bm_25.k = retrivier_k
    
#     clean_query = re.sub(r'[^\w\s]', '', query)  
#     words = clean_query.split()
    
#     if len(words) < 3:
#         raiting = bm_25.invoke(query)[:ensemble_k]
    
#     else: 

    vector_retriever = vector_store.as_retriever(search_kwargs={"k": retrivier_k})

    ensemble_retriever = EnsembleRetriever(
        retrievers=[bm_25, vector_retriever],
        weights=[0.25, 0.75]
    )

    raiting = ensemble_retriever.invoke(query)[:ensemble_k]

    results = []
    for res in raiting:
        results.append({
        "topic": res.metadata['source'],
        "full_text": res.page_content
    })

    combined_text = "\n".join(doc.page_content for doc in raiting)
    
    return results, combined_text