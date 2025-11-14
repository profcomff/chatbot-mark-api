import torch
from langchain_core.embeddings import Embeddings
from tqdm import tqdm
from transformers import XLMRobertaModel, XLMRobertaTokenizer


class E5LangChainEmbedder(Embeddings):
    """
    Кастомная модель для получения эмбедингов в пайплайне LangChain.
    """
    def __init__(self, tokenizer, model, device='cpu', embed_batch_size=8, add_prefix=False, disable_tqdm=False):
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
        for i in tqdm(
            range(0, len(texts), self.embed_batch_size),
            desc="Вычисление эмбеддингов",
            unit="batch",
            disable=self.disable_tqdm,
        ):
            batch_texts = texts[i : i + self.embed_batch_size]
            batch_dict = self.tokenizer(
                batch_texts, max_length=512, padding=True, truncation=True, return_tensors='pt'
            ).to(self.device)
            with torch.no_grad():
                outputs = self.model(**batch_dict)
                embeddings = self._average_pool(outputs.last_hidden_state, batch_dict['attention_mask'])
                embeddings = torch.nn.functional.normalize(embeddings, p=2, dim=1)
                all_embeddings.extend(embeddings.cpu().tolist())
        return all_embeddings

    def embed_query(self, text):
        if self.add_prefix:
            text = "query: " + text
        batch_dict = self.tokenizer([text], max_length=512, padding=True, truncation=True, return_tensors='pt').to(
            self.device
        )
        with torch.no_grad():
            outputs = self.model(**batch_dict)
            embeddings = self._average_pool(outputs.last_hidden_state, batch_dict['attention_mask'])
            embeddings = torch.nn.functional.normalize(embeddings, p=2, dim=1)
            return embeddings.cpu().tolist()[0]


def init_embedder():
    tokenizer = XLMRobertaTokenizer.from_pretrained("d0rj/e5-base-en-ru", use_cache=False)
    model = XLMRobertaModel.from_pretrained("d0rj/e5-base-en-ru", use_cache=False)

    return E5LangChainEmbedder(
        tokenizer=tokenizer,
        model=model,
    )


class FilteredEnsembleRetriever:
    def __init__(self, semantic_model, bm25, retriever_k=20, ensemble_k=5, weights=[0.5, 0.5], c=60):
        self.semantic_model = semantic_model  # like vector_store
        self.bm25 = bm25
        self.retriever_k = retriever_k
        self.ensemble_k = ensemble_k
        self.weights = weights
        self.bm25.k = retriever_k
        self.c = 60
        self.relevance_score = 0.8

    def _make_relevant_dict(self, documents):
        relevance_dict = {}
        for i in range(len(documents)):
            relevance_dict[documents[i][0].metadata['number_id']] = documents[i][1] > self.relevance_score
        return relevance_dict

    @staticmethod
    def _make_rank_dict(documents):
        rank_dict = {}
        for i in range(len(documents)):
            rank_dict[documents[i].metadata['number_id']] = i + 1
        return rank_dict

    @staticmethod
    def _score_2_rank(dict_score):
        sorted_items = sorted(dict_score.items(), key=lambda x: x[1], reverse=True)
        return {doc_id: rank + 1 for rank, (doc_id, _) in enumerate(sorted_items)}

    def _rank_fusion(self, ranks1, ranks2):
        all_docs = set(ranks1.keys()) | set(ranks2.keys())
        fused_scores = {}

        for doc_id in all_docs:
            rank1 = ranks1.get(doc_id, float('inf'))
            rank2 = ranks2.get(doc_id, float('inf'))

            score1 = self.weights[0] / (rank1 + self.c) if rank1 != float('inf') else 0
            score2 = self.weights[1] / (rank2 + self.c) if rank2 != float('inf') else 0

            fused_scores[doc_id] = score1 + score2

        return fused_scores

    @staticmethod
    def _filter_relevance(fusion_score, relevance_dict):
        return {
            doc_id: score
            for doc_id, score in fusion_score.items()
            if doc_id in relevance_dict and relevance_dict[doc_id]
        }

    def _get_documents_by_ids(self, doc_ids, semantic_docs, bm25_docs):
        id_to_doc = {}

        for doc, _ in semantic_docs:
            id_to_doc[doc.metadata['number_id']] = doc

        for doc in bm25_docs:
            doc_id = doc.metadata['number_id']
            if doc_id not in id_to_doc:
                id_to_doc[doc_id] = doc

        return [id_to_doc[doc_id] for doc_id in doc_ids if doc_id in id_to_doc]

    def invoke(self, query):
        docs_with_score = self.semantic_model.similarity_search_with_score(query=query, k=self.retriever_k)
        relevance_dict = self._make_relevant_dict(docs_with_score)

        bm25_docs = self.bm25.invoke(query)
        bm25_rank = self._make_rank_dict(bm25_docs)

        semantic_rank = self._score_2_rank(relevance_dict)

        fusion_score = self._rank_fusion(semantic_rank, bm25_rank)

        filtered_scores = self._filter_relevance(fusion_score, relevance_dict)
        sorted_doc_ids = sorted(filtered_scores.keys(), key=lambda x: filtered_scores[x], reverse=True)
        top_doc_ids = sorted_doc_ids[: self.ensemble_k]

        final_docs = self._get_documents_by_ids(top_doc_ids, docs_with_score, bm25_docs)

        return final_docs
