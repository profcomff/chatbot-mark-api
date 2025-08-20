import torch
from tqdm import tqdm
from langchain_core.embeddings import Embeddings
from transformers import XLMRobertaTokenizer, XLMRobertaModel

class E5LangChainEmbedder(Embeddings):
    def __init__(self, tokenizer, model, device='cpu', embed_batch_size=8,
                 add_prefix=False, disable_tqdm=False):
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
                      desc="Вычисление эмбеддингов", unit="batch", disable=self.disable_tqdm):
            batch_texts = texts[i:i + self.embed_batch_size]
            batch_dict = self.tokenizer(batch_texts, max_length=512, padding=True,
                                        truncation=True, return_tensors='pt').to(self.device)
            with torch.no_grad():
                outputs = self.model(**batch_dict)
                embeddings = self._average_pool(outputs.last_hidden_state, batch_dict['attention_mask'])
                embeddings = torch.nn.functional.normalize(embeddings, p=2, dim=1)
                all_embeddings.extend(embeddings.cpu().tolist())
        return all_embeddings

    def embed_query(self, text):
        if self.add_prefix:
            text = "query: " + text
        batch_dict = self.tokenizer([text], max_length=512, padding=True,
                                    truncation=True, return_tensors='pt').to(self.device)
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
