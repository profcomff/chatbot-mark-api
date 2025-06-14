import pandas as pd
from transformers import XLMRobertaTokenizer, XLMRobertaModel
import torch
import torch.nn.functional as F
from torch import Tensor


class Bertinskii:
    def __init__(self, device: str = 'cpu'):
        self.model_name = None
        self.device = device
        self.model = None
        self.tokenizer = None

    def load_tokenizer_model(self, model_name):
        try:
            self.model_name = model_name
            self.tokenizer = XLMRobertaTokenizer.from_pretrained(self.model_name) 
            self.model = XLMRobertaModel.from_pretrained(self.model_name).to(self.device) 
            self.model.eval()
            print(f"Loaded {self.model_name} on {self.device}")
        except Exception as e:
            print(f"Load error: {e}")
            raise

    def _average_pool(self, last_hidden_states, attention_mask):
        last_hidden = last_hidden_states.masked_fill(~attention_mask[..., None].bool(), 0.0)
        return last_hidden.sum(dim=1) / attention_mask.sum(dim=1)[..., None]

    def compute_embeddings(self, texts, batch_size=8):
        if not self.model or not self.tokenizer:
            raise RuntimeError("Call load_tokenizer_model() first")

        all_embeddings = []
        batches = range(0, len(texts), batch_size)

        for i in tqdm(batches, desc="Processing batches", unit="batch"):
            batch = texts[i:i+batch_size]
            inputs = self.tokenizer(
                batch,
                max_length=512,
                padding=True,
                truncation=True,
                return_tensors="pt"
            ).to(self.device)

            with torch.no_grad():
                outputs = self.model(**inputs)

            embeddings = self._average_pool(outputs.last_hidden_state, inputs['attention_mask'])
            all_embeddings.append(F.normalize(embeddings, p=2, dim=1))

        return torch.cat(all_embeddings, dim=0).cpu()

    def find_answer(self, user_question: str, answer_embs: torch.tensor, answers_df: pd.DataFrame, topk: int = 3) -> str:

        query_input = self.tokenizer(
            [user_question],
            max_length=512,
            padding=True,
            truncation=True,
            return_tensors="pt"
        ).to(self.device)

        with torch.no_grad():
            query_output = self.model(**query_input)

        query_emb = self._average_pool(query_output.last_hidden_state, query_input['attention_mask'])
        query_emb = F.normalize(query_emb, p=2, dim=1)

        scores = answer_embs @ query_emb.T
        top_indices = torch.topk(scores.flatten(), topk).indices.cpu().numpy()

        results = []
        for idx in top_indices:
            row = answers_df.iloc[idx]
            results.append({
            "topic": row['topik_name'],
            "full_text": row['answer']
        })

        return results, scores