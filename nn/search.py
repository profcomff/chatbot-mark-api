import pandas as pd
from transformers import XLMRobertaTokenizer, XLMRobertaModel
import torch
import torch.nn.functional as F
from torch import Tensor


def make_query_emb(text, model, tokenizer):
    query = tokenizer(
        [text], max_length=512, padding=True, truncation=True, return_tensors="pt"
    )
    outputs = model(**query)
    embeddings = average_pool(outputs.last_hidden_state, query["attention_mask"])
    embeddings = F.normalize(embeddings, p=2, dim=1)
    return embeddings


def average_pool(last_hidden_states: Tensor, attention_mask: Tensor) -> Tensor:
    last_hidden = last_hidden_states.masked_fill(~attention_mask[..., None].bool(), 0.0)
    return last_hidden.sum(dim=1) / attention_mask.sum(dim=1)[..., None]


def get_similar_context(data, query_emb, context_embs, topk=15):
    scores = context_embs @ query_emb.T
    index = torch.topk(scores.T, topk).indices
    similar_contex = list(data.iloc[index[0]])
    answer = ""
    for number, ans in enumerate(similar_contex):
        answer += f" {number} " + ans
    return answer
