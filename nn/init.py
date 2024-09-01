from transformers import XLMRobertaTokenizer, XLMRobertaModel


def load_model():
    tokenizer = XLMRobertaTokenizer.from_pretrained(
        "d0rj/e5-base-en-ru", use_cache=False
    )
    model = XLMRobertaModel.from_pretrained("d0rj/e5-base-en-ru", use_cache=False)
    return tokenizer, model
