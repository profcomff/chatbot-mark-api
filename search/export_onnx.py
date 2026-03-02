import torch
from transformers import XLMRobertaTokenizer, XLMRobertaModel

model_name = "d0rj/e5-base-en-ru"
tokenizer = XLMRobertaTokenizer.from_pretrained(model_name)
model = XLMRobertaModel.from_pretrained(model_name)
model.eval()

class E5Wrapper(torch.nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, input_ids, attention_mask):
        outputs = self.model(input_ids=input_ids, attention_mask=attention_mask)
        last_hidden = outputs.last_hidden_state

        mask = attention_mask.unsqueeze(-1).expand(last_hidden.size()).float()
        sum_embeddings = torch.sum(last_hidden * mask, dim=1)
        sum_mask = torch.clamp(mask.sum(dim=1), min=1e-9)
        embeddings = sum_embeddings / sum_mask

        embeddings = torch.nn.functional.normalize(embeddings, p=2, dim=1)
        return embeddings

wrapper = E5Wrapper(model)

dummy_text = ["query: пример текста"]
inputs = tokenizer(dummy_text, return_tensors="pt", padding="max_length", truncation=True, max_length=512)

torch.onnx.export(
    wrapper,
    (inputs["input_ids"], inputs["attention_mask"]),
    "e5_embedder.onnx",
    input_names=["input_ids", "attention_mask"],
    output_names=["embeddings"],
    dynamic_axes={
        "input_ids": {0: "batch_size", 1: "sequence_length"},
        "attention_mask": {0: "batch_size", 1: "sequence_length"},
        "embeddings": {0: "batch_size"}
    },
    opset_version=14,
    do_constant_folding=True,
)

print("ONNX модель сохранена как e5_embedder.onnx")
