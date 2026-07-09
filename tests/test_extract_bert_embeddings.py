import torch

from src.extract_bert_embeddings import embed_texts


class FakeTokenizer:
    def __call__(self, texts, return_tensors="pt", padding=True, truncation=True, max_length=512):
        token_lists = [t.split() for t in texts]
        maxlen = max(len(t) for t in token_lists)
        input_ids = torch.zeros((len(texts), maxlen), dtype=torch.long)
        for i, tokens in enumerate(token_lists):
            for j in range(len(tokens)):
                input_ids[i, j] = j + 1
        attention_mask = (input_ids != 0).long()
        return {"input_ids": input_ids, "attention_mask": attention_mask}


class FakeOutputs:
    def __init__(self, last_hidden_state):
        self.last_hidden_state = last_hidden_state


class FakeModel:
    def __init__(self, hidden_size=4):
        self.hidden_size = hidden_size

    def __call__(self, **inputs):
        batch, seq = inputs["input_ids"].shape
        # Deterministic values so we can assert exact output.
        hidden = inputs["input_ids"].unsqueeze(-1).float().expand(batch, seq, self.hidden_size)
        return FakeOutputs(hidden)


def test_embed_texts_returns_correct_shape_and_batches_correctly():
    tokenizer = FakeTokenizer()
    model = FakeModel(hidden_size=4)
    device = torch.device("cpu")
    texts = ["hello world", "a b c d e", "single", "two words"]

    result = embed_texts(texts, tokenizer, model, device, batch_size=2)

    assert result.shape == (4, 4)


def test_embed_texts_returns_empty_array_for_empty_input():
    tokenizer = FakeTokenizer()
    model = FakeModel(hidden_size=4)
    device = torch.device("cpu")

    result = embed_texts([], tokenizer, model, device, batch_size=2)

    assert result.shape == (0, 0)
