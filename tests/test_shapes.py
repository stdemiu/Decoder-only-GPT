import torch
from models.model import GPTLanguageModel


def test_logits_shape():
    vocab_size = 50
    max_seq_len = 24
    model = GPTLanguageModel(vocab_size, max_seq_len, emb_size=64, num_heads=4, num_layers=2, dropout=0.0)
    x = torch.randint(0, vocab_size, (2, 12))
    logits = model(x)
    assert logits.shape == (2, 12, vocab_size)


def test_generate_len():
    vocab_size = 30
    max_seq_len = 16
    model = GPTLanguageModel(vocab_size, max_seq_len, emb_size=32, num_heads=4, num_layers=1, dropout=0.0)
    start = torch.randint(0, vocab_size, (1, 5))
    out = model.generate(start, max_new_tokens=7)
    assert out.shape[1] == 12 
