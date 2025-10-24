import math
import torch
import torch.nn as nn


class HeadAttention(nn.Module):
    """
    Маскированная «голова» внимания (causal self-attention).
    Вход/выход: (B, S, E) -> (B, S, H)
    """
    def __init__(self, emb_size: int, head_size: int, max_seq_len: int):
        super().__init__()
        # Порядок Wk, Wq, Wv — как в задании лекции
        self.Wk = nn.Linear(emb_size, head_size, bias=False)
        self.Wq = nn.Linear(emb_size, head_size, bias=False)
        self.Wv = nn.Linear(emb_size, head_size, bias=False)

        # Нижнетреугольная маска (SxS) — заранее, макс. длины
        self.register_buffer("mask", torch.tril(torch.ones(max_seq_len, max_seq_len)))
        self.head_size = head_size

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, S, E)
        B, S, _ = x.shape
        Q = self.Wq(x)                  # (B, S, H)
        K = self.Wk(x)                  # (B, S, H)
        V = self.Wv(x)                  # (B, S, H)

        # Матрица внимания: (B, S, S)
        scores = Q @ K.transpose(-2, -1)
        scores = scores / math.sqrt(self.head_size)

        # Causal mask: запрещаем «смотреть вперёд»
        mask = self.mask[:S, :S]        # (S, S)
        scores = scores.masked_fill(mask == 0, float("-inf"))

        # Нормализация
        attn = torch.softmax(scores, dim=-1)  # (B, S, S)

        # Взвешенное суммирование значений
        out = attn @ V                 # (B, S, H)
        return out
