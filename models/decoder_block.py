import torch
import torch.nn as nn
from .multihead_attention import MultiHeadAttention
from .feed_forward import FeedForward


class DecoderBlock(nn.Module):

    def __init__(
        self,
        emb_size: int,
        num_heads: int,
        max_seq_len: int,
        dropout: float = 0.1,
        head_size: int | None = None,
    ):
        super().__init__()
        head_size = head_size or (emb_size // num_heads)
        self.ln1 = nn.LayerNorm(emb_size)
        self.attn = MultiHeadAttention(
            num_heads=num_heads,
            emb_size=emb_size,
            head_size=head_size,
            max_seq_len=max_seq_len,
            dropout=dropout,
        )
        self.ln2 = nn.LayerNorm(emb_size)
        self.ffn = FeedForward(emb_size, dropout=dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
    
        x = x + self.attn(self.ln1(x))
      
        x = x + self.ffn(self.ln2(x))
        return x
