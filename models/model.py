import torch
import torch.nn as nn
from .embeddings import TokenEmbeddings, PositionalEmbeddings
from .decoder_block import DecoderBlock


class GPTLanguageModel(nn.Module):

    def __init__(
        self,
        vocab_size: int,
        max_seq_len: int,
        emb_size: int = 128,
        num_heads: int = 4,
        num_layers: int = 2,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.tok_emb = TokenEmbeddings(vocab_size, emb_size)
        self.pos_emb = PositionalEmbeddings(max_seq_len, emb_size)
        self.blocks = nn.ModuleList([
            DecoderBlock(
                emb_size=emb_size,
                num_heads=num_heads,
                max_seq_len=max_seq_len,
                dropout=dropout,
            )
            for _ in range(num_layers)
        ])
        self.ln_f = nn.LayerNorm(emb_size)
        self.lm_head = nn.Linear(emb_size, vocab_size, bias=False)

        self.max_seq_len = max_seq_len
        self.emb_size = emb_size

    def forward(self, idx: torch.Tensor) -> torch.Tensor:

        B, S = idx.shape
        tok = self.tok_emb(idx)                
        pos = self.pos_emb(S)                
        x = tok + pos                         
        for blk in self.blocks:
            x = blk(x)
        x = self.ln_f(x)
        logits = self.lm_head(x)             
        return logits

    @torch.no_grad()
    def generate(self, idx: torch.Tensor, max_new_tokens: int = 20) -> torch.Tensor:

        for _ in range(max_new_tokens):
            idx_cond = idx[:, -self.max_seq_len:] 
            logits = self(idx_cond)              
            next_token_logits = logits[:, -1, :]  
            next_id = torch.argmax(next_token_logits, dim=-1, keepdim=True)  
            idx = torch.cat([idx, next_id], dim=1)
        return idx
