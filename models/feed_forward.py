import torch
import torch.nn as nn


class FeedForward(nn.Module):
    def __init__(self, emb_size: int, dropout: float = 0.1):
        super().__init__()
        self.fc1 = nn.Linear(emb_size, 4 * emb_size)
        self.act = nn.ReLU()
        self.fc2 = nn.Linear(4 * emb_size, emb_size)
        self.drop = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.fc1(x)
        x = self.act(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x
