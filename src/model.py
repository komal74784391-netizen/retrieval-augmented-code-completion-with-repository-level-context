"""Core model implementation for: Retrieval-Augmented Code Completion with Repository-Level Context."""
from __future__ import annotations

import torch
from torch import nn


class Encoder(nn.Module):
    def __init__(self, vocab_size: int = 32000, hidden_dim: int = 512, layers: int = 6):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, hidden_dim)
        self.blocks = nn.ModuleList(
            [
                nn.TransformerEncoderLayer(
                    d_model=hidden_dim, nhead=8, batch_first=True
                )
                for _ in range(layers)
            ]
        )
        self.norm = nn.LayerNorm(hidden_dim)

    def forward(self, input_ids, attention_mask=None):
        x = self.embed(input_ids)
        for block in self.blocks:
            x = block(x, src_key_padding_mask=attention_mask)
        return self.norm(x)


class Scorer(nn.Module):
    def __init__(self, hidden_dim: int = 512, num_labels: int = 2):
        super().__init__()
        self.dropout = nn.Dropout(0.1)
        self.head = nn.Linear(hidden_dim, num_labels)

    def forward(self, hidden_states):
        pooled = hidden_states.mean(dim=1)
        return self.head(self.dropout(pooled))


def build_model(**kwargs):
    encoder = Encoder(**{k: v for k, v in kwargs.items() if k != 'num_labels'})
    scorer = Scorer(hidden_dim=kwargs.get('hidden_dim', 512), num_labels=kwargs.get('num_labels', 2))
    return nn.Sequential(encoder, scorer)
