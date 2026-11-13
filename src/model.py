"""Model definition for the paper."""
from __future__ import annotations

import torch
from torch import nn


class ProposedModel(nn.Module):
    """Encoder + lightweight task head used in all main experiments."""

    def __init__(self, vocab_size: int = 32000, hidden_size: int = 768, num_layers: int = 6):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, hidden_size)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_size, nhead=8, batch_first=True
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.head = nn.Linear(hidden_size, 2)

    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor | None = None):
        x = self.embed(input_ids)
        x = self.encoder(x, src_key_padding_mask=attention_mask)
        pooled = x.mean(dim=1)
        return self.head(pooled)
