import torch

from src.model import ProposedModel


def test_forward_shape():
    model = ProposedModel(vocab_size=128, hidden_size=32, num_layers=1)
    batch = torch.randint(0, 128, (2, 16))
    out = model(batch)
    assert out.shape == (2, 2)
