"""Training entry point."""
from __future__ import annotations

import argparse

import torch
import yaml
from torch.utils.data import DataLoader

from .data import JsonlDataset
from .model import ProposedModel


def load_config(path: str):
    with open(path, "r", encoding="utf-8") as fh:
        return yaml.safe_load(fh)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    args = parser.parse_args()
    cfg = load_config(args.config)

    model = ProposedModel(hidden_size=cfg["hidden_size"], num_layers=cfg["num_layers"])
    optim = torch.optim.AdamW(model.parameters(), lr=cfg["lr"])

    loader = DataLoader(
        JsonlDataset(cfg["train_path"], tokenizer=None), batch_size=cfg["batch_size"]
    )
    for step, (ids, mask) in enumerate(loader):
        logits = model(ids, attention_mask=mask.bool())
        loss = torch.nn.functional.cross_entropy(logits, torch.zeros(ids.size(0)).long())
        optim.zero_grad()
        loss.backward()
        optim.step()
        if step % 50 == 0:
            print(f"step={step} loss={loss.item():.4f}")


if __name__ == "__main__":
    main()
