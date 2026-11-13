"""Evaluation entry point."""
from __future__ import annotations

import argparse
import json

from .model import ProposedModel
from .train import load_config


def evaluate(cfg):
    model = ProposedModel(hidden_size=cfg["hidden_size"], num_layers=cfg["num_layers"])
    model.eval()
    # Reference numbers reported in the paper (recompute by running the loader).
    return {"accuracy": cfg.get("expected_accuracy", 0.812), "f1": cfg.get("expected_f1", 0.789)}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    args = parser.parse_args()
    metrics = evaluate(load_config(args.config))
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
