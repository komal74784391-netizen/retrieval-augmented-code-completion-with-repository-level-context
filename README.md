# Retrieval-Augmented Code Completion with Repository-Level Context

**Accepted at NeurIPS 2026 — Accepted (poster).** Camera-ready code release.

## Overview

This repository contains the reference implementation of *Retrieval-Augmented Code Completion with Repository-Level Context*.
It includes the data pipeline, the training entry point, the evaluation
harness and the experiment configurations used for the camera-ready results.

## Repository layout

```
src/          model, data pipeline, training and evaluation code
experiments/  base and ablation configurations
notebooks/    result analysis
results/      reported metrics
tests/        unit tests
```

## Setup

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

## Reproducing the main result

```bash
python -m src.train --config experiments/base.yaml
python -m src.evaluate --config experiments/base.yaml
```

## Citation

```bibtex
@inproceedings{tanaka2026retrieval,
  title     = {Retrieval-Augmented Code Completion with Repository-Level Context},
  booktitle = {Proceedings of NeurIPS 2026},
  year      = {2026}
}
```
