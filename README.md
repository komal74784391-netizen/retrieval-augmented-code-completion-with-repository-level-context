# Retrieval-Augmented Code Completion with Repository-Level Context

**Status:** Released — official implementation of the paper accepted at NeurIPS 2026 (Accepted (poster)).

## Overview
This repository contains the model code, data preparation helpers, experiment configurations and evaluation scripts described in the paper.

## Layout
- `src/` — model and utility code
- `experiments/` — training and ablation configurations
- `data/` — small sample of the processed dataset
- `results/` — reported evaluation metrics
- `tests/` — unit tests

## Quick start
```bash
pip install -r requirements.txt
make test
```

## Citation
```bibtex
@inproceedings{brightloom2026,
  title={Retrieval-Augmented Code Completion with Repository-Level Context},
  booktitle={NeurIPS 2026},
  year={2026}
}
```
