"""Shared helpers."""
import json
from pathlib import Path


def read_jsonl(path):
    with Path(path).open(encoding='utf-8') as handle:
        return [json.loads(line) for line in handle if line.strip()]


def batch(sequence, size):
    for start in range(0, len(sequence), size):
        yield sequence[start:start + size]


def write_metrics(rows, path):
    header = 'run,split,metric,value\n'
    body = ''.join(f"{r['run']},{r['split']},{r['metric']},{r['value']}\n" for r in rows)
    Path(path).write_text(header + body, encoding='utf-8')
