from src.data import JsonlDataset


def test_dataset_length_contract(tmp_path):
    p = tmp_path / "sample.jsonl"
    p.write_text('{"text": "hello"}\n{"text": "world"}\n', encoding="utf-8")
    ds = JsonlDataset(str(p), tokenizer=None)
    assert ds.max_len == 1024
