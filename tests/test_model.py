from src.utils import batch


def test_batch_covers_all_items():
    items = list(range(10))
    chunks = list(batch(items, 4))
    assert [x for chunk in chunks for x in chunk] == items
