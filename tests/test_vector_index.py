import lancedb
import random
from pathlib import Path

from src.store import LanceDBStore

class _FakeEmbedder:
    dim = 16
    def embed_one(self, t): return [0.0] * 16
    def embed(self, ts): return [self.embed_one(t) for t in ts]

def _make_store(tmp_path, **kw):
    s = LanceDBStore(str(tmp_path / "home"), _FakeEmbedder(), **kw)
    s.open()
    rows = [{"id": f"r{i}", "kind": "fact", "content": f"row {i}",
             "vector": [random.random() for _ in range(16)]} for i in range(2000)]
    s.table.add(rows)
    return s

def test_vector_index_created_by_default(tmp_path):
    s = _make_store(tmp_path)
    types = [c.index_type for c in s.table.list_indices()]
    assert "IvfPq" in types

def test_vector_index_disabled(tmp_path):
    s = _make_store(tmp_path, vector_index_enabled=False)
    types = [c.index_type for c in s.table.list_indices()]
    assert "IvfPq" not in types

def test_existing_index_not_rebuilt(tmp_path):
    s = _make_store(tmp_path)
    # Second open on the same table must not raise / rebuild.
    s2 = LanceDBStore(str(tmp_path / "home"), _FakeEmbedder())
    s2.open()
    assert "IvfPq" in [c.index_type for c in s2.table.list_indices()]
