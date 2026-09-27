import pytest
from src.store import LanceDBStore

class _FakeEmbedder:
    dim = 4
    def embed_one(self, t): return [0.0] * 4
    def embed(self, ts): return [self.embed_one(t) for t in ts]

def test_store_path_overrides_db_path(tmp_path):
    sp = tmp_path / "shared_store"
    s = LanceDBStore("/some/hermes/home", _FakeEmbedder(), store_path=str(sp))
    assert str(s.db_path) == str(sp)

def test_empty_store_path_uses_hermes_home(tmp_path):
    s = LanceDBStore(str(tmp_path / "home"), _FakeEmbedder(), store_path="")
    assert s.db_path == tmp_path / "home" / "lancedb"

def test_store_path_default_is_empty(tmp_path):
    s = LanceDBStore(str(tmp_path / "home"), _FakeEmbedder())
    assert s.db_path == tmp_path / "home" / "lancedb"
