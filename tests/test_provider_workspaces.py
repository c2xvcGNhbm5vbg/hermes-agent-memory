import pytest

from src.provider import LanceDBMemoryProvider
from src.store import LanceDBStore


@pytest.fixture
def fake_home(tmp_path):
    home = tmp_path / "hermes_home"
    home.mkdir()
    (home / "config.yaml").write_text(
        "plugins:\n  lancedb:\n    workspaces:\n"
        "      enabled: true\n"
        "      profiles:\n"
        "        default: {write: shared, read: all}\n"
        "        ops: {write: dev, read: [dev, shared]}\n"
        "        test: {write: dev, read: [dev, shared]}\n"
    )
    return home


@pytest.fixture
def no_store_worker(monkeypatch):
    # initialize() calls self.store.start_worker() which opens a lancedb table.
    # In the shadowing test env (plugin repo top-level __init__.py shadows the
    # venv lancedb) that import fails. The bucket derivation happens BEFORE the
    # start_worker call, so we stub it to a no-op and only assert the derived
    # bucket attributes.
    monkeypatch.setattr(LanceDBStore, "start_worker", lambda self: None)


def test_provider_derives_buckets_from_identity(fake_home, monkeypatch, no_store_worker):
    import hermes_constants
    monkeypatch.setattr(hermes_constants, "get_hermes_home", lambda: fake_home)
    p = LanceDBMemoryProvider()
    p.initialize(session_id="s1", agent_identity="test", agent_workspace="hermes")
    assert p._write_bucket == "dev"
    assert p._read_buckets == ["dev", "shared"]


def test_provider_default_profile_reads_all(fake_home, monkeypatch, no_store_worker):
    import hermes_constants
    monkeypatch.setattr(hermes_constants, "get_hermes_home", lambda: fake_home)
    p = LanceDBMemoryProvider()
    p.initialize(session_id="s1", agent_identity="default", agent_workspace="hermes")
    assert p._write_bucket == "shared"
    assert p._read_buckets == []
