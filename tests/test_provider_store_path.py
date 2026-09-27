import importlib.util, sys
from pathlib import Path

def _load():
    repo = Path.home() / ".hermes" / "plugins" / "lancedb"
    pkg = "lancedb_plugin_sp_test"
    spec = importlib.util.spec_from_file_location(pkg, repo / "__init__.py", submodule_search_locations=[str(repo)])
    mod = importlib.util.module_from_spec(spec); sys.modules[pkg] = mod; spec.loader.exec_module(mod)
    return mod

def test_provider_passes_store_path_to_store(monkeypatch, tmp_path):
    # Point the config loader at a fake home whose config sets store_path.
    home = tmp_path / "hh"; home.mkdir()
    (home / "config.yaml").write_text(
        "plugins:\n  lancedb:\n    store_path: " + str(tmp_path / "shared") + "\n"
    )
    import hermes_constants
    monkeypatch.setattr(hermes_constants, "get_hermes_home", lambda: home)
    _load()
    from lancedb_plugin_sp_test.src.provider import LanceDBMemoryProvider
    p = LanceDBMemoryProvider()
    # Don't call initialize (it opens a table); just check the store path wiring.
    sp = (p._config.get("store_path") or "")
    store = p.store  # triggers the store property
    assert store.store_path == sp
    assert str(store.db_path) == str(tmp_path / "shared")
