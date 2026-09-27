import importlib.util, sys
from pathlib import Path

REPO = Path.home() / ".hermes" / "plugins" / "lancedb"

def _load():
    pkg = "lancedb_plugin_shared_test"
    if pkg in sys.modules:
        return sys.modules[pkg]
    spec = importlib.util.spec_from_file_location(pkg, REPO / "__init__.py", submodule_search_locations=[str(REPO)])
    mod = importlib.util.module_from_spec(spec); sys.modules[pkg] = mod; spec.loader.exec_module(mod)
    return mod

def _provider(home, identity):
    import hermes_constants
    hermes_constants.get_hermes_home = lambda: home
    mod = _load()
    p = mod.src.provider.LanceDBMemoryProvider()
    p.initialize("s", hermes_home=str(home), platform="cli", agent_context="primary",
                 agent_identity=identity, agent_workspace="hermes")
    return p

def test_two_profiles_share_one_store(tmp_path, monkeypatch):
    shared = tmp_path / "shared_store"
    # Both profiles point at the same store via a shared config.
    for name in ("p1", "p2"):
        h = tmp_path / name; h.mkdir()
        (h / "config.yaml").write_text(
            "plugins:\n  lancedb:\n"
            f"    store_path: {shared}\n"
            "    embedding: {provider: openai, model: bge-m3, base_url: http://127.0.0.1:8080/v1, api_key_env: OPENROUTER_API_KEY}\n"
            "    workspaces: {enabled: true, profiles: {default: {write: shared, read: all}}}\n"
        )
    p1 = _provider(tmp_path / "p1", "default")
    p2 = _provider(tmp_path / "p2", "default")
    assert p1.store.db_path == p2.store.db_path == shared
    row = p1.build_fact_row(content="SHAREDSTORE cross-profile fact", abstract="", category="general",
                           tags=[], provenance_turn_ids=[], source="shared-test")
    p1.store.add_row(row)
    # In production each profile is a separate process that opens its store
    # after the other profile has written. In this single-process test p2's
    # lancedb handle was opened before p1's write, so it is a stale snapshot;
    # force a fresh open to model p2 starting a new session.
    p2.store._table = None
    p2.store.open()
    res = p2.recall("cross-profile fact", mode="hybrid", kind="fact", limit=5)
    assert any("SHAREDSTORE" in r["content"] for r in res)
