import threading
import queue


class _FakeEmbedder:
    dim = 16

    def embed(self, texts):
        return [[random.random() for _ in range(self.dim)] for _ in texts]


def _make_store(tmp_path, name="w"):
    import sys
    from pathlib import Path
    repo = Path.home() / ".hermes" / "plugins" / "lancedb"
    import importlib.util
    pkg = "lancedb_plugin_wr_test"
    spec = importlib.util.spec_from_file_location(pkg, repo / "__init__.py", submodule_search_locations=[str(repo)])
    mod = importlib.util.module_from_spec(spec)
    sys.modules[pkg] = mod
    spec.loader.exec_module(mod)
    from src.store import LanceDBStore
    d = tmp_path / name
    d.mkdir(exist_ok=True)
    return LanceDBStore(d, _FakeEmbedder())


def test_worker_survives_failed_batch(tmp_path):
    s = _make_store(tmp_path)
    calls = {"n": 0}

    def _boom(rows):
        calls["n"] += 1
        raise RuntimeError("boom")

    s.add_rows = _boom  # type: ignore[method-assign]
    for _ in range(16):
        s._queue.put_nowait({"content": "x"})

    t = threading.Thread(target=s._worker_loop, daemon=True)
    t.start()
    t.join(timeout=3)
    assert t.is_alive(), "worker thread died on a failed batch (resilience missing)"
    assert calls["n"] >= 1, "add_rows was never called"
    s._closed.set()
    t.join(timeout=1)
    assert not t.is_alive(), "worker did not stop after _closed"
