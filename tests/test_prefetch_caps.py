"""Prefetch caps (max_items / max_chars) are user-configurable.

The auto-injected memory block is bounded: N items, each truncated to
M chars. Both caps come from the ``prefetch`` config block (defaults
5 / 500) and must be overridable via ~/.hermes/config.yaml.
"""
import pytest

from src.provider import LanceDBMemoryProvider
from src.retrieval import format_prefetch


def _row(i, length):
    return {"id": f"r{i}", "content": "x" * length, "category": "test"}


def test_format_prefetch_max_chars_truncates():
    rows = [_row(0, 101)]
    out = format_prefetch(rows, max_items=5, max_chars=100)
    # Truncated to max_chars - 3 + "..."
    assert "x" * 97 in out
    assert "x" * 98 not in out
    assert out.rstrip().endswith("...")


def test_format_prefetch_max_chars_no_truncation_when_short():
    rows = [_row(0, 50)]
    out = format_prefetch(rows, max_items=5, max_chars=100)
    assert "x" * 50 in out
    assert "..." not in out


def test_format_prefetch_max_items_limits_rows():
    rows = [_row(i, 10) for i in range(10)]
    out = format_prefetch(rows, max_items=2, max_chars=500)
    assert out.count("- (") == 2


def test_format_prefetch_defaults_unchanged():
    # Default behavior: 5 items, 500-char cap (backward compatible).
    rows = [_row(i, 1000) for i in range(10)]
    out = format_prefetch(rows)
    assert out.count("- (") == 5
    assert "x" * 497 in out
    assert "x" * 500 not in out


@pytest.fixture
def fake_home(tmp_path):
    home = tmp_path / "hermes_home"
    home.mkdir()
    (home / "config.yaml").write_text(
        "plugins:\n  lancedb:\n"
        "    prefetch:\n"
        "      max_items: 2\n"
        "      max_chars: 100\n"
    )
    return home


@pytest.fixture
def no_store_worker(monkeypatch):
    from src.store import LanceDBStore
    monkeypatch.setattr(LanceDBStore, "start_worker", lambda self: None)


def test_provider_prefetch_uses_config_caps(fake_home, monkeypatch, no_store_worker):
    import hermes_constants
    monkeypatch.setattr(hermes_constants, "get_hermes_home", lambda: fake_home)

    p = LanceDBMemoryProvider()
    p.initialize(session_id="s1", agent_identity="default", agent_workspace="hermes")

    captured = {}

    def _fake_recall(query, *, mode, kind, limit):
        return [_row(i, 1000) for i in range(10)]

    def _fake_format(rows, *, max_items, max_chars):
        captured["max_items"] = max_items
        captured["max_chars"] = max_chars
        return "ok"

    monkeypatch.setattr(p, "recall", _fake_recall)
    monkeypatch.setattr("src.provider.format_prefetch", _fake_format)

    p.queue_prefetch("query", session_id="s1")
    p._prefetch_thread.join(timeout=5)

    assert captured == {"max_items": 2, "max_chars": 100}
