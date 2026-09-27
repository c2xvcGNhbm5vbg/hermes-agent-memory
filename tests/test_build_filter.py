from src.store import build_filter

def test_read_buckets_emits_in_clause():
    f = build_filter(kind="fact", read_buckets=["dev", "shared"])
    assert "agent_workspace IN ('dev', 'shared')" in f
    assert "kind = 'fact'" in f

def test_empty_read_buckets_no_workspace_clause():
    f = build_filter(kind="fact", read_buckets=[])
    assert "agent_workspace" not in f
    assert "kind = 'fact'" in f

def test_no_read_buckets_arg_is_backward_compatible():
    # old single-workspace callers: pass read_buckets=None -> no workspace clause
    f = build_filter(kind="fact", read_buckets=None)
    assert "agent_workspace" not in f
