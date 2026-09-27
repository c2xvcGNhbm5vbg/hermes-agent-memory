from src.workspaces import resolve_workspace

CFG = {
    "workspaces": {
        "enabled": True,
        "profiles": {
            "default": {"write": "shared", "read": "all"},
            "ops": {"write": "dev", "read": ["dev", "shared"]},
            "critic": {"write": "dev", "read": ["dev", "shared"]},
            "orchestrator": {"write": "dev", "read": ["dev", "shared"]},
        },
    }
}

def test_default_profile_reads_all():
    write, read = resolve_workspace("default", CFG)
    assert write == "shared"
    assert read == []          # "all" -> empty list -> no filter -> sees everything

def test_dev_profile_write_and_read_union():
    write, read = resolve_workspace("ops", CFG)
    assert write == "dev"
    assert read == ["dev", "shared"]

def test_unmapped_profile_inherits_default():
    # no hardcoded fallback — an unmapped profile inherits the default profile's entry
    write, read = resolve_workspace("some_new_profile", CFG)
    assert write == "shared"
    assert read == []

def test_empty_identity_uses_default_profile():
    write, read = resolve_workspace("", CFG)
    assert write == "shared"
    assert read == []

def test_no_default_profile_means_no_scoping():
    # no hardcoded buckets: if 'default' itself is unmapped, fall back to no scoping
    cfg2 = {"workspaces": {"enabled": True, "profiles": {"ops": {"write": "dev", "read": ["dev", "shared"]}}}}
    write, read = resolve_workspace("x", cfg2)
    assert write == ""
    assert read == []

def test_disabled_returns_no_filter_and_empty_write():
    cfg3 = {"workspaces": {"enabled": False, "profiles": {}}}
    write, read = resolve_workspace("default", cfg3)
    assert write == ""        # stamp empty -> legacy behavior
    assert read == []         # no filter
