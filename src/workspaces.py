"""Resolve a profile (agent identity) to a write bucket + read-bucket list.

Pure logic, no lancedb import — unit-testable in isolation. The core passes
``agent_identity`` (the profile name); we map it through the ``workspaces``
config. An empty identity is treated as the ``default`` profile. ``read: all``
yields an empty list, which the filter treats as "no workspace filter"
(see everything). When ``workspaces.enabled`` is false, we return ("", []) to
preserve the legacy no-filter behavior.

No bucket names are hardcoded: buckets are emergent — they exist only when a
profile's ``write``/``read`` references them. An unmapped profile inherits the
``default`` profile's entry; if ``default`` itself is unmapped, the fallback is
no scoping (empty write, no filter).
"""
from __future__ import annotations

from typing import Any, Dict, List, Tuple


def _read_buckets(read: Any) -> List[str]:
    """Normalize a configured read value to a list. 'all' -> [] (no filter)."""
    if read == "all":
        return []
    if isinstance(read, str):
        return [read]
    if isinstance(read, (list, tuple)):
        return [str(r) for r in read if r]
    return []


def _entry_buckets(entry: Dict[str, Any]) -> Tuple[str, List[str]]:
    return str(entry.get("write") or ""), _read_buckets(entry.get("read"))


def resolve_workspace(profile: str, config: Dict[str, Any]) -> Tuple[str, List[str]]:
    """Return (write_bucket, read_buckets) for ``profile`` under ``config``."""
    ws = (config.get("workspaces") or {})
    if not ws.get("enabled", True):
        return "", []
    name = (profile or "default").strip() or "default"
    profiles = ws.get("profiles") or {}
    entry = profiles.get(name)
    if not isinstance(entry, dict):
        # No hardcoded fallback: inherit the default profile's entry, else no scoping.
        default_entry = profiles.get("default")
        if isinstance(default_entry, dict):
            return _entry_buckets(default_entry)
        return "", []
    return _entry_buckets(entry)
