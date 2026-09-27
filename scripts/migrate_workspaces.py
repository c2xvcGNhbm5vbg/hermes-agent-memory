#!/usr/bin/env python3
"""One-time migration: rewrite the agent_workspace column of existing rows.

LanceDB has no in-place UPDATE, so this deletes each matching row and
re-adds it with the new agent_workspace (and a recomputed content_hash).
Idempotent: a row whose workspace is already non-empty and not 'hermes' is
left alone.

Mapping (default):
    ""        -> shared
    "hermes"  -> shared

Usage:
    python scripts/migrate_workspaces.py --db ~/.hermes/lancedb --dry-run
    python scripts/migrate_workspaces.py --db ~/.hermes/lancedb
"""
from __future__ import annotations
import argparse
from pathlib import Path

TABLE = "memories"
MAPPING = {"": "shared", "hermes": "shared"}


def _recompute_hash(content: str, workspace: str, kind: str) -> str:
    import hashlib
    normalized = " ".join((content or "").strip().lower().split())
    payload = f"{kind}\0{workspace}\0{normalized}"
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--db", default=str(Path.home() / ".hermes" / "lancedb"))
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    import lancedb
    db = lancedb.connect(args.db)
    t = db.open_table(TABLE)
    rows = t.to_arrow().to_pylist()

    to_update = [r for r in rows if r.get("agent_workspace", "") in MAPPING]
    print(f"total rows: {len(rows)}")
    print(f"rows to remap: {len(to_update)}")
    for r in to_update:
        print(f"  {r['agent_workspace']!r} -> {MAPPING[r['agent_workspace']]!r} | {str(r.get('content'))[:50]}")

    if args.dry_run:
        print("\n[dry-run] no changes written")
        return

    if not to_update:
        print("\nnothing to do")
        return

    for r in to_update:
        new_ws = MAPPING[r.get("agent_workspace", "")]
        t.delete(f"id = '{r['id']}'")
        r["agent_workspace"] = new_ws
        r["content_hash"] = _recompute_hash(r.get("content", ""), new_ws, r.get("kind", ""))
        t.add([r])
    print(f"\nmigrated {len(to_update)} rows")


if __name__ == "__main__":
    main()
