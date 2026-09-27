from src.retrieval import collapse_near_dups

def test_collapse_keeps_highest_score_per_content():
    rows = [
        {"content": "bigboy runs 2x R9700", "_relevance_score": 0.5, "agent_workspace": "shared"},
        {"content": "bigboy runs 2x R9700", "_relevance_score": 0.9, "agent_workspace": "dev"},
        {"content": "unrelated fact", "_relevance_score": 0.7, "agent_workspace": "dev"},
    ]
    out = collapse_near_dups(rows)
    assert len(out) == 2
    kept = [r for r in out if "R9700" in r["content"]][0]
    assert kept["_relevance_score"] == 0.9   # highest-score copy kept

def test_no_dups_returns_all():
    rows = [{"content": "a", "_relevance_score": 1}, {"content": "b", "_relevance_score": 2}]
    assert len(collapse_near_dups(rows)) == 2

def test_empty_is_empty():
    assert collapse_near_dups([]) == []
