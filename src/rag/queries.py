"""
Curated key-term query set for RAG evaluation.

Queries are loaded from config.RAG_QUERIES so the list is reviewable and
editable in one place.  This module just re-exports them for convenience
and provides a helper to iterate them.
"""

import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from config import RAG_QUERIES  # noqa: E402


def get_queries():
    """Return list of query dicts: {id, query, expected_terms}."""
    return RAG_QUERIES


def get_query_by_id(query_id: str):
    """Return a single query dict by its id."""
    for q in RAG_QUERIES:
        if q["id"] == query_id:
            return q
    raise KeyError(f"Query id '{query_id}' not found in RAG_QUERIES")


if __name__ == "__main__":
    for q in get_queries():
        print(f"  [{q['id']}] {q['query']}")
        print(f"       expected terms: {q['expected_terms']}")
