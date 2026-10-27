"""
Tests for RAG retrieval and evaluation:
- Reciprocal Rank Fusion (_rrf_fuse)
- Keyword relevance proxy (is_relevant_keyword)
- Precision@k calculation (precision_at_k)
"""

import sys
import os
import pandas as pd
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
from rag.retrieve import _rrf_fuse, RRF_K
from rag.evaluate import is_relevant_keyword, precision_at_k


def test_rrf_fuse_basic():
    # Dense list
    dense = [
        {"pmid": 101, "rank": 1, "score": 0.95, "method": "dense"},
        {"pmid": 102, "rank": 2, "score": 0.85, "method": "dense"},
        {"pmid": 103, "rank": 3, "score": 0.75, "method": "dense"},
    ]
    # BM25 list: 102 is ranked 1st, 101 is ranked 2nd, 104 is ranked 3rd
    bm25 = [
        {"pmid": 102, "rank": 1, "score": 12.5, "method": "bm25"},
        {"pmid": 101, "rank": 2, "score": 10.1, "method": "bm25"},
        {"pmid": 104, "rank": 3, "score": 8.0, "method": "bm25"},
    ]

    fused = _rrf_fuse(dense, bm25, k=60)

    # 101: 1/(60+1) + 1/(60+2) = 1/61 + 1/62 = 0.016393 + 0.016129 = 0.032522
    # 102: 1/(60+2) + 1/(60+1) = 0.032522
    # Both 101 and 102 should be ahead of 103 and 104 (which only have single list hit)
    top_pmids = [item["pmid"] for item in fused[:2]]
    assert 101 in top_pmids
    assert 102 in top_pmids

    # Check method tracking
    hit_101 = next(item for item in fused if item["pmid"] == 101)
    assert hit_101["method"] == "bm25+dense"

    hit_103 = next(item for item in fused if item["pmid"] == 103)
    assert hit_103["method"] == "dense"

    # Total unique items
    assert len(fused) == 4
    assert [item["rank"] for item in fused] == [1, 2, 3, 4]


def test_is_relevant_keyword():
    title = "Efficacy of Pembrolizumab targeting PD-1 in NSCLC"
    abstract = "In this trial, anti-PD-1 therapy improved overall survival."
    terms = ["pd-1", "pembrolizumab"]

    assert is_relevant_keyword(title, abstract, terms) is True
    assert is_relevant_keyword("No relevant title", "Nothing in abstract", terms) is False
    # Case insensitivity
    assert is_relevant_keyword("anti-PD-1", "", ["pd-1"]) is True


def test_precision_at_k():
    meta = pd.DataFrame([
        {"pmid": 1, "title": "Study on PD-1 blockade", "abstract": "T cells..."},
        {"pmid": 2, "title": "CAR-T cell engineering", "abstract": "Chimeric antigen..."},
        {"pmid": 3, "title": "Targeting PD-L1 pathway", "abstract": "Tumor microenvironment..."},
        {"pmid": 4, "title": "General review on cardiology", "abstract": "Heart failure..."},
    ])

    hits = [
        {"pmid": 1},
        {"pmid": 3},
        {"pmid": 4},
        {"pmid": 2},
    ]

    terms = ["pd-1", "pd-l1"]
    # Top 2 hits (pmid 1 and 3) both contain expected terms -> 2/2 = 1.0
    p2 = precision_at_k(hits, terms, k=2, meta=meta)
    assert p2 == 1.0

    # Top 3 hits (pmid 1, 3, 4) -> 2 out of 3 -> 2/3
    p3 = precision_at_k(hits, terms, k=3, meta=meta)
    assert abs(p3 - (2.0 / 3.0)) < 1e-4
