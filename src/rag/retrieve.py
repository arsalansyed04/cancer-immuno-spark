"""
Hybrid retrieval: dense (FAISS/brute-force) + BM25, fused with
Reciprocal Rank Fusion (RRF).

Expose:
    retrieve(query, k, gold_dir, silver_dir, encoder)
        → list of dicts {pmid, title, score, method}

RRF formula: score(d) = Σ_r  1 / (60 + rank_r(d))
where rank_r is 1-indexed position in the sorted hit list for retriever r.
"""

import os
import pickle
import sys
from typing import List, Dict

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from config import (  # noqa: E402
    GOLD_DIR, SILVER_DIR,
    EMBEDDING_MODEL, RETRIEVAL_TOP_N, FINAL_TOP_K,
)

# RRF constant (standard value from Cormack et al., 2009)
RRF_K = 60


# ---------------------------------------------------------------------------
# Index loaders (cached after first load)
# ---------------------------------------------------------------------------

_faiss_cache   = {}
_bm25_cache    = {}
_meta_cache    = {}
_embed_cache   = {}


def _load_meta(silver_dir: str) -> pd.DataFrame:
    if silver_dir not in _meta_cache:
        import pyarrow.parquet as pq
        table = pq.read_table(silver_dir, columns=["pmid", "title", "abstract"])
        _meta_cache[silver_dir] = table.to_pandas()
    return _meta_cache[silver_dir]


def _load_embeddings_and_pmids(gold_dir: str):
    if gold_dir not in _embed_cache:
        npy_path   = os.path.join(gold_dir, "embeddings.npy")
        pmids_path = os.path.join(gold_dir, "embedding_pmids.parquet")
        emb  = np.load(npy_path)
        pmids = pd.read_parquet(pmids_path)["pmid"].tolist()
        _embed_cache[gold_dir] = (emb, pmids)
    return _embed_cache[gold_dir]


def _load_faiss(gold_dir: str):
    if gold_dir not in _faiss_cache:
        index_path = os.path.join(gold_dir, "faiss.index")
        np_path    = os.path.join(gold_dir, "faiss_bruteforce.npy")
        if os.path.exists(index_path):
            try:
                import faiss
                _faiss_cache[gold_dir] = ("faiss", faiss.read_index(index_path))
            except ImportError:
                _faiss_cache[gold_dir] = ("np", np.load(index_path.replace(".index", "_bruteforce.npy")))
        elif os.path.exists(np_path):
            _faiss_cache[gold_dir] = ("np", np.load(np_path))
        else:
            raise FileNotFoundError(
                f"No FAISS index found in {gold_dir}. Run src/rag/index.py first."
            )
    return _faiss_cache[gold_dir]


def _load_bm25(gold_dir: str):
    if gold_dir not in _bm25_cache:
        path = os.path.join(gold_dir, "bm25.pkl")
        with open(path, "rb") as f:
            _bm25_cache[gold_dir] = pickle.load(f)
    return _bm25_cache[gold_dir]


# ---------------------------------------------------------------------------
# Query encoding
# ---------------------------------------------------------------------------

def _encode_query(query: str, gold_dir: str, encoder: str) -> np.ndarray:
    """Encode a single query string to a unit-norm vector."""
    # Detect whether the index was built with LSA or sentence-transformers
    # by checking the embedding dimension
    emb, _ = _load_embeddings_and_pmids(gold_dir)
    dim = emb.shape[1]

    try:
        from sentence_transformers import SentenceTransformer
        from config import LSA_DIM
        if dim == LSA_DIM:
            raise ValueError("LSA dimension detected — skipping SentenceTransformer")
        model = SentenceTransformer(encoder)
        vec   = model.encode([query], normalize_embeddings=True, convert_to_numpy=True)
        return vec[0].astype(np.float32)
    except Exception:
        # LSA fallback: re-use stored corpus TF-IDF (or a fresh one)
        return _encode_query_lsa(query, gold_dir)


def _encode_query_lsa(query: str, gold_dir: str) -> np.ndarray:
    """Project query into the LSA space using a fitted vectoriser/SVD."""
    lsa_path = os.path.join(gold_dir, "lsa_model.pkl")
    if not os.path.exists(lsa_path):
        raise FileNotFoundError(
            f"LSA model not found at {lsa_path}.  "
            "Re-run src/rag/embed.py with --use-lsa to persist the model."
        )
    with open(lsa_path, "rb") as f:
        bundle = pickle.load(f)
    vec     = bundle["vectorizer"]
    svd     = bundle["svd"]
    from sklearn.preprocessing import normalize
    X = vec.transform([query])
    q = svd.transform(X)
    q = normalize(q, norm="l2")
    return q[0].astype(np.float32)


# ---------------------------------------------------------------------------
# Dense retrieval
# ---------------------------------------------------------------------------

def _dense_retrieve(query_vec: np.ndarray, gold_dir: str, top_n: int) -> List[Dict]:
    """Return top_n results from FAISS index as list of {pmid, score, rank}."""
    kind, index_obj = _load_faiss(gold_dir)
    _, pmids        = _load_embeddings_and_pmids(gold_dir)

    if kind == "faiss":
        scores, indices = index_obj.search(query_vec.reshape(1, -1), top_n)
        scores  = scores[0].tolist()
        indices = indices[0].tolist()
    else:
        # NumPy brute-force inner product
        sims    = index_obj @ query_vec
        idxs    = np.argsort(sims)[::-1][:top_n]
        indices = idxs.tolist()
        scores  = sims[idxs].tolist()

    results = []
    for rank_0, (idx, score) in enumerate(zip(indices, scores)):
        if idx < 0 or idx >= len(pmids):
            continue
        results.append({
            "pmid":   pmids[idx],
            "score":  float(score),
            "rank":   rank_0 + 1,
            "method": "dense",
        })
    return results


# ---------------------------------------------------------------------------
# BM25 retrieval
# ---------------------------------------------------------------------------

def _bm25_retrieve(query: str, gold_dir: str, top_n: int) -> List[Dict]:
    """Return top_n BM25 results."""
    bundle = _load_bm25(gold_dir)
    bm25   = bundle["bm25"]
    pmids  = bundle["pmids"]

    tokens = query.lower().split()
    scores = bm25.get_scores(tokens)
    top_idxs = np.argsort(scores)[::-1][:top_n]

    results = []
    for rank_0, idx in enumerate(top_idxs):
        results.append({
            "pmid":   pmids[idx],
            "score":  float(scores[idx]),
            "rank":   rank_0 + 1,
            "method": "bm25",
        })
    return results


# ---------------------------------------------------------------------------
# Reciprocal Rank Fusion
# ---------------------------------------------------------------------------

def _rrf_fuse(list_a: List[Dict], list_b: List[Dict], k: int = RRF_K) -> List[Dict]:
    """Fuse two ranked lists using RRF; return merged list sorted by RRF score."""
    scores: Dict[int, float] = {}
    sources: Dict[int, set] = {}

    for hits in (list_a, list_b):
        for item in hits:
            pmid = item["pmid"]
            r    = item["rank"]
            scores[pmid]  = scores.get(pmid, 0.0) + 1.0 / (k + r)
            sources.setdefault(pmid, set()).add(item["method"])

    fused = [
        {"pmid": pmid, "score": score,
         "method": "+".join(sorted(sources[pmid]))}
        for pmid, score in sorted(scores.items(), key=lambda x: -x[1])
    ]
    for i, item in enumerate(fused):
        item["rank"] = i + 1
    return fused


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def retrieve(
    query:      str,
    k:          int        = FINAL_TOP_K,
    gold_dir:   str        = GOLD_DIR,
    silver_dir: str        = SILVER_DIR,
    encoder:    str        = EMBEDDING_MODEL,
    top_n:      int        = RETRIEVAL_TOP_N,
) -> List[Dict]:
    """Hybrid retrieval: dense + BM25 fused with RRF.

    Returns list of dicts {pmid, title, score, method} ordered by RRF score.
    """
    query_vec     = _encode_query(query, gold_dir, encoder)
    dense_hits    = _dense_retrieve(query_vec, gold_dir, top_n)
    bm25_hits     = _bm25_retrieve(query, gold_dir, top_n)
    fused         = _rrf_fuse(dense_hits, bm25_hits)[:k]

    # Attach title/abstract from silver
    meta = _load_meta(silver_dir).set_index("pmid")
    for hit in fused:
        row = meta.loc[hit["pmid"]] if hit["pmid"] in meta.index else {}
        hit["title"]            = row.get("title",    "") if isinstance(row, dict) else row.get("title", "")
        hit["abstract_snippet"] = (
            str(row.get("abstract", ""))[:300]
            if not isinstance(row, dict)
            else ""
        )

    return fused
