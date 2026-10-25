"""
Build and persist a vector index (FAISS) and a BM25 index over the
embedded silver-table abstracts.

Outputs:
  gold/faiss.index   – FAISS flat inner-product index (or saved as .npy
                        brute-force if faiss is unavailable)
  gold/bm25.pkl      – serialised BM25Okapi object from rank_bm25

Run with:
  python src/rag/index.py \\
      --silver  output/silver/pubmed_filtered \\
      --gold    output/gold
"""

import argparse
import os
import pickle
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from config import GOLD_DIR, SILVER_DIR  # noqa: E402


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def build_arg_parser():
    p = argparse.ArgumentParser(
        description="Build FAISS vector index and BM25 index."
    )
    p.add_argument("--silver", default=SILVER_DIR)
    p.add_argument("--gold",   default=GOLD_DIR)
    return p


# ---------------------------------------------------------------------------
# FAISS index
# ---------------------------------------------------------------------------

def build_faiss_index(embeddings: np.ndarray, gold_dir: str):
    """Build an inner-product (cosine on L2-normalised vectors) FAISS index."""
    index_path = os.path.join(gold_dir, "faiss.index")
    try:
        import faiss
        dim   = embeddings.shape[1]
        index = faiss.IndexFlatIP(dim)  # inner product = cosine for unit vectors
        index.add(embeddings)
        faiss.write_index(index, index_path)
        print(f"[index] FAISS flat-IP index built ({index.ntotal} vectors) -> {index_path}")
    except ImportError:
        # Brute-force fallback: just save the embedding matrix
        np_path = os.path.join(gold_dir, "faiss_bruteforce.npy")
        np.save(np_path, embeddings)
        print(f"[index] faiss not installed; saved brute-force matrix -> {np_path}")


# ---------------------------------------------------------------------------
# BM25 index
# ---------------------------------------------------------------------------

def build_bm25_index(silver_dir: str, gold_dir: str) -> list:
    """Build a BM25Okapi index over tokenised abstract+title text."""
    import pyarrow.parquet as pq
    from rank_bm25 import BM25Okapi

    table = pq.read_table(silver_dir, columns=["pmid", "title", "abstract"])
    df    = table.to_pandas()
    df["text"] = (
        df["title"].fillna("") + " " + df["abstract"].fillna("")
    ).str.lower()

    # Simple whitespace tokeniser (matches embed.py's text preparation)
    corpus = [text.split() for text in df["text"].tolist()]
    bm25   = BM25Okapi(corpus)

    bm25_path = os.path.join(gold_dir, "bm25.pkl")
    with open(bm25_path, "wb") as f:
        pickle.dump({"bm25": bm25, "pmids": df["pmid"].tolist()}, f)
    print(f"[index] BM25 index built ({len(corpus)} documents) -> {bm25_path}")
    return df["pmid"].tolist()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def build_indexes(silver_dir: str, gold_dir: str):
    """Build both indexes from cached embeddings + silver table."""
    os.makedirs(gold_dir, exist_ok=True)

    npy_path = os.path.join(gold_dir, "embeddings.npy")
    if not os.path.exists(npy_path):
        raise FileNotFoundError(
            f"embeddings.npy not found at {npy_path}.  "
            "Run src/rag/embed.py first."
        )
    embeddings = np.load(npy_path)
    build_faiss_index(embeddings, gold_dir)
    build_bm25_index(silver_dir, gold_dir)


def main():
    args = build_arg_parser().parse_args()
    build_indexes(args.silver, args.gold)


if __name__ == "__main__":
    main()
