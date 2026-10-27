"""
Embed PubMed silver-table abstracts to dense vectors.

Primary encoder: biomedical sentence-transformer (configured in config.py).
Fallback:        200-dim LSA (TruncatedSVD over TF-IDF) — runs fully offline
                 without downloading any model weights.

Outputs:
  gold/embeddings.npy          – float32 array shape (N, D)
  gold/embedding_pmids.parquet – pmid ordering matching rows of embeddings.npy

Run with:
  python src/rag/embed.py \\
      --silver output/silver/pubmed_filtered \\
      --output output/gold \\
      [--encoder pritamdeka/S-PubMedBert-MS-MARCO] \\
      [--use-lsa]
"""

import argparse
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from config import (  # noqa: E402
    GOLD_DIR, SILVER_DIR,
    EMBEDDING_MODEL, LSA_DIM,
)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def build_arg_parser():
    p = argparse.ArgumentParser(
        description="Encode silver-table abstracts to dense embeddings."
    )
    p.add_argument("--silver",  default=SILVER_DIR,
                   help="Parquet directory (silver/pubmed_filtered)")
    p.add_argument("--output",  default=GOLD_DIR,
                   help="Output directory (gold/)")
    p.add_argument("--encoder", default=EMBEDDING_MODEL,
                   help="sentence-transformers model name or path")
    p.add_argument("--use-lsa", action="store_true",
                   help="Force LSA fallback even if sentence-transformers is available")
    p.add_argument("--batch",   type=int, default=64,
                   help="Batch size for sentence-transformer encoding")
    return p


# ---------------------------------------------------------------------------
# Loader
# ---------------------------------------------------------------------------

def load_silver(silver_dir: str) -> pd.DataFrame:
    """Load the silver parquet table into a pandas DataFrame.

    Only the columns needed for encoding are read.
    """
    import pyarrow.parquet as pq
    table = pq.read_table(
        silver_dir,
        columns=["pmid", "title", "abstract"],
    )
    df = table.to_pandas()
    df["text"] = (
        df["title"].fillna("") + ". " + df["abstract"].fillna("")
    ).str.strip()
    return df


# ---------------------------------------------------------------------------
# Encoders
# ---------------------------------------------------------------------------

def encode_with_sentence_transformers(texts, model_name: str, batch_size: int) -> np.ndarray:
    """Encode texts with a sentence-transformers model (L2-normalised)."""
    from sentence_transformers import SentenceTransformer
    print(f"[embed] Loading sentence-transformer: {model_name}")
    model = SentenceTransformer(model_name)
    print(f"[embed] Encoding {len(texts)} texts (batch={batch_size}) ...")
    embeddings = model.encode(
        texts,
        batch_size=batch_size,
        show_progress_bar=True,
        normalize_embeddings=True,   # L2-normalise for inner-product FAISS
        convert_to_numpy=True,
    )
    return embeddings.astype(np.float32)


def encode_with_lsa(texts, n_components: int = LSA_DIM) -> np.ndarray:
    """Fallback: 200-dim TruncatedSVD (LSA) over TF-IDF, L2-normalised.

    No network access required; runs entirely offline.
    """
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.decomposition import TruncatedSVD
    from sklearn.preprocessing import normalize

    print(f"[embed] Using LSA fallback ({n_components} dims) ...")
    vec = TfidfVectorizer(
        max_features=20_000,
        min_df=2,
        max_df=0.95,
        sublinear_tf=True,
        strip_accents="unicode",
        analyzer="word",
        token_pattern=r"(?u)\b[a-z][a-z0-9\-]{1,}\b",
    )
    X = vec.fit_transform(texts)
    svd = TruncatedSVD(n_components=n_components, random_state=42)
    X_lsa = svd.fit_transform(X)
    X_lsa = normalize(X_lsa, norm="l2")
    print(f"[embed] LSA explained variance ratio sum: "
          f"{svd.explained_variance_ratio_.sum():.3f}")
    return X_lsa.astype(np.float32)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def embed(silver_dir: str, output_dir: str, encoder: str,
          use_lsa: bool = False, batch_size: int = 64):
    """Full embedding pipeline.  Returns (embeddings, pmids)."""
    os.makedirs(output_dir, exist_ok=True)

    npy_path   = os.path.join(output_dir, "embeddings.npy")
    pmids_path = os.path.join(output_dir, "embedding_pmids.parquet")

    print(f"[embed] Loading silver table from {silver_dir} ...")
    df = load_silver(silver_dir)
    texts = df["text"].tolist()
    pmids = df["pmid"].tolist()
    print(f"[embed] {len(texts)} papers to embed")

    if use_lsa:
        embeddings = encode_with_lsa(texts)
    else:
        try:
            embeddings = encode_with_sentence_transformers(texts, encoder, batch_size)
        except Exception as e:
            print(f"[embed] sentence-transformers failed ({e}); falling back to LSA")
            embeddings = encode_with_lsa(texts)

    np.save(npy_path, embeddings)
    pd.DataFrame({"pmid": pmids}).to_parquet(pmids_path, index=False)

    print(f"[embed] Saved embeddings {embeddings.shape} -> {npy_path}")
    print(f"[embed] Saved PMID mapping -> {pmids_path}")
    return embeddings, pmids


def main():
    args = build_arg_parser().parse_args()
    embed(args.silver, args.output, args.encoder, args.use_lsa, args.batch)


if __name__ == "__main__":
    main()
