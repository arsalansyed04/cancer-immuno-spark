"""
Paper importance scoring (report section 4.5, Layer L3).

Formula:
  Importance(p) = 100 × [0.35·r(p) + 0.25·w(p) + 0.25·e(p) + 0.15·y(p)]
                       / max_over_corpus

Components (each min-max scaled to [0, 1]):
  r(p) — max cosine similarity of paper p to any RAG key-term query (dense)
  w(p) — summed TF-IDF weight of the 20-term key lexicon in the paper
  e(p) — θ-vector of paper dotted with max(Δ share, 0) per topic
  y(p) — recency (publication year, min-max scaled)

Weights live in config.IMPORTANCE_WEIGHTS and can be overridden via CLI.

NOTE: "importance" means relevance to and momentum within the tracked
themes (immunotherapy/checkpoint/CAR-T/etc.), NOT citation impact.

Outputs (gold/):
  paper_importance.parquet  — full table with all component scores
  paper_importance.csv      — same as CSV
  top10_importance.csv      — top-10 papers
  weight_sensitivity.csv    — rank correlation when each weight is ±0.1

Run with:
  python src/importance.py \\
      --topics      output/gold/paper_topics.parquet \\
      --tfidf-vocab output/gold/topic_terms.csv \\
      --topic-delta output/gold/topic_delta.csv \\
      --gold        output/gold \\
      --silver      output/silver/pubmed_filtered \\
      [--use-epitope-density]
"""

import argparse
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(__file__))
from config import (  # noqa: E402
    GOLD_DIR, SILVER_DIR, EMBEDDING_MODEL,
    IMPORTANCE_WEIGHTS, IMPORTANCE_KEY_LEXICON,
    FINAL_TOP_K,
)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def build_arg_parser():
    p = argparse.ArgumentParser(
        description="Compute paper importance score (0-100)."
    )
    p.add_argument("--topics",       required=True,
                   help="gold/paper_topics.parquet")
    p.add_argument("--tfidf-vocab",  required=True,
                   help="gold/topic_terms.csv (term, weight columns)")
    p.add_argument("--topic-delta",  required=True,
                   help="gold/topic_delta.csv (from trend_analysis.py)")
    p.add_argument("--gold",         default=GOLD_DIR)
    p.add_argument("--silver",       default=SILVER_DIR)
    p.add_argument("--encoder",      default=EMBEDDING_MODEL)
    p.add_argument("--wr",           type=float, default=IMPORTANCE_WEIGHTS["r"])
    p.add_argument("--ww",           type=float, default=IMPORTANCE_WEIGHTS["w"])
    p.add_argument("--we",           type=float, default=IMPORTANCE_WEIGHTS["e"])
    p.add_argument("--wy",           type=float, default=IMPORTANCE_WEIGHTS["y"])
    p.add_argument("--use-epitope-density", action="store_true",
                   help="Add normalised epitope density as extra component (weight 0 by default)")
    p.add_argument("--epitope-weight", type=float, default=0.0,
                   help="Weight for epitope-density component (default 0 = no effect)")
    return p


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _minmax(arr: np.ndarray) -> np.ndarray:
    lo, hi = arr.min(), arr.max()
    if hi == lo:
        return np.zeros_like(arr, dtype=float)
    return (arr - lo) / (hi - lo)


def _load_silver_meta(silver_dir: str) -> pd.DataFrame:
    import pyarrow.parquet as pq
    table = pq.read_table(silver_dir, columns=["pmid", "title", "journal", "pub_year"])
    return table.to_pandas()


# ---------------------------------------------------------------------------
# Component r: max RAG cosine similarity
# ---------------------------------------------------------------------------

def compute_r(pmids: list, gold_dir: str, silver_dir: str, encoder: str) -> np.ndarray:
    """For each paper, compute max cosine similarity to any key-term query."""
    sys.path.insert(0, os.path.dirname(__file__))
    from rag.retrieve import _encode_query, _load_embeddings_and_pmids
    from rag.queries  import get_queries

    try:
        emb, emb_pmids = _load_embeddings_and_pmids(gold_dir)
    except FileNotFoundError:
        print("[importance] embeddings.npy not found — r component = 0")
        return np.zeros(len(pmids))

    pmid_to_idx = {p: i for i, p in enumerate(emb_pmids)}
    queries     = get_queries()

    # Encode all queries
    query_vecs = []
    for q in queries:
        try:
            qv = _encode_query(q["query"], gold_dir, encoder)
            query_vecs.append(qv)
        except Exception as e:
            print(f"[importance] Could not encode query '{q['id']}': {e}")

    if not query_vecs:
        return np.zeros(len(pmids))

    Q = np.stack(query_vecs, axis=0)   # (n_queries, D)

    scores = np.zeros(len(pmids))
    for i, pmid in enumerate(pmids):
        idx = pmid_to_idx.get(pmid)
        if idx is None:
            continue
        p_vec = emb[idx]               # (D,)
        sims  = Q @ p_vec              # cosine (unit vectors)
        scores[i] = float(sims.max())

    return scores


# ---------------------------------------------------------------------------
# Component w: TF-IDF weight of key-lexicon terms
# ---------------------------------------------------------------------------

def compute_w(pmids: list, tfidf_vocab_path: str, silver_dir: str) -> np.ndarray:
    """Sum TF-IDF weights of key-lexicon terms per paper.

    The TF-IDF vectors are stored implicitly via topic_terms.csv (which has
    term weights per topic, not per paper).  We approximate w(p) as the
    sum of IDF weights (from topic_terms.csv) of key-lexicon terms that
    appear in the paper's abstract/title, reweighted by the paper's
    within-vocabulary term frequency.

    Because we don't persist per-paper TF-IDF vectors (they are consumed by
    LDA), we use a simpler proxy: count of key-lexicon term occurrences in
    the lowercased abstract+title, weighted by the term's average IDF from
    topic_terms.csv.
    """
    import re
    import pyarrow.parquet as pq

    # Build idf map from topic_terms.csv
    topic_terms = pd.read_csv(tfidf_vocab_path)
    idf_map = (
        topic_terms.groupby("term")["weight"].mean().to_dict()
    )

    # Load abstracts
    table = pq.read_table(silver_dir, columns=["pmid", "title", "abstract"])
    meta  = table.to_pandas().set_index("pmid")

    pmid_set = set(pmids)
    w_scores = np.zeros(len(pmids))

    for i, pmid in enumerate(pmids):
        if pmid not in meta.index:
            continue
        row  = meta.loc[pmid]
        text = (str(row.get("title", "")) + " " + str(row.get("abstract", ""))).lower()
        score = 0.0
        for term in IMPORTANCE_KEY_LEXICON:
            count = len(re.findall(r"\b" + re.escape(term) + r"\b", text))
            if count > 0:
                idf = idf_map.get(term, 1.0)
                score += count * idf
        w_scores[i] = score

    return w_scores


# ---------------------------------------------------------------------------
# Component e: θ·max(Δ_share, 0) per topic
# ---------------------------------------------------------------------------

def compute_e(theta_matrix: np.ndarray, topic_delta_path: str) -> np.ndarray:
    """Dot each paper's θ with the vector of positive topic growths."""
    delta_df  = pd.read_csv(topic_delta_path)
    # delta_pp is in percentage points; normalise to [0,1] for dot product
    n_topics  = theta_matrix.shape[1]
    growth    = np.zeros(n_topics)
    for _, row in delta_df.iterrows():
        tid = int(row["topic_id"])
        if tid < n_topics:
            growth[tid] = max(float(row["delta_pp"]) / 100.0, 0.0)

    return theta_matrix @ growth   # (N,)


# ---------------------------------------------------------------------------
# Component y: recency
# ---------------------------------------------------------------------------

def compute_y(pub_years: np.ndarray) -> np.ndarray:
    return _minmax(pub_years.astype(float))


# ---------------------------------------------------------------------------
# Epitope density (optional)
# ---------------------------------------------------------------------------

def compute_epitope_density(pmids: list, gold_dir: str) -> np.ndarray:
    ep_path = os.path.join(gold_dir, "epitope_density.parquet")
    if not os.path.exists(ep_path):
        print("[importance] epitope_density.parquet not found — density component = 0")
        return np.zeros(len(pmids))
    ep_df = pd.read_parquet(ep_path).set_index("pmid")
    ep    = np.zeros(len(pmids))
    for i, pmid in enumerate(pmids):
        if pmid in ep_df.index:
            ep[i] = float(ep_df.loc[pmid, "n_distinct_epitopes"])
    return ep


# ---------------------------------------------------------------------------
# Weight sensitivity analysis
# ---------------------------------------------------------------------------

def weight_sensitivity(df: pd.DataFrame, weights: dict, perturbation: float = 0.1) -> pd.DataFrame:
    """Compute Spearman rank correlation of top-50 when each weight is ±perturbed."""
    from scipy.stats import spearmanr

    top50      = df.nlargest(50, "importance")
    base_ranks = top50["importance"].rank(ascending=False).values

    rows = []
    for key in weights:
        for direction in (+perturbation, -perturbation):
            new_w = dict(weights)
            new_w[key] = max(0.0, new_w[key] + direction)
            # Renormalise to sum 1
            total = sum(new_w.values())
            new_w = {k: v / total for k, v in new_w.items()}

            r_sc = _minmax(top50["r"].values)
            w_sc = _minmax(top50["w"].values)
            e_sc = _minmax(top50["e"].values)
            y_sc = _minmax(top50["y"].values)
            new_imp = (
                new_w["r"] * r_sc +
                new_w["w"] * w_sc +
                new_w["e"] * e_sc +
                new_w["y"] * y_sc
            )
            new_ranks = new_imp.argsort()[::-1].argsort() + 1
            rho, _ = spearmanr(base_ranks, new_ranks)
            rows.append({
                "perturbed_weight": key,
                "direction": f"{'+' if direction > 0 else ''}{direction:+.1f}",
                "new_value": round(new_w[key], 3),
                "spearman_rho": round(rho, 4),
            })

    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    args = build_arg_parser().parse_args()
    os.makedirs(args.gold, exist_ok=True)

    weights = {
        "r": args.wr,
        "w": args.ww,
        "e": args.we,
        "y": args.wy,
    }

    # --- Load paper_topics.parquet -------------------------------------------
    topics_df = pd.read_parquet(args.topics)
    pmids     = topics_df["pmid"].tolist()

    # theta is stored as array<double>; convert to numpy matrix
    theta_matrix = np.stack(topics_df["theta"].values).astype(float)
    pub_years    = topics_df["pub_year"].fillna(0).values.astype(float)

    print(f"[importance] {len(pmids)} papers loaded from paper_topics.parquet")

    # --- Compute components --------------------------------------------------
    print("[importance] Computing r (RAG similarity) ...")
    r_raw = compute_r(pmids, args.gold, args.silver, args.encoder)

    print("[importance] Computing w (TF-IDF key-lexicon weight) ...")
    w_raw = compute_w(pmids, args.tfidf_vocab, args.silver)

    print("[importance] Computing e (topic growth alignment) ...")
    e_raw = compute_e(theta_matrix, args.topic_delta)

    print("[importance] Computing y (recency) ...")
    y_raw = compute_y(pub_years)

    # --- Min-max scale each component ----------------------------------------
    r_sc = _minmax(r_raw)
    w_sc = _minmax(w_raw)
    e_sc = _minmax(e_raw)
    y_sc = _minmax(y_raw)

    # --- Weighted sum --------------------------------------------------------
    raw_score = (
        weights["r"] * r_sc +
        weights["w"] * w_sc +
        weights["e"] * e_sc +
        weights["y"] * y_sc
    )

    # Optional epitope density
    ep_component = None
    if args.use_epitope_density and args.epitope_weight > 0:
        ep_raw  = compute_epitope_density(pmids, args.gold)
        ep_sc   = _minmax(ep_raw)
        raw_score = raw_score + args.epitope_weight * ep_sc
        ep_component = ep_sc
        print(f"[importance] Epitope density added (weight={args.epitope_weight})")

    # Divide by max → top paper = 100
    max_score = raw_score.max() if raw_score.max() > 0 else 1.0
    importance = (raw_score / max_score) * 100.0

    # --- Assemble output DataFrame -------------------------------------------
    meta = _load_silver_meta(args.silver).set_index("pmid")
    out_rows = []
    for i, pmid in enumerate(pmids):
        row = meta.loc[pmid] if pmid in meta.index else {}
        out_rows.append({
            "pmid":           pmid,
            "title":          topics_df.iloc[i]["title"],
            "journal":        topics_df.iloc[i]["journal"],
            "pub_year":       int(pub_years[i]),
            "dominant_topic": int(topics_df.iloc[i]["dominant_topic"]),
            "r":              round(float(r_sc[i]), 4),
            "w":              round(float(w_sc[i]), 4),
            "e":              round(float(e_sc[i]), 4),
            "y":              round(float(y_sc[i]), 4),
            "importance":     round(float(importance[i]), 2),
        })

    out_df = pd.DataFrame(out_rows).sort_values("importance", ascending=False)

    out_df.to_parquet(os.path.join(args.gold, "paper_importance.parquet"), index=False)
    out_df.to_csv(os.path.join(args.gold, "paper_importance.csv"), index=False)

    top10 = out_df.head(10)
    top10.to_csv(os.path.join(args.gold, "top10_importance.csv"), index=False)

    print("\n[importance] Top-10 papers by importance score:")
    for _, r in top10.iterrows():
        print(f"  {r['importance']:6.2f}  PMID={r['pmid']}  {str(r['title'])[:70]}")

    # --- Weight sensitivity --------------------------------------------------
    try:
        sens_df = weight_sensitivity(out_df, weights)
        sens_df.to_csv(os.path.join(args.gold, "weight_sensitivity.csv"), index=False)
        print("[importance] Wrote weight_sensitivity.csv")
    except Exception as e:
        print(f"[importance] Weight sensitivity failed (scipy missing?): {e}")

    # --- Epitope vs topic correlation (optional) ----------------------------
    if args.use_epitope_density:
        ep_raw = compute_epitope_density(pmids, args.gold)
        corr_rows = []
        for t in range(theta_matrix.shape[1]):
            theta_col = theta_matrix[:, t]
            if theta_col.std() == 0 or ep_raw.std() == 0:
                rho = 0.0
            else:
                from scipy.stats import spearmanr
                rho, _ = spearmanr(theta_col, ep_raw)
            corr_rows.append({"topic_id": t, "spearman_rho_vs_epitope_density": round(rho, 4)})
        pd.DataFrame(corr_rows).to_csv(
            os.path.join(args.gold, "epitope_vs_topic.csv"), index=False
        )
        print("[importance] Wrote epitope_vs_topic.csv")

    print(f"[importance] All outputs written to {args.gold}")


if __name__ == "__main__":
    main()
