"""
Evaluate retrieval quality with Precision@10.

Three methods are compared:
  dense   — FAISS (or NumPy brute-force) only
  bm25    — BM25 only
  hybrid  — RRF-fused dense + BM25

Relevance proxy (keyword-based):
  A hit is counted as relevant if its title or abstract contains at least
  one of the query's expected_terms.  This is an imperfect proxy — it
  cannot reward papers that discuss the concept without using the exact
  keywords (paraphrase retrieval), and it may penalise correct retrievals
  that use synonyms.  This limitation is documented in the README.

Outputs:
  gold/retrieval_eval.csv           — query, method, p_at_10, n_hits
  gold/retrieval_results.parquet    — query, rank, pmid, score, method
  gold/relevance_labels_template.csv— query, pmid, title, abstract_snippet,
                                       relevant (empty; for human labelling)

Run with:
  python src/rag/evaluate.py \\
      --silver output/silver/pubmed_filtered \\
      --gold   output/gold \\
      [--human-labels output/gold/relevance_labels_human.csv]
"""

import argparse
import csv
import os
import re
import sys
from typing import List, Dict

import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from config import (  # noqa: E402
    GOLD_DIR, SILVER_DIR, EMBEDDING_MODEL, FINAL_TOP_K, RETRIEVAL_TOP_N
)

sys.path.insert(0, os.path.dirname(__file__))
from queries  import get_queries   # noqa: E402
from retrieve import (              # noqa: E402
    retrieve,
    _encode_query, _dense_retrieve, _bm25_retrieve,
    _load_meta,
)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def build_arg_parser():
    p = argparse.ArgumentParser(
        description="Evaluate retrieval Precision@10 with keyword proxy."
    )
    p.add_argument("--silver",       default=SILVER_DIR)
    p.add_argument("--gold",         default=GOLD_DIR)
    p.add_argument("--encoder",      default=EMBEDDING_MODEL)
    p.add_argument("--k",            type=int, default=FINAL_TOP_K)
    p.add_argument("--top-n",        type=int, default=RETRIEVAL_TOP_N)
    p.add_argument("--human-labels", default=None,
                   help="Path to human-labelled CSV (optional; adds nDCG@10)")
    return p


# ---------------------------------------------------------------------------
# Keyword relevance proxy
# ---------------------------------------------------------------------------

def is_relevant_keyword(title: str, abstract: str, expected_terms: List[str]) -> bool:
    """Return True if any expected term appears in title or abstract (case-insensitive)."""
    text = (title + " " + abstract).lower()
    return any(re.search(re.escape(t.lower()), text) for t in expected_terms)


# ---------------------------------------------------------------------------
# Precision@k
# ---------------------------------------------------------------------------

def precision_at_k(hits: List[Dict], expected_terms: List[str], k: int,
                   meta: pd.DataFrame) -> float:
    """Compute P@k using the keyword proxy."""
    meta_idx = meta.set_index("pmid") if "pmid" in meta.columns else meta
    n_rel    = 0
    for item in hits[:k]:
        pmid = item["pmid"]
        row  = meta_idx.loc[pmid] if pmid in meta_idx.index else None
        if row is not None:
            title    = str(row.get("title", "")) if isinstance(row, dict) else str(row["title"]) if "title" in row else ""
            abstract = str(row.get("abstract", "")) if isinstance(row, dict) else str(row["abstract"]) if "abstract" in row else ""
            if is_relevant_keyword(title, abstract, expected_terms):
                n_rel += 1
    return n_rel / min(k, len(hits)) if hits else 0.0


# ---------------------------------------------------------------------------
# Human-label nDCG (optional)
# ---------------------------------------------------------------------------

def ndcg_from_labels(hits, human_df: pd.DataFrame, query_id: str, k: int) -> float:
    """Compute nDCG@k from binary human relevance labels if available."""
    import math
    sub = human_df[(human_df["query"] == query_id) & (human_df["relevant"].notna())]
    rel_map = {row["pmid"]: int(row["relevant"]) for _, row in sub.iterrows()}

    dcg  = 0.0
    idcg = 0.0
    for i, item in enumerate(hits[:k], start=1):
        rel  = rel_map.get(item["pmid"], 0)
        dcg += rel / math.log2(i + 1)
    for i, val in enumerate(sorted(rel_map.values(), reverse=True)[:k], start=1):
        idcg += val / math.log2(i + 1)

    return dcg / idcg if idcg > 0 else 0.0


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def evaluate(silver_dir, gold_dir, encoder, k, top_n, human_labels_path=None):
    os.makedirs(gold_dir, exist_ok=True)
    meta    = _load_meta(silver_dir)
    queries = get_queries()

    human_df = None
    if human_labels_path and os.path.exists(human_labels_path):
        human_df = pd.read_csv(human_labels_path)
        print(f"[evaluate] Loaded human labels from {human_labels_path}")

    eval_rows    = []
    result_rows  = []
    template_rows = []

    for q in queries:
        qid           = q["id"]
        query_text    = q["query"]
        expected      = q["expected_terms"]

        print(f"\n[evaluate] Query: {qid}")

        # Dense
        try:
            qvec   = _encode_query(query_text, gold_dir, encoder)
            d_hits = _dense_retrieve(qvec, gold_dir, top_n)
            p_d    = precision_at_k(d_hits, expected, k, meta)
            eval_rows.append({
                "query": qid, "method": "dense",
                "p_at_10": round(p_d, 4), "n_hits": min(k, len(d_hits)),
            })
            print(f"  dense  P@{k} = {p_d:.4f}")
        except Exception as e:
            print(f"  dense failed: {e}")
            d_hits, p_d = [], 0.0

        # BM25
        b_hits = _bm25_retrieve(query_text, gold_dir, top_n)
        p_b    = precision_at_k(b_hits, expected, k, meta)
        eval_rows.append({
            "query": qid, "method": "bm25",
            "p_at_10": round(p_b, 4), "n_hits": min(k, len(b_hits)),
        })
        print(f"  bm25   P@{k} = {p_b:.4f}")

        # Hybrid
        h_hits = retrieve(query_text, k=k, gold_dir=gold_dir,
                          silver_dir=silver_dir, encoder=encoder, top_n=top_n)
        p_h    = precision_at_k(h_hits, expected, k, meta)
        eval_rows.append({
            "query": qid, "method": "hybrid",
            "p_at_10": round(p_h, 4), "n_hits": min(k, len(h_hits)),
        })
        print(f"  hybrid P@{k} = {p_h:.4f}")

        if human_df is not None:
            ndcg = ndcg_from_labels(h_hits, human_df, qid, k)
            print(f"  hybrid nDCG@{k} (human) = {ndcg:.4f}")

        # Store result rows (hybrid)
        meta_idx = meta.set_index("pmid")
        for item in h_hits:
            pmid = item["pmid"]
            row  = meta_idx.loc[pmid] if pmid in meta_idx.index else {}
            result_rows.append({
                "query": qid, "rank": item["rank"], "pmid": pmid,
                "score": item["score"], "method": item["method"],
            })
            template_rows.append({
                "query": qid, "pmid": pmid,
                "title": item.get("title", ""),
                "abstract_snippet": item.get("abstract_snippet", ""),
                "relevant": "",
            })

    # Write retrieval_eval.csv
    eval_df = pd.DataFrame(eval_rows)
    eval_df.to_csv(os.path.join(gold_dir, "retrieval_eval.csv"), index=False)
    print(f"\n[evaluate] Wrote retrieval_eval.csv")

    # Write retrieval_results.parquet
    res_df = pd.DataFrame(result_rows)
    res_df.to_parquet(os.path.join(gold_dir, "retrieval_results.parquet"), index=False)
    print(f"[evaluate] Wrote retrieval_results.parquet")

    # Write human-label template
    tmpl_df = pd.DataFrame(template_rows)
    tmpl_path = os.path.join(gold_dir, "relevance_labels_template.csv")
    tmpl_df.to_csv(tmpl_path, index=False)
    print(
        f"[evaluate] Wrote relevance_labels_template.csv\n"
        "  -> Fill the 'relevant' column (1=relevant, 0=not) and re-run with\n"
        "     --human-labels <path> to obtain nDCG@10.\n"
        "  NOTE: The keyword proxy cannot reward paraphrase retrieval and may\n"
        "  under-report P@10 for papers using synonyms."
    )

    return eval_df


def main():
    args = build_arg_parser().parse_args()
    evaluate(
        silver_dir=args.silver,
        gold_dir=args.gold,
        encoder=args.encoder,
        k=args.k,
        top_n=args.top_n,
        human_labels_path=args.human_labels,
    )


if __name__ == "__main__":
    main()
