"""
Trend engine: year-wise topic share, Δ share, emerging/declining ranking,
and key-term prevalence over time.

Reads:
  gold/paper_topics.parquet  (pmid, pub_year, dominant_topic, theta)
  silver/pubmed_filtered     (pmid, pub_year, title, abstract, mesh_terms)

Writes to gold/:
  topic_share_by_year.csv    — mean θ share and paper count per (year, topic)
  topic_delta.csv            — Δ share early vs late window, emerging/declining label
  year_counts.csv            — papers per year (with flag for years below threshold)
  term_prevalence_by_year.csv— % of abstracts per year matching each key-term regex
  papers_per_topic.csv       — total papers by dominant topic (if not already written)

Run with:
  spark-submit src/trend_analysis.py \\
      --topics  output/gold/paper_topics.parquet \\
      --silver  output/silver/pubmed_filtered \\
      --output  output/gold
"""

import argparse
import os
import shutil
import sys

from pyspark.sql import functions as F
from pyspark.ml.functions import vector_to_array

sys.path.append(os.path.dirname(__file__))
from config import (  # noqa: E402
    get_spark,
    GOLD_DIR, SILVER_DIR,
    TREND_EARLY_WINDOW, TREND_LATE_WINDOW,
    TREND_EMERGING_THRESHOLD, TREND_DECLINING_THRESHOLD,
    MIN_PAPERS_PER_YEAR,
    KEY_TERM_PATTERNS,
    LDA_FINAL_K,
)


# ---------------------------------------------------------------------------
# Helper
# ---------------------------------------------------------------------------

def write_csv(df, output_path: str):
    """Write a Spark DataFrame as a single CSV file."""
    temp_dir = output_path + "_tmp"
    df.coalesce(1).write.mode("overwrite").option("header", True).csv(temp_dir)
    part_file = next(
        name for name in os.listdir(temp_dir)
        if name.startswith("part-") and name.endswith(".csv")
    )
    os.replace(os.path.join(temp_dir, part_file), output_path)
    shutil.rmtree(temp_dir)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def build_arg_parser():
    p = argparse.ArgumentParser(
        description="Compute year-wise topic trends and key-term prevalence."
    )
    p.add_argument("--topics",  required=True,
                   help="gold/paper_topics.parquet from topic_modeling.py")
    p.add_argument("--silver",  required=True,
                   help="silver/pubmed_filtered parquet dir")
    p.add_argument("--output",  default=GOLD_DIR)
    p.add_argument("--early-start", type=int, default=TREND_EARLY_WINDOW[0])
    p.add_argument("--early-end",   type=int, default=TREND_EARLY_WINDOW[1])
    p.add_argument("--late-start",  type=int, default=TREND_LATE_WINDOW[0])
    p.add_argument("--late-end",    type=int, default=TREND_LATE_WINDOW[1])
    p.add_argument("--min-papers",  type=int, default=MIN_PAPERS_PER_YEAR,
                   help="Minimum papers/year to include in trend computation")
    p.add_argument("--n-topics",    type=int, default=LDA_FINAL_K)
    return p


# ---------------------------------------------------------------------------
# Step 1 – year counts and valid-year filter
# ---------------------------------------------------------------------------

def compute_year_counts(topics_df, min_papers, output_dir):
    """Count papers per year; flag years with too few papers."""
    year_counts = (
        topics_df.groupBy("pub_year")
        .agg(F.count("pmid").alias("num_papers"))
        .withColumn("sufficient", F.col("num_papers") >= min_papers)
        .orderBy("pub_year")
    )
    write_csv(year_counts, os.path.join(output_dir, "year_counts.csv"))
    valid_years = {
        row["pub_year"]
        for row in year_counts.collect()
        if row["sufficient"]
    }
    flagged = [
        row["pub_year"]
        for row in year_counts.collect()
        if not row["sufficient"]
    ]
    print(f"[trend_analysis] Years with ≥{min_papers} papers: {sorted(valid_years)}")
    if flagged:
        print(f"[trend_analysis] Flagged (excluded from trend): {sorted(flagged)}")
    return valid_years


# ---------------------------------------------------------------------------
# Step 2 – topic share by year
# ---------------------------------------------------------------------------

def compute_topic_share_by_year(topics_df, n_topics, valid_years, output_dir):
    """Mean θ share per (year, topic) for years with sufficient papers.

    theta is array<double> of length k.  We use posexplode to get
    (year, topic_index, theta_i) rows, then average within each group.
    """
    # Filter to valid years
    valid_df = topics_df.filter(F.col("pub_year").isin(valid_years))

    # posexplode the theta array → (pos=topic_index, col=theta_value)
    exploded = valid_df.select(
        "pmid", "pub_year",
        F.posexplode("theta").alias("topic_id", "theta_value"),
    )

    share = (
        exploded.groupBy("pub_year", "topic_id")
        .agg(
            F.avg("theta_value").alias("mean_theta_share"),
            F.count("pmid").alias("num_papers"),
        )
        .orderBy("pub_year", "topic_id")
    )
    write_csv(share, os.path.join(output_dir, "topic_share_by_year.csv"))
    print("[trend_analysis] Wrote topic_share_by_year.csv")
    return share


# ---------------------------------------------------------------------------
# Step 3 – Δ share (emerging / declining)
# ---------------------------------------------------------------------------

def compute_topic_delta(share_df, early_window, late_window,
                        emerging_threshold, declining_threshold, output_dir):
    """Mean share in early window vs late window, with emerging/declining label."""
    early_start, early_end = early_window
    late_start,  late_end  = late_window

    early = (
        share_df.filter(
            F.col("pub_year").between(early_start, early_end)
        )
        .groupBy("topic_id")
        .agg(F.avg("mean_theta_share").alias("early_share"))
    )

    late = (
        share_df.filter(
            F.col("pub_year").between(late_start, late_end)
        )
        .groupBy("topic_id")
        .agg(F.avg("mean_theta_share").alias("late_share"))
    )

    delta = (
        early.join(late, on="topic_id", how="outer")
        .fillna(0.0, subset=["early_share", "late_share"])
        .withColumn("delta_pp",
                    (F.col("late_share") - F.col("early_share")) * 100)
        .withColumn(
            "trend",
            F.when(F.col("delta_pp") >= emerging_threshold,  "emerging")
             .when(F.col("delta_pp") <= declining_threshold, "declining")
             .otherwise("stable"),
        )
        .orderBy(F.desc("delta_pp"))
    )
    write_csv(delta, os.path.join(output_dir, "topic_delta.csv"))

    # Console summary
    rows = delta.collect()
    emerging  = [r for r in rows if r["trend"] == "emerging"]
    declining = [r for r in rows if r["trend"] == "declining"]
    print("[trend_analysis] Top EMERGING topics (by Δ pp):")
    for r in emerging:
        print(f"  topic {r['topic_id']:2d}: Δ={r['delta_pp']:+.2f} pp "
              f"(early={r['early_share']:.4f}, late={r['late_share']:.4f})")
    print("[trend_analysis] Top DECLINING topics:")
    for r in declining:
        print(f"  topic {r['topic_id']:2d}: Δ={r['delta_pp']:+.2f} pp "
              f"(early={r['early_share']:.4f}, late={r['late_share']:.4f})")
    print("[trend_analysis] Wrote topic_delta.csv")
    return delta


# ---------------------------------------------------------------------------
# Step 4 – key-term prevalence by year
# ---------------------------------------------------------------------------

def compute_term_prevalence(silver_df, valid_years, key_term_patterns, output_dir):
    """Percentage of abstracts per year matching each key-term regex.

    Patterns use rlike() on lower-cased (title + " " + abstract).
    Only valid years (≥ min_papers) are included.
    """
    valid_df = silver_df.filter(F.col("pub_year").isin(valid_years))

    # Combined text field (lower-cased)
    combined = valid_df.withColumn(
        "text",
        F.lower(F.concat_ws(" ", F.col("title"), F.col("abstract"))),
    ).select("pmid", "pub_year", "text")

    rows = []
    for term_name, pattern in key_term_patterns.items():
        matched = combined.withColumn(
            "hit", F.col("text").rlike(pattern).cast("int")
        )
        per_year = (
            matched.groupBy("pub_year")
            .agg(
                F.count("pmid").alias("total"),
                F.sum("hit").alias("n_match"),
            )
            .withColumn("pct_match", F.round(F.col("n_match") / F.col("total") * 100, 2))
            .withColumn("term", F.lit(term_name))
            .select("pub_year", "term", "total", "n_match", "pct_match")
        )
        rows.append(per_year)
        print(f"[trend_analysis] Term '{term_name}': computed prevalence per year")

    if not rows:
        print("[trend_analysis] WARNING: no key-term patterns — skipping prevalence")
        return

    from functools import reduce
    from pyspark.sql import DataFrame

    combined_rows = reduce(DataFrame.union, rows)
    combined_rows = combined_rows.orderBy("term", "pub_year")
    write_csv(combined_rows,
              os.path.join(output_dir, "term_prevalence_by_year.csv"))
    print("[trend_analysis] Wrote term_prevalence_by_year.csv")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    args  = build_arg_parser().parse_args()
    os.makedirs(args.output, exist_ok=True)

    early_window    = (args.early_start, args.early_end)
    late_window     = (args.late_start,  args.late_end)

    spark = get_spark("cancer-immunology-trend-analysis")

    # --- Load inputs ---------------------------------------------------------
    topics_df = spark.read.parquet(args.topics)
    silver_df = spark.read.parquet(args.silver)

    total_papers = topics_df.count()
    print(f"[trend_analysis] {total_papers} papers in topic table")

    # --- Step 1: year counts -------------------------------------------------
    valid_years = compute_year_counts(topics_df, args.min_papers, args.output)

    if not valid_years:
        print(
            "[trend_analysis] WARNING: no year meets the min-papers threshold "
            f"({args.min_papers}).  Proceeding with all years for trend "
            "computation (counts will be small — validation set artefact)."
        )
        valid_years = set(
            r["pub_year"] for r in topics_df.select("pub_year").distinct().collect()
            if r["pub_year"] is not None
        )

    # --- Step 2: topic share by year -----------------------------------------
    share_df = compute_topic_share_by_year(
        topics_df, args.n_topics, valid_years, args.output
    )

    # --- Step 3: Δ share ------------------------------------------------------
    compute_topic_delta(
        share_df,
        early_window, late_window,
        TREND_EMERGING_THRESHOLD, TREND_DECLINING_THRESHOLD,
        args.output,
    )

    # --- Step 4: key-term prevalence -----------------------------------------
    compute_term_prevalence(silver_df, valid_years, KEY_TERM_PATTERNS, args.output)

    print(f"[trend_analysis] All trend outputs written to {args.output}")
    spark.stop()


if __name__ == "__main__":
    main()
