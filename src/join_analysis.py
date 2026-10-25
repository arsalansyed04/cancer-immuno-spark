"""
Join MeSH-filtered PubMed cancer-immunology articles with IEDB curated
records on PMID, then compute distributed analytics:

  1. Epitope/antigen counts per cancer-immunology MeSH term (de-duplicated)
  2. Publication + distinct curated-epitope volume per year
  3. Top antigens (objects) referenced in the matched literature
  4. Epitope density per paper (gold/epitope_density.parquet)

Fan-out fix (report section 4.6)
----------------------------------
The original code exploded mesh_terms on the full multi-joined table,
inflating counts because each paper×epitope×mesh-term row appears once
per mesh heading.  The fixed pipeline:

  1. Builds a de-duplicated paper–epitope table first
     (pmid, curated_epitope_id, antigen_name) – one row per unique epitope
     per paper.
  2. Counts with countDistinct("curated_epitope_id").
  3. Only then joins back to mesh_terms / explodes per mesh term.

Row-count and distinct-count assertions are logged at every join so
inflation is immediately visible.

Run with:
  spark-submit src/join_analysis.py \\
      --pubmed  output/silver/pubmed_filtered \\
      --iedb    output/iedb_tables \\
      --output  output/gold
"""

import argparse
import os
import shutil
import sys

from pyspark.sql import functions as F

sys.path.append(os.path.dirname(__file__))
from config import GOLD_DIR, get_spark  # noqa: E402


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


def _log_counts(label, df, pmid_col="pmid", epitope_col=None):
    """Log row count, distinct PMID, and (optionally) distinct epitope count."""
    n_rows = df.count()
    n_pmid = df.select(pmid_col).distinct().count()
    msg = f"[join_analysis] {label}: {n_rows} rows, {n_pmid} distinct PMIDs"
    if epitope_col and epitope_col in df.columns:
        n_ep = df.select(epitope_col).distinct().count()
        msg += f", {n_ep} distinct {epitope_col}"
    print(msg)
    return n_rows


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def build_arg_parser():
    p = argparse.ArgumentParser(
        description="Join PubMed silver table with IEDB, write gold analytics."
    )
    p.add_argument("--pubmed",  required=True,
                   help="Parquet dir from pubmed_parser.py (silver)")
    p.add_argument("--iedb",    required=True,
                   help="Parquet dir from iedb_loader.py (one subfolder per table)")
    p.add_argument("--output",  default=GOLD_DIR,
                   help="Output directory for gold CSV / Parquet files")
    return p


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    args = build_arg_parser().parse_args()
    spark = get_spark("cancer-immunology-join-analysis")
    os.makedirs(args.output, exist_ok=True)

    # ------------------------------------------------------------------
    # Load silver PubMed table
    # ------------------------------------------------------------------
    pubmed = spark.read.parquet(args.pubmed)
    _log_counts("pubmed silver", pubmed)

    # ------------------------------------------------------------------
    # Load IEDB tables
    # ------------------------------------------------------------------
    article         = spark.read.parquet(os.path.join(args.iedb, "article"))
    curated_epitope = spark.read.parquet(os.path.join(args.iedb, "curated_epitope"))
    obj             = spark.read.parquet(os.path.join(args.iedb, "object"))

    _log_counts("iedb:article", article, pmid_col="pubmed_id")

    # ------------------------------------------------------------------
    # Step 1: PubMed ⟕ article (by PMID)
    # ------------------------------------------------------------------
    joined_refs = pubmed.join(
        article.select(
            F.col("pubmed_id").cast("long").alias("pmid"),
            F.col("reference_id"),
        ),
        on="pmid",
        how="inner",
    )
    _log_counts("pubmed⋈article", joined_refs)

    # ------------------------------------------------------------------
    # Step 2: Build de-duplicated paper–epitope table BEFORE any explode
    # FAN-OUT FIX: one row per (pmid, curated_epitope_id)
    # ------------------------------------------------------------------
    raw_curated = (
        joined_refs
        .join(curated_epitope, on="reference_id", how="inner")
        .join(
            obj.select(
                F.col("object_id"),
                F.col("object_description").alias("antigen_name"),
            ),
            F.col("e_object_id") == F.col("object_id"),
            how="left",
        )
    )
    _log_counts("raw curated (before dedup)", raw_curated,
                epitope_col="curated_epitope_id")

    # De-duplicate on (pmid, curated_epitope_id) — the unit of curation
    dedup = (
        raw_curated
        .select("pmid", "curated_epitope_id", "antigen_name",
                "pub_year", "mesh_terms")
        .dropDuplicates(["pmid", "curated_epitope_id"])
    )
    _log_counts("dedup paper-epitope", dedup, epitope_col="curated_epitope_id")

    dedup.cache()

    # Persist joined table for ad-hoc queries
    dedup.write.mode("overwrite").parquet(
        os.path.join(args.output, "joined_papers.parquet")
    )

    # ------------------------------------------------------------------
    # Step 3: Epitope density per paper (gold/epitope_density.parquet)
    # ------------------------------------------------------------------
    epitope_density = (
        dedup.groupBy("pmid")
        .agg(F.countDistinct("curated_epitope_id").alias("n_distinct_epitopes"))
        .orderBy(F.desc("n_distinct_epitopes"))
    )
    epitope_density.write.mode("overwrite").parquet(
        os.path.join(args.output, "epitope_density.parquet")
    )
    print(f"[join_analysis] epitope_density.parquet written "
          f"({epitope_density.count()} papers with IEDB epitopes)")

    # ------------------------------------------------------------------
    # Analytics 1: Epitope mentions per MeSH term  (fan-out-fixed)
    # Explode mesh_terms AFTER de-duplication.
    # ------------------------------------------------------------------
    exploded_mesh = dedup.select(
        "pmid", "curated_epitope_id", "antigen_name",
        F.explode_outer("mesh_terms").alias("mesh_term"),
    )
    mesh_counts = (
        exploded_mesh
        .groupBy("mesh_term")
        .agg(
            F.countDistinct("pmid").alias("num_papers"),
            F.countDistinct("curated_epitope_id").alias("num_distinct_epitopes"),
        )
        .orderBy(F.desc("num_distinct_epitopes"))
    )
    write_csv(mesh_counts,
              os.path.join(args.output, "epitope_counts_by_mesh_term.csv"))
    print("[join_analysis] Wrote epitope_counts_by_mesh_term.csv")

    # ------------------------------------------------------------------
    # Analytics 2: Publications + epitopes per year
    # ------------------------------------------------------------------
    per_year = (
        dedup.groupBy("pub_year")
        .agg(
            F.countDistinct("pmid").alias("num_papers"),
            F.countDistinct("curated_epitope_id").alias("num_distinct_epitopes"),
        )
        .orderBy("pub_year")
    )
    write_csv(per_year, os.path.join(args.output, "publications_per_year.csv"))
    print("[join_analysis] Wrote publications_per_year.csv")

    # ------------------------------------------------------------------
    # Analytics 3: Top antigens
    # ------------------------------------------------------------------
    top_antigens = (
        dedup.filter(F.col("antigen_name").isNotNull())
        .groupBy("antigen_name")
        .agg(F.countDistinct("pmid").alias("num_papers"))
        .orderBy(F.desc("num_papers"))
        .limit(100)
    )
    write_csv(top_antigens, os.path.join(args.output, "top_antigens.csv"))
    print("[join_analysis] Wrote top_antigens.csv")

    matched = joined_refs.select("pmid").distinct().count()
    print(f"[join_analysis] {matched} distinct PubMed articles matched to IEDB.")
    print(f"[join_analysis] All analytics written to {args.output}")

    spark.stop()


if __name__ == "__main__":
    main()
