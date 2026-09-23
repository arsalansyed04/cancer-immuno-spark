"""
Join the MeSH-filtered PubMed cancer-immunology articles with IEDB curated
records on PMID, then compute a few distributed analytics:

  1. epitope/antigen mention counts per cancer-immunology MeSH term
  2. publication + curated-epitope volume per year
  3. top antigens (objects) referenced in the matched literature

Run with:
  spark-submit join_analysis.py \
      --pubmed output/pubmed_filtered --iedb output/iedb_tables --output output/analysis
"""

import argparse
import os
import sys
import shutil

from pyspark.sql import functions as F

sys.path.append(os.path.dirname(__file__))
from config import get_spark  # noqa: E402


def write_csv(df, output_path):
    """Write a Spark DataFrame as a single CSV file."""
    temp_dir = output_path + "_tmp"

    df.coalesce(1).write.mode("overwrite").option("header", True).csv(temp_dir)

    part_file = next(
        name for name in os.listdir(temp_dir)
        if name.startswith("part-") and name.endswith(".csv")
    )

    os.replace(
        os.path.join(temp_dir, part_file),
        output_path,
    )

    shutil.rmtree(temp_dir)

def build_arg_parser():
    p = argparse.ArgumentParser()
    p.add_argument("--pubmed", required=True, help="Parquet dir from pubmed_parser.py")
    p.add_argument("--iedb", required=True, help="Parquet dir from iedb_loader.py (has subfolders per table)")
    p.add_argument("--output", required=True)
    return p


def main():
    args = build_arg_parser().parse_args()
    spark = get_spark("cancer-immunology-join-analysis")

    pubmed = spark.read.parquet(args.pubmed)

    article = spark.read.parquet(os.path.join(args.iedb, "article"))
    curated_epitope = spark.read.parquet(os.path.join(args.iedb, "curated_epitope"))
    obj = spark.read.parquet(os.path.join(args.iedb, "object"))

    # --- Join PubMed articles to IEDB references by PMID -----------------
    # article.pubmed_id holds the PMID and article.reference_id links
    # the article to the IEDB reference.
    joined_refs = pubmed.join(
        article.select(
            F.col("pubmed_id").cast("long").alias("pmid"),
            F.col("reference_id"),
        ),
        on="pmid",
        how="inner",
    )

    # curated_epitope links references to curated epitope/object records.
    curated = (
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

    curated.cache()
    joined_count = curated.select("pmid").distinct().count()
    print(f"[join_analysis] {joined_count} distinct PubMed articles matched to IEDB curated records.")

    curated.write.mode("overwrite").parquet(os.path.join(args.output, "joined_papers.parquet"))

    # --- 1. Epitope mentions per MeSH term --------------------------------
    exploded_mesh = curated.select("pmid", "antigen_name", F.explode_outer("mesh_terms").alias("mesh_term"))
    mesh_counts = (
        exploded_mesh.groupBy("mesh_term")
        .agg(F.countDistinct("pmid").alias("num_papers"), F.count("antigen_name").alias("num_epitope_mentions"))
        .orderBy(F.desc("num_epitope_mentions"))
    )
    write_csv(mesh_counts, os.path.join(args.output, "epitope_counts_by_mesh_term.csv"))

    # --- 2. Publications + curated epitopes per year ----------------------
    per_year = (
        curated.groupBy("pub_year")
        .agg(F.countDistinct("pmid").alias("num_papers"), F.count("antigen_name").alias("num_epitope_mentions"))
        .orderBy("pub_year")
    )
    write_csv(per_year, os.path.join(args.output, "publications_per_year.csv"))

    # --- 3. Top antigens in the matched literature -------------------------
    top_antigens = (
        curated.filter(F.col("antigen_name").isNotNull())
        .groupBy("antigen_name")
        .agg(F.countDistinct("pmid").alias("num_papers"))
        .orderBy(F.desc("num_papers"))
        .limit(100)
    )
    write_csv(top_antigens, os.path.join(args.output, "top_antigens.csv"))

    print(f"[join_analysis] Wrote CSV summaries and joined_papers.parquet to {args.output}")
    spark.stop()


if __name__ == "__main__":
    main()
