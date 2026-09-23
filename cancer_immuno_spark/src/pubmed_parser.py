"""
Parse PubMed baseline XML (*.xml.gz) with Spark and filter to records whose
MeSH headings match the cancer-immunology term list in config.py.

Each PubMed baseline file is a <PubmedArticleSet> containing many
<PubmedArticle> records. We use the spark-xml package (`com.databricks:spark-xml`)
to parse them into a DataFrame in a single distributed pass, no need to
manually walk the XML tree with lxml (that would be single-threaded).

Run with:
  spark-submit --packages com.databricks:spark-xml_2.12:0.18.0 \
      pubmed_parser.py --input data/pubmed_baseline --output output/pubmed_filtered
"""

import argparse
import os
import sys

from pyspark.sql import functions as F
from pyspark.sql.types import ArrayType, StringType

sys.path.append(os.path.dirname(__file__))
from config import CANCER_IMMUNOLOGY_MESH_TERMS, get_spark  # noqa: E402


def build_arg_parser():
    p = argparse.ArgumentParser()
    p.add_argument("--input", required=True, help="Directory of pubmed*.xml.gz files")
    p.add_argument("--output", required=True, help="Output parquet directory")
    return p


def parse_pubmed(spark, input_glob: str):
    """Load PubMed baseline XML into a DataFrame of one row per PubmedArticle."""
    df = (
        spark.read.format("xml")
        .option("rowTag", "PubmedArticle")
        .option("compression", "gzip")
        .load(input_glob)
    )
    return df


def extract_fields(df):
    """Pull out PMID, title, abstract, year, journal, and MeSH heading list."""
    article = df["MedlineCitation.Article"]
    mesh_list = df["MedlineCitation.MeshHeadingList.MeshHeading"]

    out = df.select(
        df["MedlineCitation.PMID._VALUE"].cast("long").alias("pmid"),
        article["ArticleTitle"].alias("title"),
        # Abstract text can be a single string or an array of AbstractText
        # sections (structured abstracts) depending on the record.
        article["Abstract"]["AbstractText"].alias("abstract_raw"),
        article["Journal"]["JournalIssue"]["PubDate"]["Year"].alias("pub_year"),
        article["Journal"]["Title"].alias("journal"),
        mesh_list.alias("mesh_raw"),
    )
    return out


def normalize_mesh(df):
    """
    mesh_raw is an array of structs with DescriptorName (._VALUE) and
    optional QualifierName. Build a flat array of "Descriptor/Qualifier"
    strings (matching the format used in CANCER_IMMUNOLOGY_MESH_TERMS),
    plus the bare descriptor names, so filtering is robust either way.
    """
    def flatten_mesh(mesh_array):
        # mesh_array elements: {DescriptorName: {_VALUE}, QualifierName: {_VALUE} or list}
        terms = []
        if mesh_array is None:
            return terms
        for heading in mesh_array:
            try:
                descriptor = heading["DescriptorName"]["_VALUE"]
            except (TypeError, KeyError):
                continue
            terms.append(descriptor)
            qualifiers = heading.get("QualifierName") if isinstance(heading, dict) else None
            if qualifiers:
                if isinstance(qualifiers, list):
                    for q in qualifiers:
                        qv = q.get("_VALUE") if isinstance(q, dict) else q
                        if qv:
                            terms.append(f"{descriptor}/{qv}")
                else:
                    qv = qualifiers.get("_VALUE") if isinstance(qualifiers, dict) else qualifiers
                    if qv:
                        terms.append(f"{descriptor}/{qv}")
        return terms

    flatten_udf = F.udf(flatten_mesh, ArrayType(StringType()))
    return df.withColumn("mesh_terms", flatten_udf(F.col("mesh_raw"))).drop("mesh_raw")


def flatten_abstract(df):
    """abstract_raw may be a string, a struct, or an array of structs."""
    def to_text(val):
        if val is None:
            return None
        if isinstance(val, str):
            return val
        if isinstance(val, list):
            parts = []
            for v in val:
                if isinstance(v, dict):
                    parts.append(v.get("_VALUE", ""))
                elif isinstance(v, str):
                    parts.append(v)
            return " ".join(p for p in parts if p)
        if isinstance(val, dict):
            return val.get("_VALUE")
        return str(val)

    to_text_udf = F.udf(to_text, StringType())
    return df.withColumn("abstract", to_text_udf(F.col("abstract_raw"))).drop("abstract_raw")


def filter_by_mesh(df, mesh_terms):
    """Keep rows where any mesh_terms entry matches (exactly, or by
    descriptor prefix) any term in our target list."""
    bare_terms = {t.split("/")[0] for t in mesh_terms}
    target_terms = set(mesh_terms) | bare_terms

    def matches(term_array):
        if not term_array:
            return False
        return any(t in target_terms for t in term_array)

    matches_udf = F.udf(matches, "boolean")
    return df.filter(matches_udf(F.col("mesh_terms")))


def main():
    args = build_arg_parser().parse_args()
    spark = get_spark("pubmed-cancer-immunology-parser")

    input_glob = os.path.join(args.input, "*.xml.gz")
    raw = parse_pubmed(spark, input_glob)
    fields = extract_fields(raw)
    with_mesh = normalize_mesh(fields)
    with_abstract = flatten_abstract(with_mesh)

    filtered = filter_by_mesh(with_abstract, CANCER_IMMUNOLOGY_MESH_TERMS)
    filtered = filtered.filter(F.col("pmid").isNotNull())

    count = filtered.count()
    print(f"[pubmed_parser] Matched {count} cancer-immunology articles.")

    filtered.write.mode("overwrite").parquet(args.output)
    print(f"[pubmed_parser] Wrote filtered results to {args.output}")

    spark.stop()


if __name__ == "__main__":
    main()
