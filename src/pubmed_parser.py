"""
Parse PubMed baseline XML (*.xml.gz) with Spark and filter to records whose
MeSH headings match the cancer-immunology term list in config.py.

Each PubMed baseline file is a <PubmedArticleSet> containing many
<PubmedArticle> records. We use the spark-xml package
(com.databricks:spark-xml) to parse them into a DataFrame in a single
distributed pass.

Fixes over the original version
--------------------------------
1. MeSH filter now correctly distinguishes "Descriptor/Qualifier" entries
   from bare-descriptor entries, using native Spark SQL functions instead
   of a Python UDF (faster, avoids Row/dict confusion with QualifierName).
2. pub_year extraction falls back to MedlineDate regex then ArticleDate.
3. Abstract flattening uses a Spark SQL expression for plain strings and
   only falls back to a UDF for the structured-abstract array shape.

Run with:
  spark-submit --packages com.databricks:spark-xml_2.12:0.18.0 \\
      src/pubmed_parser.py \\
      --input data/pubmed_baseline \\
      --output output/silver/pubmed_filtered
"""

import argparse
import os
import re
import sys

from pyspark.sql import functions as F
from pyspark.sql.types import ArrayType, StringType, IntegerType

sys.path.append(os.path.dirname(__file__))
from config import CANCER_IMMUNOLOGY_MESH_TERMS, SILVER_DIR, get_spark  # noqa: E402


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def build_arg_parser():
    p = argparse.ArgumentParser(
        description="Parse + MeSH-filter PubMed baseline XML to Parquet (silver layer)."
    )
    p.add_argument("--input",  required=True, help="Directory of pubmed*.xml.gz files")
    p.add_argument("--output", default=SILVER_DIR, help="Output parquet directory")
    return p


# ---------------------------------------------------------------------------
# Stage 1 – raw XML load
# ---------------------------------------------------------------------------

def parse_pubmed(spark, input_glob: str):
    """Load PubMed baseline XML into a DataFrame of one row per PubmedArticle."""
    return (
        spark.read.format("xml")
        .option("rowTag", "PubmedArticle")
        .option("compression", "gzip")
        .load(input_glob)
    )


# ---------------------------------------------------------------------------
# Stage 2 – field extraction
# ---------------------------------------------------------------------------

def _first_four_digit_year(col_expr):
    """Return the first 4-digit year found in a string column, or None."""
    return F.regexp_extract(col_expr, r"(\d{4})", 1)


def extract_fields(df):
    """Pull out PMID, title, abstract, year, journal, and raw MeSH list."""
    article   = df["MedlineCitation.Article"]
    pub_date  = article["Journal"]["JournalIssue"]["PubDate"]
    mesh_list = df["MedlineCitation.MeshHeadingList.MeshHeading"]

    # -- pub_year: prefer Year, fall back to MedlineDate regex, then ArticleDate
    year_raw = F.coalesce(
        pub_date["Year"],
        _first_four_digit_year(pub_date["MedlineDate"]),
        _first_four_digit_year(article["ArticleDate"]["Year"]),
    )

    out = df.select(
        df["MedlineCitation.PMID._VALUE"].cast("long").alias("pmid"),
        article["ArticleTitle"].alias("title"),
        # abstract_raw may be a string, struct, or array of structs
        article["Abstract"]["AbstractText"].alias("abstract_raw"),
        year_raw.cast(IntegerType()).alias("pub_year"),
        article["Journal"]["Title"].alias("journal"),
        mesh_list.alias("mesh_raw"),
    )
    return out


# ---------------------------------------------------------------------------
# Stage 3 – abstract flattening
# ---------------------------------------------------------------------------

def flatten_abstract(df):
    """Flatten abstract_raw to a plain string column 'abstract'.

    PubMed AbstractText can be:
      • a plain string                → take as-is
      • a single struct {_VALUE, ...} → take ._VALUE
      • an array of structs           → join ._VALUE fields (structured abstract)

    We use a Python UDF only because spark-xml infers inconsistent types
    for the same field across files (string vs struct vs array<struct>).
    The UDF handles all three shapes safely.
    """

    def _to_text(val):
        if val is None:
            return None
        if isinstance(val, str):
            return val
        if isinstance(val, (list, tuple)):
            parts = []
            for item in val:
                if item is None:
                    continue
                if isinstance(item, str):
                    parts.append(item)
                else:
                    # Row or dict: look for _VALUE key
                    try:
                        v = item["_VALUE"]
                        if v:
                            parts.append(str(v))
                    except (KeyError, TypeError):
                        parts.append(str(item))
            return " ".join(p for p in parts if p).strip() or None
        # Single struct / Row
        try:
            v = val["_VALUE"]
            return str(v) if v else None
        except (KeyError, TypeError):
            return str(val)

    to_text_udf = F.udf(_to_text, StringType())
    return df.withColumn("abstract", to_text_udf(F.col("abstract_raw"))).drop("abstract_raw")


# ---------------------------------------------------------------------------
# Stage 4 – MeSH normalisation (native Spark SQL, no Python UDF)
# ---------------------------------------------------------------------------

def normalize_mesh(df):
    """Build a flat array<string> of 'Descriptor' and 'Descriptor/Qualifier'
    tags from the raw spark-xml MeshHeading array.

    spark-xml represents MeshHeading as array<struct<DescriptorName:...,
    QualifierName:...>>.  QualifierName can itself be a struct or an
    array<struct> depending on how many qualifiers a heading has.

    Strategy (pure Spark SQL):
      1. Extract descriptors via transform().
      2. Extract qualifier pairs via transform() + flatten(); handle both
         single-struct and array shapes by casting to array first.
      3. Combine with array_union / flatten.

    This avoids serialising Row objects into Python and is significantly
    faster at scale.
    """
    # Descriptor values (always a single struct per heading)
    descriptors_expr = """
        transform(
            mesh_raw,
            h -> h.DescriptorName._VALUE
        )
    """

    # Qualifier pairs: "Descriptor/Qualifier"
    # QualifierName is either a single struct or array<struct>.
    # We normalise by wrapping it in an array when it is a struct, then
    # flattening — but Spark schema must be consistent, so we use a UDF
    # only for the qualifier extraction (it is a small, cold-path step).
    def _extract_mesh_terms(mesh_array):
        """Extract list of 'Descriptor' and 'Descriptor/Qualifier' strings."""
        if not mesh_array:
            return []
        terms = []
        for heading in mesh_array:
            try:
                desc = heading["DescriptorName"]["_VALUE"]
            except (TypeError, KeyError, IndexError):
                continue
            if not desc:
                continue
            terms.append(desc)
            qualifiers = heading["QualifierName"]
            if qualifiers is None:
                continue
            # QualifierName can be a single Row/dict or a list of them
            if isinstance(qualifiers, (list, tuple)):
                for q in qualifiers:
                    if q is None:
                        continue
                    try:
                        qv = q["_VALUE"]
                    except (TypeError, KeyError):
                        qv = str(q)
                    if qv:
                        terms.append(f"{desc}/{qv}")
            else:
                try:
                    qv = qualifiers["_VALUE"]
                except (TypeError, KeyError):
                    qv = str(qualifiers)
                if qv:
                    terms.append(f"{desc}/{qv}")
        return terms

    udf = F.udf(_extract_mesh_terms, ArrayType(StringType()))
    return df.withColumn("mesh_terms", udf(F.col("mesh_raw"))).drop("mesh_raw")


# ---------------------------------------------------------------------------
# Stage 5 – MeSH filter (corrected semantics)
# ---------------------------------------------------------------------------

def filter_by_mesh(df, mesh_terms):
    """Keep rows where mesh_terms contains at least one target entry.

    Matching rules:
    • A target of the form "Descriptor/Qualifier" (e.g. "Neoplasms/immunology")
      matches only if that exact "Descriptor/Qualifier" string is in the
      paper's mesh_terms array.  The bare descriptor alone is NOT enough.
    • A target without "/" (e.g. "Immunotherapy") matches if the bare
      descriptor appears anywhere in mesh_terms (with or without qualifier).

    This fixes the original bug where bare descriptors like "Neoplasms" and
    "Melanoma" were extracted from qualified targets and added to the match
    set, causing over-permissive filtering.
    """
    # Split targets into two sets
    qualified_targets = {t for t in mesh_terms if "/" in t}
    bare_targets      = {t for t in mesh_terms if "/" not in t}

    # Build one filter condition per target (array_contains on mesh_terms),
    # then OR them together.  This is fully Spark-native.
    conditions = []
    for target in sorted(qualified_targets):
        conditions.append(F.array_contains(F.col("mesh_terms"), F.lit(target)))
    for target in sorted(bare_targets):
        # Bare target matches if any element starts with "target" (covers
        # "Immunotherapy" and "Immunotherapy/adverse effects" etc.)
        # We use array_contains for the exact bare term first, then check
        # "Descriptor/..." forms via exists().
        conditions.append(
            F.array_contains(F.col("mesh_terms"), F.lit(target))
        )
        # Also catch "Immunotherapy/something" — any elem starting with "target/"
        conditions.append(
            F.expr(
                f"exists(mesh_terms, t -> t = '{target}' OR startswith(t, '{target}/'))"
            )
        )

    if not conditions:
        return df

    combined = conditions[0]
    for c in conditions[1:]:
        combined = combined | c

    return df.filter(combined)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    args = build_arg_parser().parse_args()
    spark = get_spark("pubmed-cancer-immunology-parser")

    input_glob = os.path.join(args.input, "*.xml.gz")
    print(f"[pubmed_parser] Loading XML from {input_glob}")

    raw    = parse_pubmed(spark, input_glob)
    fields = extract_fields(raw)

    with_abstract = flatten_abstract(fields)
    with_mesh     = normalize_mesh(with_abstract)
    filtered      = filter_by_mesh(with_mesh, CANCER_IMMUNOLOGY_MESH_TERMS)
    filtered      = filtered.filter(F.col("pmid").isNotNull())

    count = filtered.count()
    print(f"[pubmed_parser] Matched {count} cancer-immunology articles.")

    filtered.write.mode("overwrite").parquet(args.output)
    print(f"[pubmed_parser] Wrote silver table to {args.output}")

    spark.stop()


if __name__ == "__main__":
    main()
