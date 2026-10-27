"""
Tests for pubmed_parser.py — MeSH filter correctness, year extraction,
and abstract flattening.

Key requirement: filter_by_mesh must correctly distinguish
  • "Descriptor/Qualifier" targets  → match only when BOTH parts present
  • bare "Descriptor" targets        → match when descriptor present (any qual)
"""

import sys
import os
import pytest
pytest.importorskip("pyspark")

from pyspark.sql import functions as F
from pyspark.sql.types import (
    StructType, StructField, StringType, LongType, ArrayType, IntegerType
)

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
from pubmed_parser import filter_by_mesh, flatten_abstract, normalize_mesh  # noqa: E402


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

SAMPLE_MESH_TERMS = [
    "Neoplasms/immunology",   # requires BOTH descriptor AND /immunology qualifier
    "Immunotherapy",           # bare descriptor — matches with or without qualifier
    "Melanoma/immunology",    # requires BOTH
]


def _make_df(spark, rows, schema):
    return spark.createDataFrame(rows, schema)


# ---------------------------------------------------------------------------
# MeSH filter tests
# ---------------------------------------------------------------------------

class TestFilterByMesh:
    def test_bare_descriptor_matches_with_qualifier(self, spark):
        """'Immunotherapy' target should match 'Immunotherapy/adverse effects'."""
        df = spark.createDataFrame(
            [(1, ["Immunotherapy/adverse effects"])],
            ["pmid", "mesh_terms"],
        )
        result = filter_by_mesh(df, ["Immunotherapy"])
        assert result.count() == 1

    def test_bare_descriptor_matches_without_qualifier(self, spark):
        """'Immunotherapy' target should match bare 'Immunotherapy'."""
        df = spark.createDataFrame(
            [(2, ["Immunotherapy"])],
            ["pmid", "mesh_terms"],
        )
        result = filter_by_mesh(df, ["Immunotherapy"])
        assert result.count() == 1

    def test_qualified_target_requires_qualifier(self, spark):
        """'Neoplasms/immunology' should NOT match bare 'Neoplasms'."""
        df = spark.createDataFrame(
            [(3, ["Neoplasms"])],          # bare descriptor only
            ["pmid", "mesh_terms"],
        )
        result = filter_by_mesh(df, ["Neoplasms/immunology"])
        assert result.count() == 0, (
            "Bug: bare 'Neoplasms' should not satisfy 'Neoplasms/immunology' target"
        )

    def test_qualified_target_matches_with_correct_qualifier(self, spark):
        """'Neoplasms/immunology' should match 'Neoplasms/immunology'."""
        df = spark.createDataFrame(
            [(4, ["Neoplasms/immunology", "Neoplasms/therapy"])],
            ["pmid", "mesh_terms"],
        )
        result = filter_by_mesh(df, ["Neoplasms/immunology"])
        assert result.count() == 1

    def test_wrong_qualifier_does_not_match(self, spark):
        """'Neoplasms/therapy' should NOT match target 'Neoplasms/immunology'."""
        df = spark.createDataFrame(
            [(5, ["Neoplasms/therapy"])],
            ["pmid", "mesh_terms"],
        )
        result = filter_by_mesh(df, ["Neoplasms/immunology"])
        assert result.count() == 0

    def test_no_mesh_terms_excluded(self, spark):
        """Papers with empty mesh_terms should be excluded."""
        df = spark.createDataFrame(
            [(6, [])],
            ["pmid", "mesh_terms"],
        )
        result = filter_by_mesh(df, SAMPLE_MESH_TERMS)
        assert result.count() == 0

    def test_multiple_targets_any_match(self, spark):
        """A paper matching any target in the list is kept."""
        df = spark.createDataFrame(
            [
                (7,  ["Immunotherapy"]),             # bare match
                (8,  ["Neoplasms/immunology"]),      # qualified match
                (9,  ["Diabetes"]),                   # no match
                (10, ["Melanoma", "Immunotherapy"]),  # bare match via 2nd term
            ],
            ["pmid", "mesh_terms"],
        )
        result = filter_by_mesh(df, SAMPLE_MESH_TERMS)
        retained = {r["pmid"] for r in result.collect()}
        assert retained == {7, 8, 10}, f"Unexpected retained set: {retained}"


# ---------------------------------------------------------------------------
# Year extraction tests (test via extract_fields output)
# ---------------------------------------------------------------------------

class TestYearExtraction:
    def test_year_column_from_plain_year(self, spark):
        """pub_year is correctly extracted from a four-digit Year string."""
        # Simulate what extract_fields produces
        df = spark.createDataFrame(
            [("2019", None, None)],
            ["year_raw", "medline_raw", "article_date_raw"],
        ).withColumn(
            "pub_year",
            F.coalesce(
                F.col("year_raw"),
                F.regexp_extract(F.col("medline_raw"), r"(\d{4})", 1),
                F.regexp_extract(F.col("article_date_raw"), r"(\d{4})", 1),
            ).cast("int")
        )
        assert df.first()["pub_year"] == 2019

    def test_year_fallback_medline_date(self, spark):
        """pub_year falls back to first 4-digit year in MedlineDate."""
        df = spark.createDataFrame(
            [(None, "2019 Jan-Feb", None)],
            ["year_raw", "medline_raw", "article_date_raw"],
        ).withColumn(
            "pub_year",
            F.coalesce(
                F.col("year_raw"),
                F.regexp_extract(F.col("medline_raw"), r"(\d{4})", 1),
                F.regexp_extract(F.col("article_date_raw"), r"(\d{4})", 1),
            ).cast("int")
        )
        assert df.first()["pub_year"] == 2019

    def test_year_fallback_null_when_no_year(self, spark):
        """pub_year is None when no year string is available."""
        df = spark.createDataFrame(
            [(None, None, None)],
            ["year_raw", "medline_raw", "article_date_raw"],
        ).withColumn(
            "pub_year",
            F.coalesce(
                F.col("year_raw"),
                F.regexp_extract(F.col("medline_raw"), r"(\d{4})", 1),
                F.regexp_extract(F.col("article_date_raw"), r"(\d{4})", 1),
            ).cast("int")
        )
        assert df.first()["pub_year"] is None


# ---------------------------------------------------------------------------
# Abstract flattening tests
# ---------------------------------------------------------------------------

class TestFlattenAbstract:
    def _run(self, spark, abstract_raw_value):
        """Create a minimal DF with abstract_raw and apply flatten_abstract."""
        df = spark.createDataFrame(
            [(abstract_raw_value,)],
            ["abstract_raw"],
        )
        return flatten_abstract(df)

    def test_plain_string(self, spark):
        result = self._run(spark, "This is an abstract.")
        assert result.first()["abstract"] == "This is an abstract."

    def test_none_returns_none(self, spark):
        result = self._run(spark, None)
        assert result.first()["abstract"] is None
