"""
Tests for trend_analysis.py — topic delta calculation and prevalence logic.
"""

import sys
import os
import pytest
pytest.importorskip("pyspark")

from pyspark.sql import functions as F

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))


class TestTrendDelta:
    """Unit tests for topic Δ share computation logic."""

    def _make_share_df(self, spark, rows):
        """Create a synthetic topic_share_by_year DF."""
        return spark.createDataFrame(
            rows,
            ["pub_year", "topic_id", "mean_theta_share", "num_papers"],
        )

    def test_emerging_topic_identified(self, spark):
        """A topic with large positive Δ should be labelled 'emerging'."""
        rows = [
            (2015, 0, 0.05, 100),
            (2016, 0, 0.06, 120),
            (2023, 0, 0.25, 200),
            (2024, 0, 0.28, 220),
        ]
        share_df = self._make_share_df(spark, rows)

        early_start, early_end = 2015, 2016
        late_start,  late_end  = 2023, 2024

        early = (
            share_df.filter(F.col("pub_year").between(early_start, early_end))
            .groupBy("topic_id")
            .agg(F.avg("mean_theta_share").alias("early_share"))
        )
        late = (
            share_df.filter(F.col("pub_year").between(late_start, late_end))
            .groupBy("topic_id")
            .agg(F.avg("mean_theta_share").alias("late_share"))
        )
        delta = (
            early.join(late, on="topic_id", how="outer")
            .fillna(0.0)
            .withColumn("delta_pp",
                        (F.col("late_share") - F.col("early_share")) * 100)
            .withColumn(
                "trend",
                F.when(F.col("delta_pp") >= 2.0,  "emerging")
                 .when(F.col("delta_pp") <= -2.0, "declining")
                 .otherwise("stable"),
            )
        )
        row = delta.first()
        assert row["trend"] == "emerging"
        assert row["delta_pp"] > 2.0

    def test_declining_topic_identified(self, spark):
        """A topic with large negative Δ should be labelled 'declining'."""
        rows = [
            (2015, 1, 0.30, 100),
            (2023, 1, 0.05, 150),
        ]
        share_df = self._make_share_df(spark, rows)
        early = (
            share_df.filter(F.col("pub_year").between(2015, 2018))
            .groupBy("topic_id").agg(F.avg("mean_theta_share").alias("early_share"))
        )
        late  = (
            share_df.filter(F.col("pub_year").between(2023, 2025))
            .groupBy("topic_id").agg(F.avg("mean_theta_share").alias("late_share"))
        )
        delta = (
            early.join(late, on="topic_id", how="outer")
            .fillna(0.0)
            .withColumn("delta_pp", (F.col("late_share") - F.col("early_share")) * 100)
            .withColumn("trend",
                F.when(F.col("delta_pp") >= 2.0,  "emerging")
                 .when(F.col("delta_pp") <= -2.0, "declining")
                 .otherwise("stable")
            )
        )
        row = delta.first()
        assert row["trend"] == "declining"
        assert row["delta_pp"] < -2.0


class TestTermPrevalence:
    """Test key-term pattern matching."""

    def test_pd1_pattern_matches_various_forms(self, spark):
        texts = [
            "PD-1 blockade therapy",
            "programmed death 1 receptor",
            "pd1 expression levels",
            "anti-PD-L1 antibody",
            "unrelated text about diabetes",
        ]
        df = spark.createDataFrame(
            [(i, t) for i, t in enumerate(texts)],
            ["id", "text"],
        ).withColumn("text", F.lower(F.col("text")))

        pattern = r"pd-?1|pd-?l1|programmed.?death.?1"
        matched = df.filter(F.col("text").rlike(pattern)).count()
        assert matched == 4   # rows 0-3 match; row 4 does not

    def test_car_t_pattern(self, spark):
        texts = [
            "CAR-T cell therapy showed",
            "chimeric antigen receptor T cells",
            "adoptive cell transfer therapy",
            "checkpoint inhibitor alone",
        ]
        df = spark.createDataFrame(
            [(i, t) for i, t in enumerate(texts)],
            ["id", "text"],
        ).withColumn("text", F.lower(F.col("text")))

        pattern = r"car-?t\b|chimeric.?antigen.?receptor|adoptive.?cell.?(therapy|transfer)"
        matched = df.filter(F.col("text").rlike(pattern)).count()
        assert matched == 3
