# conftest.py — shared pytest fixtures for the cancer-immuno-spark test suite

import pytest


@pytest.fixture(scope="session")
def spark():
    """Local-mode SparkSession shared across the test session."""
    try:
        from pyspark.sql import SparkSession
    except ImportError:
        pytest.skip("pyspark is not installed in the current environment")

    session = (
        SparkSession.builder
        .master("local[2]")
        .appName("cancer-immuno-spark-tests")
        .config("spark.sql.shuffle.partitions", "2")
        .config("spark.ui.enabled", "false")
        .getOrCreate()
    )
    session.sparkContext.setLogLevel("ERROR")
    yield session
    session.stop()
