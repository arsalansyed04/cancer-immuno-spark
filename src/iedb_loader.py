"""
Load relevant IEDB tables from a MySQL database (populated from
iedb_public.sql.gz) into Spark via JDBC, and write them as Parquet
for fast downstream joins.

IEDB column reference (iedb_public release 3.x):
  article.article_id, article.pubmed_id, article.reference_id,
  article.authors, article.article_title, article.journal_name
  curated_epitope.curated_epitope_id, curated_epitope.reference_id,
  curated_epitope.e_object_id
  object.object_id, object.object_description

Verify against your imported dump with `DESCRIBE <table>;` and update
IEDB_TABLES in src/config.py if column names differ.

Run with:
  spark-submit --jars /opt/spark/jars_extra/mysql-connector-j-8.4.0.jar \\
      src/iedb_loader.py \\
      --jdbc-url "jdbc:mysql://localhost:3306/iedb_public" \\
      --user spark --password sparkpw \\
      --output output/iedb_tables
"""

import argparse
import os
import sys

sys.path.append(os.path.dirname(__file__))
from config import IEDB_TABLES, get_spark  # noqa: E402


def build_arg_parser():
    p = argparse.ArgumentParser(
        description="Load IEDB MySQL tables to Parquet (for JDBC pipeline)."
    )
    p.add_argument("--jdbc-url",        required=True)
    p.add_argument("--user",            required=True)
    p.add_argument("--password",        required=True)
    p.add_argument("--output",          required=True,
                   help="Output directory; one subfolder per table")
    p.add_argument("--num-partitions",  type=int, default=8)
    return p


def read_table(spark, jdbc_url, user, password, table,
               partition_column=None, num_partitions=8):
    """Read a MySQL table into a Spark DataFrame via JDBC."""
    reader = (
        spark.read.format("jdbc")
        .option("url",       jdbc_url)
        .option("dbtable",   table)
        .option("user",      user)
        .option("password",  password)
        .option("driver",    "com.mysql.cj.jdbc.Driver")
        .option("fetchsize", "10000")
    )
    if partition_column:
        bounds = (
            spark.read.format("jdbc")
            .option("url",      jdbc_url)
            .option("query",    f"SELECT MIN({partition_column}) lo, "
                                f"MAX({partition_column}) hi FROM {table}")
            .option("user",     user)
            .option("password", password)
            .option("driver",   "com.mysql.cj.jdbc.Driver")
            .load()
            .collect()[0]
        )
        if bounds["lo"] is not None and bounds["hi"] is not None:
            reader = (
                reader
                .option("partitionColumn", partition_column)
                .option("lowerBound",      str(bounds["lo"]))
                .option("upperBound",      str(bounds["hi"]))
                .option("numPartitions",   str(num_partitions))
            )
    return reader.load()


def main():
    args  = build_arg_parser().parse_args()
    spark = get_spark("iedb-loader")

    # Numeric primary key hints for parallel partitioned reads
    pk_hints = {
        "article":         "article_id",
        "curated_epitope": "curated_epitope_id",
        "epitope":         "epitope_id",
        "object":          "object_id",
    }

    for logical_name, table in IEDB_TABLES.items():
        print(f"[iedb_loader] Reading table '{table}' ...")
        try:
            df = read_table(
                spark, args.jdbc_url, args.user, args.password,
                table,
                partition_column=pk_hints.get(logical_name),
                num_partitions=args.num_partitions,
            )
        except Exception as e:
            print(f"[iedb_loader] WARNING: failed to read '{table}': {e}")
            print("  -> Verify table/column names against your dump "
                  "(`SHOW TABLES; DESCRIBE <table>;`) and update "
                  "IEDB_TABLES in src/config.py.")
            continue

        out_path = os.path.join(args.output, logical_name)
        df.write.mode("overwrite").parquet(out_path)
        n = df.count()
        print(f"[iedb_loader] {table} -> {out_path} ({n} rows, "
              f"{df.rdd.getNumPartitions()} partitions)")

    spark.stop()


if __name__ == "__main__":
    main()
