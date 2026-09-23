"""
Load relevant IEDB tables from a MySQL database (populated from
iedb_public.sql.gz) into Spark via JDBC, and write them out as parquet for
fast downstream joins.

Run with:
  spark-submit --jars /opt/spark/jars_extra/mysql-connector-j-8.4.0.jar \
      iedb_loader.py --jdbc-url "jdbc:mysql://localhost:3306/iedb_public" \
      --user spark --password sparkpw --output output/iedb_tables
"""

import argparse
import os
import sys

sys.path.append(os.path.dirname(__file__))
from config import IEDB_TABLES, get_spark  # noqa: E402


def build_arg_parser():
    p = argparse.ArgumentParser()
    p.add_argument("--jdbc-url", required=True, help='e.g. "jdbc:mysql://localhost:3306/iedb_public"')
    p.add_argument("--user", required=True)
    p.add_argument("--password", required=True)
    p.add_argument("--output", required=True, help="Output parquet directory (one subdir per table)")
    p.add_argument(
        "--num-partitions",
        type=int,
        default=8,
        help="Parallel JDBC read partitions for large tables (epitope/curation_object)",
    )
    return p


def read_table(spark, jdbc_url, user, password, table, partition_column=None, num_partitions=8):
    reader = (
        spark.read.format("jdbc")
        .option("url", jdbc_url)
        .option("dbtable", table)
        .option("user", user)
        .option("password", password)
        .option("driver", "com.mysql.cj.jdbc.Driver")
        .option("fetchsize", "10000")
    )
    if partition_column:
        # Parallelize the read across executors using a numeric PK range.
        bounds = (
            spark.read.format("jdbc")
            .option("url", jdbc_url)
            .option("query", f"SELECT MIN({partition_column}) lo, MAX({partition_column}) hi FROM {table}")
            .option("user", user)
            .option("password", password)
            .option("driver", "com.mysql.cj.jdbc.Driver")
            .load()
            .collect()[0]
        )
        if bounds["lo"] is not None and bounds["hi"] is not None:
            reader = (
                reader.option("partitionColumn", partition_column)
                .option("lowerBound", str(bounds["lo"]))
                .option("upperBound", str(bounds["hi"]))
                .option("numPartitions", str(num_partitions))
            )
    return reader.load()


def main():
    args = build_arg_parser().parse_args()
    spark = get_spark("iedb-loader")

    # Primary-key column guesses for parallel partitioned reads; adjust if
    # your dump's schema differs (check with DESCRIBE <table>;).
    pk_hints = {
        "article": "article_id",
        "curated_epitope": "curated_epitope_id",
        "epitope": "epitope_id",
        "object": "object_id",
    }

    for logical_name, table in IEDB_TABLES.items():
        print(f"[iedb_loader] Reading table '{table}' ...")
        try:
            df = read_table(
                spark,
                args.jdbc_url,
                args.user,
                args.password,
                table,
                partition_column=pk_hints.get(logical_name),
                num_partitions=args.num_partitions,
            )
        except Exception as e:  # noqa: BLE001
            print(f"[iedb_loader] WARNING: failed to read '{table}': {e}")
            print("  -> Check table name / column names in src/config.py against your dump.")
            continue

        out_path = os.path.join(args.output, logical_name)
        df.write.mode("overwrite").parquet(out_path)
        print(f"[iedb_loader] Wrote {table} -> {out_path} ({df.rdd.getNumPartitions()} partitions)")

    spark.stop()


if __name__ == "__main__":
    main()
