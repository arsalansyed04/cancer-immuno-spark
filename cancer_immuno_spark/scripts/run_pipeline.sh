#!/usr/bin/env bash
# Orchestrates the full pipeline: PubMed parse -> IEDB load -> join/analysis.
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"

MYSQL_URL="${MYSQL_URL:-jdbc:mysql://localhost:3306/iedb_public}"
MYSQL_USER="${MYSQL_USER:-spark}"
MYSQL_PASS="${MYSQL_PASS:-sparkpw}"
MYSQL_JAR="${MYSQL_JAR:-/opt/spark/jars_extra/mysql-connector-j-8.4.0.jar}"
PUBMED_INPUT="${PUBMED_INPUT:-data/pubmed_baseline}"
PUBMED_OUTPUT="${PUBMED_OUTPUT:-output/pubmed_filtered}"
echo "== Step 1: Parsing + filtering PubMed baseline XML =="
spark-submit \
  --packages com.databricks:spark-xml_2.12:0.18.0 \
  --driver-memory 4g --executor-memory 4g \
  src/pubmed_parser.py \
  --input "$PUBMED_INPUT" \
  --output "$PUBMED_OUTPUT"

echo "== Step 2: Loading IEDB tables from MySQL =="
spark-submit \
  --jars "$MYSQL_JAR" \
  --driver-memory 4g \
  src/iedb_loader.py \
  --jdbc-url "$MYSQL_URL" \
  --user "$MYSQL_USER" \
  --password "$MYSQL_PASS" \
  --output output/iedb_tables

echo "== Step 3: Joining + analyzing =="
spark-submit \
  --driver-memory 4g --executor-memory 4g \
  src/join_analysis.py \
  --pubmed "$PUBMED_OUTPUT" \
  --iedb output/iedb_tables \
  --output output/analysis

echo "== Done. Results in output/analysis/ =="
