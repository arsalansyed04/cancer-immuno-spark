#!/usr/bin/env bash
# Run the full pipeline on the 10 DA3 validation files.
#
# Prerequisites:
#   - spark-submit on PATH  (SPARK_HOME set)
#   - Python environment with requirements.txt + requirements-rag.txt
#   - wget (for downloading)
#   - (optional) MySQL with IEDB dump for the join stage
#
# Usage:
#   bash scripts/run_validation.sh
#   SKIP_IEDB=1 SKIP_JOIN=1 bash scripts/run_validation.sh

set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"

echo "================================================================"
echo "== DA3 Validation run: 10 PubMed baseline files"
echo "================================================================"

# Step 1: Download the 10 validation files
echo "-- Downloading validation files ..."
bash scripts/download_pubmed.sh --validation

# Step 2: Run the full pipeline (skip IEDB by default unless MySQL is set up)
PUBMED_N_FILES=0 \
PUBMED_DIR=data/pubmed_baseline \
SILVER_DIR=output/silver/pubmed_filtered \
GOLD_DIR=output/gold \
SKIP_DOWNLOAD=1 \
    bash scripts/run_pipeline.sh

echo ""
echo "================================================================"
echo "== Validation run complete.  Outputs in output/gold/"
echo "================================================================"
echo "To view the dashboard: streamlit run dashboard/app.py"
