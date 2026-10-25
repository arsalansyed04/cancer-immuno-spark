#!/usr/bin/env bash
# Download PubMed baseline XML files.
#
# Usage:
#   bash scripts/download_pubmed.sh [options]
#
# Options:
#   -n N             Download first N files (default: 10)
#   --files f1,f2,…  Download specific basenames (e.g. pubmed26n0850,pubmed26n0950)
#   --validation     Download the 10 DA3 validation files (config.VALIDATION_FILES)
#   --all            Download all baseline files (tens of GB — be patient)
#
# Files are saved to data/pubmed_baseline/ relative to the repo root.

set -euo pipefail

BASE_URL="https://ftp.ncbi.nlm.nih.gov/pubmed/baseline/"
REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
OUT_DIR="${PUBMED_DIR:-$REPO_ROOT/data/pubmed_baseline}"
mkdir -p "$OUT_DIR"

# Default: first 10 files
MODE="count"
COUNT=10
FILE_LIST=""

while [[ $# -gt 0 ]]; do
    case "$1" in
        -n)       MODE="count";      COUNT="$2";    shift 2 ;;
        --files)  MODE="explicit";   FILE_LIST="$2"; shift 2 ;;
        --validation)
                  MODE="explicit"
                  FILE_LIST="pubmed26n0850,pubmed26n0950,pubmed26n1030,pubmed26n1100,pubmed26n1160,pubmed26n1210,pubmed26n1250,pubmed26n1290,pubmed26n1320,pubmed26n1334"
                  shift ;;
        --all)    MODE="all";        shift ;;
        *) echo "Unknown option: $1"; exit 1 ;;
    esac
done

if [[ "$MODE" == "explicit" ]]; then
    # Build filenames from comma-separated basenames
    LISTING=""
    IFS=',' read -ra BASES <<< "$FILE_LIST"
    for base in "${BASES[@]}"; do
        base="${base// /}"           # strip spaces
        LISTING="$LISTING ${base}.xml.gz"
    done
    LISTING="${LISTING# }"           # trim leading space
else
    echo "Fetching file listing from $BASE_URL ..."
    LISTING=$(curl -s "$BASE_URL" | grep -oE 'pubmed[0-9]+n[0-9]+\.xml\.gz' | sort -u)
    if [[ "$MODE" == "count" ]]; then
        LISTING=$(echo "$LISTING" | head -n "$COUNT")
    fi
fi

COUNT_FILES=$(echo "$LISTING" | wc -w)
echo "Downloading ${COUNT_FILES} file(s) into $OUT_DIR ..."

for f in $LISTING; do
    # Strip directory prefix if present
    f="${f##*/}"
    if [[ -f "$OUT_DIR/$f" ]]; then
        echo "  skip (exists): $f"
        continue
    fi
    echo "  fetching: $f"
    wget -q --show-progress -O "$OUT_DIR/$f" "${BASE_URL}${f}"
    # MD5 integrity check
    wget -q -O "$OUT_DIR/$f.md5" "${BASE_URL}${f}.md5" || true
    if [[ -f "$OUT_DIR/$f.md5" ]]; then
        (cd "$OUT_DIR" && md5sum -c "$f.md5") \
            || echo "  WARNING: md5 mismatch for $f"
    fi
done

echo "Done. Files in $OUT_DIR:"
ls -lh "$OUT_DIR"/*.xml.gz 2>/dev/null || echo "(none downloaded)"
