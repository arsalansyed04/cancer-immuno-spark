#!/usr/bin/env bash
# Download PubMed baseline XML files.
# Usage: download_pubmed.sh [N]
#   N = number of files to fetch (default 10, for testing). Pass "all" for
#       the full baseline (~1,700 files, tens of GB).
set -euo pipefail

BASE_URL="https://ftp.ncbi.nlm.nih.gov/pubmed/baseline/"
OUT_DIR="$(dirname "$0")/../data/pubmed_baseline"
mkdir -p "$OUT_DIR"

N="${1:-10}"

echo "Fetching file listing from $BASE_URL ..."
LISTING=$(curl -s "$BASE_URL" | grep -oE 'pubmed[0-9]+n[0-9]+\.xml\.gz' | sort -u)

if [[ "$N" != "all" ]]; then
    LISTING=$(echo "$LISTING" | head -n "$N")
fi

echo "Downloading $(echo "$LISTING" | wc -l) file(s) into $OUT_DIR ..."
for f in $LISTING; do
    if [[ -f "$OUT_DIR/$f" ]]; then
        echo "  skip (exists): $f"
        continue
    fi
    echo "  fetching: $f"
    wget -q --show-progress -O "$OUT_DIR/$f" "${BASE_URL}${f}"
    # Optional integrity check against NCBI's md5 sidecar file
    wget -q -O "$OUT_DIR/$f.md5" "${BASE_URL}${f}.md5" || true
    if [[ -f "$OUT_DIR/$f.md5" ]]; then
        (cd "$OUT_DIR" && md5sum -c "$f.md5") || echo "  WARNING: md5 mismatch for $f"
    fi
done

echo "Done. Files in $OUT_DIR:"
ls -la "$OUT_DIR"
