#!/usr/bin/env bash
# Full pipeline orchestration: PubMed parse → IEDB load → join → topics
# → trends → RAG (embed, index, evaluate) → importance → run summary.
#
# Each stage is skippable with environment variables:
#   SKIP_DOWNLOAD=1    SKIP_PARSE=1    SKIP_IEDB=1     SKIP_JOIN=1
#   SKIP_TOPICS=1      SKIP_TRENDS=1   SKIP_EMBED=1    SKIP_INDEX=1
#   SKIP_EVAL=1        SKIP_IMPORTANCE=1
#
# Fail-fast on any error; per-stage wall-clock is logged to
#   output/gold/run_summary.json
#
# Usage:
#   bash scripts/run_pipeline.sh
#   SKIP_IEDB=1 SKIP_JOIN=1 bash scripts/run_pipeline.sh

set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"

# ----- configurable paths (override via env) --------------------------------
PUBMED_DIR="${PUBMED_DIR:-data/pubmed_baseline}"
SILVER_DIR="${SILVER_DIR:-output/silver/pubmed_filtered}"
IEDB_TABLES="${IEDB_TABLES_DIR:-output/iedb_tables}"
GOLD_DIR="${GOLD_DIR:-output/gold}"
MYSQL_URL="${MYSQL_URL:-jdbc:mysql://localhost:3306/iedb_public}"
MYSQL_USER="${MYSQL_USER:-spark}"
MYSQL_PASS="${MYSQL_PASS:-sparkpw}"
MYSQL_JAR="${MYSQL_JAR:-/opt/spark/jars_extra/mysql-connector-j-8.4.0.jar}"
SPARK_SUBMIT="${SPARK_SUBMIT:-spark-submit}"
SPARK_XML_PKG="com.databricks:spark-xml_2.12:0.18.0"
DRIVER_MEM="${DRIVER_MEM:-4g}"
EXEC_MEM="${EXEC_MEM:-4g}"

mkdir -p "$GOLD_DIR"

# ----- timing helpers -------------------------------------------------------
declare -A STAGE_TIMES
_ts() { date +%s; }
_elapsed() { echo $(( $(_ts) - $1 )); }

run_stage() {
    local name="$1"; shift
    if [[ "${!name:-0}" == "1" || "${SKIP_ALL:-0}" == "1" ]]; then
        echo "-- skipping $name (SKIP_${name}=1)"
        return 0
    fi
    echo ""
    echo "================================================================"
    echo "== $name"
    echo "================================================================"
    local t0
    t0=$(_ts)
    "$@"
    STAGE_TIMES["$name"]=$(_elapsed "$t0")
    echo "-- $name done in ${STAGE_TIMES[$name]}s"
}

# ----- stages ---------------------------------------------------------------

download() {
    bash scripts/download_pubmed.sh "${PUBMED_N_FILES:-10}"
}

parse() {
    $SPARK_SUBMIT \
      --packages "$SPARK_XML_PKG" \
      --driver-memory "$DRIVER_MEM" --executor-memory "$EXEC_MEM" \
      src/pubmed_parser.py \
      --input  "$PUBMED_DIR" \
      --output "$SILVER_DIR"
}

load_iedb() {
    if [[ ! -f "$MYSQL_JAR" ]]; then
        echo "[run_pipeline] MYSQL_JAR not found at $MYSQL_JAR — skipping IEDB load"
        return 0
    fi
    $SPARK_SUBMIT \
      --jars "$MYSQL_JAR" \
      --driver-memory "$DRIVER_MEM" \
      src/iedb_loader.py \
      --jdbc-url "$MYSQL_URL" \
      --user     "$MYSQL_USER" \
      --password "$MYSQL_PASS" \
      --output   "$IEDB_TABLES"
}

join() {
    if [[ ! -d "$IEDB_TABLES/article" ]]; then
        echo "[run_pipeline] IEDB tables not found — skipping join"
        return 0
    fi
    $SPARK_SUBMIT \
      --driver-memory "$DRIVER_MEM" --executor-memory "$EXEC_MEM" \
      src/join_analysis.py \
      --pubmed  "$SILVER_DIR" \
      --iedb    "$IEDB_TABLES" \
      --output  "$GOLD_DIR"
}

topics() {
    $SPARK_SUBMIT \
      --driver-memory "$DRIVER_MEM" --executor-memory "$EXEC_MEM" \
      src/topic_modeling.py \
      --input  "$SILVER_DIR" \
      --output "$GOLD_DIR"
}

trends() {
    $SPARK_SUBMIT \
      --driver-memory "$DRIVER_MEM" --executor-memory "$EXEC_MEM" \
      src/trend_analysis.py \
      --topics "$GOLD_DIR/paper_topics.parquet" \
      --silver "$SILVER_DIR" \
      --output "$GOLD_DIR"
}

embed_rag() {
    python src/rag/embed.py \
      --silver "$SILVER_DIR" \
      --output "$GOLD_DIR"
}

index_rag() {
    python src/rag/index.py \
      --silver "$SILVER_DIR" \
      --gold   "$GOLD_DIR"
}

eval_rag() {
    python src/rag/evaluate.py \
      --silver "$SILVER_DIR" \
      --gold   "$GOLD_DIR"
}

importance() {
    python src/importance.py \
      --topics      "$GOLD_DIR/paper_topics.parquet" \
      --tfidf-vocab "$GOLD_DIR/topic_terms.csv" \
      --topic-delta "$GOLD_DIR/topic_delta.csv" \
      --gold        "$GOLD_DIR" \
      --silver      "$SILVER_DIR"
}

write_summary() {
    # Write run_summary.json to gold/
    python - <<'PYEOF'
import json, os, glob, time

gold = os.environ.get("GOLD_DIR", "output/gold")

# Count retained papers
try:
    import pyarrow.parquet as pq
    pt = pq.read_metadata(os.path.join(gold, "paper_topics.parquet"))
    retained = pt.num_rows
except Exception:
    retained = None

# Years covered
try:
    import pandas as pd
    yc = pd.read_csv(os.path.join(gold, "year_counts.csv"))
    years = sorted(yc["pub_year"].dropna().unique().tolist())
    years_covered = f"{int(years[0])}-{int(years[-1])}" if years else "unknown"
except Exception:
    years_covered = "unknown"

# Vocab size
try:
    import pandas as pd
    tt = pd.read_csv(os.path.join(gold, "topic_terms.csv"))
    vocab_size = tt["term"].nunique()
except Exception:
    vocab_size = None

summary = {
    "generated_at":   time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    "spark_master":   os.environ.get("SPARK_MASTER", "local[*]"),
    "files_processed": len(glob.glob("data/pubmed_baseline/*.xml.gz")),
    "records_scanned": None,      # fill from pubmed_parser log if available
    "papers_retained": retained,
    "vocab_size":      vocab_size,
    "years_covered":   years_covered,
    "stage_runtimes":  {},
}
os.makedirs(gold, exist_ok=True)
with open(os.path.join(gold, "run_summary.json"), "w") as f:
    json.dump(summary, f, indent=2)
print(f"[run_pipeline] Wrote run_summary.json to {gold}")
PYEOF
}

# ----- run ------------------------------------------------------------------
T0=$(_ts)

run_stage SKIP_DOWNLOAD  download
run_stage SKIP_PARSE     parse
run_stage SKIP_IEDB      load_iedb
run_stage SKIP_JOIN      join
run_stage SKIP_TOPICS    topics
run_stage SKIP_TRENDS    trends
run_stage SKIP_EMBED     embed_rag
run_stage SKIP_INDEX     index_rag
run_stage SKIP_EVAL      eval_rag
run_stage SKIP_IMPORTANCE importance

# Write run summary with collected timings
GOLD_DIR="$GOLD_DIR" python - <<PYEOF
import json, os
gold = os.environ["GOLD_DIR"]
path = os.path.join(gold, "run_summary.json")
with open(path) as f:
    summary = json.load(f)
summary["total_wall_clock_s"] = $(( $(_ts) - $T0 ))
with open(path, "w") as f:
    json.dump(summary, f, indent=2)
PYEOF

echo ""
echo "================================================================"
echo "== Pipeline complete in $(( $(_ts) - $T0 ))s"
echo "== Gold outputs in $GOLD_DIR"
echo "================================================================"
