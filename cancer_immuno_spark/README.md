# Distributed Analytics of Cancer Immunology Literature (Apache Spark)

Pipeline that fuses two sources at scale:

1. **PubMed Baseline (XML)** — `https://ftp.ncbi.nlm.nih.gov/pubmed/baseline/`
   Parsed with Spark, filtered to cancer-immunology MeSH terms.
2. **IEDB full dump (MySQL .sql.gz)** — `https://www.iedb.org/doc/iedb_public.sql.gz`
   Imported into MySQL, read into Spark via JDBC.

The two are joined on **PubMed ID (PMID)**, since IEDB's `reference` table stores
the PMID of the paper each epitope/assay was curated from. That gives you, per
paper: which MeSH terms it was tagged with, and which epitopes/antigens/assays
IEDB curators extracted from it — enabling questions like "which antigens are
most studied in melanoma-immunotherapy literature" or "epitope discovery rate
over publication years for lung cancer."

```
                 ┌────────────────────────┐        ┌──────────────────────────┐
                 │ PubMed baseline *.xml.gz│        │ iedb_public.sql.gz       │
                 └───────────┬─────────────┘        └────────────┬─────────────┘
                             │ spark-xml                          │ mysql import
                             ▼                                    ▼
                 pubmed_parser.py  ──parquet──►  iedb_loader.py ──parquet──►
                             │                                    │
                             └─────────────┬──────────────────────┘
                                           ▼
                                  join_analysis.py
                                           │
                                           ▼
                              output/*.parquet + *.csv summaries
```

## 1. Installation

Tested on Ubuntu 22.04/24.04. Run as a user with sudo.

### 1.1 Java (Spark requires JDK 8/11/17)
```bash
sudo apt-get update
sudo apt-get install -y openjdk-17-jdk
java -version
```

### 1.2 Apache Spark
```bash
SPARK_VERSION=3.5.3
HADOOP_VERSION=3
cd /opt
sudo wget https://downloads.apache.org/spark/spark-${SPARK_VERSION}/spark-${SPARK_VERSION}-bin-hadoop${HADOOP_VERSION}.tgz
sudo tar -xzf spark-${SPARK_VERSION}-bin-hadoop${HADOOP_VERSION}.tgz
sudo ln -s spark-${SPARK_VERSION}-bin-hadoop${HADOOP_VERSION} spark

echo 'export SPARK_HOME=/opt/spark' >> ~/.bashrc
echo 'export PATH=$SPARK_HOME/bin:$SPARK_HOME/sbin:$PATH' >> ~/.bashrc
echo 'export PYSPARK_PYTHON=python3' >> ~/.bashrc
source ~/.bashrc
```

For a multi-node cluster, install the same Spark build on every worker and
start it with `sbin/start-master.sh` / `sbin/start-worker.sh spark://MASTER:7077`,
then submit jobs with `--master spark://MASTER:7077`. Everything below also
works unmodified with `--master local[*]` on a single machine.

### 1.3 Python environment
```bash
sudo apt-get install -y python3-pip python3-venv
python3 -m venv ~/venvs/cancer-immuno
source ~/venvs/cancer-immuno/bin/activate
pip install -r requirements.txt
```

### 1.4 MySQL/MariaDB (to host the IEDB dump) + JDBC driver
```bash
sudo apt-get install -y mariadb-server mariadb-client
sudo systemctl enable --now mariadb
sudo mysql -e "CREATE DATABASE iedb_public; CREATE USER 'spark'@'%' IDENTIFIED BY 'sparkpw'; GRANT ALL ON iedb_public.* TO 'spark'@'%'; FLUSH PRIVILEGES;"

# JDBC connector Spark will use to read MySQL
mkdir -p /opt/spark/jars_extra
wget -O /opt/spark/jars_extra/mysql-connector-j-8.4.0.jar \
  https://repo1.maven.org/maven2/com/mysql/mysql-connector-j/8.4.0/mysql-connector-j-8.4.0.jar
```

### 1.5 spark-xml package (for PubMed XML parsing)
No manual jar download needed — it's pulled at submit time via `--packages`
(see the spark-submit commands below), Maven coordinate:
`com.databricks:spark-xml_2.12:0.18.0`.

## 2. Get the data

```bash
bash scripts/download_pubmed.sh        # pulls a subset (or all) of the baseline
bash scripts/setup_iedb.sh             # downloads + imports the IEDB dump into MySQL
```

`download_pubmed.sh` takes an optional file-count argument since the full
baseline is ~1,700 gzipped XML files (~40GB+) — start small while testing.

## 3. Run the pipeline

```bash
source ~/venvs/cancer-immuno/bin/activate

# Step 1: parse + MeSH-filter PubMed XML -> parquet
spark-submit \
  --packages com.databricks:spark-xml_2.12:0.18.0 \
  --driver-memory 4g --executor-memory 4g \
  src/pubmed_parser.py --input data/pubmed_baseline --output output/pubmed_filtered

# Step 2: load IEDB tables from MySQL -> parquet
spark-submit \
  --jars /opt/spark/jars_extra/mysql-connector-j-8.4.0.jar \
  --driver-memory 4g \
  src/iedb_loader.py --jdbc-url "jdbc:mysql://localhost:3306/iedb_public" \
  --user spark --password sparkpw --output output/iedb_tables

# Step 3: join PubMed (by MeSH term) with IEDB (by PMID) + analyze
spark-submit \
  --driver-memory 4g --executor-memory 4g \
  src/join_analysis.py \
  --pubmed output/pubmed_filtered --iedb output/iedb_tables --output output/analysis
```

Or run all three with `bash scripts/run_pipeline.sh`.

## 4. What you get in `output/analysis/`

- `epitope_counts_by_mesh_term.csv` — epitope/antigen mentions per cancer-immunology MeSH term
- `publications_per_year.csv` — literature + curated-epitope volume over time
- `top_antigens.csv` — most-studied antigens in the matched literature
- `joined_papers.parquet` — full joined table for further ad-hoc Spark SQL

## 5. Notes / things to adjust

- **MeSH term list**: edit `CANCER_IMMUNOLOGY_MESH_TERMS` in `src/config.py` — the
  provided list is a reasonable starting set (see below) but MeSH curation is
  broad; refine to your scope (e.g., add specific cancer types).
- **IEDB schema**: the IEDB public MySQL dump's exact table/column names shift
  slightly between releases. `src/iedb_loader.py` expects tables named
  `reference`, `curation_object`, `epitope`, `object` (typical IEDB schema) with
  a `pubmed_id` column on `reference`. Run `SHOW TABLES;` / `DESCRIBE reference;`
  after import and adjust the `IEDB_TABLES` dict in `src/config.py` if your
  dump differs.
- **Scaling**: XML parsing is the expensive step. Partition by input files
  (Spark does this automatically per `.xml.gz`), and increase
  `--num-executors`/`--executor-cores` on a real cluster.
