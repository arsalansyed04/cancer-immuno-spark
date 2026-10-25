# Distributed Topic-Trend and RAG-Based Importance Analytics of Cancer Immunology Literature using Apache Spark

Distributed big-data pipeline fusing **PubMed Central / MEDLINE baseline XML archives** with the **Immune Epitope Database (IEDB)** to discover temporal research trends, cluster latent scientific topics, score paper importance, and enable grounded hybrid retrieval (RAG).

Built for **Apache Spark 3.5.3 (PySpark)**, **FAISS**, **Rank-BM25**, and **Streamlit**.

---

## 1. System Architecture (5-Layer Pipeline)

```
===================================================================================================
LAYER L1: INGESTION & MEDALLION ETL (PySpark + spark-xml + JDBC)
  [PubMed Baseline XML.gz]                     [IEDB MySQL Dump]
             │                                         │
             ▼ spark-xml                               ▼ JDBC
      pubmed_parser.py                           iedb_loader.py
             │                                         │
    output/silver/pubmed_filtered               output/silver/iedb_tables/
             └──────────────────┬──────────────────────┘
                                ▼
                         join_analysis.py  (de-duplicated epitope join; no fan-out)
                                │
                    output/gold/epitope_counts_by_mesh_term.csv
                    output/gold/epitope_density.parquet
───────────────────────────────────────────────────────────────────────────────────────────────────
LAYER L2: DISTRIBUTED TOPIC MODELING & CLUSTERING (Spark MLlib)
                                │
                        topic_modeling.py
    - CountVectorizer (minDF=5, maxDF=0.5, vocabSize=5000) + Scientific Stop-Words
    - Distributed LDA Sweep (k=5, 10, 15) with Perplexity & Log-Perplexity
    - Final LDA (k=10): Full θ-mixture per paper
    - K-means (k=10, L2-normalized TF-IDF) with Cosine Silhouette & Cluster Sizes
                                │
            output/gold/paper_topics.parquet (pmid, title, theta, dominant_topic)
            output/gold/topic_terms.csv & lda_sweep_metrics.csv
───────────────────────────────────────────────────────────────────────────────────────────────────
LAYER L3: TEMPORAL TREND ENGINE (PySpark)
                                │
                        trend_analysis.py
    - Topic share by year & Δ percentage points (2015–2018 early vs. 2023–2025 late window)
    - Trend categorization: emerging (Δ > +0.5pp), declining (Δ < -0.5pp), stable
    - Key-term longitudinal prevalence (PD-1, CAR-T, neoantigen, TMB, ctDNA, resistance)
                                │
            output/gold/topic_share_by_year.csv
            output/gold/topic_delta.csv
            output/gold/term_prevalence_by_year.csv
───────────────────────────────────────────────────────────────────────────────────────────────────
LAYER L4: RAG & IMPORTANCE SCORING LAYER (SentenceTransformers / FAISS / BM25 / Anthropic)
                                │
         ┌──────────────────────┴──────────────────────┐
         ▼                                             ▼
     rag/embed.py + rag/index.py                 importance.py
  Dense (Biomedical/LSA fallback) + BM25     Importance(p) = 100 × [0.35 r + 0.25 w + 0.25 e + 0.15 y] / max
         │                                             │
   rag/retrieve.py                                     │
  Hybrid RRF Fusion: 1 / (60 + rank)                   │
         │                                             │
   rag/evaluate.py                                     │
  Precision@10 (Dense vs BM25 vs Hybrid)               │
         │                                             │
   rag/summarize.py (optional Anthropic LLM)           │
                                │                      │
            output/gold/retrieval_eval.csv             output/gold/paper_importance.parquet
            output/gold/faiss.index & bm25.pkl         output/gold/top10_importance.csv
───────────────────────────────────────────────────────────────────────────────────────────────────
LAYER L5: SERVING & VISUALIZATION (Streamlit + Matplotlib)
         ┌─────────────────────────────────────────────┐
         ▼                                             ▼
  dashboard/app.py                            scripts/make_figures.py
  (Interactive KPI, Trends, Retrieval)        (Publication Fig 2–6 PNG exports)
===================================================================================================
```

---

## 2. Installation & Requirements

The project uses modular dependency files to keep core Spark execution decoupled from local neural serving:

```bash
# Core Spark processing (PySpark, pandas, pyarrow, pytest)
pip install -r requirements.txt

# RAG & Embeddings (FAISS, rank-bm25, sentence-transformers, scikit-learn)
pip install -r requirements-rag.txt

# Interactive Dashboard (Streamlit, plotly)
pip install -r requirements-dashboard.txt
```

### System Prerequisites
- **Java**: JDK 17 (recommended: Eclipse Temurin OpenJDK 17)
- **Apache Spark**: Spark 3.5.3 pre-built with Hadoop 3
- **Database (Optional for full IEDB)**: MySQL 8.0 or MariaDB 10.6+

---

## 3. Data Ingestion & Setup

### 3.1 Download PubMed Baseline Files
Use `scripts/download_pubmed.sh` to download files from NCBI FTP:
```bash
# Download the 10 sampled validation baseline files
bash scripts/download_pubmed.sh --validation

# Or download specific files
bash scripts/download_pubmed.sh --files pubmed26n0850,pubmed26n0950

# Or download first N files
bash scripts/download_pubmed.sh -n 5
```

### 3.2 Setup IEDB Database (Optional)
```bash
bash scripts/setup_iedb.sh
```

---

## 4. Running the Pipeline

### 4.1 One-Command Validation Run
To execute the complete end-to-end pipeline across all 10 validation files:
```bash
bash scripts/run_validation.sh
```

### 4.2 Standard Pipeline Orchestration
To execute all stages or skip individual stages using environment flags:
```bash
# Run entire pipeline
bash scripts/run_pipeline.sh

# Skip download and IEDB import
SKIP_DOWNLOAD=1 SKIP_IEDB=1 bash scripts/run_pipeline.sh
```

### 4.3 Running Individual Stages via `spark-submit` / Python

```bash
# 1. PubMed Parsing & MeSH Filtering (Silver)
spark-submit \
  --packages com.databricks:spark-xml_2.12:0.18.0 \
  --driver-memory 4g --executor-memory 4g \
  src/pubmed_parser.py \
  --input data/pubmed_baseline \
  --output output/silver/pubmed_filtered

# 2. IEDB Ingestion (Silver)
spark-submit \
  --jars /opt/spark/jars_extra/mysql-connector-j-8.4.0.jar \
  --driver-memory 4g \
  src/iedb_loader.py \
  --jdbc-url "jdbc:mysql://localhost:3306/iedb_public" \
  --user spark --password sparkpw \
  --output output/silver/iedb_tables

# 3. De-duplicated Join Analysis (Gold)
spark-submit \
  --driver-memory 4g --executor-memory 4g \
  src/join_analysis.py \
  --pubmed output/silver/pubmed_filtered \
  --iedb output/silver/iedb_tables \
  --output output/gold

# 4. Distributed Topic Modeling (Spark MLlib LDA & K-Means)
spark-submit \
  --driver-memory 4g --executor-memory 4g \
  src/topic_modeling.py \
  --input output/silver/pubmed_filtered \
  --output output/gold

# 5. Temporal Trend Analysis
spark-submit \
  --driver-memory 4g \
  src/trend_analysis.py \
  --topics output/gold/paper_topics.parquet \
  --silver output/silver/pubmed_filtered \
  --output output/gold

# 6. RAG Embedding & Indexing
python src/rag/embed.py --silver output/silver/pubmed_filtered --gold output/gold
python src/rag/index.py --gold output/gold

# 7. Retrieval Evaluation (Precision@10)
python src/rag/evaluate.py --silver output/silver/pubmed_filtered --gold output/gold

# 8. Paper Importance Scoring
python src/importance.py \
  --topics output/gold/paper_topics.parquet \
  --tfidf-vocab output/gold/topic_terms.csv \
  --topic-delta output/gold/topic_delta.csv \
  --gold output/gold \
  --silver output/silver/pubmed_filtered

# 9. Publication Figure Generation
python scripts/make_figures.py --gold output/gold --out docs/figures
```

---

## 5. Interactive Dashboard

Launch the Streamlit visualization app:
```bash
streamlit run dashboard/app.py
```
**Features**:
- **Overview KPI**: Files scanned, filtered cancer-immunology corpus, vocab size, years active.
- **Topic Trends**: Interactive area and line charts of topic proportions over time.
- **Key Terms**: Longitudinal prevalence trajectories for checkpoint inhibitors, CAR-T, neoantigens, ctDNA, and resistance.
- **Top Papers**: Table of top-scoring papers with direct PMID links to PubMed.
- **Retrieval Explorer**: Hybrid RRF search interface comparing dense and BM25 hits.

---

## 6. Output Tables Reference

| Medallion Layer | File Path | Format | Description |
|---|---|---|---|
| **Silver** | `output/silver/pubmed_filtered` | Parquet | Cleaned cancer immunology papers (`pmid`, `title`, `abstract`, `pub_year`, `journal`, `mesh_terms`) |
| **Silver** | `output/silver/iedb_tables/` | Parquet | Filtered IEDB tables (`article`, `curated_epitope`, `object`) |
| **Gold** | `output/gold/epitope_counts_by_mesh_term.csv` | CSV | Distinct curated epitope counts per MeSH heading |
| **Gold** | `output/gold/epitope_density.parquet` | Parquet | Epitope count per PMID (`pmid`, `n_distinct_epitopes`) |
| **Gold** | `output/gold/paper_topics.parquet` | Parquet | Final LDA topic mixture `theta` (array<double>) and `dominant_topic` |
| **Gold** | `output/gold/lda_sweep_metrics.csv` | CSV | Perplexity and log-perplexity for k ∈ {5, 10, 15} |
| **Gold** | `output/gold/topic_terms.csv` | CSV | Top 20 terms and weights for each topic |
| **Gold** | `output/gold/kmeans_cluster_sizes.csv` | CSV | Document counts per K-Means cluster (silhouette logged) |
| **Gold** | `output/gold/topic_share_by_year.csv` | CSV | Mean θ topic share by publication year |
| **Gold** | `output/gold/topic_delta.csv` | CSV | Early (2015-18) vs Late (2023-25) Δ share and emerging/declining tag |
| **Gold** | `output/gold/term_prevalence_by_year.csv` | CSV | Annual percentage of abstracts containing key terms |
| **Gold** | `output/gold/retrieval_eval.csv` | CSV | Precision@10 for Dense, BM25, and Hybrid RRF across 5 queries |
| **Gold** | `output/gold/paper_importance.parquet` | Parquet | Importance score (0–100) with component breakdowns (`r`, `w`, `e`, `y`) |
| **Gold** | `output/gold/top10_importance.csv` | CSV | Top 10 ranked papers across corpus |
| **Gold** | `output/gold/run_summary.json` | JSON | End-to-end execution metadata, timings per stage, and environment specs |

---

## 7. Configuration Reference (`src/config.py`)

All parameters are centrally managed in `src/config.py`:
- **MeSH Filters**: 17 domain MeSH terms with strict qualifier handling (`Neoplasms/immunology`, `Melanoma/immunology`, etc.).
- **Scientific Stop-Words**: Tailored filtering of non-informative academic boilerplate (`patients`, `results`, `study`, `cells`, `treatment`) while safeguarding domain terms (`pd-1`, `car-t`, `ctdna`).
- **Trend Windows**: Early window (2015–2018), Late window (2023–2025), Minimum papers per year threshold (50 papers).
- **Importance Weights**:
  $$\text{Importance}(p) = 100 \times \frac{0.35 \cdot r(p) + 0.25 \cdot w(p) + 0.25 \cdot e(p) + 0.15 \cdot y(p)}{\max_{p'} \text{Score}(p')}$$
  - $r(p)$: Max cosine similarity to key queries (RAG).
  - $w(p)$: Summed TF-IDF weight of key domain terms.
  - $e(p)$: Topic growth alignment ($\theta \cdot \max(\Delta, 0)$).
  - $y(p)$: Recency (min-max scaled publication year).

---

## 8. Limitations & Methodology Notes

1. **Importance Score Context**: The importance metric measures *thematic alignment and research momentum* within contemporary cancer immunotherapy topics, **not historical citation count**.
2. **Keyword Relevance Proxy**: RAG evaluation uses a keyword presence proxy on abstracts. While fast and reproducible, it does not reward paraphrase retrieval. A human-in-the-loop template (`gold/relevance_labels_template.csv`) is generated to allow manual scoring.
3. **IEDB Join Scale**: The join between PubMed and IEDB matches on PMID. In sampled validation subsets, matched PMIDs reflect literature sampled in both sources; full corpus runs require complete database indexing.
