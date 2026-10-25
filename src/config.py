"""Shared configuration for the cancer-immunology Spark pipeline.

All thresholds, weights, lexicons, and paths are defined here so every
module can import them as constants and they can be overridden via CLI args.
"""

import os

# ---------------------------------------------------------------------------
# Medallion path roots  (relative to the project root directory)
# ---------------------------------------------------------------------------
# bronze: optional raw-parsed (not always written)
# silver: MeSH-filtered, quality-checked PubMed records
# gold:   all derived analytics tables
BRONZE_DIR = os.environ.get("BRONZE_DIR", "output/bronze")
SILVER_DIR = os.environ.get("SILVER_DIR", "output/silver/pubmed_filtered")
GOLD_DIR   = os.environ.get("GOLD_DIR",   "output/gold")

# ---------------------------------------------------------------------------
# PubMed validation file-set (10 sampled baseline files for the DA3 run)
# ---------------------------------------------------------------------------
VALIDATION_FILES = [
    "pubmed26n0850",
    "pubmed26n0950",
    "pubmed26n1030",
    "pubmed26n1100",
    "pubmed26n1160",
    "pubmed26n1210",
    "pubmed26n1250",
    "pubmed26n1290",
    "pubmed26n1320",
    "pubmed26n1334",
]

# ---------------------------------------------------------------------------
# MeSH terms used to filter PubMed baseline records
# ---------------------------------------------------------------------------
# Terms with a "/" qualifier (e.g. "Neoplasms/immunology") only match records
# where that exact descriptor+qualifier pair is tagged.  Terms without "/"
# (e.g. "Immunotherapy") match on the descriptor alone regardless of qualifier.
CANCER_IMMUNOLOGY_MESH_TERMS = [
    "Neoplasms/immunology",
    "Immunotherapy",
    "Immunotherapy, Adoptive",
    "Cancer Vaccines",
    "Tumor Escape",
    "T-Lymphocytes, Tumor-Infiltrating",
    "Antigens, Neoplasm",
    "Immune Checkpoint Inhibitors",
    "Programmed Cell Death 1 Receptor",
    "CTLA-4 Antigen",
    "Receptors, Chimeric Antigen",
    "Tumor Microenvironment",
    "Neoplasms/therapy",
    "Immunologic Surveillance",
    "Epitopes, T-Lymphocyte",
    "Melanoma/immunology",
    "Lung Neoplasms/immunology",
]

# ---------------------------------------------------------------------------
# IEDB table names as they appear in the public MySQL dump
# ---------------------------------------------------------------------------
# Verify against your imported dump with `DESCRIBE <table>;` and edit if
# your release uses different names/columns.
# Key columns used downstream:
#   article.pubmed_id  article.reference_id
#   curated_epitope.reference_id  curated_epitope.curated_epitope_id
#   curated_epitope.e_object_id
#   object.object_id  object.object_description
IEDB_TABLES = {
    "article":         "article",
    "curated_epitope": "curated_epitope",
    "epitope":         "epitope",
    "object":          "object",
}

# ---------------------------------------------------------------------------
# Vocabulary / TF-IDF parameters
# ---------------------------------------------------------------------------
VOCAB_SIZE = 5000
MIN_DF     = 5      # minimum document frequency (count)
MAX_DF     = 0.50   # maximum document frequency (fraction)

# ---------------------------------------------------------------------------
# Generic scientific stop-words (added on top of Spark's English stop-words)
# ---------------------------------------------------------------------------
# Do NOT include domain-relevant terms (pd-1, pd-l1, car-t, tme, etc.).
# These are words so common in biomedical prose that they carry no
# discriminative signal for topic modelling.
SCIENTIFIC_STOP_WORDS = [
    # study language
    "patients", "patient", "results", "result", "study", "studies",
    "analysis", "data", "treatment", "treated", "using", "used",
    "show", "shown", "however", "based", "may", "also", "although",
    "associated", "significantly", "significant", "compared",
    "observed", "found", "increased", "decreased", "expression",
    "however", "therefore", "furthermore", "moreover", "thus",
    # generic biology
    "cells", "cell", "cancer", "tumor", "tumour", "tumors", "tumours",
    "mice", "mouse", "vitro", "vivo", "human", "patients",
    "clinical", "response", "responses", "anti",
    # filler
    "role", "effect", "effects", "including", "levels", "high",
    "low", "new", "novel", "potential", "via", "within", "among",
    "without", "across", "between", "against", "either",
    "these", "those", "their", "they", "were", "have", "been",
    "with", "from", "that", "this", "which", "both", "all", "not",
    # numbers/units that survive tokenisation
    "95", "ci", "hr", "95ci", "pvalue",
]

# ---------------------------------------------------------------------------
# Key-term lexicon for trend analysis (regex patterns, case-insensitive)
# ---------------------------------------------------------------------------
# Used in trend_analysis.py term_prevalence_by_year.csv
# Patterns support both US and UK spelling (tumor/tumour) via alternation.
KEY_TERM_PATTERNS = {
    "PD-1/PD-L1":               r"pd-?1|pd-?l1|programmed.?death.?1|programmed.?death.?ligand",
    "CTLA-4":                   r"ctla-?4|ipilimumab|anti.?ctla",
    "CAR-T":                    r"car-?t\b|chimeric.?antigen.?receptor|adoptive.?cell.?(therapy|transfer)",
    "Neoantigen":               r"neoantigen|neoantigens|tumou?r.?mutation.?burden|tumou?r.?neoepitope",
    "TMB":                      r"\btmb\b|tumou?r.?mutational.?burden|mutation.?burden",
    "ctDNA":                    r"\bctdna\b|circulating.?tumou?r.?dna|cell.?free.?dna|liquid.?biopsy",
    "TME":                      r"\btme\b|tumou?r.?microenvironment|tumor.?microenvironment",
    "Resistance":               r"\bresistance\b|acquired.?resistance|innate.?resistance|immune.?evasion",
    "Checkpoint inhibitor":     r"checkpoint.?(inhibitor|blockade|therapy)|anti.?pd.?[1l]|immune.?checkpoint",
    "Biomarker":                r"\bbiomarker\b|predictive.?biomarker|prognostic.?biomarker",
}

# ---------------------------------------------------------------------------
# 20-term key lexicon for importance scoring (w component)
# ---------------------------------------------------------------------------
# These are the terms whose TF-IDF weight in a paper contributes to w(p).
IMPORTANCE_KEY_LEXICON = [
    "pd-1", "pd-l1", "ctla-4", "car-t", "neoantigen",
    "tmb", "ctdna", "immunotherapy", "checkpoint", "blockade",
    "t-cell", "tcell", "microenvironment", "resistance", "biomarker",
    "pembrolizumab", "nivolumab", "ipilimumab", "adoptive", "epitope",
]

# ---------------------------------------------------------------------------
# Importance score weights  (must sum to 1.0)
# ---------------------------------------------------------------------------
# r: max cosine similarity to any key-term query (RAG retrieval score)
# w: TF-IDF weight of key lexicon terms in the paper
# e: topic-mixture θ weighted by positive topic growth (Δ share)
# y: recency (min–max scaled pub_year)
IMPORTANCE_WEIGHTS = {
    "r": 0.35,
    "w": 0.25,
    "e": 0.25,
    "y": 0.15,
}

# ---------------------------------------------------------------------------
# Trend analysis windows and thresholds
# ---------------------------------------------------------------------------
TREND_EARLY_WINDOW  = (2015, 2018)   # (start_year, end_year) inclusive
TREND_LATE_WINDOW   = (2023, 2025)
TREND_EMERGING_THRESHOLD  =  2.0    # Δ percentage points above this → emerging
TREND_DECLINING_THRESHOLD = -2.0    # Δ pp below this → declining
MIN_PAPERS_PER_YEAR = 50            # years below this count are flagged / excluded

# ---------------------------------------------------------------------------
# LDA sweep parameters
# ---------------------------------------------------------------------------
LDA_K_VALUES   = [5, 10, 15]
LDA_FINAL_K    = 10
LDA_MAX_ITER   = 20
LDA_SEED       = 42
KMEANS_K       = 10

# ---------------------------------------------------------------------------
# RAG / embedding parameters
# ---------------------------------------------------------------------------
# Primary encoder (biomedical sentence transformer).
# Falls back to LSA (200-dim TruncatedSVD over TF-IDF) when unavailable.
EMBEDDING_MODEL = "pritamdeka/S-PubMedBert-MS-MARCO"
EMBEDDING_DIM   = 768          # dimension for the chosen model
LSA_DIM         = 200          # fallback TruncatedSVD dimension
FAISS_INDEX_TYPE = "flat"      # "flat" = brute-force (exact), use "ivf" on large corpora
RETRIEVAL_TOP_N  = 20          # candidates from each retriever before RRF fusion
FINAL_TOP_K      = 10          # results returned to the user / evaluation

# ---------------------------------------------------------------------------
# Key-term queries for RAG evaluation (query text + expected terms)
# ---------------------------------------------------------------------------
RAG_QUERIES = [
    {
        "id":             "pd1_pdl1_blockade",
        "query":          "PD-1 PD-L1 checkpoint blockade anti-tumor immune response",
        "expected_terms": ["pd-1", "pd-l1", "pd1", "pdl1", "checkpoint", "blockade",
                           "pembrolizumab", "nivolumab"],
    },
    {
        "id":             "cart_cell_therapy",
        "query":          "CAR-T chimeric antigen receptor T cell adoptive transfer therapy",
        "expected_terms": ["car-t", "car t", "chimeric antigen", "adoptive cell",
                           "adoptive transfer"],
    },
    {
        "id":             "neoantigen_vaccine",
        "query":          "neoantigen personalised cancer vaccine tumour mutational burden",
        "expected_terms": ["neoantigen", "neoantigens", "neoepitope", "personali"],
    },
    {
        "id":             "tmb_biomarker",
        "query":          "tumour mutational burden TMB biomarker immunotherapy response prediction",
        "expected_terms": ["tmb", "mutational burden", "biomarker", "predictive"],
    },
    {
        "id":             "ctdna_liquid_biopsy",
        "query":          "circulating tumour DNA ctDNA liquid biopsy minimal residual disease",
        "expected_terms": ["ctdna", "circulating tumour", "circulating tumor",
                           "liquid biopsy", "cell-free dna"],
    },
    {
        "id":             "resistance_mechanisms",
        "query":          "acquired resistance immune checkpoint inhibitor tumour microenvironment",
        "expected_terms": ["resistance", "immune evasion", "microenvironment", "tme"],
    },
]


# ---------------------------------------------------------------------------
# Spark session factory
# ---------------------------------------------------------------------------
def get_spark(app_name: str):
    """Return (or reuse) a SparkSession configured for local or cluster mode.

    All pipeline modules call this factory; set SPARK_MASTER env var to
    override the default local[*].
    """
    from pyspark.sql import SparkSession
    master = os.environ.get("SPARK_MASTER", "local[*]")
    return (
        SparkSession.builder
        .master(master)
        .appName(app_name)
        .config("spark.sql.shuffle.partitions", "200")
        .config("spark.sql.parquet.compression.codec", "snappy")
        # Allow vector UDFs from MLlib to cross Python boundary
        .config("spark.sql.execution.arrow.pyspark.enabled", "true")
        .getOrCreate()
    )
