"""Shared configuration for the cancer-immunology Spark pipeline."""

from pyspark.sql import SparkSession

# --- MeSH terms used to filter PubMed baseline records ------------------
# Adjust to taste. These target the intersection of oncology + immunology.
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

# --- IEDB table names as they appear in the public MySQL dump -----------
# Verify against your imported dump with `DESCRIBE <table>;` and edit if
# your release uses different names/columns.
IEDB_TABLES = {
    "article": "article",            # has pubmed_id, title, authors, journal
    "curated_epitope": "curated_epitope", # links reference -> epitope/object records
    "epitope": "epitope",                 # epitope definitions
    "object": "object",                   # antigen/source-organism info
}


def get_spark(app_name: str) -> SparkSession:
    return (
        SparkSession.builder.appName(app_name)
        .config("spark.sql.shuffle.partitions", "200")
        .config("spark.sql.parquet.compression.codec", "snappy")
        .getOrCreate()
    )
