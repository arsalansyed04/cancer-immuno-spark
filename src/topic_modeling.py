"""
Branch A: unsupervised topic modeling over the MeSH-filtered PubMed corpus
produced by pubmed_parser.py.

Pipeline:
  tokenize → stop-word removal (English + scientific) → CountVectorizer+IDF
  → LDA sweep (k=5,10,15) → final LDA (k=10, full θ saved) → K-means

Changes from v1
---------------
• Vocabulary: minDF=5, maxDF=0.50 (fraction), vocabSize=5000
• Stop-words: English default + scientific list from config.py
• LDA sweep writes logPerplexity AND perplexity (exp(logPerplexity))
• Final LDA stores full topicDistribution array (theta) as array<double>
• topic_terms.csv and topic_labels.csv are produced
• K-means silhouette is computed and logged
• argmax uses vector_to_array + array_position (no Python UDF)
• cluster_sizes.csv is written

Run with:
  spark-submit src/topic_modeling.py \\
      --input  output/silver/pubmed_filtered \\
      --output output/gold \\
      --lda-ks 5,10,15 --final-k 10 --kmeans-k 10
"""

import argparse
import math
import os
import shutil
import sys

# Python 3.12 compatibility for PySpark 3.5.x
import setuptools

from pyspark.sql import functions as F
from pyspark.ml.feature import (
    RegexTokenizer,
    StopWordsRemover,
    CountVectorizer,
    IDF,
    Normalizer,
)
from pyspark.ml.clustering import LDA, KMeans
from pyspark.ml.evaluation import ClusteringEvaluator
from pyspark.ml.functions import vector_to_array

sys.path.append(os.path.dirname(__file__))
from config import (  # noqa: E402
    get_spark,
    GOLD_DIR,
    VOCAB_SIZE, MIN_DF, MAX_DF,
    SCIENTIFIC_STOP_WORDS,
    LDA_K_VALUES, LDA_FINAL_K, LDA_MAX_ITER, LDA_SEED,
    KMEANS_K,
)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def build_arg_parser():
    p = argparse.ArgumentParser(
        description="Topic modeling (LDA + K-means) over filtered PubMed abstracts."
    )
    p.add_argument("--input",   required=True,
                   help="Parquet dir from pubmed_parser.py (silver)")
    p.add_argument("--output",  default=GOLD_DIR,
                   help="Output directory for CSV / Parquet results (gold)")
    p.add_argument("--lda-ks",  default=",".join(str(k) for k in LDA_K_VALUES),
                   help="Comma-separated LDA k values to sweep")
    p.add_argument("--final-k", type=int, default=LDA_FINAL_K,
                   help="Final LDA k for per-paper topic assignments")
    p.add_argument("--kmeans-k", type=int, default=KMEANS_K,
                   help="K for K-means clustering")
    p.add_argument("--vocab-size", type=int, default=VOCAB_SIZE)
    p.add_argument("--min-df",     type=float, default=MIN_DF)
    p.add_argument("--max-df",     type=float, default=MAX_DF)
    return p


# ---------------------------------------------------------------------------
# Helper
# ---------------------------------------------------------------------------

def write_csv(df, output_path: str):
    """Write a Spark DataFrame as a single CSV file (same pattern as join_analysis.py)."""
    temp_dir = output_path + "_tmp"
    df.coalesce(1).write.mode("overwrite").option("header", True).csv(temp_dir)
    part_file = next(
        name for name in os.listdir(temp_dir)
        if name.startswith("part-") and name.endswith(".csv")
    )
    os.replace(os.path.join(temp_dir, part_file), output_path)
    shutil.rmtree(temp_dir)


# ---------------------------------------------------------------------------
# Stage 1 – tokenise and remove stop-words
# ---------------------------------------------------------------------------

def tokenize_corpus(spark, df):
    """Clean biomedical text while preserving internal hyphens."""
    cleaned = df.withColumn(
        "clean_abstract",
        F.regexp_replace(
            F.lower(F.col("abstract")),
            r"[^a-z0-9\s-]",
            " ",
        ),
    ).filter(
        F.col("clean_abstract").isNotNull()
        & (F.length(F.col("clean_abstract")) > 10)
    )

    # Split on whitespace, preserving terms such as pd-l1 and car-t.
    tokenizer = RegexTokenizer(
        inputCol="clean_abstract",
        outputCol="raw_tokens",
        pattern=r"\s+",
        gaps=True,
        minTokenLength=2,
        toLowercase=True,
    )
    tokenized = tokenizer.transform(cleaned)

    # Remove stray hyphens at the start/end of tokens.
    tokenized = tokenized.withColumn(
        "raw_tokens",
        F.expr(
            "filter(transform(raw_tokens, x -> regexp_replace(x, '^-+|-+$', '')), "
            "x -> x RLIKE '[a-z0-9]')"
        ),
    )

    english_sw = StopWordsRemover.loadDefaultStopWords("english")
    all_sw = list(set(english_sw) | set(SCIENTIFIC_STOP_WORDS))

    remover = StopWordsRemover(
        inputCol="raw_tokens",
        outputCol="filtered_tokens",
        stopWords=all_sw,
    )
    return remover.transform(tokenized)


# ---------------------------------------------------------------------------
# Stage 2 – TF-IDF
# ---------------------------------------------------------------------------

def build_tfidf(df, vocab_size, min_df, max_df):
    """CountVectorizer + IDF.  Returns (tfidf_df, cv_model, idf_model)."""
    cv = CountVectorizer(
        inputCol="filtered_tokens",
        outputCol="tf",
        vocabSize=vocab_size,
        minDF=min_df,
        maxDF=max_df,
    )
    cv_model  = cv.fit(df)
    tf_df     = cv_model.transform(df)

    idf       = IDF(inputCol="tf", outputCol="features")
    idf_model = idf.fit(tf_df)
    tfidf_df  = idf_model.transform(tf_df)

    return tfidf_df, cv_model, idf_model


# ---------------------------------------------------------------------------
# Stage 3 – LDA sweep
# ---------------------------------------------------------------------------

def lda_sweep(tfidf_df, cv_model, k_values, max_terms_per_topic=10, max_iter=LDA_MAX_ITER):
    """Fit LDA at each k.  Returns list of result dicts."""
    vocab   = cv_model.vocabulary
    results = []

    for k in k_values:
        print(f"[topic_modeling] Fitting LDA k={k} ...")
        lda   = LDA(featuresCol="features", k=k, maxIter=max_iter, seed=LDA_SEED)
        model = lda.fit(tfidf_df)

        log_ll   = model.logLikelihood(tfidf_df)
        log_perp = model.logPerplexity(tfidf_df)
        perp     = math.exp(log_perp)

        topics     = model.describeTopics(maxTermsPerTopic=max_terms_per_topic).collect()
        topic_rows = []
        for row in topics:
            terms = [vocab[i] for i in row["termIndices"] if i < len(vocab)]
            weights = list(row["termWeights"])
            topic_rows.append((row["topic"], terms, weights))

        print(f"[topic_modeling] k={k}: logLikelihood={log_ll:.2f}, "
              f"logPerplexity={log_perp:.4f}, perplexity={perp:.2f}")
        for tid, terms, _ in topic_rows:
            print(f"  topic {tid}: {', '.join(terms[:8])}")

        results.append({
            "k":              k,
            "log_likelihood": log_ll,
            "log_perplexity": log_perp,
            "perplexity":     perp,
            "topics":         topic_rows,
        })

    return results


# ---------------------------------------------------------------------------
# Stage 4 – final LDA (full θ output)
# ---------------------------------------------------------------------------

def fit_final_lda(tfidf_df, cv_model, k, max_iter=LDA_MAX_ITER):
    """Fit LDA at the chosen k; return assignments with full topic mixture."""
    print(f"[topic_modeling] Fitting final LDA k={k} ...")
    lda   = LDA(featuresCol="features", k=k, maxIter=max_iter, seed=LDA_SEED)
    model = lda.fit(tfidf_df)
    transformed = model.transform(tfidf_df)  # adds topicDistribution column

    # Convert topicDistribution (DenseVector) to array<double>
    with_theta = transformed.withColumn(
        "theta", vector_to_array(F.col("topicDistribution")).cast("array<double>")
    )

    # dominant_topic = argmax(theta) — native Spark SQL, no Python UDF
    # array_position returns 1-indexed; subtract 1 for 0-indexed topic id
    with_dominant = with_theta.withColumn(
        "dominant_topic",
        F.expr("cast(array_position(theta, array_max(theta)) - 1 as int)"),
    )

    assignments = with_dominant.select(
        "pmid", "title", "pub_year", "journal",
        "dominant_topic", "theta",
    )
    return assignments, model


# ---------------------------------------------------------------------------
# Stage 5 – K-means
# ---------------------------------------------------------------------------

def run_kmeans(tfidf_df, k):
    """K-means on L2-normalised TF-IDF.  Returns (clustered_df, silhouette)."""
    normalizer = Normalizer(inputCol="features", outputCol="norm_features", p=2.0)
    normalized = normalizer.transform(tfidf_df)

    km    = KMeans(featuresCol="norm_features", k=k, seed=LDA_SEED)
    model = km.fit(normalized)
    clustered = model.transform(normalized)

    evaluator = ClusteringEvaluator(
        featuresCol="norm_features",
        predictionCol="prediction",
        metricName="silhouette",
        distanceMeasure="cosine",
    )
    silhouette = evaluator.evaluate(clustered)
    print(f"[topic_modeling] K-means k={k}: silhouette (cosine) = {silhouette:.4f}")

    return clustered, silhouette


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    args   = build_arg_parser().parse_args()
    k_vals = [int(k.strip()) for k in args.lda_ks.split(",") if k.strip()]
    os.makedirs(args.output, exist_ok=True)

    spark = get_spark("cancer-immunology-topic-modeling")

    df = spark.read.parquet(args.input).filter(F.col("abstract").isNotNull())
    total = df.count()
    print(f"[topic_modeling] {total} filtered papers loaded from {args.input}")

    # --- Tokenise -----------------------------------------------------------
    tokenized = tokenize_corpus(spark, df)

    # --- TF-IDF -------------------------------------------------------------
    tfidf_df, cv_model, idf_model = build_tfidf(
        tokenized, args.vocab_size, args.min_df, args.max_df
    )
    tfidf_df.cache()

    # Save reusable TF-IDF features for independent K-means tuning.
    tfidf_export = os.path.join(args.output, "tfidf_features.parquet")
    (
        tfidf_df.select("pmid", "title", "pub_year", "features")
        .write.mode("overwrite")
        .parquet(tfidf_export)
    )
    print(f"[topic_modeling] Saved reusable TF-IDF features to {tfidf_export}")

    vocab_size = len(cv_model.vocabulary)
    print(f"[topic_modeling] Vocabulary size after filtering: {vocab_size}")

    # --- LDA sweep ----------------------------------------------------------
    sweep_results = lda_sweep(tfidf_df, cv_model, k_vals)

    # lda_sweep_metrics.csv: k, log_likelihood, log_perplexity, perplexity
    metrics_rows = [
        (r["k"], float(r["log_likelihood"]), float(r["log_perplexity"]),
         float(r["perplexity"]))
        for r in sweep_results
    ]
    metrics_df = spark.createDataFrame(
        metrics_rows, ["k", "log_likelihood", "log_perplexity", "perplexity"]
    )
    write_csv(metrics_df, os.path.join(args.output, "lda_sweep_metrics.csv"))
    print("[topic_modeling] Wrote lda_sweep_metrics.csv")

    # lda_sweep_topics.csv: k, topic_id, rank, term, weight
    topic_rows = []
    for r in sweep_results:
        for topic_id, terms, weights in r["topics"]:
            for rank, (term, weight) in enumerate(zip(terms, weights)):
                topic_rows.append((r["k"], topic_id, rank, term, float(weight)))
    sweep_topics_df = spark.createDataFrame(
        topic_rows, ["k", "topic_id", "rank", "term", "weight"]
    )
    write_csv(sweep_topics_df, os.path.join(args.output, "lda_sweep_topics.csv"))
    print("[topic_modeling] Wrote lda_sweep_topics.csv")

    # --- Final LDA ----------------------------------------------------------
    assignments, final_model = fit_final_lda(tfidf_df, cv_model, args.final_k)
    assignments.write.mode("overwrite").parquet(
        os.path.join(args.output, "paper_topics.parquet")
    )
    print(f"[topic_modeling] Wrote paper_topics.parquet (k={args.final_k})")

    # topic_terms.csv: topic_id, rank, term, weight
    vocab = cv_model.vocabulary
    final_topics = final_model.describeTopics(maxTermsPerTopic=20).collect()
    term_rows = []
    for row in final_topics:
        for rank, (idx, w) in enumerate(zip(row["termIndices"], row["termWeights"])):
            if idx < len(vocab):
                term_rows.append((int(row["topic"]), rank, vocab[idx], float(w)))
    topic_terms_df = spark.createDataFrame(
        term_rows, ["topic_id", "rank", "term", "weight"]
    )
    write_csv(topic_terms_df, os.path.join(args.output, "topic_terms.csv"))
    print("[topic_modeling] Wrote topic_terms.csv")

    # Generate provisional labels from the highest-weight topic terms.
    label_rows = []
    for topic_id in range(args.final_k):
        terms = [
            row["term"]
            for row in (
                topic_terms_df
                .filter(F.col("topic_id") == topic_id)
                .orderBy("rank")
                .select("term")
                .limit(5)
                .collect()
            )
            if row["term"] and row["term"].strip()
        ]
        top_terms = ", ".join(terms)
        label = " / ".join(terms[:3]) if terms else "Needs review"
        label_rows.append((topic_id, label, top_terms))

    labels_df = spark.createDataFrame(
        label_rows, ["topic_id", "label", "top_terms"]
    )
    write_csv(labels_df, os.path.join(args.output, "topic_labels.csv"))
    print("[topic_modeling] Wrote topic_labels.csv with provisional labels")

    # papers_per_topic.csv
    ppt = (
        assignments.groupBy("dominant_topic")
        .agg(F.count("pmid").alias("num_papers"))
        .orderBy("dominant_topic")
    )
    write_csv(ppt, os.path.join(args.output, "papers_per_topic.csv"))
    print("[topic_modeling] Wrote papers_per_topic.csv")

    # --- K-means ------------------------------------------------------------
    clustered, silhouette = run_kmeans(tfidf_df, args.kmeans_k)

    cluster_assignments = clustered.select(
        "pmid", "title", "pub_year",
        F.col("prediction").alias("cluster"),
    )
    cluster_assignments.write.mode("overwrite").parquet(
        os.path.join(args.output, "paper_clusters.parquet")
    )
    print(f"[topic_modeling] Wrote paper_clusters.parquet (k={args.kmeans_k})")

    # cluster_sizes.csv
    cluster_sizes = (
        cluster_assignments.groupBy("cluster")
        .agg(F.count("pmid").alias("num_papers"))
        .orderBy("cluster")
    )
    write_csv(cluster_sizes, os.path.join(args.output, "cluster_sizes.csv"))

    # silhouette_score.csv (single-row)
    sil_df = spark.createDataFrame(
        [(args.kmeans_k, float(silhouette))],
        ["kmeans_k", "silhouette_cosine"],
    )
    write_csv(sil_df, os.path.join(args.output, "silhouette_score.csv"))
    print(f"[topic_modeling] K-means silhouette={silhouette:.4f}; "
          "wrote cluster_sizes.csv + silhouette_score.csv")

    print(f"[topic_modeling] All stages complete.  Output in {args.output}")
    spark.stop()


if __name__ == "__main__":
    main()
