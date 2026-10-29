import csv
import os

from pyspark.ml.clustering import KMeans
from pyspark.ml.evaluation import ClusteringEvaluator
from pyspark.ml.feature import Normalizer
from pyspark.sql import functions as F

from config import get_spark

INPUT = "output/gold/tfidf_features.parquet"
OUTPUT = "output/gold/kmeans_tuning.csv"
K_VALUES = [5, 8, 10, 12, 15]
SEED = 42


def main():
    spark = get_spark("cancer-immunology-kmeans-tuning")

    try:
        df = spark.read.parquet(INPUT).select(
            "pmid", "title", "pub_year", "features"
        ).cache()

        total = df.count()
        print(f"Loaded {total} papers from saved TF-IDF features.")

        normalized = Normalizer(
            inputCol="features",
            outputCol="norm_features",
            p=2.0,
        ).transform(df).cache()

        evaluator = ClusteringEvaluator(
            featuresCol="norm_features",
            predictionCol="prediction",
            metricName="silhouette",
            distanceMeasure="cosine",
        )

        results = []

        for k in K_VALUES:
            print(f"\nEvaluating K={k} ...")

            model = KMeans(
                featuresCol="norm_features",
                predictionCol="prediction",
                k=k,
                seed=SEED,
                maxIter=30,
                initMode="k-means||",
            ).fit(normalized)

            clustered = model.transform(normalized).cache()
            silhouette = evaluator.evaluate(clustered)

            sizes = [
                row["count"]
                for row in (
                    clustered.groupBy("prediction")
                    .count()
                    .collect()
                )
            ]

            results.append({
                "k": k,
                "silhouette_cosine": float(silhouette),
                "smallest_cluster": min(sizes),
                "largest_cluster": max(sizes),
                "clusters_with_under_10_papers": sum(
                    n < 10 for n in sizes
                ),
            })

            print(
                f"K={k}: silhouette={silhouette:.6f}, "
                f"smallest={min(sizes)}, largest={max(sizes)}, "
                f"clusters_under_10={sum(n < 10 for n in sizes)}"
            )

            clustered.unpersist()

        with open(OUTPUT, "w", newline="") as f:
            writer = csv.DictWriter(
                f,
                fieldnames=[
                    "k",
                    "silhouette_cosine",
                    "smallest_cluster",
                    "largest_cluster",
                    "clusters_with_under_10_papers",
                ],
            )
            writer.writeheader()
            writer.writerows(results)

        print(f"\nResults saved to {OUTPUT}")

    finally:
        spark.stop()


if __name__ == "__main__":
    main()
