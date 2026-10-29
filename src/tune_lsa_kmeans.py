import csv
import os

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score


EMBEDDINGS = "output/gold/embeddings.npy"
PMIDS = "output/gold/embedding_pmids.parquet"
OUTPUT = "output/gold/lsa_kmeans_tuning.csv"
K_VALUES = [5, 8, 10, 12, 15]
SEED = 42


def main():
    X = np.load(EMBEDDINGS)
    pmids = pd.read_parquet(PMIDS)

    if len(X) != len(pmids):
        raise ValueError(
            f"Embedding/PMID mismatch: {len(X)} vs {len(pmids)}"
        )

    print(f"Loaded {len(X)} embeddings with {X.shape[1]} dimensions.")

    results = []

    for k in K_VALUES:
        print(f"\nEvaluating LSA K-means, K={k} ...")

        model = KMeans(
            n_clusters=k,
            random_state=SEED,
            n_init=10,
            max_iter=300,
        )
        labels = model.fit_predict(X)

        # Exact silhouette computation is manageable for this corpus,
        # but use a reproducible sample to keep memory requirements low.
        score = silhouette_score(
            X,
            labels,
            metric="cosine",
            sample_size=min(1000, len(X)),
            random_state=SEED,
        )

        sizes = np.bincount(labels, minlength=k)

        row = {
            "k": k,
            "silhouette_cosine_sampled": float(score),
            "smallest_cluster": int(sizes.min()),
            "largest_cluster": int(sizes.max()),
            "clusters_under_10_papers": int((sizes < 10).sum()),
        }
        results.append(row)

        print(
            f"K={k}: silhouette={score:.6f}, "
            f"smallest={sizes.min()}, largest={sizes.max()}, "
            f"clusters_under_10={(sizes < 10).sum()}"
        )

    with open(OUTPUT, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=results[0].keys())
        writer.writeheader()
        writer.writerows(results)

    print(f"\nSaved results to {OUTPUT}")


if __name__ == "__main__":
    main()
