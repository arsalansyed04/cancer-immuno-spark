import numpy as np
import pandas as pd
import pyarrow.parquet as pq
from sklearn.cluster import KMeans

EMBEDDINGS = "output/gold/embeddings.npy"
PMIDS = "output/gold/embedding_pmids.parquet"
SILVER = "output/silver/pubmed_filtered"
OUTPUT = "output/gold/lsa_cluster_representatives.csv"

X = np.load(EMBEDDINGS)
papers = pd.read_parquet(PMIDS)

# Match the assignment order to the embedding order.
model = KMeans(n_clusters=10, random_state=42, n_init=10, max_iter=300)
labels = model.fit_predict(X)

papers["cluster"] = labels

# Load titles and abstracts for inspection.
table = pq.read_table(SILVER, columns=["pmid", "title", "abstract"])
metadata = table.to_pandas()
metadata["pmid"] = metadata["pmid"].astype(str)
papers["pmid"] = papers["pmid"].astype(str)

papers = papers.merge(metadata, on="pmid", how="left", validate="one_to_one")

# Normalized LSA embeddings allow cosine similarity via dot products.
norms = np.linalg.norm(X, axis=1, keepdims=True)
X_norm = X / np.maximum(norms, 1e-12)

representatives = []

for cluster_id in sorted(papers["cluster"].unique()):
    indices = np.flatnonzero(labels == cluster_id)
    centroid = X_norm[indices].mean(axis=0)
    centroid /= max(np.linalg.norm(centroid), 1e-12)

    similarities = X_norm[indices] @ centroid
    selected = indices[np.argsort(similarities)[-5:][::-1]]

    for idx in selected:
        row = papers.iloc[idx]
        representatives.append({
            "cluster": int(cluster_id),
            "pmid": row["pmid"],
            "title": row["title"],
            "cosine_to_centroid": float(X_norm[idx] @ centroid),
            "abstract_preview": str(row["abstract"] or "")[:500],
        })

result = pd.DataFrame(representatives)
result.to_csv(OUTPUT, index=False)

print(f"Saved representative papers to {OUTPUT}")
for cluster_id, group in result.groupby("cluster"):
    print(f"\n=== CLUSTER {cluster_id} ===")
    for _, row in group.iterrows():
        print(f"- PMID {row['pmid']}: {row['title']}")
