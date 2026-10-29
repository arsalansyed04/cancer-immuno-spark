import numpy as np
import pandas as pd
from sklearn.cluster import KMeans

EMBEDDINGS = "output/gold/embeddings.npy"
PMIDS = "output/gold/embedding_pmids.parquet"
OUTPUT = "output/gold/lsa_paper_clusters.parquet"

X = np.load(EMBEDDINGS)
papers = pd.read_parquet(PMIDS)

model = KMeans(n_clusters=10, random_state=42, n_init=10, max_iter=300)
labels = model.fit_predict(X)

papers["cluster"] = labels
papers.to_parquet(OUTPUT, index=False)

print("Saved:", OUTPUT)
print("\nCluster sizes:")
print(papers.groupby("cluster").size().sort_index())
