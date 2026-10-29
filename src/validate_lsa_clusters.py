import pandas as pd

CLUSTERS = "output/gold/lsa_paper_clusters.parquet"
SILVER = "output/silver/pubmed_filtered"

clusters = pd.read_parquet(CLUSTERS)
papers = pd.read_parquet(SILVER)

df = clusters.merge(
    papers[["pmid", "title", "abstract"]],
    on="pmid",
    how="left",
)

print("Total papers:", len(df))
print("\nPapers per cluster:")
print(df.groupby("cluster").size().sort_index())

for cluster_id in [3, 9]:
    subset = df[df["cluster"] == cluster_id]
    print(f"\n{'=' * 70}")
    print(f"CLUSTER {cluster_id}: {len(subset)} papers")
    print("=" * 70)

    for _, row in subset.sample(
        n=min(25, len(subset)), random_state=42
    ).iterrows():
        print(f"\nPMID: {row['pmid']}")
        print(f"Title: {row['title']}")
