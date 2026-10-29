import pandas as pd
import re

SILVER = "output/silver/pubmed_filtered"
OUTPUT = "output/gold/cancer_scope_audit.csv"

df = pd.read_parquet(SILVER)
df["title"] = df["title"].fillna("")
df["abstract"] = df["abstract"].fillna("")
df["text"] = (df["title"] + " " + df["abstract"]).str.lower()

# Broad cancer-related terms; this is a screening heuristic, not a
# definitive scientific relevance classifier.
cancer_pattern = re.compile(
    r"\b(cancer|cancers|tumou?r|oncolog\w*|carcinoma|"
    r"leukaemia|leukemia|lymphoma|melanoma|sarcoma|"
    r"glioblastoma|neoplasm\w*|malignan\w*|metasta\w*|"
    r"anticancer|anti-tumou?r|anti-cancer|"
    r"immuno-oncology|neoantigen\w*)\b",
    re.I,
)

df["cancer_term_found"] = df["text"].apply(
    lambda text: bool(cancer_pattern.search(text))
)

# Flag records without explicit cancer-related terminology for review.
df["scope_flag"] = df["cancer_term_found"].map({
    True: "Cancer terminology found",
    False: "Review: no explicit cancer terminology found",
})

df[[
    "pmid", "title", "pub_year", "scope_flag"
]].to_csv(OUTPUT, index=False)

print("Total papers:", len(df))
print("\nScreening counts:")
print(df["scope_flag"].value_counts().to_string())
print("\nReview sample (up to 30 records):")

review = df[~df["cancer_term_found"]]
for _, row in review.head(30).iterrows():
    print(f"\nPMID: {row['pmid']}")
    print(f"Title: {row['title']}")

print(f"\nFull audit saved to: {OUTPUT}")
print("No papers were deleted or changed.")
