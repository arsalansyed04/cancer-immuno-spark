import pandas as pd
import re

INPUT = "output/gold/cancer_scope_audit.csv"
SILVER = "output/silver/pubmed_filtered"
OUTPUT = "output/gold/cancer_scope_review.csv"

audit = pd.read_csv(INPUT)
papers = pd.read_parquet(SILVER)

df = audit.merge(
    papers[["pmid", "abstract"]],
    on="pmid",
    how="left",
)
df["title"] = df["title"].fillna("")
df["abstract"] = df["abstract"].fillna("")
df["text"] = (df["title"] + " " + df["abstract"]).str.lower()

# Cancer-specific indicators that may have been missed by the first audit.
cancer_specific = re.compile(
    r"\b("
    r"multiple myeloma|myeloma|"
    r"nsclc|sclc|"
    r"tumou?r|carcinoma|leukaemia|leukemia|lymphoma|"
    r"melanoma|sarcoma|glioblastoma|neoplasm|"
    r"neoantigen|onco(gen|genic|protein|virus)|"
    r"anti[- ]?tumou?r|anti[- ]?cancer|"
    r"cancer immunotherap\w*|tumou?r microenvironment|"
    r"tumou?r cell|tumou?r growth|"
    r"pd-?1|pd-?l1|ctla-?4|car[- ]?t"
    r")\b",
    re.I,
)

broader_immune = re.compile(
    r"\b("
    r"t.cell|b.cell|t.cell receptor|epitope|immunotherap\w*|"
    r"immunopeptidome|hla[- ]|antigen.present\w*|"
    r"immune response|immune checkpoint|"
    r"cytotoxic|dendritic cell|"
    r"vaccine|antibody|immunomodulat\w*"
    r")\b",
    re.I,
)

def classify(text):
    if cancer_specific.search(text):
        return "Likely cancer-related — verify"
    if broader_immune.search(text):
        return "Broader immunology — assess relevance"
    return "Potentially off-topic — review"

review = df[df["scope_flag"].str.startswith("Review:")].copy()
review["suggested_category"] = review["text"].apply(classify)
review["manual_decision"] = ""

review[[
    "pmid", "title", "abstract",
    "suggested_category", "manual_decision"
]].to_csv(OUTPUT, index=False)

print("Flagged papers:", len(review))
print("\nSuggested categories:")
print(review["suggested_category"].value_counts().to_string())
print("\nExamples by category:")
for category, group in review.groupby("suggested_category"):
    print(f"\n--- {category} ({len(group)}) ---")
    for title in group["title"].head(8):
        print("-", title)

print(f"\nReview file saved to: {OUTPUT}")
print("These are heuristic suggestions, not verified classifications.")
print("No source data or clustering outputs were modified.")
