"""RAG layer for cancer-immunology literature retrieval (Task 4).

Modules:
    embed    – encode abstracts to dense vectors (or LSA fallback)
    index    – FAISS vector index + BM25 index
    retrieve – hybrid dense+BM25 retrieval with RRF fusion
    queries  – curated key-term query set
    evaluate – Precision@10 evaluation
    summarize– LLM summaries grounded in retrieved abstracts (optional)
"""
