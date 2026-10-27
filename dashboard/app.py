"""
Streamlit dashboard for the cancer-immunology Spark pipeline outputs.

Reads gold/ tables only.  Run with:
  streamlit run dashboard/app.py

Set GOLD_DIR environment variable to override the default gold/ path.

Tabs:
  Overview     — KPI row from run_summary.json
  Topic Trends — share by year, emerging/declining bar chart
  Key Terms    — term prevalence multi-line chart
  Papers       — top-N importance table with PubMed links
  Retrieval    — query explorer (pick a query, see top-10 hybrid results)
"""

import json
import os
import sys
import math

import pandas as pd
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go

# Allow running from repo root or dashboard/ folder
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
from config import GOLD_DIR, RAG_QUERIES  # noqa: E402

# ---------------------------------------------------------------------------
# Page config
# ---------------------------------------------------------------------------

st.set_page_config(
    page_title="Cancer Immunology Analytics",
    page_icon="🧬",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ---------------------------------------------------------------------------
# Sidebar
# ---------------------------------------------------------------------------

with st.sidebar:
    st.header("⚙️ Settings")
    gold_dir = st.text_input(
        "Gold directory",
        value=os.environ.get("GOLD_DIR", GOLD_DIR),
    )
    top_n_papers = st.slider("Top-N papers to display", 5, 50, 10)
    st.markdown("---")
    st.markdown("**Pipeline:** [GitHub](https://github.com/arsalansyed04/cancer-immuno-spark)")


# ---------------------------------------------------------------------------
# Loaders (cached)
# ---------------------------------------------------------------------------

@st.cache_data(ttl=300)
def load_csv(path):
    if not os.path.exists(path):
        return None
    return pd.read_csv(path)


@st.cache_data(ttl=300)
def load_parquet(path):
    if not os.path.exists(path):
        return None
    return pd.read_parquet(path)


@st.cache_data(ttl=300)
def load_run_summary(gold_dir):
    path = os.path.join(gold_dir, "run_summary.json")
    if not os.path.exists(path):
        return {}
    with open(path) as f:
        return json.load(f)


def _check_gold(gold_dir):
    """Return True and list of present tables; False with missing list."""
    required = [
        "paper_topics.parquet",
        "topic_share_by_year.csv",
        "topic_delta.csv",
        "paper_importance.parquet",
    ]
    missing = [f for f in required if not os.path.exists(os.path.join(gold_dir, f))]
    return len(missing) == 0, missing


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

ok, missing = _check_gold(gold_dir)
if not ok:
    st.error(
        f"Gold tables not found in **{gold_dir}**.\n\n"
        f"Missing: `{'`, `'.join(missing)}`\n\n"
        "Run `bash scripts/run_validation.sh` to generate all outputs."
    )
    st.stop()

tab_overview, tab_trends, tab_terms, tab_papers, tab_retrieval = st.tabs([
    "🏠 Overview", "📈 Topic Trends", "🔍 Key Terms",
    "📄 Top Papers", "🔎 Retrieval Explorer",
])

# ---------------------------------------------------------------------------
# Tab 1: Overview
# ---------------------------------------------------------------------------

with tab_overview:
    st.title("🧬 Cancer Immunology Analytics Dashboard")
    st.markdown(
        "Distributed Topic-Trend and RAG-Based Importance Analytics of "
        "Cancer Immunology Literature using Apache Spark (DA3 project)."
    )

    summary = load_run_summary(gold_dir)

    # KPI metrics from run_summary.json
    col1, col2, col3, col4, col5 = st.columns(5)
    col1.metric("Files processed",    summary.get("files_processed", "—"))
    col2.metric("Records scanned",    f"{summary.get('records_scanned', '—'):,}" if isinstance(summary.get('records_scanned'), int) else "—")
    col3.metric("Papers retained",    f"{summary.get('papers_retained', '—'):,}" if isinstance(summary.get('papers_retained'), int) else "—")
    col4.metric("Vocabulary size",    summary.get("vocab_size", "—"))
    col5.metric("Years covered",      summary.get("years_covered", "—"))

    if summary:
        st.markdown("### ⏱️ Stage runtimes")
        stages = summary.get("stage_runtimes", {})
        if stages:
            runtime_df = pd.DataFrame(
                [{"Stage": k, "Wall-clock (s)": v} for k, v in stages.items()]
            )
            st.dataframe(runtime_df, use_container_width=True)

    # Architecture diagram
    st.markdown("### 🏗️ Pipeline architecture")
    st.markdown("""
```
L1 Raw data        PubMed baseline XML  ←→  IEDB MySQL dump
       ↓                                         ↓
L2 Silver          pubmed_parser.py          iedb_loader.py
  (MeSH filtered)                                ↓
       ↓                              join_analysis.py (fan-out fixed)
L3 Analytics       topic_modeling.py → LDA/K-means/TF-IDF
                   trend_analysis.py → topic trends / key-term prevalence
                   importance.py     → paper importance score (0-100)
       ↓
L4 RAG             embed.py → index.py → retrieve.py (hybrid RRF)
                   evaluate.py → P@10 table
       ↓
L5 Serving         dashboard/app.py (this page)
```
""")

# ---------------------------------------------------------------------------
# Tab 2: Topic Trends
# ---------------------------------------------------------------------------

with tab_trends:
    st.header("📈 Topic Trends Over Time")

    share_df = load_csv(os.path.join(gold_dir, "topic_share_by_year.csv"))
    delta_df = load_csv(os.path.join(gold_dir, "topic_delta.csv"))
    labels_df = load_csv(os.path.join(gold_dir, "topic_labels.csv"))

    if share_df is not None and not share_df.empty:
        # Load topic labels if available
        label_map = {}
        if labels_df is not None:
            for _, row in labels_df.iterrows():
                label_map[int(row["topic_id"])] = (
                    row["label"] if row["label"] != "unlabelled" else f"Topic {row['topic_id']}"
                )
        share_df["topic_label"] = share_df["topic_id"].apply(
            lambda t: label_map.get(int(t), f"Topic {t}")
        )

        # Year filter
        years = sorted(share_df["pub_year"].dropna().unique().tolist())
        min_yr, max_yr = st.select_slider(
            "Year range", options=years, value=(years[0], years[-1])
        )
        filtered = share_df[
            (share_df["pub_year"] >= min_yr) & (share_df["pub_year"] <= max_yr)
        ]

        fig = px.line(
            filtered, x="pub_year", y="mean_theta_share",
            color="topic_label", markers=True,
            title="Mean Topic Share (θ) by Year",
            labels={"pub_year": "Year", "mean_theta_share": "Mean θ share",
                    "topic_label": "Topic"},
        )
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("topic_share_by_year.csv not found or empty.")

    if delta_df is not None and not delta_df.empty:
        st.markdown("### Emerging vs Declining Topics (Δ percentage points)")
        delta_df["topic_label"] = delta_df["topic_id"].apply(
            lambda t: label_map.get(int(t), f"Topic {t}") if labels_df is not None else f"Topic {t}"
        )
        delta_df["color"] = delta_df["trend"].map(
            {"emerging": "#2ecc71", "stable": "#95a5a6", "declining": "#e74c3c"}
        )
        fig2 = px.bar(
            delta_df.sort_values("delta_pp", ascending=True),
            x="delta_pp", y="topic_label", orientation="h",
            color="trend",
            color_discrete_map={"emerging": "#2ecc71", "stable": "#95a5a6", "declining": "#e74c3c"},
            title="Topic Δ Share (late window − early window, pp)",
            labels={"delta_pp": "Δ share (pp)", "topic_label": "Topic"},
        )
        st.plotly_chart(fig2, use_container_width=True)

# ---------------------------------------------------------------------------
# Tab 3: Key Terms
# ---------------------------------------------------------------------------

with tab_terms:
    st.header("🔍 Key-Term Prevalence Over Time")
    prevalence_df = load_csv(os.path.join(gold_dir, "term_prevalence_by_year.csv"))

    if prevalence_df is not None and not prevalence_df.empty:
        all_terms = sorted(prevalence_df["term"].unique().tolist())
        selected  = st.multiselect("Select terms", all_terms, default=all_terms[:5])
        sub = prevalence_df[prevalence_df["term"].isin(selected)]
        fig = px.line(
            sub, x="pub_year", y="pct_match", color="term", markers=True,
            title="% of Abstracts Mentioning Each Key Term by Year",
            labels={"pub_year": "Year", "pct_match": "% papers", "term": "Term"},
        )
        st.plotly_chart(fig, use_container_width=True)
        st.dataframe(prevalence_df, use_container_width=True)
    else:
        st.info("term_prevalence_by_year.csv not found or empty.")

# ---------------------------------------------------------------------------
# Tab 4: Top Papers
# ---------------------------------------------------------------------------

with tab_papers:
    st.header("📄 Top Papers by Importance Score")
    imp_df = load_parquet(os.path.join(gold_dir, "paper_importance.parquet"))
    ppt_df = load_csv(os.path.join(gold_dir, "papers_per_topic.csv"))

    if imp_df is not None and not imp_df.empty:
        top_df = imp_df.nlargest(top_n_papers, "importance")

        # Add PubMed link
        def pubmed_link(pmid):
            return f"https://pubmed.ncbi.nlm.nih.gov/{int(pmid)}/"

        display_df = top_df[
            ["rank" if "rank" in top_df.columns else "importance",
             "pmid", "title", "journal", "pub_year",
             "dominant_topic", "importance", "r", "w", "e", "y"]
        ].copy() if "rank" in top_df.columns else top_df[
            ["pmid", "title", "journal", "pub_year",
             "dominant_topic", "importance", "r", "w", "e", "y"]
        ].copy()
        display_df.insert(0, "PubMed", display_df["pmid"].apply(pubmed_link))

        st.dataframe(
            display_df.style.background_gradient(subset=["importance"], cmap="YlOrRd"),
            use_container_width=True,
        )

        # Papers per topic
        if ppt_df is not None:
            st.markdown("### Papers per dominant topic")
            fig = px.bar(
                ppt_df.sort_values("dominant_topic"),
                x="dominant_topic", y="num_papers",
                title="Papers per Dominant Topic",
                labels={"dominant_topic": "Topic ID", "num_papers": "# Papers"},
            )
            st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("paper_importance.parquet not found or empty.")

# ---------------------------------------------------------------------------
# Tab 5: Retrieval Explorer
# ---------------------------------------------------------------------------

with tab_retrieval:
    st.header("🔎 Retrieval Explorer")
    eval_df = load_csv(os.path.join(gold_dir, "retrieval_eval.csv"))
    res_df  = load_parquet(os.path.join(gold_dir, "retrieval_results.parquet"))

    if eval_df is not None and not eval_df.empty:
        st.markdown("### Precision@10 Summary")
        pivot = eval_df.pivot(index="query", columns="method", values="p_at_10")
        st.dataframe(pivot, use_container_width=True)

    query_ids = [q["id"] for q in RAG_QUERIES]
    selected_q = st.selectbox("Pick a query", query_ids)

    q_meta = next((q for q in RAG_QUERIES if q["id"] == selected_q), None)
    if q_meta:
        st.markdown(f"**Query text:** _{q_meta['query']}_")

    if res_df is not None and not res_df.empty:
        hits = res_df[res_df["query"] == selected_q].sort_values("rank")
        if not hits.empty:
            st.dataframe(hits, use_container_width=True)
        else:
            st.info(f"No results for query '{selected_q}' in retrieval_results.parquet.")
    else:
        st.info("retrieval_results.parquet not found. Run src/rag/evaluate.py first.")
