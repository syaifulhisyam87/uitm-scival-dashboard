"""Streamlit dashboard for bibliometric exploration.

Run with:
    streamlit run biblio_dashboard.py

The dashboard expects a merged CSV produced by ``biblio_preprocess.py``
(saved as ``merged_biblio.csv`` by default). Adjust ``MERGED_CSV_PATH`` if
needed.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st

MERGED_CSV_PATH = Path("merged_biblio.csv")
DEFAULT_MULTI_SEPARATOR = ";"


# ---------------------------------------------------------------------------
# Helper functions
# ---------------------------------------------------------------------------
def load_data(path: Path) -> pd.DataFrame:
    """Load the merged CSV file, raising an error if missing."""

    if not path.exists():
        raise FileNotFoundError(
            f"Merged file not found: {path}. Please run biblio_preprocess.py first."
        )
    return pd.read_csv(path)


@st.cache_data(show_spinner=False)
def get_data(path: Path) -> pd.DataFrame:
    """Cache the loaded dataset for smoother interaction."""

    df = load_data(path)
    # Convert key columns to consistent types
    df["Year"] = pd.to_numeric(df["Year"], errors="coerce").astype("Int64")
    df["Citations"] = pd.to_numeric(df["Citations"], errors="coerce").fillna(0)
    df["APC"] = pd.to_numeric(df.get("APC"), errors="coerce")
    for col in ["Authors", "AuthorKeywords", "SDG"]:
        if col in df.columns:
            df[col] = df[col].fillna("")
    return df


def explode_multivalue(series: pd.Series, separator: str = DEFAULT_MULTI_SEPARATOR) -> pd.Series:
    """Split a semicolon-delimited Series into a long Series of values."""

    return (
        series.fillna("")
        .astype(str)
        .str.split(separator)
        .explode()
        .str.strip()
        .replace("", pd.NA)
        .dropna()
    )


def top_n_counts(series: pd.Series, n: int) -> pd.DataFrame:
    """Return a DataFrame with labels and counts for the top ``n`` values."""

    counts = series.value_counts().head(n)
    return counts.reset_index().rename(columns={"index": "label", 0: "count"})


def make_bar(data: pd.DataFrame, x: str, y: str, title: str, orientation: str = "v"):
    """Create a bar chart with consistent styling."""

    fig = px.bar(data, x=x, y=y, orientation=orientation, title=title)
    fig.update_layout(margin=dict(l=10, r=10, t=40, b=10))
    return fig


# ---------------------------------------------------------------------------
# Layout
# ---------------------------------------------------------------------------
st.set_page_config(page_title="Bibliometric Dashboard", layout="wide")
st.title("Bibliometric Dashboard")

# Load data
try:
    df = get_data(MERGED_CSV_PATH)
except FileNotFoundError as exc:
    st.error(str(exc))
    st.stop()

# Sidebar filters
st.sidebar.header("Filters")
years = df["Year"].dropna().astype(int)
if years.empty:
    year_min, year_max = 2000, 2024
else:
    year_min, year_max = years.min(), years.max()

selected_years = st.sidebar.slider("Year range", min_value=int(year_min), max_value=int(year_max), value=(int(year_min), int(year_max)))

all_doc_types = ["All"] + sorted(df["DocumentType"].dropna().unique())
doc_type = st.sidebar.selectbox("Document type", options=all_doc_types)

if "OpenAccess" in df.columns:
    oa_options = ["All"] + sorted(df["OpenAccess"].dropna().unique())
    oa_filter = st.sidebar.multiselect("Open Access category", oa_options, default=["All"])
else:
    oa_filter = ["All"]

if "SDG" in df.columns:
    sdg_values = explode_multivalue(df["SDG"]).unique()
    sdg_options = ["All"] + sorted([v for v in sdg_values if isinstance(v, str)])
    sdg_filter = st.sidebar.multiselect("SDG", sdg_options, default=["All"])
else:
    sdg_filter = ["All"]

# Apply filters
mask = (
    df["Year"].between(selected_years[0], selected_years[1], inclusive="both")
)
if doc_type != "All":
    mask &= df["DocumentType"] == doc_type
if "All" not in oa_filter:
    mask &= df["OpenAccess"].isin(oa_filter)
if "All" not in sdg_filter and "SDG" in df.columns:
    sdg_exploded = explode_multivalue(df["SDG"])
    sdg_mask = sdg_exploded.isin(sdg_filter)
    mask &= sdg_mask.groupby(level=0).transform("any")

filtered = df[mask].copy()

# Summary metrics
c1, c2, c3, c4 = st.columns(4)
c1.metric("Records", f"{len(filtered):,}")
c2.metric("Total citations", f"{int(filtered['Citations'].sum()):,}")

if "OpenAccess" in filtered.columns:
    oa_counts = filtered["OpenAccess"].value_counts()
    open_total = oa_counts.get("Gold", 0) + oa_counts.get("Green", 0) + oa_counts.get("Hybrid", 0) + oa_counts.get("Bronze", 0)
    oa_share = (open_total / oa_counts.sum() * 100) if oa_counts.sum() else 0
    c3.metric("Open Access share", f"{oa_share:.1f}%")
else:
    c3.metric("Open Access share", "N/A")

if filtered["APC"].notna().any():
    c4.metric("Average APC", f"${filtered['APC'].mean():,.0f}")
else:
    c4.metric("Average APC", "N/A")

st.markdown("---")

# ---------------------------------------------------------------------------
# Core bibliometrics
# ---------------------------------------------------------------------------
st.subheader("Core Bibliometrics")
col1, col2 = st.columns(2)

pubs_per_year = filtered.groupby("Year").size().reset_index(name="count")
col1.plotly_chart(make_bar(pubs_per_year, x="Year", y="count", title="Publications per Year"), use_container_width=True)

col2.plotly_chart(
    px.histogram(filtered, x="Citations", nbins=30, title="Citation Histogram").update_layout(margin=dict(l=10, r=10, t=40, b=10)),
    use_container_width=True,
)

col3, col4 = st.columns(2)
top_journals = top_n_counts(filtered["SourceTitle"].dropna(), 10)
col3.plotly_chart(make_bar(top_journals, x="count", y="label", title="Top Journals / Sources", orientation="h"), use_container_width=True)

authors_long = explode_multivalue(filtered["Authors"])
top_authors = top_n_counts(authors_long, 10)
col4.plotly_chart(make_bar(top_authors, x="count", y="label", title="Top Authors", orientation="h"), use_container_width=True)

keywords_long = explode_multivalue(filtered["AuthorKeywords"])
top_keywords = top_n_counts(keywords_long, 20)
st.plotly_chart(make_bar(top_keywords, x="count", y="label", title="Top Author Keywords", orientation="h"), use_container_width=True)

# ---------------------------------------------------------------------------
# SDG analysis
# ---------------------------------------------------------------------------
st.subheader("SDG Analysis")
if "SDG" in filtered.columns and filtered["SDG"].notna().any():
    sdg_series = explode_multivalue(filtered["SDG"])
    top_n = st.slider("Top N SDGs", min_value=5, max_value=30, value=10, step=5)
    top_sdg = top_n_counts(sdg_series, top_n)
    st.plotly_chart(make_bar(top_sdg, x="count", y="label", title="Publications by SDG", orientation="h"), use_container_width=True)

    sdg_year = explode_multivalue(filtered["SDG"]).to_frame("SDG")
    sdg_year["Year"] = filtered.loc[sdg_year.index, "Year"].values
    sdg_year = sdg_year.dropna(subset=["Year", "SDG"])
    if not sdg_year.empty:
        heatmap_data = sdg_year.groupby(["Year", "SDG"]).size().reset_index(name="count")
        heatmap = px.density_heatmap(heatmap_data, x="Year", y="SDG", z="count", title="Publications by SDG over Time")
        heatmap.update_layout(margin=dict(l=10, r=10, t=40, b=10))
        st.plotly_chart(heatmap, use_container_width=True)
else:
    st.info("No SDG data available.")

# ---------------------------------------------------------------------------
# Open Access analysis
# ---------------------------------------------------------------------------
st.subheader("Open Access Analysis")
if "OpenAccess" in filtered.columns:
    oa_counts = filtered["OpenAccess"].value_counts().reset_index()
    oa_counts.columns = ["OpenAccess", "count"]
    st.plotly_chart(make_bar(oa_counts, x="OpenAccess", y="count", title="Publications by Open Access Category"), use_container_width=True)

    oa_year = filtered.groupby(["Year", "OpenAccess"]).size().reset_index(name="count")
    if not oa_year.empty:
        stacked = px.bar(oa_year, x="Year", y="count", color="OpenAccess", title="Open Access by Year")
        stacked.update_layout(margin=dict(l=10, r=10, t=40, b=10))
        st.plotly_chart(stacked, use_container_width=True)
else:
    st.info("No Open Access data available.")

# ---------------------------------------------------------------------------
# APC analysis
# ---------------------------------------------------------------------------
st.subheader("APC Analysis")
apc_data = filtered.dropna(subset=["APC"])
if not apc_data.empty:
    c1, c2 = st.columns(2)
    c1.plotly_chart(
        px.histogram(apc_data, x="APC", nbins=30, title="APC Distribution").update_layout(margin=dict(l=10, r=10, t=40, b=10)),
        use_container_width=True,
    )
    if "OpenAccess" in apc_data.columns:
        c2.plotly_chart(
            px.box(apc_data, x="OpenAccess", y="APC", title="APC by Open Access Category").update_layout(margin=dict(l=10, r=10, t=40, b=10)),
            use_container_width=True,
        )

    top_sources = apc_data.groupby("SourceTitle")["APC"].mean().sort_values(ascending=False).head(10).reset_index()
    st.plotly_chart(make_bar(top_sources, x="APC", y="SourceTitle", title="Mean APC by Journal/Source (Top 10)", orientation="h"), use_container_width=True)
else:
    st.info("No APC data available.")

st.markdown("---")
st.caption("Dashboard generated from merged Web of Science and Scopus data. Adjust filters to explore subsets.")
