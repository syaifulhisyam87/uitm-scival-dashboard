"""Preprocess Web of Science and Scopus CSV exports.

This script harmonises fields, merges both sources, computes basic
bibliometric indicators (including SDG, Open Access, and APC summaries),
and writes a cleaned CSV file for dashboarding.

Usage:
    python biblio_preprocess.py

Adjust the ``WOS_CSV_PATH`` and ``SCOPUS_CSV_PATH`` variables near the top of
this file to point to your exports. The merged output will be written to
``MERGED_OUTPUT_PATH``.
"""
from __future__ import annotations

import math
from pathlib import Path
from typing import Iterable, Optional

import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
# Update these paths to match your local CSV exports.
WOS_CSV_PATH = Path("wos_export.csv")
SCOPUS_CSV_PATH = Path("scopus_export.csv")
MERGED_OUTPUT_PATH = Path("merged_biblio.csv")

# Default separator used by WoS/Scopus for multi-value fields.
DEFAULT_MULTI_SEPARATOR = ";"

# Histogram bins for citation and APC summaries.
CITATION_BINS = [0, 1, 5, 10, 20, 50, 100, 200, 500, np.inf]
APC_BINS = [0, 500, 1000, 2000, 5000, 10000, np.inf]


# ---------------------------------------------------------------------------
# Helper functions
# ---------------------------------------------------------------------------
def clean_string(value: Optional[str]) -> Optional[str]:
    """Trim whitespace and normalise empty strings to ``None``.

    Parameters
    ----------
    value:
        A scalar string or None.

    Returns
    -------
    Optional[str]
        Cleaned string or None if empty/NaN.
    """

    if value is None or (isinstance(value, float) and math.isnan(value)):
        return None
    text = str(value).strip()
    return text if text else None


def explode_multivalue_column(df: pd.DataFrame, column: str, separator: str = DEFAULT_MULTI_SEPARATOR) -> pd.DataFrame:
    """Explode a multi-value column into one value per row.

    Parameters
    ----------
    df:
        DataFrame containing the column.
    column:
        Column name to explode.
    separator:
        Delimiter used to split values (default is semicolon).

    Returns
    -------
    pd.DataFrame
        A new DataFrame with the specified column exploded and other columns duplicated.
    """

    if column not in df.columns:
        return pd.DataFrame(columns=df.columns)

    tmp = df.copy()
    tmp[column] = (
        tmp[column]
        .fillna("")
        .astype(str)
        .str.split(separator)
        .apply(lambda items: [clean_string(v) for v in items if clean_string(v)])
    )
    exploded = tmp.explode(column)
    return exploded.dropna(subset=[column])


def top_n_frequency(series: pd.Series, n: int) -> pd.Series:
    """Return the top ``n`` value counts from a Series, sorted descending."""

    return series.dropna().value_counts().head(n)


def combine_columns(df: pd.DataFrame, candidate_columns: Iterable[str], separator: str = DEFAULT_MULTI_SEPARATOR) -> pd.Series:
    """Combine multiple candidate columns into a single Series.

    Strings are split on ``separator``, cleaned, deduplicated, and then
    re-joined with ``separator``.
    """

    collected = []
    for col in candidate_columns:
        if col in df.columns:
            parts = (
                df[col]
                .fillna("")
                .astype(str)
                .str.split(separator)
                .apply(lambda values: [clean_string(v) for v in values if clean_string(v)])
            )
            collected.append(parts)

    if not collected:
        return pd.Series([np.nan] * len(df))

    stacked = pd.concat(collected, axis=1)

    def merge_row(row: pd.Series) -> Optional[str]:
        values: list[str] = []
        for cell in row.dropna():
            values.extend(cell)
        unique_values = []
        for val in values:
            if val and val not in unique_values:
                unique_values.append(val)
        return separator.join(unique_values) if unique_values else None

    return stacked.apply(merge_row, axis=1)


def detect_columns(df: pd.DataFrame, keywords: Iterable[str]) -> list[str]:
    """Find columns whose lowercase name contains any of the keywords."""

    lowered = {col: col.lower() for col in df.columns}
    return [col for col, low in lowered.items() if any(key in low for key in keywords)]


def map_open_access(value: Optional[str]) -> str:
    """Map a raw OA indicator into a standard category.

    Categories follow a simplified taxonomy: Gold, Green, Hybrid, Bronze,
    Closed, or Unknown.
    """

    if value is None:
        return "Unknown"

    text = value.strip().lower()
    if not text:
        return "Unknown"

    if any(word in text for word in ["gold", "doaj", "oa gold"]):
        return "Gold"
    if "hybrid" in text:
        return "Hybrid"
    if "green" in text or "repository" in text:
        return "Green"
    if "bronze" in text:
        return "Bronze"
    if any(word in text for word in ["closed", "subscription", "paywalled", "non-oa"]):
        return "Closed"

    # Common Scopus codes
    if text in {"oa", "open", "open access"}:
        return "Gold"
    if text in {"none", "no", "n/a"}:
        return "Closed"

    return "Unknown"


def first_valid(series: pd.Series) -> Optional[str]:
    """Return the first non-null value from a Series."""

    for val in series:
        if pd.notna(val) and str(val).strip():
            return str(val)
    return None


def to_numeric(series: pd.Series) -> pd.Series:
    """Convert a Series to numeric, coercing errors to NaN."""

    return pd.to_numeric(series, errors="coerce")


# ---------------------------------------------------------------------------
# Harmonisation logic
# ---------------------------------------------------------------------------
def harmonise_wos(df: pd.DataFrame) -> pd.DataFrame:
    """Standardise Web of Science fields."""

    rename_map = {
        "PY": "Year",
        "SO": "SourceTitle",
        "AU": "Authors",
        "DE": "AuthorKeywords",
        "ID": "IndexKeywords",
        "TC": "Citations",
        "DT": "DocumentType",
    }
    df = df.rename(columns={k: v for k, v in rename_map.items() if k in df.columns})
    df["Source"] = "WoS"

    # SDG fields
    sdg_cols = detect_columns(df, ["sdg", "sustainable development"])
    df["SDG"] = combine_columns(df, sdg_cols)

    # Open Access fields
    oa_cols = detect_columns(df, ["open access", "oa", "access type"])
    df["OpenAccess"] = combine_columns(df, oa_cols)

    # APC fields
    apc_cols = detect_columns(df, ["apc", "article processing", "publication fee"])
    if apc_cols:
        df["APC"] = to_numeric(df[apc_cols].apply(first_valid, axis=1))
    else:
        df["APC"] = np.nan

    return df


def harmonise_scopus(df: pd.DataFrame) -> pd.DataFrame:
    """Standardise Scopus fields."""

    rename_map = {
        "Year": "Year",
        "Source title": "SourceTitle",
        "Authors": "Authors",
        "Author Keywords": "AuthorKeywords",
        "Index Keywords": "IndexKeywords",
        "Cited by": "Citations",
        "Document Type": "DocumentType",
    }
    df = df.rename(columns={k: v for k, v in rename_map.items() if k in df.columns})
    df["Source"] = "Scopus"

    sdg_cols = detect_columns(df, ["sdg", "sustainable development"])
    df["SDG"] = combine_columns(df, sdg_cols)

    oa_cols = detect_columns(df, ["open access", "oa", "access type"])
    df["OpenAccess"] = combine_columns(df, oa_cols)

    apc_cols = detect_columns(df, ["apc", "article processing", "publication fee"])
    if apc_cols:
        df["APC"] = to_numeric(df[apc_cols].apply(first_valid, axis=1))
    else:
        df["APC"] = np.nan

    return df


def harmonise_sources(wos_path: Path, scopus_path: Path) -> pd.DataFrame:
    """Read and harmonise WoS and Scopus CSV exports."""

    frames = []
    if wos_path.exists():
        wos_df = pd.read_csv(wos_path)
        frames.append(harmonise_wos(wos_df))
    else:
        print(f"[WARN] WoS file not found: {wos_path}")

    if scopus_path.exists():
        scopus_df = pd.read_csv(scopus_path)
        frames.append(harmonise_scopus(scopus_df))
    else:
        print(f"[WARN] Scopus file not found: {scopus_path}")

    if not frames:
        raise FileNotFoundError("No input files were found. Please update the paths.")

    merged = pd.concat(frames, ignore_index=True, sort=False)

    # Ensure essential columns exist
    for col in [
        "Year",
        "SourceTitle",
        "Authors",
        "AuthorKeywords",
        "IndexKeywords",
        "Citations",
        "DocumentType",
        "Source",
        "SDG",
        "OpenAccess",
        "APC",
    ]:
        if col not in merged.columns:
            merged[col] = np.nan

    merged["Year"] = pd.to_numeric(merged["Year"], errors="coerce").astype("Int64")
    merged["Citations"] = pd.to_numeric(merged["Citations"], errors="coerce").fillna(0).astype(int)
    merged["OpenAccess"] = merged["OpenAccess"].apply(map_open_access)
    merged["APC"] = to_numeric(merged["APC"])

    # Clean string columns
    for col in ["SourceTitle", "Authors", "AuthorKeywords", "IndexKeywords", "DocumentType", "SDG"]:
        merged[col] = merged[col].apply(clean_string)

    return merged


# ---------------------------------------------------------------------------
# Reporting helpers
# ---------------------------------------------------------------------------
def print_header(title: str) -> None:
    """Print a formatted section header."""

    print("\n" + "=" * 80)
    print(title)
    print("=" * 80)


def summarize_core_indicators(df: pd.DataFrame) -> None:
    """Print core bibliometric indicators."""

    print_header("Publications per Year")
    print(df["Year"].value_counts().sort_index())

    print_header("Top 10 Journals / Sources")
    print(top_n_frequency(df["SourceTitle"], 10))

    print_header("Top 10 Authors")
    authors_exploded = explode_multivalue_column(df, "Authors")
    print(top_n_frequency(authors_exploded["Authors"], 10))

    print_header("Top 20 Author Keywords")
    keywords_exploded = explode_multivalue_column(df, "AuthorKeywords")
    print(top_n_frequency(keywords_exploded["AuthorKeywords"], 20))

    print_header("Citation Statistics")
    print(df["Citations"].describe())
    print("\nCitation Histogram (counts per bucket):")
    labels = pd.IntervalIndex.from_breaks(CITATION_BINS)
    hist = pd.cut(df["Citations"], bins=CITATION_BINS, include_lowest=True)
    hist_counts = hist.value_counts().sort_index()
    hist_counts.index = labels
    print(hist_counts)


def summarize_sdg(df: pd.DataFrame) -> None:
    """Print SDG publication counts and cross-tabs."""

    print_header("SDG Analysis")
    sdg_exploded = explode_multivalue_column(df, "SDG")
    if sdg_exploded.empty:
        print("No SDG data found.")
        return

    print("Top SDGs by publication count:")
    print(top_n_frequency(sdg_exploded["SDG"], 20))

    print("\nPublications per SDG per Year:")
    sdg_year = sdg_exploded.pivot_table(index="Year", columns="SDG", aggfunc="size", fill_value=0)
    print(sdg_year)


def summarize_open_access(df: pd.DataFrame) -> None:
    """Print Open Access counts and shares."""

    print_header("Open Access Analysis")
    counts = df["OpenAccess"].value_counts(dropna=False)
    print("Counts by OpenAccess category:")
    print(counts)

    shares = (counts / len(df) * 100).round(2)
    print("\nShare (%) by category:")
    print(shares)

    print("\nOpenAccess by Year:")
    oa_year = df.pivot_table(index="Year", columns="OpenAccess", aggfunc="size", fill_value=0)
    print(oa_year)


def summarize_apc(df: pd.DataFrame) -> None:
    """Print APC statistics and optional breakdowns."""

    print_header("APC Analysis")
    apc_non_null = df.dropna(subset=["APC"])
    if apc_non_null.empty:
        print("No APC data found.")
        return

    print("APC basic statistics:")
    print(apc_non_null["APC"].describe())

    print("\nAPC histogram (counts per bucket):")
    labels = pd.IntervalIndex.from_breaks(APC_BINS)
    apc_hist = pd.cut(apc_non_null["APC"], bins=APC_BINS, include_lowest=True).value_counts().sort_index()
    apc_hist.index = labels
    print(apc_hist)

    print("\nAverage APC by OpenAccess category:")
    print(apc_non_null.groupby("OpenAccess")["APC"].mean().sort_values(ascending=False))

    print("\nAverage APC by SourceTitle (top 10):")
    print(apc_non_null.groupby("SourceTitle")["APC"].mean().sort_values(ascending=False).head(10))


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------
def main() -> None:
    merged = harmonise_sources(WOS_CSV_PATH, SCOPUS_CSV_PATH)
    merged.to_csv(MERGED_OUTPUT_PATH, index=False)
    print(f"Merged dataset saved to {MERGED_OUTPUT_PATH.resolve()}")

    summarize_core_indicators(merged)
    summarize_sdg(merged)
    summarize_open_access(merged)
    summarize_apc(merged)


if __name__ == "__main__":
    main()
