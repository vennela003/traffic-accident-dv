"""
Data Loader and Preprocessing Pipeline for US Accidents Dataset.
Optimized for high-speed streaming, chunked sampling, null/duplicate cleaning,
and Parquet caching.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Callable, Optional, Sequence, Tuple
import numpy as np
import pandas as pd
import streamlit as st

# Default file paths
PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_RAW_CSV = PROJECT_ROOT / "data" / "US_Accidents_March23.csv"
DEFAULT_PARQUET = PROJECT_ROOT / "data" / "us_accidents_sampled.parquet"

REQUIRED_COLUMNS = [
    "Severity",
    "Start_Time",
    "End_Time",
    "Start_Lat",
    "Start_Lng",
    "City",
    "State",
    "Weather_Condition",
]

DAY_ORDER = [
    "Monday",
    "Tuesday",
    "Wednesday",
    "Thursday",
    "Friday",
    "Saturday",
    "Sunday",
]


def clean_chunk(chunk: pd.DataFrame) -> pd.DataFrame:
    """Clean a single chunk of data."""
    # Drop rows missing crucial fields
    chunk = chunk.dropna(subset=["Start_Lat", "Start_Lng", "Start_Time", "State", "Severity"])

    # Impute categorical text fields
    chunk["City"] = chunk["City"].fillna("Unknown").astype(str).str.strip()
    chunk["Weather_Condition"] = (
        chunk["Weather_Condition"].fillna("Unknown").astype(str).str.strip()
    )

    # Valid bounding box for continental US, Alaska, and Hawaii
    valid_coords = (
        (chunk["Start_Lat"] >= 18.0)
        & (chunk["Start_Lat"] <= 72.0)
        & (chunk["Start_Lng"] >= -180.0)
        & (chunk["Start_Lng"] <= -60.0)
    )
    chunk = chunk[valid_coords]
    return chunk


def process_and_cache_dataset(
    raw_csv_path: Path | str = DEFAULT_RAW_CSV,
    parquet_path: Path | str = DEFAULT_PARQUET,
    sample_fraction: float = 0.05,
    chunk_size: int = 250_000,
    random_state: int = 42,
    progress_callback: Optional[Callable[[float, str], None]] = None,
) -> pd.DataFrame:
    """
    Reads the 7.7M row CSV in chunks, performs uniform random sampling across
    all chunks, cleans missing values and duplicates, extracts temporal features,
    and caches the result to an optimized Parquet file.
    """
    raw_csv_path = Path(raw_csv_path)
    parquet_path = Path(parquet_path)

    if not raw_csv_path.exists():
        raise FileNotFoundError(f"Raw dataset not found at: {raw_csv_path}")

    parquet_path.parent.mkdir(parents=True, exist_ok=True)

    sampled_chunks: list[pd.DataFrame] = []
    total_raw_rows = 0
    total_sampled_rows = 0

    if progress_callback:
        progress_callback(0.05, "Reading and sampling dataset in chunks...")

    # Read in chunks
    estimated_total_chunks = 31  # ~7.7M / 250k chunks
    for idx, chunk in enumerate(
        pd.read_csv(
            raw_csv_path,
            usecols=REQUIRED_COLUMNS,
            chunksize=chunk_size,
            low_memory=False,
        )
    ):
        total_raw_rows += len(chunk)
        cleaned = clean_chunk(chunk)

        # Uniform sample from this chunk
        sampled = cleaned.sample(
            frac=sample_fraction,
            random_state=random_state + idx,
        )
        sampled_chunks.append(sampled)
        total_sampled_rows += len(sampled)

        if progress_callback:
            progress = min(0.1 + (idx + 1) / estimated_total_chunks * 0.65, 0.75)
            progress_callback(
                progress,
                f"Sampled {total_sampled_rows:,} rows from {total_raw_rows:,} processed...",
            )

    if progress_callback:
        progress_callback(0.8, "Merging sampled chunks and removing duplicates...")

    df = pd.concat(sampled_chunks, ignore_index=True)
    df = df.drop_duplicates(subset=["Start_Time", "Start_Lat", "Start_Lng", "State"])

    if progress_callback:
        progress_callback(0.85, "Parsing datetimes and extracting temporal features...")

    # Datetime parsing
    df["Start_Time"] = pd.to_datetime(df["Start_Time"], errors="coerce")
    df["End_Time"] = pd.to_datetime(df["End_Time"], errors="coerce")
    df = df.dropna(subset=["Start_Time"])

    # Feature extraction
    df["Hour"] = df["Start_Time"].dt.hour.astype("int8")
    df["Day_Name"] = pd.Categorical(
        df["Start_Time"].dt.day_name(),
        categories=DAY_ORDER,
        ordered=True,
    )
    df["Day_of_Week"] = df["Start_Time"].dt.dayofweek.astype("int8")
    df["Year"] = df["Start_Time"].dt.year.astype("int16")
    df["Month"] = df["Start_Time"].dt.month.astype("int8")
    df["Date"] = pd.to_datetime(df["Start_Time"].dt.date)

    # Incident duration clipped to realistic max 24 hours (1440 min)
    duration = (df["End_Time"] - df["Start_Time"]).dt.total_seconds() / 60.0
    df["Duration_Minutes"] = duration.clip(lower=0, upper=1440).fillna(30.0).astype("float32")

    # Downcast datatypes for memory and load speed
    df["Severity"] = df["Severity"].astype("int8")
    df["Start_Lat"] = df["Start_Lat"].astype("float32")
    df["Start_Lng"] = df["Start_Lng"].astype("float32")
    df["State"] = df["State"].astype("category")
    df["City"] = df["City"].astype("string")
    df["Weather_Condition"] = df["Weather_Condition"].astype("category")

    # Sort chronologically
    df = df.sort_values("Start_Time").reset_index(drop=True)

    if progress_callback:
        progress_callback(0.95, "Saving optimized Parquet cache...")

    df.to_parquet(parquet_path, engine="pyarrow", compression="snappy", index=False)

    if progress_callback:
        progress_callback(1.0, f"Ready! Cached {len(df):,} cleaned rows to {parquet_path.name}")

    return df


@st.cache_data(show_spinner=False)
def load_dataset(
    parquet_path: str = str(DEFAULT_PARQUET),
    raw_csv_path: str = str(DEFAULT_RAW_CSV),
    sample_fraction: float = 0.05,
    force_reload: bool = False,
) -> pd.DataFrame:
    """
    Loads dataset from Parquet cache if available, otherwise generates it
    from the raw CSV file.
    """
    parquet_file = Path(parquet_path)
    raw_file = Path(raw_csv_path)

    if parquet_file.exists() and not force_reload:
        df = pd.read_parquet(parquet_file)
        # Ensure category order and datetimes are preserved
        if "Day_Name" in df.columns:
            df["Day_Name"] = pd.Categorical(df["Day_Name"], categories=DAY_ORDER, ordered=True)
        if "Date" in df.columns:
            df["Date"] = pd.to_datetime(df["Date"])
        return df

    return process_and_cache_dataset(
        raw_csv_path=raw_file,
        parquet_path=parquet_file,
        sample_fraction=sample_fraction,
    )


def filter_data(
    df: pd.DataFrame,
    states: Optional[Sequence[str]] = None,
    date_range: Optional[Tuple[pd.Timestamp, pd.Timestamp]] = None,
    severities: Optional[Sequence[int]] = None,
    weather_conditions: Optional[Sequence[str]] = None,
) -> pd.DataFrame:
    """Filter DataFrame according to user sidebar selections."""
    filtered = df

    if states and "All" not in states:
        filtered = filtered[filtered["State"].isin(states)]

    if severities:
        filtered = filtered[filtered["Severity"].isin(severities)]

    if date_range and len(date_range) == 2:
        start_date, end_date = date_range
        # Compare as datetime
        filtered = filtered[
            (filtered["Date"] >= pd.to_datetime(start_date))
            & (filtered["Date"] <= pd.to_datetime(end_date))
        ]

    if weather_conditions and "All" not in weather_conditions:
        filtered = filtered[filtered["Weather_Condition"].isin(weather_conditions)]

    return filtered
