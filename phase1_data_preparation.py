"""
Phase 1: Data preparation and verification for Cyclistic project.

This script assumes the consolidated dataset has (at minimum) the following columns:

    ride_id                object
    rideable_type          object
    started_at             object
    ended_at               object
    start_station_name     object
    start_station_id       object
    end_station_name       object
    end_station_id         object
    start_lat              float64
    start_lng              float64
    end_lat                float64
    end_lng                float64
    member_casual          object
    started_at_clean       string[python]
    ended_at_clean         string[python]
    started_at_parsed      datetime64[ns]
    ended_at_parsed        datetime64[ns]

Phase 1 responsibilities:
- Load data
- Ensure clean datetime columns and parsed datetimes are consistent
- Compute trip duration (sec, min)
- Print verification summaries
- Save Phase 1 dataset for later phases
"""

import argparse
import os
from typing import List

import numpy as np
import pandas as pd


# -------------------------------------------------------------------
# Configuration
# -------------------------------------------------------------------

EXPECTED_COLUMNS: List[str] = [
    "ride_id",
    "rideable_type",
    "started_at",
    "ended_at",
    "start_station_name",
    "start_station_id",
    "end_station_name",
    "end_station_id",
    "start_lat",
    "start_lng",
    "end_lat",
    "end_lng",
    "member_casual",
    "started_at_clean",
    "ended_at_clean",
    "started_at_parsed",
    "ended_at_parsed",
]


# -------------------------------------------------------------------
# Core helpers
# -------------------------------------------------------------------

def load_dataset(input_path: str) -> pd.DataFrame:
    """
    Load the consolidated dataset from parquet or CSV.
    """
    ext = os.path.splitext(input_path)[1].lower()
    if ext == ".parquet":
        df = pd.read_parquet(input_path)
    elif ext == ".csv":
        df = pd.read_csv(input_path)
    else:
        raise ValueError(f"Unsupported file extension: {ext} (use .parquet or .csv)")
    return df


def ensure_clean_and_parsed_datetimes(df: pd.DataFrame) -> pd.DataFrame:
    """
    Ensure the following columns are in a consistent and usable state:

        started_at          (raw object/string)
        ended_at
        started_at_clean    (string[python])
        ended_at_clean
        started_at_parsed   (datetime64[ns])
        ended_at_parsed

    Logic:
    - Make sure *_clean exist; if missing, derive from raw started_at/ended_at.
    - Ensure *_clean are pandas StringDtype (string[python]).
    - Ensure *_parsed are datetime64[ns].
    - For rows where *_parsed is NaT but *_clean is non-null, parse from *_clean.
    - As a last resort, parse from raw started_at/ended_at if needed.
    """
    df = df.copy()

    pairs = [
        ("started_at", "started_at_clean", "started_at_parsed"),
        ("ended_at", "ended_at_clean", "ended_at_parsed"),
    ]

    for raw_col, clean_col, parsed_col in pairs:
        # 1) Ensure raw column exists
        if raw_col not in df.columns:
            raise KeyError(f"Required raw datetime column '{raw_col}' not found.")

        # 2) Ensure clean column exists; if not, create from raw
        if clean_col not in df.columns:
            print(f"[INFO] Creating clean column '{clean_col}' from '{raw_col}'...")
            df[clean_col] = df[raw_col].astype("string[python]")
        else:
            # Ensure StringDtype
            if not pd.api.types.is_string_dtype(df[clean_col].dtype):
                print(f"[INFO] Converting '{clean_col}' to string[python] dtype...")
                df[clean_col] = df[clean_col].astype("string[python]")

        # 3) Ensure parsed column exists; if not, create it
        if parsed_col not in df.columns:
            print(f"[INFO] Creating parsed column '{parsed_col}' from '{clean_col}'...")
            df[parsed_col] = pd.to_datetime(df[clean_col], errors="coerce")

        # 4) Ensure parsed column is datetime64[ns]
        if not np.issubdtype(df[parsed_col].dtype, np.datetime64):
            print(f"[INFO] Converting '{parsed_col}' to datetime...")
            df[parsed_col] = pd.to_datetime(df[parsed_col], errors="coerce")

        # 5) Fill NaT in parsed column from clean strings
        missing_before = df[parsed_col].isna().sum()
        mask_clean = df[parsed_col].isna() & df[clean_col].notna()
        if mask_clean.any():
            df.loc[mask_clean, parsed_col] = pd.to_datetime(
                df.loc[mask_clean, clean_col], errors="coerce"
            )

        # 6) As a last resort, fill remaining NaT from raw column
        mask_raw = df[parsed_col].isna() & df[raw_col].notna()
        if mask_raw.any():
            df.loc[mask_raw, parsed_col] = pd.to_datetime(
                df.loc[mask_raw, raw_col], errors="coerce"
            )

        missing_after = df[parsed_col].isna().sum()

        print(
            f"[INFO] Column '{parsed_col}': "
            f"missing before={missing_before:,}, after={missing_after:,} "
            f"(fixed {missing_before - missing_after:,})"
        )

    return df


def validate_columns_and_types(df: pd.DataFrame) -> None:
    """
    Validate:
    - All EXPECTED_COLUMNS are present.
    - started_at_clean / ended_at_clean are string dtype.
    - started_at_parsed / ended_at_parsed are datetime64[ns].
    """
    missing = [c for c in EXPECTED_COLUMNS if c not in df.columns]
    if missing:
        raise ValueError(f"Missing expected columns: {missing}")

    # String dtype checks
    for col in ["started_at_clean", "ended_at_clean"]:
        if not pd.api.types.is_string_dtype(df[col].dtype):
            raise TypeError(
                f"Column '{col}' must be a string dtype, found {df[col].dtype}"
            )

    # Datetime dtype checks
    for col in ["started_at_parsed", "ended_at_parsed"]:
        if not np.issubdtype(df[col].dtype, np.datetime64):
            raise TypeError(
                f"Column '{col}' must be datetime64[ns], found {df[col].dtype}"
            )

    # Log remaining nulls in parsed columns
    for col in ["started_at_parsed", "ended_at_parsed"]:
        nulls = df[col].isna().sum()
        print(f"[CHECK] Null count in '{col}': {nulls:,}")


def add_duration_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add trip duration features based on the parsed datetime columns:

        trip_duration_sec
        trip_duration_min
    """
    df = df.copy()

    delta = df["ended_at_parsed"] - df["started_at_parsed"]
    df["trip_duration_sec"] = delta.dt.total_seconds()
    df["trip_duration_min"] = df["trip_duration_sec"] / 60.0

    return df


def summarize_dataset(df: pd.DataFrame) -> None:
    """
    Print key checks so we can confirm Phase 1 is correct before moving on.

    - Shape
    - Date range
    - Null counts for key columns
    - Trip duration summary
    - Rider type / bike type breakdown
    - Sample rows
    """
    print("\n" + "=" * 80)
    print("PHASE 1 SUMMARY: DATASET OVERVIEW")
    print("=" * 80)

    # 1) Shape
    print("\n[1] SHAPE")
    print(f"  Rows   : {len(df):,}")
    print(f"  Columns: {len(df.columns)}")

    # 2) Date range
    print("\n[2] DATE RANGE (started_at_parsed)")
    started_min = df["started_at_parsed"].min()
    started_max = df["started_at_parsed"].max()
    print(f"  Min: {started_min}")
    print(f"  Max: {started_max}")

    # 3) Null counts for key columns
    key_cols = [
        "ride_id",
        "rideable_type",
        "start_station_id",
        "end_station_id",
        "member_casual",
        "started_at_parsed",
        "ended_at_parsed",
    ]
    print("\n[3] NULL COUNTS (key columns)")
    print(df[key_cols].isna().sum())

    # 4) Trip duration summary
    if "trip_duration_min" in df.columns:
        print("\n[4] TRIP DURATION SUMMARY (minutes)")
        valid_mask = df["trip_duration_min"].notna()
        desc = df.loc[valid_mask, "trip_duration_min"].describe(
            percentiles=[0.01, 0.05, 0.5, 0.95, 0.99]
        )
        print(desc)

        negative = (df["trip_duration_min"] < 0).sum()
        very_long = (df["trip_duration_min"] > 24 * 60).sum()

        print("\n    Sanity checks:")
        print(f"      Negative durations (< 0 min): {negative:,}")
        print(f"      > 24 hours (> 1440 min)     : {very_long:,}")

    # 5) Rider type breakdown
    if "member_casual" in df.columns:
        print("\n[5] RIDER TYPE BREAKDOWN (member_casual)")
        print(df["member_casual"].value_counts(dropna=False))

    # 6) Bike type breakdown
    if "rideable_type" in df.columns:
        print("\n[6] BIKE TYPE BREAKDOWN (rideable_type)")
        print(df["rideable_type"].value_counts(dropna=False))

    # 7) Basic station stats
    if "start_station_id" in df.columns:
        print("\n[7] TOP 5 START STATIONS (by trip count)")
        print(df["start_station_id"].value_counts().head(5))

    # 8) Sample rows
    print("\n[8] SAMPLE ROWS")
    print(df.head(5))

    print("\n" + "=" * 80)
    print("End of Phase 1 summary. If everything above looks reasonable,")
    print("we are safe to proceed to Phase 2 (feature engineering / modeling).")
    print("=" * 80 + "\n")


# -------------------------------------------------------------------
# Saving and CLI
# -------------------------------------------------------------------

def save_dataset(df: pd.DataFrame, output_path: str) -> None:
    """
    Save the processed dataset. Format is inferred from file extension.
    """
    dir_name = os.path.dirname(output_path)
    if dir_name:
        os.makedirs(dir_name, exist_ok=True)

    ext = os.path.splitext(output_path)[1].lower()
    if ext == ".parquet":
        df.to_parquet(output_path, index=False)
    elif ext == ".csv":
        df.to_csv(output_path, index=False)
    else:
        raise ValueError(
            f"Unsupported output extension: {ext} (use .parquet or .csv)"
        )

    print(f"\n[INFO] Phase 1 dataset saved to: {output_path}")


def filter_bad_durations(df: pd.DataFrame) -> pd.DataFrame:
    """
    Remove trips with clearly invalid durations:
    - Negative durations
    - Durations > 24 hours

    Returns the cleaned dataframe.
    """
    df = df.copy()

    mask_negative = df["trip_duration_min"] < 0
    mask_over_24h = df["trip_duration_min"] > 24 * 60

    bad_count = (mask_negative | mask_over_24h).sum()
    print(f"[INFO] Removing {bad_count:,} trips with invalid durations.")

    df = df[~(mask_negative | mask_over_24h)]

    return df



def main() -> None:

    input_path = "data/full_bike_raw_dataset.parquet"
    df = load_dataset(input_path)

    print("[INFO] Ensuring clean and parsed datetime columns are consistent...")
    df = ensure_clean_and_parsed_datetimes(df)

    print("[INFO] Validating columns and dtypes...")
    validate_columns_and_types(df)

    print("[INFO] Adding duration features based on started_at_parsed / ended_at_parsed...")
    df = add_duration_features(df)

    print("[INFO] Filtering trips with invalid durations (negative or > 24h)...")
    df = filter_bad_durations(df)

    print("[INFO] Running Phase 1 summary checks...")
    summarize_dataset(df)

    print("[INFO] Saving processed Phase 1 dataset...")
    output_path = "data/processed/full_bike_dataset_phase1.parquet"
    save_dataset(df, output_path)


if __name__ == "__main__":
    main()
