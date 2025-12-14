"""
Phase 2: Feature Engineering for Cyclistic Multi-Task Project

Input (from Phase 1):
    - data/processed/full_bike_dataset_phase1.parquet (by default)

Expected minimum columns:
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
    trip_duration_sec      float64
    trip_duration_min      float64

Phase 2 responsibilities:
    - Remove trips with clearly invalid durations
      (negative or > 24 hours)
    - Engineer time-based features:
        * date, year, month, day, hour, minute, weekday, is_weekend
    - Engineer simple behavioral feature:
        * is_roundtrip (start_station_id == end_station_id)
    - Optionally add a log-transformed duration:
        * trip_duration_min_log1p
    - Summarize and save as Phase 2 dataset.
"""

import argparse
import os
from typing import List

import numpy as np
import pandas as pd


# -------------------------------------------------------------------
# Configuration
# -------------------------------------------------------------------

REQUIRED_COLUMNS: List[str] = [
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
    "trip_duration_sec",
    "trip_duration_min",
]


# -------------------------------------------------------------------
# Core helpers
# -------------------------------------------------------------------

def load_dataset(input_path: str) -> pd.DataFrame:
    """
    Load the Phase 1 dataset from parquet or CSV.
    """
    ext = os.path.splitext(input_path)[1].lower()
    if ext == ".parquet":
        df = pd.read_parquet(input_path)
    elif ext == ".csv":
        df = pd.read_csv(input_path)
    else:
        raise ValueError(f"Unsupported file extension: {ext} (use .parquet or .csv)")
    return df


def validate_phase1_input(df: pd.DataFrame) -> None:
    """
    Ensure all required columns from Phase 1 are present in the input.
    """
    missing = [c for c in REQUIRED_COLUMNS if c not in df.columns]
    if missing:
        raise ValueError(
            f"Phase 2 expected the following columns from Phase 1, "
            f"but they are missing: {missing}"
        )

    # Basic dtype checks for datetime columns
    for col in ["started_at_parsed", "ended_at_parsed"]:
        if not np.issubdtype(df[col].dtype, np.datetime64):
            raise TypeError(
                f"Column '{col}' must be datetime64[ns], found dtype={df[col].dtype}"
            )


def filter_bad_durations(df: pd.DataFrame) -> pd.DataFrame:
    """
    Remove trips with clearly invalid durations:
        - Negative durations
        - Durations > 24 hours

    These are treated as data errors, not "interesting" anomalies.
    """
    df = df.copy()

    # Count before
    n_before = len(df)

    mask_negative = df["trip_duration_min"] < 0
    mask_over_24h = df["trip_duration_min"] > 24 * 60

    bad_mask = mask_negative | mask_over_24h
    bad_count = bad_mask.sum()

    print(f"[INFO] Found {bad_count:,} trips with invalid durations.")
    if bad_count > 0:
        df = df[~bad_mask].copy()
        n_after = len(df)
        print(
            f"[INFO] Removed invalid duration trips. "
            f"Rows before={n_before:,}, after={n_after:,} "
            f"(dropped {n_before - n_after:,})."
        )
    else:
        print("[INFO] No invalid durations found; no rows removed.")

    return df


def add_time_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add time-based features derived from started_at_parsed:

        - date        (date only, object dtype)
        - year        (int)
        - month       (int)
        - day         (int)
        - hour        (int)
        - minute      (int)
        - weekday     (0=Monday,...,6=Sunday)
        - is_weekend  (bool)
    """
    df = df.copy()

    dt = df["started_at_parsed"]

    df["date"] = dt.dt.date
    df["year"] = dt.dt.year.astype("int16")
    df["month"] = dt.dt.month.astype("int8")
    df["day"] = dt.dt.day.astype("int8")
    df["hour"] = dt.dt.hour.astype("int8")
    df["minute"] = dt.dt.minute.astype("int8")
    df["weekday"] = dt.dt.weekday.astype("int8")  # Monday=0, Sunday=6
    df["is_weekend"] = df["weekday"].isin([5, 6])  # Saturday, Sunday

    return df


def add_behavior_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add simple behavioral features:

        - is_roundtrip: True if start_station_id == end_station_id and not null
        - trip_duration_min_log1p: log1p of duration in minutes (for modeling)
    """
    df = df.copy()

    # is_roundtrip: same station, ignoring rows with missing station IDs
    same_station = (
        df["start_station_id"].notna()
        & df["end_station_id"].notna()
        & (df["start_station_id"] == df["end_station_id"])
    )
    df["is_roundtrip"] = same_station

    # log1p duration (avoid log(0) issues)
    # Only for non-negative durations (negative ones should already be removed).
    df["trip_duration_min_log1p"] = np.log1p(df["trip_duration_min"].clip(lower=0))

    return df


def summarize_phase2(df: pd.DataFrame) -> None:
    """
    Print key summaries to verify Phase 2 worked as intended.
    """
    print("\n" + "=" * 80)
    print("PHASE 2 SUMMARY: FEATURE ENGINEERING OVERVIEW")
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

    # 3) Duration sanity after filtering
    print("\n[3] TRIP DURATION SUMMARY AFTER FILTERING (minutes)")
    desc = df["trip_duration_min"].describe(
        percentiles=[0.01, 0.05, 0.5, 0.95, 0.99]
    )
    print(desc)

    negative = (df["trip_duration_min"] < 0).sum()
    over_24h = (df["trip_duration_min"] > 24 * 60).sum()
    print("\n    Sanity checks (post-filter):")
    print(f"      Negative durations (< 0 min): {negative:,}")
    print(f"      > 24 hours (> 1440 min)     : {over_24h:,}")

    # 4) Time feature distributions
    print("\n[4] TIME FEATURE CHECKS")
    print("  Year counts:")
    print(df["year"].value_counts().sort_index())
    print("\n  Month counts:")
    print(df["month"].value_counts().sort_index())
    print("\n  Weekday counts (0=Mon,...,6=Sun):")
    print(df["weekday"].value_counts().sort_index())
    print("\n  is_weekend value counts:")
    print(df["is_weekend"].value_counts())

    # 5) Rider / bike type breakdown (unchanged, but useful sanity check)
    print("\n[5] RIDER TYPE BREAKDOWN (member_casual)")
    print(df["member_casual"].value_counts(dropna=False))

    print("\n[6] BIKE TYPE BREAKDOWN (rideable_type)")
    print(df["rideable_type"].value_counts(dropna=False))

    # 7) Roundtrip vs non-roundtrip trips
    if "is_roundtrip" in df.columns:
        print("\n[7] ROUNDTRIP FLAG")
        print(df["is_roundtrip"].value_counts())

    # 8) Sample rows (with new features)
    print("\n[8] SAMPLE ROWS (with engineered features)")
    cols_to_show = [
        "ride_id",
        "member_casual",
        "rideable_type",
        "started_at_parsed",
        "ended_at_parsed",
        "trip_duration_min",
        "year",
        "month",
        "day",
        "hour",
        "weekday",
        "is_weekend",
        "is_roundtrip",
    ]
    cols_to_show = [c for c in cols_to_show if c in df.columns]
    print(df[cols_to_show].head(5))

    print("\n" + "=" * 80)
    print("End of Phase 2 summary. If everything above looks reasonable,")
    print("the dataset is ready for Phase 3 (task-specific modeling pipelines).")
    print("=" * 80 + "\n")


def save_dataset(df: pd.DataFrame, output_path: str) -> None:
    """
    Save the Phase 2 dataset. Format is inferred from file extension.
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

    print(f"\n[INFO] Phase 2 dataset saved to: {output_path}")


# -------------------------------------------------------------------
# CLI
# -------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Phase 2: Feature engineering for Cyclistic project."
    )
    parser.add_argument(
        "--input-path",
        "-i",
        type=str,
        default="data/processed/full_bike_dataset_phase1.parquet",
        help="Path to the Phase 1 dataset (parquet or CSV).",
    )
    parser.add_argument(
        "--output-path",
        "-o",
        type=str,
        default="data/processed/full_bike_dataset_phase2.parquet",
        help="Where to save the Phase 2 dataset (extension determines format).",
    )
    return parser.parse_args()


def main() -> None:

    input_path = "data/processed/full_bike_dataset_phase1.parquet"
    print(f"[INFO] Loading Phase 1 dataset from: {input_path}")
    df = load_dataset(input_path)

    print("[INFO] Validating Phase 1 input schema...")
    validate_phase1_input(df)

    print("[INFO] Filtering trips with invalid durations (negative or > 24h)...")
    df = filter_bad_durations(df)

    print("[INFO] Adding time-based features...")
    df = add_time_features(df)

    print("[INFO] Adding behavioral features (is_roundtrip, log-duration)...")
    df = add_behavior_features(df)

    print("[INFO] Running Phase 2 summary checks...")
    summarize_phase2(df)

    print("[INFO] Saving Phase 2 dataset...")
    output_path = "data/processed/full_bike_dataset_phase2.parquet"
    save_dataset(df, output_path)


if __name__ == "__main__":
    main()
