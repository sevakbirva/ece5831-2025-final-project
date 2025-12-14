"""
Phase 3: Build Model-Ready Datasets for Hierarchical VAE

Input (from Phase 2):
    data/processed/full_bike_dataset_phase2.parquet  (by default)

Expected columns (at minimum):
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
    trip_duration_min_log1p float64 (from Phase 2)
    year, month, day       int
    hour, minute           int
    weekday                int
    is_weekend             bool
    is_roundtrip           bool
    date                   (python date; from Phase 2)

Phase 3 responsibilities:
    - Compute a simple "demand contribution" target:
        * start_station_day_share = trips at station on that day / total trips that day
    - Encode categorical features into integer indices:
        * start_station_id_idx
        * end_station_id_idx
        * rideable_type_idx
        * member_casual_idx
    - Add simple integer versions of booleans:
        * is_weekend_int, is_roundtrip_int
    - Time-based split into train/val/test:
        * Train: trips before 2025-02-01
        * Val:   [2025-02-01, 2025-03-01)
        * Test:  trips on/after 2025-03-01
    - Compute normalization stats on train split for selected numeric features and
      add normalized columns with suffix "_norm".
    - Save:
        * train/val/test parquet files
        * phase3_artifacts.pkl with:
            - category_mappings
            - numeric_scalers
            - feature_config
            - split_boundaries
"""

import argparse
import os
import pickle
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd


# -------------------------------------------------------------------
# Configuration
# -------------------------------------------------------------------

REQUIRED_PHASE2_COLUMNS: List[str] = [
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
    "trip_duration_min_log1p",
    "year",
    "month",
    "day",
    "hour",
    "minute",
    "weekday",
    "is_weekend",
    "is_roundtrip",
    "date",
]

CATEGORICAL_COLUMNS: List[str] = [
    "start_station_id",
    "end_station_id",
    "rideable_type",
    "member_casual",
]

NUMERIC_FOR_SCALING: List[str] = [
    "trip_duration_min",
    "trip_duration_min_log1p",
    "start_lat",
    "start_lng",
    "end_lat",
    "end_lng",
]

# Time-based split boundaries (can be adjusted if needed)
SPLIT_BOUNDARIES = {
    "train_end": "2025-02-01",  # train: < 2025-02-01
    "val_end": "2025-03-01",    # val: [2025-02-01, 2025-03-01), test: >= 2025-03-01
}


# -------------------------------------------------------------------
# Helpers
# -------------------------------------------------------------------

def load_dataset(input_path: str) -> pd.DataFrame:
    """Load Phase 2 dataset from parquet or CSV."""
    ext = os.path.splitext(input_path)[1].lower()
    if ext == ".parquet":
        df = pd.read_parquet(input_path)
    elif ext == ".csv":
        df = pd.read_csv(input_path)
    else:
        raise ValueError(f"Unsupported file extension: {ext} (use .parquet or .csv)")
    return df


def validate_phase2_df(df: pd.DataFrame) -> None:
    """Ensure all required Phase 2 columns are present."""
    missing = [c for c in REQUIRED_PHASE2_COLUMNS if c not in df.columns]
    if missing:
        raise ValueError(
            f"Phase 3 expected the following columns from Phase 2, "
            f"but they are missing: {missing}"
        )

    # Basic dtype check
    if not np.issubdtype(df["started_at_parsed"].dtype, np.datetime64):
        raise TypeError(
            f"'started_at_parsed' must be datetime64[ns], "
            f"found {df['started_at_parsed'].dtype}"
        )


def add_demand_contribution(df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute a simple "demand contribution" target for each trip:

        start_station_day_share =
            (trips starting at station s on date d)
            / (all trips starting anywhere on date d)

    For rows with missing start_station_id, share defaults to 0.0.
    """
    df = df.copy()

    # Count trips per (date, start_station_id)
    group_counts = (
        df.groupby(["date", "start_station_id"], observed=True)
        .size()
        .rename("start_station_day_count")
    )
    df = df.join(group_counts, on=["date", "start_station_id"])

    # Total trips per day
    day_totals = (
        df.groupby("date", observed=True)["start_station_day_count"]
        .transform("sum")
    )

    df["start_station_day_share"] = (
        df["start_station_day_count"] / day_totals.replace(0, np.nan)
    )

    df["start_station_day_share"] = df["start_station_day_share"].fillna(0.0)

    return df


def build_category_mapping(series: pd.Series, unknown_token: str = "<UNK>") -> Tuple[pd.Series, Dict[str, int]]:
    """
    Build a mapping from category value -> integer index, including an <UNK> token.

    - Converts to string dtype.
    - Replaces NaN with <UNK>.
    - Ensures <UNK> has a defined index (0).
    - Returns encoded series (int32) and mapping dict.
    """
    s = series.astype("string[python]")
    s = s.fillna(unknown_token)

    unique_vals = list(pd.Series(s.unique()))
    # Ensure unknown_token is at index 0
    unique_vals = [unknown_token] + [v for v in unique_vals if v != unknown_token]

    mapping = {val: idx for idx, val in enumerate(unique_vals)}
    encoded = s.map(mapping).astype("int32")

    return encoded, mapping


def encode_categorical_columns(df: pd.DataFrame, cat_cols: List[str]) -> Tuple[pd.DataFrame, Dict[str, Dict[str, int]]]:
    """
    Encode all categorical columns into integer index columns with suffix '_idx'.

    Returns:
        - updated dataframe
        - dict: {col_name: mapping_dict}
    """
    df = df.copy()
    mappings: Dict[str, Dict[str, int]] = {}

    for col in cat_cols:
        if col not in df.columns:
            raise KeyError(f"Categorical column '{col}' not found in dataframe.")

        print(f"[INFO] Encoding categorical column '{col}'...")
        encoded, mapping = build_category_mapping(df[col])
        idx_col = f"{col}_idx"
        df[idx_col] = encoded
        mappings[col] = mapping

    return df, mappings


def add_boolean_ints(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add integer versions of boolean flags:
        - is_weekend_int
        - is_roundtrip_int
    """
    df = df.copy()

    df["is_weekend_int"] = df["is_weekend"].astype("int8")
    df["is_roundtrip_int"] = df["is_roundtrip"].astype("int8")

    return df


def time_based_split(df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Time-based train/val/test split using started_at_parsed:

        train: started_at_parsed <  train_end
        val:   train_end <= started_at_parsed < val_end
        test:  started_at_parsed >= val_end
    """
    df = df.copy()
    train_end = pd.to_datetime(SPLIT_BOUNDARIES["train_end"])
    val_end = pd.to_datetime(SPLIT_BOUNDARIES["val_end"])

    ts = df["started_at_parsed"]

    train_mask = ts < train_end
    val_mask = (ts >= train_end) & (ts < val_end)
    test_mask = ts >= val_end

    train_df = df[train_mask].copy()
    val_df = df[val_mask].copy()
    test_df = df[test_mask].copy()

    print("[INFO] Time-based split sizes:")
    print(f"  Train: {len(train_df):,}")
    print(f"  Val  : {len(val_df):,}")
    print(f"  Test : {len(test_df):,}")

    if len(train_df) == 0 or len(val_df) == 0 or len(test_df) == 0:
        raise RuntimeError("One of the splits is empty. Adjust SPLIT_BOUNDARIES or check data coverage.")

    return train_df, val_df, test_df

def impute_missing_numeric_and_targets(
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    test_df: pd.DataFrame,
    numeric_cols: List[str],
    target_numeric_cols: List[str],
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Impute missing values in numeric feature columns and numeric targets
    using train-split means.

    - numeric_cols: features that will be normalized
    - target_numeric_cols: numeric targets like duration_log1p, demand_share

    Imputation is done BEFORE normalization so that scaling works correctly.
    """
    train_df = train_df.copy()
    val_df = val_df.copy()
    test_df = test_df.copy()

    # Compute means on train only
    means: Dict[str, float] = {}
    for col in numeric_cols + target_numeric_cols:
        if col not in train_df.columns:
            raise KeyError(f"Column '{col}' not found in train_df during imputation.")
        means[col] = float(train_df[col].mean())
        # If column is entirely NaN for some reason, default to 0
        if np.isnan(means[col]):
            means[col] = 0.0
        print(f"[INFO] Imputation mean for '{col}': {means[col]:.6f}")

    # Apply to all splits
    for df, split_name in [
        (train_df, "train"),
        (val_df, "val"),
        (test_df, "test"),
    ]:
        for col in numeric_cols + target_numeric_cols:
            before = df[col].isna().sum()
            if before > 0:
                print(f"[INFO] Imputing {before:,} NaNs in '{col}' for {split_name} split.")
                df[col] = df[col].fillna(means[col])

    return train_df, val_df, test_df


def compute_numeric_scalers(train_df: pd.DataFrame, numeric_cols: List[str]) -> Dict[str, Dict[str, float]]:
    """
    Compute mean and std for each numeric column using the train split.

    Returns:
        - dict: {col_name: {"mean": float, "std": float}}
    """
    scalers: Dict[str, Dict[str, float]] = {}
    for col in numeric_cols:
        if col not in train_df.columns:
            raise KeyError(f"Numeric column '{col}' not found in train dataframe.")

        vals = train_df[col].astype("float64")
        mean = float(vals.mean())
        std = float(vals.std())

        if std == 0.0 or np.isnan(std):
            std = 1.0

        scalers[col] = {"mean": mean, "std": std}
        print(f"[INFO] Numeric scaler for '{col}': mean={mean:.4f}, std={std:.4f}")

    return scalers


def apply_numeric_scalers(
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    test_df: pd.DataFrame,
    scalers: Dict[str, Dict[str, float]],
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Add normalized versions of numeric columns to each split using train-based scalers.

    New columns have suffix '_norm'.
    """
    for col, stats in scalers.items():
        mean = stats["mean"]
        std = stats["std"]
        norm_col = f"{col}_norm"

        for split_df, split_name in [
            (train_df, "train"),
            (val_df, "val"),
            (test_df, "test"),
        ]:
            if col not in split_df.columns:
                raise KeyError(f"Column '{col}' missing from {split_name} split during scaling.")

            split_df[norm_col] = ((split_df[col].astype("float64") - mean) / std).astype("float32")

    return train_df, val_df, test_df


def save_splits(train_df: pd.DataFrame, val_df: pd.DataFrame, test_df: pd.DataFrame, output_dir: str) -> None:
    """Save train, val, test splits as parquet files."""
    os.makedirs(output_dir, exist_ok=True)

    train_path = os.path.join(output_dir, "train.parquet")
    val_path = os.path.join(output_dir, "val.parquet")
    test_path = os.path.join(output_dir, "test.parquet")

    train_df.to_parquet(train_path, index=False)
    val_df.to_parquet(val_path, index=False)
    test_df.to_parquet(test_path, index=False)

    print(f"[INFO] Saved train split to: {train_path}")
    print(f"[INFO] Saved val split   to: {val_path}")
    print(f"[INFO] Saved test split  to: {test_path}")


def save_artifacts(
    artifacts: Dict,
    output_dir: str,
    filename: str = "phase3_artifacts.pkl",
) -> None:
    """Save mappings, scalers, and feature config as a pickle."""
    os.makedirs(output_dir, exist_ok=True)
    path = os.path.join(output_dir, filename)
    with open(path, "wb") as f:
        pickle.dump(artifacts, f)
    print(f"[INFO] Saved artifacts to: {path}")


def summarize_phase3(
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    test_df: pd.DataFrame,
    feature_config: Dict,
) -> None:
    """Print a concise summary of splits and feature dimensions."""
    print("\n" + "=" * 80)
    print("PHASE 3 SUMMARY: MODEL-READY DATASETS")
    print("=" * 80)

    print("\n[1] SPLIT SIZES")
    print(f"  Train: {len(train_df):,}")
    print(f"  Val  : {len(val_df):,}")
    print(f"  Test : {len(test_df):,}")

    print("\n[2] FEATURE CONFIG")
    print("  Categorical index features:", feature_config["categorical_feature_cols"])
    print("  Numeric features (normalized):", feature_config["numeric_feature_cols"])
    print("  Targets:", feature_config["target_cols"])

    print("\n[3] SAMPLE ROW (train)")
    sample_cols = (
        feature_config["categorical_feature_cols"]
        + feature_config["numeric_feature_cols"]
        + feature_config["target_cols"]
    )
    sample_cols = [c for c in sample_cols if c in train_df.columns]
    print(train_df[sample_cols].head(5))

    print("\n" + "=" * 80)
    print("End of Phase 3 summary. Splits and artifacts are ready for the HVAE training.")
    print("=" * 80 + "\n")


# -------------------------------------------------------------------
# CLI
# -------------------------------------------------------------------



def main() -> None:

    input_path = "data/processed/full_bike_dataset_phase2.parquet"
    print(f"[INFO] Loading Phase 2 dataset from: {input_path}")
    df = load_dataset(input_path)

    print("[INFO] Validating Phase 2 schema...")
    validate_phase2_df(df)

    print("[INFO] Computing demand contribution target (start_station_day_share)...")
    df = add_demand_contribution(df)

    print("[INFO] Encoding categorical columns...")
    df, category_mappings = encode_categorical_columns(df, CATEGORICAL_COLUMNS)

    print("[INFO] Adding integer versions of boolean flags...")
    df = add_boolean_ints(df)

    print("[INFO] Performing time-based train/val/test split...")
    train_df, val_df, test_df = time_based_split(df)

    target_numeric_cols = [
        "trip_duration_min",        # not used directly in loss but safe to clean
        "trip_duration_min_log1p",  # used as duration target
        "start_station_day_share",  # used as demand target
    ]

    print("[INFO] Imputing missing numeric features and targets using train means...")
    train_df, val_df, test_df = impute_missing_numeric_and_targets(
        train_df,
        val_df,
        test_df,
        numeric_cols=NUMERIC_FOR_SCALING,
        target_numeric_cols=target_numeric_cols,
    )

    for split_df, split_name in [
        (train_df, "train"),
        (val_df, "val"),
        (test_df, "test"),
    ]:
        before_clip_min = (split_df["start_station_day_share"] < 0).sum()
        before_clip_max = (split_df["start_station_day_share"] > 1).sum()
        if before_clip_min or before_clip_max:
            print(
                f"[INFO] Clipping start_station_day_share in {split_name} split: "
                f"<0 count={before_clip_min}, >1 count={before_clip_max}"
            )
        split_df["start_station_day_share"] = split_df["start_station_day_share"].clip(0.0, 1.0)

    print("[INFO] Computing numeric scalers from train split...")
    numeric_scalers = compute_numeric_scalers(train_df, NUMERIC_FOR_SCALING)

    print("[INFO] Applying numeric scalers to all splits...")
    train_df, val_df, test_df = apply_numeric_scalers(
        train_df, val_df, test_df, numeric_scalers
    )

    # Define feature sets for HVAE training
    categorical_feature_cols = [
        "start_station_id_idx",
        "end_station_id_idx",
        "rideable_type_idx",
        "member_casual_idx",
    ]

    numeric_feature_cols = [
        "trip_duration_min_norm",
        "trip_duration_min_log1p_norm",
        "start_lat_norm",
        "start_lng_norm",
        "end_lat_norm",
        "end_lng_norm",
        "hour",               # unnormalized; treated as integer feature
        "weekday",            # unnormalized; treated as integer feature
        "is_weekend_int",
        "is_roundtrip_int",
    ]

    target_cols = [
        "trip_duration_min",            # raw duration
        "trip_duration_min_log1p",      # log-transformed duration
        "start_station_day_share",      # demand contribution
        "rideable_type_idx",            # for bike-type prediction
    ]

    feature_config = {
        "categorical_feature_cols": categorical_feature_cols,
        "numeric_feature_cols": numeric_feature_cols,
        "target_cols": target_cols,
    }

    artifacts = {
        "category_mappings": category_mappings,
        "numeric_scalers": numeric_scalers,
        "feature_config": feature_config,
        "split_boundaries": SPLIT_BOUNDARIES,
    }

    print("[INFO] Saving train/val/test splits...")
    output_dir = "data/model_ready"
    save_splits(train_df, val_df, test_df, output_dir)

    print("[INFO] Saving Phase 3 artifacts...")
    save_artifacts(artifacts, output_dir)

    print("[INFO] Running Phase 3 summary...")
    summarize_phase3(train_df, val_df, test_df, feature_config)


if __name__ == "__main__":
    main()
