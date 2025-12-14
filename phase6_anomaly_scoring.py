"""
Phase 6: Anomaly Scoring with Hierarchical VAE

This script:
- Loads Phase 3 model-ready data and artifacts
- Loads the trained HVAE model (Phase 5, best checkpoint)
- Computes anomaly scores for a chosen split (train/val/test)
  using reconstruction-based error (no anomaly labels)
- Saves a new parquet file with anomaly scores and ranks

Usage example:

    python phase6_anomaly_scoring.py \
        --data-dir data/model_ready \
        --checkpoint-path checkpoints/hvae_v2/best_model.pt \
        --split test \
        --output-dir data/anomaly_scores \
        --batch-size 4096 \
        --device cuda
"""

import argparse
import os
import pickle
from typing import Dict, Any, List

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader

from phase4_hvae_model import HVAEConfig, HierarchicalVAE


# -------------------------------------------------------------------
# Dataset for anomaly scoring
# -------------------------------------------------------------------

class BikeHVAEAnomalyDataset(Dataset):
    """
    Dataset for running HVAE anomaly scoring on a single split parquet file.

    Expects the parquet file to contain at least:
      - Categorical index feature inputs:
          start_station_id_idx
          end_station_id_idx
          member_casual_idx

      - Numeric features (from feature_config["numeric_feature_cols"]):
          e.g., trip_duration_min_norm, trip_duration_min_log1p_norm,
                start_lat_norm, start_lng_norm, end_lat_norm, end_lng_norm,
                hour, weekday, is_weekend_int, is_roundtrip_int

      - Targets needed for reconstruction error:
          trip_duration_min_log1p
          start_station_day_share
          rideable_type_idx
    """

    def __init__(self, parquet_path: str, feature_config: Dict[str, Any]):
        super().__init__()
        self.parquet_path = parquet_path
        self.feature_config = feature_config

        df = pd.read_parquet(parquet_path)

        # Categorical columns used as inputs (NO rideable_type_idx to avoid leakage)
        self.cat_cols: List[str] = [
            "start_station_id_idx",
            "end_station_id_idx",
            "member_casual_idx",
        ]

        # Numeric input columns (normalized features)
        self.num_cols: List[str] = feature_config["numeric_feature_cols"]

        # Targets used for reconstruction / anomaly score
        self.target_cols = {
            "duration_log1p": "trip_duration_min_log1p",
            "demand_share": "start_station_day_share",
            "rideable_type": "rideable_type_idx",
        }

        # Sanity checks
        missing_cat = [c for c in self.cat_cols if c not in df.columns]
        missing_num = [c for c in self.num_cols if c not in df.columns]
        missing_tgt = [c for c in self.target_cols.values() if c not in df.columns]
        if missing_cat or missing_num or missing_tgt:
            raise ValueError(
                f"Missing columns in {parquet_path}.\n"
                f"  Missing categorical: {missing_cat}\n"
                f"  Missing numeric: {missing_num}\n"
                f"  Missing targets: {missing_tgt}"
            )

        # Store arrays for fast access
        self.x_cat = df[self.cat_cols].to_numpy(dtype=np.int64)
        self.x_num = df[self.num_cols].to_numpy(dtype=np.float32)

        self.y_duration_log1p = df[self.target_cols["duration_log1p"]].to_numpy(dtype=np.float32)
        self.y_demand_share = df[self.target_cols["demand_share"]].to_numpy(dtype=np.float32)
        self.y_rideable_type = df[self.target_cols["rideable_type"]].to_numpy(dtype=np.int64)

        self.n_samples = self.x_cat.shape[0]

    def __len__(self) -> int:
        return self.n_samples

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        x_cat = torch.from_numpy(self.x_cat[idx])          # LongTensor [3]
        x_num = torch.from_numpy(self.x_num[idx])          # FloatTensor [D_num]

        targets = {
            "duration_log1p": torch.tensor(self.y_duration_log1p[idx], dtype=torch.float32),
            "demand_share": torch.tensor(self.y_demand_share[idx], dtype=torch.float32),
            "rideable_type": torch.tensor(self.y_rideable_type[idx], dtype=torch.long),
        }

        return {
            "x_cat": x_cat,
            "x_num": x_num,
            "targets": targets,
        }


# -------------------------------------------------------------------
# Model loading
# -------------------------------------------------------------------

def load_artifacts(artifacts_path: str) -> Dict[str, Any]:
    print(f"[INFO] Loading Phase 3 artifacts from: {artifacts_path}")
    with open(artifacts_path, "rb") as f:
        artifacts = pickle.load(f)
    return artifacts


def build_model_from_artifacts(
    artifacts: Dict[str, Any],
    checkpoint_path: str,
    device: torch.device,
) -> HierarchicalVAE:
    """
    Re-create the HVAE model with the same architecture as in Phase 5
    and load trained weights from checkpoint.
    """
    category_mappings = artifacts["category_mappings"]
    feature_config = artifacts["feature_config"]

    num_start_stations = len(category_mappings["start_station_id"])
    num_end_stations = len(category_mappings["end_station_id"])
    num_ride_types = len(category_mappings["rideable_type"])
    num_member_types = len(category_mappings["member_casual"])
    num_numeric_features = len(feature_config["numeric_feature_cols"])

    print("[INFO] Category sizes for model reconstruction:")
    print(f"  num_start_stations: {num_start_stations}")
    print(f"  num_end_stations  : {num_end_stations}")
    print(f"  num_ride_types    : {num_ride_types}")
    print(f"  num_member_types  : {num_member_types}")
    print(f"  num_numeric_features: {num_numeric_features}")

    # These must match the config used in your best training run (Phase 5, v2)
    config = HVAEConfig(
        num_start_stations=num_start_stations,
        num_end_stations=num_end_stations,
        num_ride_types=num_ride_types,
        num_member_types=num_member_types,
        num_numeric_features=num_numeric_features,
        emb_dim_station=32,
        emb_dim_ride_type=8,
        emb_dim_member=4,
        latent_dim_global=16,
        latent_dim_individual=16,
        encoder_hidden_dim=256,
        decoder_hidden_dim=256,
        w_duration=0.3,
        w_demand=0.0,
        w_ride_type=3.0,
        beta_global=0.1,
        beta_individual=0.1,
    )

    model = HierarchicalVAE(config).to(device)
    print(f"[INFO] Loading model checkpoint from: {checkpoint_path}")
    ckpt = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()

    return model


# -------------------------------------------------------------------
# Anomaly scoring loop
# -------------------------------------------------------------------

def compute_anomaly_scores(
    model: HierarchicalVAE,
    dataset: BikeHVAEAnomalyDataset,
    batch_size: int,
    device: torch.device,
) -> np.ndarray:
    """
    Run model.compute_anomaly_score over the entire dataset.

    Returns:
        scores: numpy array of shape [N_samples]
    """
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=4,
        pin_memory=True if device.type == "cuda" else False,
    )

    scores_list: List[np.ndarray] = []

    print(f"[INFO] Computing anomaly scores on {len(dataset):,} samples...")
    with torch.no_grad():
        for i, batch in enumerate(loader):
            x_cat = batch["x_cat"].to(device)
            x_num = batch["x_num"].to(device)
            targets = {k: v.to(device) for k, v in batch["targets"].items()}

            batch_scores = model.compute_anomaly_score(
                {"x_cat": x_cat, "x_num": x_num, "targets": targets}
            )  # FloatTensor [B]

            scores_list.append(batch_scores.cpu().numpy())

            if (i + 1) % 50 == 0:
                print(f"  [INFO] Processed { (i + 1) * batch_size:,} samples...")

    scores = np.concatenate(scores_list, axis=0)
    assert scores.shape[0] == len(dataset), "Mismatch between scores and dataset length."

    return scores


# -------------------------------------------------------------------
# Argument parsing and main
# -------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Phase 6: Compute anomaly scores with HVAE on Cyclistic data."
    )
    parser.add_argument(
        "--data-dir",
        type=str,
        default="data/model_ready",
        help="Directory with train/val/test parquet and phase3_artifacts.pkl",
    )
    parser.add_argument(
        "--checkpoint-path",
        type=str,
        default="checkpoints/hvae_v2/best_model.pt",
        help="Path to the trained HVAE checkpoint (.pt).",
    )
    parser.add_argument(
        "--split",
        type=str,
        default="test",
        choices=["train", "val", "test"],
        help="Which split to run anomaly scoring on.",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="data/anomaly_scores",
        help="Directory to save parquet with anomaly scores.",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=4096,
        help="Batch size for anomaly scoring.",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda",
        help="Device to use: 'cuda' or 'cpu'.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    print(f"[INFO] Using device: {device}")

    artifacts_path = os.path.join(args.data_dir, "phase3_artifacts.pkl")
    split_path = os.path.join(args.data_dir, f"{args.split}.parquet")

    if not os.path.exists(split_path):
        raise FileNotFoundError(f"Split file not found: {split_path}")

    # Load artifacts and model
    artifacts = load_artifacts(artifacts_path)
    feature_config = artifacts["feature_config"]
    model = build_model_from_artifacts(artifacts, args.checkpoint_path, device)

    # Build dataset
    print(f"[INFO] Building anomaly dataset for split='{args.split}'...")
    dataset = BikeHVAEAnomalyDataset(split_path, feature_config)

    # Compute anomaly scores
    scores = compute_anomaly_scores(model, dataset, args.batch_size, device)

    # Load original dataframe and attach scores
    df = pd.read_parquet(split_path)
    df["anomaly_score"] = scores

    # Higher score = more anomalous; create rank (1=most anomalous)
    df["anomaly_rank"] = df["anomaly_score"].rank(method="dense", ascending=False).astype(int)

    # Basic summary
    print("\n" + "=" * 80)
    print(f"ANOMALY SCORE SUMMARY ({args.split} split)")
    print("=" * 80)

    desc = df["anomaly_score"].describe(percentiles=[0.5, 0.9, 0.95, 0.99])
    print(desc)

    # Show a few most anomalous trips (top 10)
    print("\n[TOP 10 MOST ANOMALOUS TRIPS]")
    cols_to_show = [
        "ride_id",
        "started_at_parsed",
        "start_station_id",
        "end_station_id",
        "member_casual",
        "rideable_type",
        "trip_duration_min",
        "start_station_day_share",
        "anomaly_score",
        "anomaly_rank",
    ]
    cols_to_show = [c for c in cols_to_show if c in df.columns]
    top_10 = df.sort_values("anomaly_score", ascending=False).head(10)
    print(top_10[cols_to_show])

    # Save to parquet
    os.makedirs(args.output_dir, exist_ok=True)
    out_path = os.path.join(args.output_dir, f"anomaly_scores_{args.split}.parquet")
    df.to_parquet(out_path, index=False)
    print(f"\n[INFO] Saved anomaly-scored {args.split} split to: {out_path}")


if __name__ == "__main__":
    main()
