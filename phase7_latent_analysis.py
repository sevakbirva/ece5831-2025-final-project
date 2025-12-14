"""
Phase 7: Latent Intent & Anomaly Pattern Analysis

This script:
- Loads Phase 3 artifacts and the trained HVAE (best checkpoint)
- Loads anomaly_scores_<split>.parquet (with anomaly_score + anomaly_rank)
- Extracts latent vectors (z_global, z_individual) for each trip
- Runs k-means clustering on z_global to get "intent clusters"
- Joins everything into a single parquet file
- Prints cluster-level summaries (trip counts, mean duration, ride mix, anomaly stats)

Usage example:

    python phase7_latent_analysis.py \
        --data-dir data/model_ready \
        --anomaly-path data/anomaly_scores/anomaly_scores_test.parquet \
        --checkpoint-path checkpoints/hvae_v2/best_model.pt \
        --split test \
        --output-dir data/latent_analysis \
        --num-clusters 8 \
        --batch-size 4096 \
        --device cuda
"""

import argparse
import os
import pickle
from typing import Dict, Any, List, Tuple

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from sklearn.cluster import KMeans

from phase4_hvae_model import HVAEConfig, HierarchicalVAE


# -------------------------------------------------------------------
# Dataset for latent extraction
# -------------------------------------------------------------------

class BikeHVAELatentDataset(Dataset):
    """
    Dataset used to feed (x_cat, x_num) to HVAE for latent extraction.

    Expects the parquet to contain:
      - Categorical inputs:
          start_station_id_idx
          end_station_id_idx
          member_casual_idx

      - Numeric inputs:
          feature_config["numeric_feature_cols"]
    """

    def __init__(self, parquet_path: str, feature_config: Dict[str, Any]):
        super().__init__()
        self.parquet_path = parquet_path
        self.feature_config = feature_config

        df = pd.read_parquet(parquet_path)

        self.cat_cols: List[str] = [
            "start_station_id_idx",
            "end_station_id_idx",
            "member_casual_idx",
        ]
        self.num_cols: List[str] = feature_config["numeric_feature_cols"]

        missing_cat = [c for c in self.cat_cols if c not in df.columns]
        missing_num = [c for c in self.num_cols if c not in df.columns]
        if missing_cat or missing_num:
            raise ValueError(
                f"Missing columns in {parquet_path}.\n"
                f"  Missing categorical: {missing_cat}\n"
                f"  Missing numeric: {missing_num}"
            )

        self.x_cat = df[self.cat_cols].to_numpy(dtype=np.int64)
        self.x_num = df[self.num_cols].to_numpy(dtype=np.float32)
        self.n_samples = self.x_cat.shape[0]

    def __len__(self) -> int:
        return self.n_samples

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        return {
            "x_cat": torch.from_numpy(self.x_cat[idx]),
            "x_num": torch.from_numpy(self.x_num[idx]),
        }


# -------------------------------------------------------------------
# Utilities: artifacts + model
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

    # Match Phase 5 (v2) config
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
# Latent extraction
# -------------------------------------------------------------------

def extract_latents(
    model: HierarchicalVAE,
    dataset: BikeHVAELatentDataset,
    batch_size: int,
    device: torch.device,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Extract mean latents (z_global, z_individual) for all trips.

    Returns:
        z_global:      [N, latent_dim_global]
        z_individual:  [N, latent_dim_individual]
    """
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=4,
        pin_memory=True if device.type == "cuda" else False,
    )

    z_g_list: List[np.ndarray] = []
    z_i_list: List[np.ndarray] = []

    print(f"[INFO] Extracting latents for {len(dataset):,} samples...")
    with torch.no_grad():
        for i, batch in enumerate(loader):
            x_cat = batch["x_cat"].to(device)
            x_num = batch["x_num"].to(device)

            # Forward pass
            outputs, latents = model.forward(x_cat, x_num)
            # Use mean of q(z|x) for deterministic embedding
            z_global = latents["mu_global"]        # [B, latent_dim_global]
            z_ind = latents["mu_individual"]       # [B, latent_dim_individual]

            z_g_list.append(z_global.cpu().numpy())
            z_i_list.append(z_ind.cpu().numpy())

            if (i + 1) % 50 == 0:
                print(f"  [INFO] Processed { (i + 1) * batch_size:,} samples...")

    z_global_all = np.concatenate(z_g_list, axis=0)
    z_ind_all = np.concatenate(z_i_list, axis=0)

    assert z_global_all.shape[0] == len(dataset)
    assert z_ind_all.shape[0] == len(dataset)

    print(f"[INFO] Latent shapes: z_global={z_global_all.shape}, z_individual={z_ind_all.shape}")
    return z_global_all, z_ind_all


# -------------------------------------------------------------------
# Argument parsing and main
# -------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Phase 7: Latent intent and anomaly pattern analysis with HVAE."
    )
    parser.add_argument(
        "--data-dir",
        type=str,
        default="data/model_ready",
        help="Directory with phase3_artifacts.pkl",
    )
    parser.add_argument(
        "--anomaly-path",
        type=str,
        default="data/anomaly_scores/anomaly_scores_test.parquet",
        help="Path to anomaly_scores_<split>.parquet.",
    )
    parser.add_argument(
        "--checkpoint-path",
        type=str,
        default="checkpoints/hvae_v2/best_model.pt",
        help="Path to trained HVAE checkpoint.",
    )
    parser.add_argument(
        "--split",
        type=str,
        default="test",
        help="Split label (for naming only).",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="data/latent_analysis",
        help="Directory to save enriched parquet with latents and clusters.",
    )
    parser.add_argument(
        "--num-clusters",
        type=int,
        default=8,
        help="Number of k-means clusters in z_global space.",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=4096,
        help="Batch size for latent extraction.",
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
    if not os.path.exists(artifacts_path):
        raise FileNotFoundError(f"Artifacts file not found: {artifacts_path}")
    if not os.path.exists(args.anomaly_path):
        raise FileNotFoundError(f"Anomaly parquet not found: {args.anomaly_path}")

    # Load artifacts and model
    artifacts = load_artifacts(artifacts_path)
    feature_config = artifacts["feature_config"]
    model = build_model_from_artifacts(artifacts, args.checkpoint_path, device)

    # Dataset for latent extraction
    print(f"[INFO] Building latent dataset from: {args.anomaly_path}")
    latent_dataset = BikeHVAELatentDataset(args.anomaly_path, feature_config)

    # Extract latents
    z_global_all, z_ind_all = extract_latents(model, latent_dataset, args.batch_size, device)

    # Load anomaly dataframe and attach latents
    df = pd.read_parquet(args.anomaly_path)

    # Add latent columns
    z_g_dim = z_global_all.shape[1]
    z_i_dim = z_ind_all.shape[1]

    for d in range(z_g_dim):
        df[f"z_global_{d}"] = z_global_all[:, d]
    for d in range(z_i_dim):
        df[f"z_ind_{d}"] = z_ind_all[:, d]

    # ------------------------------------------------------------------
    # K-means clustering in global latent space
    # ------------------------------------------------------------------
    print(f"[INFO] Running k-means with K={args.num_clusters} on z_global...")
    kmeans = KMeans(
        n_clusters=args.num_clusters,
        random_state=42,
        n_init=10,
    )
    cluster_labels = kmeans.fit_predict(z_global_all)
    df["intent_cluster"] = cluster_labels

    # ------------------------------------------------------------------
    # Cluster-level summaries
    # ------------------------------------------------------------------
    print("\n" + "=" * 80)
    print(f"CLUSTER SUMMARY (split={args.split}, K={args.num_clusters})")
    print("=" * 80)

    # Ensure these columns exist before summarizing
    cols_available = df.columns

    # Basic count per cluster
    cluster_counts = df["intent_cluster"].value_counts().sort_index()
    print("\n[Cluster sizes]")
    print(cluster_counts)

    # Mean duration & anomaly score per cluster
    if "trip_duration_min" in cols_available and "anomaly_score" in cols_available:
        summary = df.groupby("intent_cluster").agg(
            n_trips=("ride_id", "count"),
            mean_duration=("trip_duration_min", "mean"),
            median_duration=("trip_duration_min", "median"),
            mean_anom=("anomaly_score", "mean"),
            p95_anom=("anomaly_score", lambda x: x.quantile(0.95)),
        )
        print("\n[Cluster: duration & anomaly stats]")
        print(summary)

    # Rideable_type distribution per cluster (if available)
    if "rideable_type" in cols_available:
        print("\n[Cluster: rideable_type distribution (% within cluster)]")
        ride_tab = pd.crosstab(df["intent_cluster"], df["rideable_type"], normalize="index") * 100
        print(ride_tab.round(2))

    # Member vs casual per cluster (if available)
    if "member_casual" in cols_available:
        print("\n[Cluster: member_casual distribution (% within cluster)]")
        rider_tab = pd.crosstab(df["intent_cluster"], df["member_casual"], normalize="index") * 100
        print(rider_tab.round(2))

    # ------------------------------------------------------------------
    # Save enriched dataframe
    # ------------------------------------------------------------------
    os.makedirs(args.output_dir, exist_ok=True)
    out_path = os.path.join(args.output_dir, f"latent_analysis_{args.split}.parquet")
    df.to_parquet(out_path, index=False)
    print(f"\n[INFO] Saved latent-enriched {args.split} split to: {out_path}")


if __name__ == "__main__":
    main()
