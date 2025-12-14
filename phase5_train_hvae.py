"""
Phase 5: Training Script for Hierarchical VAE (HVAE)

This script:
- Loads Phase 3 model-ready data and artifacts
- Builds PyTorch Dataset / DataLoader for train/val/test
- Instantiates the HierarchicalVAE model (Phase 4)
- Trains with validation monitoring, early stopping, and checkpointing

Usage:

    python phase5_train_hvae.py \
        --data-dir data/model_ready \
        --checkpoint-dir checkpoints/hvae_v1 \
        --epochs 20 \
        --batch-size 2048 \
        --lr 1e-3
"""

import argparse
import os
import pickle
from typing import Dict, Any

import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader

# Import model + config from Phase 4
from phase4_hvae_model import HVAEConfig, HierarchicalVAE


# -------------------------------------------------------------------
# Dataset
# -------------------------------------------------------------------

class BikeHVAEDataset(Dataset):
    """
    PyTorch Dataset for HVAE using Phase 3 model-ready parquet files.

    Expects the parquet file to contain at least:
      - Categorical index features:
          start_station_id_idx
          end_station_id_idx
          rideable_type_idx
          member_casual_idx

      - Numeric features (as per feature_config["numeric_feature_cols"]):
          e.g., trip_duration_min_norm, trip_duration_min_log1p_norm,
                start_lat_norm, start_lng_norm, end_lat_norm, end_lng_norm,
                hour, weekday, is_weekend_int, is_roundtrip_int

      - Targets:
          trip_duration_min_log1p
          start_station_day_share
          rideable_type_idx
    """

    def __init__(
        self,
        parquet_path: str,
        feature_config: Dict[str, Any],
    ):
        super().__init__()
        self.parquet_path = parquet_path
        self.feature_config = feature_config

        # Load dataframe
        df = pd.read_parquet(parquet_path)

        # Categorical feature columns (indices)
        self.cat_cols = [
            "start_station_id_idx",
            "end_station_id_idx",
            "member_casual_idx",
        ]
        # Numeric feature columns
        self.num_cols = feature_config["numeric_feature_cols"]

        # Targets: we only need these 3 for the HVAE loss
        # trip_duration_min_log1p, start_station_day_share, rideable_type_idx
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

        # Store as numpy arrays for fast indexing
        self.x_cat = df[self.cat_cols].to_numpy(dtype=np.int64)
        self.x_num = df[self.num_cols].to_numpy(dtype=np.float32)

        self.y_duration_log1p = df[self.target_cols["duration_log1p"]].to_numpy(dtype=np.float32)
        self.y_demand_share = df[self.target_cols["demand_share"]].to_numpy(dtype=np.float32)
        self.y_rideable_type = df[self.target_cols["rideable_type"]].to_numpy(dtype=np.int64)

        self.n_samples = self.x_cat.shape[0]

    def __len__(self) -> int:
        return self.n_samples

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        x_cat = torch.from_numpy(self.x_cat[idx])          # LongTensor [4]
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
# Training / evaluation utilities
# -------------------------------------------------------------------

def train_one_epoch(
    model: HierarchicalVAE,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
) -> Dict[str, float]:
    model.train()
    running = {
        "loss": 0.0,
        "loss_duration": 0.0,
        "loss_demand": 0.0,
        "loss_ride_type": 0.0,
        "kl_global": 0.0,
        "kl_individual": 0.0,
        "correct_ride": 0,
        "n": 0,
    }

    for batch in loader:
        x_cat = batch["x_cat"].to(device)
        x_num = batch["x_num"].to(device)
        targets = {k: v.to(device) for k, v in batch["targets"].items()}

        optimizer.zero_grad()

        loss_dict = model.compute_loss(
            {"x_cat": x_cat, "x_num": x_num, "targets": targets}
        )
        loss = loss_dict["loss"]
        # Compute rideable type accuracy on this batch
        with torch.no_grad():
            outputs, _ = model.forward(x_cat, x_num)
            preds = outputs["ride_logits"].argmax(dim=-1)
            correct = (preds == targets["rideable_type"]).sum().item()
            running["correct_ride"] += correct
        loss.backward()

        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()

        bs = x_cat.size(0)
        running["n"] += bs
        for k in ["loss", "loss_duration", "loss_demand", "loss_ride_type", "kl_global", "kl_individual"]:
            running[k] += loss_dict[k].item() * bs

    # Average over all samples
    n = running["n"]
    for k in ["loss", "loss_duration", "loss_demand", "loss_ride_type", "kl_global", "kl_individual"]:
        running[k] /= max(1, n)

    # Add accuracy
    running["ride_acc"] = running["correct_ride"] / max(1, n)

    return running


def evaluate(
    model: HierarchicalVAE,
    loader: DataLoader,
    device: torch.device,
) -> Dict[str, float]:
    model.eval()
    running = {
        "loss": 0.0,
        "loss_duration": 0.0,
        "loss_demand": 0.0,
        "loss_ride_type": 0.0,
        "kl_global": 0.0,
        "kl_individual": 0.0,
        "correct_ride": 0,
        "n": 0,
    }

    with torch.no_grad():
        for batch in loader:
            x_cat = batch["x_cat"].to(device)
            x_num = batch["x_num"].to(device)
            targets = {k: v.to(device) for k, v in batch["targets"].items()}

            loss_dict = model.compute_loss(
                {"x_cat": x_cat, "x_num": x_num, "targets": targets}
            )
            outputs, _ = model.forward(x_cat, x_num)
            preds = outputs["ride_logits"].argmax(dim=-1)
            correct = (preds == targets["rideable_type"]).sum().item()
            running["correct_ride"] += correct
            bs = x_cat.size(0)
            running["n"] += bs
            for k in ["loss", "loss_duration", "loss_demand", "loss_ride_type", "kl_global", "kl_individual"]:
                running[k] += loss_dict[k].item() * bs

    n = running["n"]
    for k in ["loss", "loss_duration", "loss_demand", "loss_ride_type", "kl_global", "kl_individual"]:
        running[k] /= max(1, n)

    running["ride_acc"] = running["correct_ride"] / max(1, n)

    return running


def save_checkpoint(
    model: HierarchicalVAE,
    optimizer: torch.optim.Optimizer,
    epoch: int,
    val_metrics: Dict[str, float],
    checkpoint_dir: str,
    is_best: bool = False,
) -> None:
    os.makedirs(checkpoint_dir, exist_ok=True)
    ckpt = {
        "epoch": epoch,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "val_metrics": val_metrics,
    }
    path = os.path.join(checkpoint_dir, f"checkpoint_epoch_{epoch}.pt")
    torch.save(ckpt, path)
    print(f"[INFO] Saved checkpoint: {path}")

    if is_best:
        best_path = os.path.join(checkpoint_dir, "best_model.pt")
        torch.save(ckpt, best_path)
        print(f"[INFO] Updated best model: {best_path}")


# -------------------------------------------------------------------
# Argument parsing
# -------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Phase 5: Train Hierarchical VAE on Cyclistic data."
    )
    parser.add_argument(
        "--data-dir",
        type=str,
        default="data/model_ready",
        help="Directory containing train/val/test parquet and phase3_artifacts.pkl",
    )
    parser.add_argument(
        "--checkpoint-dir",
        type=str,
        default="checkpoints/hvae_v1",
        help="Directory to save checkpoints.",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=20,
        help="Number of training epochs.",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=2048,
        help="Mini-batch size.",
    )
    parser.add_argument(
        "--lr",
        type=float,
        default=5e-4,
        help="Learning rate.",
    )
    parser.add_argument(
        "--weight-decay",
        type=float,
        default=1e-4,
        help="Weight decay for optimizer.",
    )
    parser.add_argument(
        "--num-workers",
        type=int,
        default=4,
        help="Number of workers for DataLoader.",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda",
        help="Device to use: 'cuda' or 'cpu'.",
    )
    parser.add_argument(
        "--patience",
        type=int,
        default=5,
        help="Early stopping patience (epochs without improvement).",
    )
    return parser.parse_args()


# -------------------------------------------------------------------
# Main training routine
# -------------------------------------------------------------------

def main() -> None:
    args = parse_args()

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    print(f"[INFO] Using device: {device}")

    # ------------------------------------------------------------------
    # Load artifacts & splits
    # ------------------------------------------------------------------
    artifacts_path = os.path.join(args.data_dir, "phase3_artifacts.pkl")
    train_path = os.path.join(args.data_dir, "train.parquet")
    val_path = os.path.join(args.data_dir, "val.parquet")
    test_path = os.path.join(args.data_dir, "test.parquet")

    print(f"[INFO] Loading Phase 3 artifacts from: {artifacts_path}")
    with open(artifacts_path, "rb") as f:
        artifacts = pickle.load(f)

    category_mappings = artifacts["category_mappings"]
    feature_config = artifacts["feature_config"]

    num_start_stations = len(category_mappings["start_station_id"])
    num_end_stations = len(category_mappings["end_station_id"])
    num_ride_types = len(category_mappings["rideable_type"])
    num_member_types = len(category_mappings["member_casual"])

    num_numeric_features = len(feature_config["numeric_feature_cols"])

    print("[INFO] Category sizes:")
    print(f"  num_start_stations: {num_start_stations}")
    print(f"  num_end_stations  : {num_end_stations}")
    print(f"  num_ride_types    : {num_ride_types}")
    print(f"  num_member_types  : {num_member_types}")
    print(f"  num_numeric_features: {num_numeric_features}")

    # ------------------------------------------------------------------
    # Build Datasets and DataLoaders
    # ------------------------------------------------------------------
    print("[INFO] Building datasets...")
    train_dataset = BikeHVAEDataset(train_path, feature_config)
    val_dataset = BikeHVAEDataset(val_path, feature_config)
    test_dataset = BikeHVAEDataset(test_path, feature_config)

    print("[INFO] Building DataLoaders...")
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=True,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True,
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True,
    )

    # ------------------------------------------------------------------
    # Instantiate model
    # ------------------------------------------------------------------
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
        w_ride_type=4.0,
        w_duration=0.2,
        w_demand=2.0,
        beta_global=0.05,
        beta_individual=0.05
    )

    model = HierarchicalVAE(config).to(device)
    print(model)

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=args.lr,
        weight_decay=args.weight_decay,
    )

    # Optional: LR scheduler
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=2, verbose=True
    )

    # ------------------------------------------------------------------
    # Training loop with early stopping
    # ------------------------------------------------------------------
    best_val_loss = float("inf")
    epochs_without_improvement = 0

    for epoch in range(1, args.epochs + 1):
        print(f"\n===== Epoch {epoch}/{args.epochs} =====")

        train_metrics = train_one_epoch(model, train_loader, optimizer, device)
        val_metrics = evaluate(model, val_loader, device)

        print("[TRAIN] "
              f"loss={train_metrics['loss']:.4f}, "
              f"dur={train_metrics['loss_duration']:.4f}, "
              f"dem={train_metrics['loss_demand']:.4f}, "
              f"ride={train_metrics['loss_ride_type']:.4f}, "
              f"kl_g={train_metrics['kl_global']:.4f}, "
              f"kl_i={train_metrics['kl_individual']:.4f}, "
              f"ride_acc={train_metrics['ride_acc']*100:.2f}%")

        print("[VAL]   "
              f"loss={val_metrics['loss']:.4f}, "
              f"dur={val_metrics['loss_duration']:.4f}, "
              f"dem={val_metrics['loss_demand']:.4f}, "
              f"ride={val_metrics['loss_ride_type']:.4f}, "
              f"kl_g={val_metrics['kl_global']:.4f}, "
              f"kl_i={val_metrics['kl_individual']:.4f}, "
              f"ride_acc={val_metrics['ride_acc']*100:.2f}%")

        # Step scheduler on validation loss
        scheduler.step(val_metrics["loss"])

        # Check for improvement
        if val_metrics["loss"] < best_val_loss - 1e-4:
            best_val_loss = val_metrics["loss"]
            epochs_without_improvement = 0
            save_checkpoint(
                model,
                optimizer,
                epoch,
                val_metrics,
                checkpoint_dir=args.checkpoint_dir,
                is_best=True,
            )
        else:
            epochs_without_improvement += 1
            print(f"[INFO] No improvement for {epochs_without_improvement} epoch(s).")

        # Always save a regular checkpoint
        save_checkpoint(
            model,
            optimizer,
            epoch,
            val_metrics,
            checkpoint_dir=args.checkpoint_dir,
            is_best=False,
        )

        # Early stopping
        if epochs_without_improvement >= args.patience:
            print(f"[INFO] Early stopping triggered after {args.patience} epochs without improvement.")
            break

    # ------------------------------------------------------------------
    # Final evaluation on test set (using best model)
    # ------------------------------------------------------------------
    best_model_path = os.path.join(args.checkpoint_dir, "best_model.pt")
    if os.path.exists(best_model_path):
        print(f"[INFO] Loading best model from: {best_model_path}")
        ckpt = torch.load(best_model_path, map_location=device)
        model.load_state_dict(ckpt["model_state_dict"])
    else:
        print("[WARN] Best model checkpoint not found; using last epoch model.")

    test_metrics = evaluate(model, test_loader, device)
    print("\n===== Final Test Metrics =====")
    print("[TEST] "
          f"loss={test_metrics['loss']:.4f}, "
          f"dur={test_metrics['loss_duration']:.4f}, "
          f"dem={test_metrics['loss_demand']:.4f}, "
          f"ride={test_metrics['loss_ride_type']:.4f}, "
          f"kl_g={test_metrics['kl_global']:.4f}, "
          f"kl_i={test_metrics['kl_individual']:.4f}, "
          f"ride_acc={test_metrics['ride_acc']*100:.2f}%")



if __name__ == "__main__":
    main()
