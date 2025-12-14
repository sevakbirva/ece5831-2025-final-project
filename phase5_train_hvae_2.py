import os
import math
import pickle
import argparse
from dataclasses import asdict
from typing import Dict, Any, Optional, List, Tuple

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader

# Updated Phase 4 model (must match the new interface)
from phase4_hvae_model import HVAEConfig, HierarchicalVAE


# ----------------------------
# Reproducibility
# ----------------------------
def seed_everything(seed: int = 42) -> None:
    import random
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = False
    torch.backends.cudnn.benchmark = True


# ----------------------------
# Dataset (from DataFrames)
# ----------------------------
class BikeHVAEDatasetFromDF(Dataset):
    """
    Inputs (NO leakage):
      x_cat: [start_station_id_idx, member_casual_idx]
      x_num: numeric context features (filtered)
    Targets:
      duration_log1p: trip_duration_min_log1p
      destination: end_station_id_idx
      rideable_type: rideable_type_idx
    """

    def __init__(
        self,
        df: pd.DataFrame,
        numeric_cols: List[str],
        enforce_no_leakage: bool = True,
    ):
        super().__init__()
        self.df = df.reset_index(drop=True)

        # Inputs
        self.cat_cols = ["start_station_id_idx", "member_casual_idx"]

        # Targets
        self.target_cols = {
            "duration_log1p": "trip_duration_min_log1p",
            "destination": "end_station_id_idx",
            "rideable_type": "rideable_type_idx",
        }

        # Sanity checks
        missing_cat = [c for c in self.cat_cols if c not in self.df.columns]
        missing_tgt = [c for c in self.target_cols.values() if c not in self.df.columns]
        if missing_cat or missing_tgt:
            raise ValueError(
                "Missing required columns.\n"
                f"  Missing categorical: {missing_cat}\n"
                f"  Missing targets: {missing_tgt}\n"
            )

        # Optional leakage filtering
        numeric_cols = list(numeric_cols)  # copy
        removed = []

        if enforce_no_leakage:
            # Destination leakage
            for c in ["end_lat_norm", "end_lng_norm"]:
                if c in numeric_cols:
                    numeric_cols.remove(c)
                    removed.append(c)

            # Duration leakage: any duration-derived numeric inputs must be removed
            for c in ["trip_duration_min_norm", "trip_duration_min_log1p_norm"]:
                if c in numeric_cols:
                    numeric_cols.remove(c)
                    removed.append(c)

        # Ensure numeric cols exist
        missing_num = [c for c in numeric_cols if c not in self.df.columns]
        if missing_num:
            raise ValueError(
                "Missing numeric columns requested from artifacts.\n"
                f"  Missing numeric: {missing_num}\n"
            )

        self.num_cols = numeric_cols

        if enforce_no_leakage and removed:
            print(f"[INFO] Leakage guard removed numeric cols: {removed}")
        print(f"[INFO] Using {len(self.num_cols)} numeric cols for x_num.")

        # Store arrays
        self.x_cat = self.df[self.cat_cols].to_numpy(dtype=np.int64)
        self.x_num = self.df[self.num_cols].to_numpy(dtype=np.float32)

        self.y_dur = self.df[self.target_cols["duration_log1p"]].to_numpy(dtype=np.float32)
        self.y_dest = self.df[self.target_cols["destination"]].to_numpy(dtype=np.int64)
        self.y_ride = self.df[self.target_cols["rideable_type"]].to_numpy(dtype=np.int64)

        # Optional: if you want duration metrics in minutes
        self.has_duration_min = "trip_duration_min" in self.df.columns
        if self.has_duration_min:
            self.y_dur_min = self.df["trip_duration_min"].to_numpy(dtype=np.float32)
        else:
            self.y_dur_min = None

    def __len__(self) -> int:
        return self.x_cat.shape[0]

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        x_cat = torch.from_numpy(self.x_cat[idx])  # [2]
        x_num = torch.from_numpy(self.x_num[idx])  # [D]

        targets = {
            "duration_log1p": torch.tensor(self.y_dur[idx], dtype=torch.float32),
            "destination": torch.tensor(self.y_dest[idx], dtype=torch.long),
            "rideable_type": torch.tensor(self.y_ride[idx], dtype=torch.long),
        }

        item = {"x_cat": x_cat, "x_num": x_num, "targets": targets}

        if self.has_duration_min:
            item["duration_min"] = torch.tensor(self.y_dur_min[idx], dtype=torch.float32)

        return item


# ----------------------------
# Metrics helpers
# ----------------------------
@torch.no_grad()
def batch_duration_metrics_minutes(duration_mu_log1p: torch.Tensor, y_min: torch.Tensor) -> Tuple[float, float]:
    """
    Convert predicted log1p(duration) mean to minutes via expm1, then compute MAE/RMSE in minutes.
    """
    mu = torch.clamp(duration_mu_log1p, min=-5.0, max=8.0)
    pred_min = torch.expm1(mu).clamp(min=0.0)
    err = pred_min - y_min
    mae = err.abs().mean().item()
    rmse = torch.sqrt((err ** 2).mean()).item()
    return mae, rmse


# ----------------------------
# Train / Eval
# ----------------------------
def train_one_epoch(model, loader, optimizer, device) -> Dict[str, float]:
    model.train()
    running = {
        "loss": 0.0,
        "loss_duration": 0.0,
        "loss_destination": 0.0,
        "loss_ride_type": 0.0,
        "kl_global": 0.0,
        "kl_individual": 0.0,
        "correct_dest": 0,
        "correct_ride": 0,
        "dur_mae_min": 0.0,
        "dur_rmse_min": 0.0,
        "n_dur_min": 0,
        "n": 0,
    }

    for batch in loader:
        x_cat = batch["x_cat"].to(device, non_blocking=True)
        x_num = batch["x_num"].to(device, non_blocking=True)
        targets = {k: v.to(device, non_blocking=True) for k, v in batch["targets"].items()}

        optimizer.zero_grad(set_to_none=True)

        loss_dict = model.compute_loss({"x_cat": x_cat, "x_num": x_num, "targets": targets})
        loss = loss_dict["loss"]
        loss.backward()

        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()

        with torch.no_grad():
            outputs, _ = model.forward(x_cat, x_num)
            dest_preds = outputs["dest_logits"].argmax(dim=-1)
            ride_preds = outputs["ride_logits"].argmax(dim=-1)
            running["correct_dest"] += (dest_preds == targets["destination"]).sum().item()
            running["correct_ride"] += (ride_preds == targets["rideable_type"]).sum().item()

            # Optional duration metrics (minutes) if provided
            if "duration_min" in batch:
                y_min = batch["duration_min"].to(device, non_blocking=True)
                mae, rmse = batch_duration_metrics_minutes(outputs["duration_mu"], y_min)
                bs = x_cat.size(0)
                running["dur_mae_min"] += mae * bs
                running["dur_rmse_min"] += rmse * bs
                running["n_dur_min"] += bs

        bs = x_cat.size(0)
        running["n"] += bs
        for k in ["loss", "loss_duration", "loss_destination", "loss_ride_type", "kl_global", "kl_individual"]:
            running[k] += loss_dict[k].item() * bs

    n = max(1, running["n"])
    for k in ["loss", "loss_duration", "loss_destination", "loss_ride_type", "kl_global", "kl_individual"]:
        running[k] /= n

    running["dest_acc"] = running["correct_dest"] / n
    running["ride_acc"] = running["correct_ride"] / n

    if running["n_dur_min"] > 0:
        running["dur_mae_min"] /= running["n_dur_min"]
        running["dur_rmse_min"] /= running["n_dur_min"]
    else:
        running["dur_mae_min"] = float("nan")
        running["dur_rmse_min"] = float("nan")

    return running


@torch.no_grad()
def evaluate(model, loader, device) -> Dict[str, float]:
    model.eval()
    running = {
        "loss": 0.0,
        "loss_duration": 0.0,
        "loss_destination": 0.0,
        "loss_ride_type": 0.0,
        "kl_global": 0.0,
        "kl_individual": 0.0,
        "correct_dest": 0,
        "correct_ride": 0,
        "dur_mae_min": 0.0,
        "dur_rmse_min": 0.0,
        "n_dur_min": 0,
        "n": 0,
    }

    for batch in loader:
        x_cat = batch["x_cat"].to(device, non_blocking=True)
        x_num = batch["x_num"].to(device, non_blocking=True)
        targets = {k: v.to(device, non_blocking=True) for k, v in batch["targets"].items()}

        loss_dict = model.compute_loss({"x_cat": x_cat, "x_num": x_num, "targets": targets})
        outputs, _ = model.forward(x_cat, x_num)

        dest_preds = outputs["dest_logits"].argmax(dim=-1)
        ride_preds = outputs["ride_logits"].argmax(dim=-1)
        running["correct_dest"] += (dest_preds == targets["destination"]).sum().item()
        running["correct_ride"] += (ride_preds == targets["rideable_type"]).sum().item()

        if "duration_min" in batch:
            y_min = batch["duration_min"].to(device, non_blocking=True)
            mae, rmse = batch_duration_metrics_minutes(outputs["duration_mu"], y_min)
            bs = x_cat.size(0)
            running["dur_mae_min"] += mae * bs
            running["dur_rmse_min"] += rmse * bs
            running["n_dur_min"] += bs

        bs = x_cat.size(0)
        running["n"] += bs
        for k in ["loss", "loss_duration", "loss_destination", "loss_ride_type", "kl_global", "kl_individual"]:
            running[k] += loss_dict[k].item() * bs

    n = max(1, running["n"])
    for k in ["loss", "loss_duration", "loss_destination", "loss_ride_type", "kl_global", "kl_individual"]:
        running[k] /= n

    running["dest_acc"] = running["correct_dest"] / n
    running["ride_acc"] = running["correct_ride"] / n

    if running["n_dur_min"] > 0:
        running["dur_mae_min"] /= running["n_dur_min"]
        running["dur_rmse_min"] /= running["n_dur_min"]
    else:
        running["dur_mae_min"] = float("nan")
        running["dur_rmse_min"] = float("nan")

    return running


# ----------------------------
# Checkpointing
# ----------------------------
def save_checkpoint(
    checkpoint_dir: str,
    epoch: int,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    val_metrics: Dict[str, float],
    config: HVAEConfig,
    is_best: bool,
) -> None:
    os.makedirs(checkpoint_dir, exist_ok=True)
    ckpt = {
        "epoch": epoch,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "val_metrics": val_metrics,
        "config": asdict(config),
    }

    path = os.path.join(checkpoint_dir, f"checkpoint_epoch_{epoch}.pt")
    torch.save(ckpt, path)
    print(f"[INFO] Saved checkpoint: {path}")

    if is_best:
        best_path = os.path.join(checkpoint_dir, "best_model.pt")
        torch.save(ckpt, best_path)
        print(f"[INFO] Updated best model: {best_path}")


# ----------------------------
# Args
# ----------------------------
def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser("Phase 5: Train updated HVAE (duration + destination + ride type).")
    p.add_argument("--data-dir", type=str, default="data/model_ready")
    p.add_argument("--artifacts", type=str, default="phase3_artifacts.pkl")
    p.add_argument("--checkpoint-dir", type=str, default="checkpoints/hvae_v3")
    p.add_argument("--epochs", type=int, default=12)
    p.add_argument("--batch-size", type=int, default=4096)
    p.add_argument("--lr", type=float, default=5e-4)
    p.add_argument("--weight-decay", type=float, default=1e-4)
    p.add_argument("--num-workers", type=int, default=4)
    p.add_argument("--device", type=str, default="cuda")
    p.add_argument("--patience", type=int, default=4)
    p.add_argument("--seed", type=int, default=42)

    # leakage guard (recommended ON)
    p.add_argument("--no-leakage", action="store_true", help="Enforce leakage guard (recommended).")

    # model hyperparams / weights
    p.add_argument("--enc-hidden", type=int, default=256)
    p.add_argument("--dec-hidden", type=int, default=256)
    p.add_argument("--z-global", type=int, default=16)
    p.add_argument("--z-ind", type=int, default=16)

    p.add_argument("--w-duration", type=float, default=0.2)
    p.add_argument("--w-dest", type=float, default=1.0)
    p.add_argument("--w-ride", type=float, default=2.0)
    p.add_argument("--beta-g", type=float, default=0.05)
    p.add_argument("--beta-i", type=float, default=0.05)

    return p.parse_args()


# ----------------------------
# Main
# ----------------------------
def main() -> None:
    args = parse_args()
    seed_everything(args.seed)

    device = torch.device(args.device if (args.device == "cuda" and torch.cuda.is_available()) else "cpu")
    print(f"[INFO] Using device: {device}")

    # ----------------------------
    # Load artifacts
    # ----------------------------
    artifacts_path = os.path.join(args.data_dir, args.artifacts)
    print(f"[INFO] Loading Phase 3 artifacts from: {artifacts_path}")
    with open(artifacts_path, "rb") as f:
        artifacts = pickle.load(f)

    category_mappings = artifacts["category_mappings"]
    feature_config = artifacts["feature_config"]

    num_start_stations = len(category_mappings["start_station_id"])
    num_end_stations = len(category_mappings["end_station_id"])
    num_ride_types = len(category_mappings["rideable_type"])
    num_member_types = len(category_mappings["member_casual"])

    numeric_cols = feature_config["numeric_feature_cols"]
    print("[INFO] Category sizes:")
    print(f"  num_start_stations: {num_start_stations}")
    print(f"  num_end_stations  : {num_end_stations}")
    print(f"  num_ride_types    : {num_ride_types}")
    print(f"  num_member_types  : {num_member_types}")
    print(f"  numeric_feature_cols (raw): {len(numeric_cols)}")

    # ----------------------------
    # Load your existing splits (as requested)
    # ----------------------------
    train_df = pd.read_parquet(os.path.join(args.data_dir, "train.parquet"))
    val_df   = pd.read_parquet(os.path.join(args.data_dir, "val.parquet"))
    test_df  = pd.read_parquet(os.path.join(args.data_dir, "test.parquet"))

    print(f"[INFO] Train n={len(train_df):,}  Val n={len(val_df):,}  Test n={len(test_df):,}")

    # ----------------------------
    # Build datasets
    # ----------------------------
    enforce_no_leakage = args.no_leakage  # recommended True
    if enforce_no_leakage:
        print("[INFO] Leakage guard: ON (recommended).")
    else:
        print("[WARN] Leakage guard: OFF. You may be leaking targets via x_num.")

    train_ds = BikeHVAEDatasetFromDF(train_df, numeric_cols, enforce_no_leakage=enforce_no_leakage)
    val_ds   = BikeHVAEDatasetFromDF(val_df,   numeric_cols, enforce_no_leakage=enforce_no_leakage)
    test_ds  = BikeHVAEDatasetFromDF(test_df,  numeric_cols, enforce_no_leakage=enforce_no_leakage)

    # Use the filtered num cols from dataset
    num_numeric_features = len(train_ds.num_cols)

    train_loader = DataLoader(
        train_ds, batch_size=args.batch_size, shuffle=True,
        num_workers=args.num_workers, pin_memory=True, drop_last=False
    )
    val_loader = DataLoader(
        val_ds, batch_size=args.batch_size, shuffle=False,
        num_workers=args.num_workers, pin_memory=True, drop_last=False
    )
    test_loader = DataLoader(
        test_ds, batch_size=args.batch_size, shuffle=False,
        num_workers=args.num_workers, pin_memory=True, drop_last=False
    )

    # ----------------------------
    # Instantiate model
    # ----------------------------
    config = HVAEConfig(
        num_start_stations=num_start_stations,
        num_end_stations=num_end_stations,
        num_ride_types=num_ride_types,
        num_member_types=num_member_types,
        num_numeric_features=num_numeric_features,
        emb_dim_station=32,
        emb_dim_member=4,
        latent_dim_global=args.z_global,
        latent_dim_individual=args.z_ind,
        encoder_hidden_dim=args.enc_hidden,
        decoder_hidden_dim=args.dec_hidden,
        w_duration=args.w_duration,
        w_destination=args.w_dest,
        w_ride_type=args.w_ride,
        beta_global=args.beta_g,
        beta_individual=args.beta_i,
    )

    model = HierarchicalVAE(config).to(device)
    print(model)

    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=2, verbose=True
    )

    # ----------------------------
    # Training loop
    # ----------------------------
    best_val_loss = float("inf")
    epochs_no_improve = 0

    for epoch in range(1, args.epochs + 1):
        print(f"\n===== Epoch {epoch}/{args.epochs} =====")

        train_m = train_one_epoch(model, train_loader, optimizer, device)
        val_m   = evaluate(model, val_loader, device)

        print(
            "[TRAIN] "
            f"loss={train_m['loss']:.4f}, "
            f"dur={train_m['loss_duration']:.4f}, "
            f"dest={train_m['loss_destination']:.4f}, "
            f"ride={train_m['loss_ride_type']:.4f}, "
            f"kl_g={train_m['kl_global']:.4f}, "
            f"kl_i={train_m['kl_individual']:.4f}, "
            f"dest_acc={train_m['dest_acc']*100:.2f}%, "
            f"ride_acc={train_m['ride_acc']*100:.2f}%, "
            f"dur_mae_min={train_m['dur_mae_min']:.2f}, "
            f"dur_rmse_min={train_m['dur_rmse_min']:.2f}"
        )

        print(
            "[VAL]   "
            f"loss={val_m['loss']:.4f}, "
            f"dur={val_m['loss_duration']:.4f}, "
            f"dest={val_m['loss_destination']:.4f}, "
            f"ride={val_m['loss_ride_type']:.4f}, "
            f"kl_g={val_m['kl_global']:.4f}, "
            f"kl_i={val_m['kl_individual']:.4f}, "
            f"dest_acc={val_m['dest_acc']*100:.2f}%, "
            f"ride_acc={val_m['ride_acc']*100:.2f}%, "
            f"dur_mae_min={val_m['dur_mae_min']:.2f}, "
            f"dur_rmse_min={val_m['dur_rmse_min']:.2f}"
        )

        scheduler.step(val_m["loss"])

        improved = val_m["loss"] < best_val_loss - 1e-4
        if improved:
            best_val_loss = val_m["loss"]
            epochs_no_improve = 0
            save_checkpoint(
                args.checkpoint_dir, epoch, model, optimizer, val_m, config, is_best=True
            )
        else:
            epochs_no_improve += 1
            print(f"[INFO] No improvement for {epochs_no_improve} epoch(s).")

        # Always save checkpoint
        save_checkpoint(args.checkpoint_dir, epoch, model, optimizer, val_m, config, is_best=False)

        if epochs_no_improve >= args.patience:
            print(f"[INFO] Early stopping triggered (patience={args.patience}).")
            break

    # ----------------------------
    # Final test using best model
    # ----------------------------
    best_path = os.path.join(args.checkpoint_dir, "best_model.pt")
    if os.path.exists(best_path):
        print(f"\n[INFO] Loading best model from: {best_path}")
        ckpt = torch.load(best_path, map_location=device)
        model.load_state_dict(ckpt["model_state_dict"])
    else:
        print("\n[WARN] best_model.pt not found; evaluating last model.")

    test_m = evaluate(model, test_loader, device)
    print("\n===== Final Test Metrics =====")
    print(
        "[TEST]  "
        f"loss={test_m['loss']:.4f}, "
        f"dur={test_m['loss_duration']:.4f}, "
        f"dest={test_m['loss_destination']:.4f}, "
        f"ride={test_m['loss_ride_type']:.4f}, "
        f"kl_g={test_m['kl_global']:.4f}, "
        f"kl_i={test_m['kl_individual']:.4f}, "
        f"dest_acc={test_m['dest_acc']*100:.2f}%, "
        f"ride_acc={test_m['ride_acc']*100:.2f}%, "
        f"dur_mae_min={test_m['dur_mae_min']:.2f}, "
        f"dur_rmse_min={test_m['dur_rmse_min']:.2f}"
    )


if __name__ == "__main__":
    main()
