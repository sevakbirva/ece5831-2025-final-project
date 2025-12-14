"""
STM #2: Single-task duration prediction (regression)

- Inputs: start_station_id_idx, end_station_id_idx, member_casual_idx + numeric_feature_cols
- Target: trip_duration_min_log1p
- Loss: Gaussian NLL with clamping
- Metrics: NLL (log1p), RMSE/MAE/R2 in minutes (after expm1)

Run example:
  python phase5_train_stm_duration.py \
    --data-dir data/model_ready \
    --epochs 10 \
    --batch-size 4096 \
    --lr 1e-3 \
    --train-fraction 0.10 \
    --device cuda
"""

import argparse
import os
import pickle
from dataclasses import dataclass
from typing import Dict, Any, Optional

import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader


# -----------------------------
# Dataset
# -----------------------------
class BikeDurationDataset(Dataset):
    """
    Expects Phase-3 parquet with:
      - start_station_id_idx, end_station_id_idx, member_casual_idx
      - trip_duration_min_log1p (target)
      - numeric cols from feature_config["numeric_feature_cols"]
    """
    def __init__(
        self,
        parquet_path: str,
        feature_config: Dict[str, Any],
        fraction: Optional[float] = None,
        max_rows: Optional[int] = None,
        seed: int = 123,
    ):
        super().__init__()
        df = pd.read_parquet(parquet_path)

        self.cat_cols = ["start_station_id_idx", "end_station_id_idx", "member_casual_idx"]
        self.num_cols = feature_config["numeric_feature_cols"]
        self.y_col = "trip_duration_min_log1p"

        missing = [c for c in (self.cat_cols + self.num_cols + [self.y_col]) if c not in df.columns]
        if missing:
            raise ValueError(f"Missing columns in {parquet_path}: {missing}")

        n = len(df)
        rng = np.random.default_rng(seed)

        if max_rows is not None:
            k = min(int(max_rows), n)
            idx = rng.choice(n, size=k, replace=False)
            df = df.iloc[idx].reset_index(drop=True)
        elif fraction is not None:
            if not (0.0 < fraction <= 1.0):
                raise ValueError("--train-fraction must be in (0, 1].")
            k = max(1, int(n * fraction))
            idx = rng.choice(n, size=k, replace=False)
            df = df.iloc[idx].reset_index(drop=True)

        self.x_cat = df[self.cat_cols].to_numpy(dtype=np.int64)
        self.x_num = df[self.num_cols].to_numpy(dtype=np.float32)
        self.y = df[self.y_col].to_numpy(dtype=np.float32)

    def __len__(self) -> int:
        return self.x_cat.shape[0]

    def __getitem__(self, i: int) -> Dict[str, torch.Tensor]:
        return {
            "x_cat": torch.from_numpy(self.x_cat[i]),                 # [3]
            "x_num": torch.from_numpy(self.x_num[i]),                 # [D]
            "y_log1p": torch.tensor(self.y[i], dtype=torch.float32),  # []
        }


# -----------------------------
# Model
# -----------------------------
@dataclass
class STMConfig:
    num_start_stations: int
    num_end_stations: int
    num_member_types: int
    num_numeric_features: int

    emb_dim_station: int = 16
    emb_dim_member: int = 4
    hidden_dim: int = 128
    dropout: float = 0.10


class DurationSTM(nn.Module):
    """
    Outputs mu/logvar for log1p(duration_minutes)
    """
    def __init__(self, cfg: STMConfig):
        super().__init__()
        self.cfg = cfg
        print("cfg", cfg)

        self.emb_start = nn.Embedding(cfg.num_start_stations, cfg.emb_dim_station)
        self.emb_end = nn.Embedding(cfg.num_end_stations, cfg.emb_dim_station)
        self.emb_member = nn.Embedding(cfg.num_member_types, cfg.emb_dim_member)

        in_dim = 2 * cfg.emb_dim_station + cfg.emb_dim_member + cfg.num_numeric_features

        self.backbone = nn.Sequential(
            nn.Linear(in_dim, cfg.hidden_dim),
            nn.ReLU(),
            nn.Dropout(cfg.dropout),
            nn.Linear(cfg.hidden_dim, cfg.hidden_dim),
            nn.ReLU(),
            nn.Dropout(cfg.dropout),
        )
        self.head = nn.Linear(cfg.hidden_dim, 2)  # mu, logvar

    def forward(self, x_cat: torch.Tensor, x_num: torch.Tensor) -> Dict[str, torch.Tensor]:
        s = self.emb_start(x_cat[:, 0])
        e = self.emb_end(x_cat[:, 1])
        m = self.emb_member(x_cat[:, 2])
        x = torch.cat([s, e, m, x_num], dim=-1)
        h = self.backbone(x)
        out = self.head(h)
        mu = out[:, 0]
        logvar = out[:, 1]
        return {"mu": mu, "logvar": logvar}


def gaussian_nll_clamped(mu: torch.Tensor, logvar: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    """
    Gaussian NLL on log1p duration with stability clamps (mirrors HVAE approach).
    """
    y = y.view_as(mu)
    mu = torch.clamp(mu, -10.0, 10.0)
    logvar = torch.clamp(logvar, -5.0, 5.0)
    var = torch.exp(logvar).clamp(min=1e-6)
    nll = 0.5 * (np.log(2.0 * np.pi) + logvar + (y - mu).pow(2) / var)
    return nll.mean()


@torch.no_grad()
def eval_metrics(model: nn.Module, loader: DataLoader, device: torch.device) -> Dict[str, float]:
    model.eval()

    n = 0
    nll_sum = 0.0

    # For minute-scale metrics
    y_true_min = []
    y_pred_min = []

    for batch in loader:
        x_cat = batch["x_cat"].to(device)
        x_num = batch["x_num"].to(device)
        y = batch["y_log1p"].to(device)

        out = model(x_cat, x_num)
        mu = out["mu"]
        logvar = out["logvar"]

        nll = gaussian_nll_clamped(mu, logvar, y)
        bs = y.size(0)
        n += bs
        nll_sum += nll.item() * bs

        # Convert to minutes for interpretability
        y_min = torch.expm1(y).clamp(min=0.0)
        pred_min = torch.expm1(mu).clamp(min=0.0)

        y_true_min.append(y_min.detach().cpu().numpy())
        y_pred_min.append(pred_min.detach().cpu().numpy())

    y_true = np.concatenate(y_true_min) if y_true_min else np.array([])
    y_pred = np.concatenate(y_pred_min) if y_pred_min else np.array([])

    if y_true.size > 0:
        err = y_pred - y_true
        rmse = float(np.sqrt(np.mean(err**2)))
        mae = float(np.mean(np.abs(err)))
        # R2
        ss_res = float(np.sum((y_true - y_pred) ** 2))
        ss_tot = float(np.sum((y_true - np.mean(y_true)) ** 2)) + 1e-12
        r2 = 1.0 - ss_res / ss_tot
    else:
        rmse, mae, r2 = float("nan"), float("nan"), float("nan")

    return {
        "nll_log1p": nll_sum / max(1, n),
        "rmse_min": rmse,
        "mae_min": mae,
        "r2_min": r2,
        "n": float(n),
    }


def train_one_epoch(model, loader, optimizer, device, log_every: int = 200) -> Dict[str, float]:
    model.train()
    n = 0
    loss_sum = 0.0

    for step, batch in enumerate(loader, start=1):
        x_cat = batch["x_cat"].to(device)
        x_num = batch["x_num"].to(device)
        y = batch["y_log1p"].to(device)

        optimizer.zero_grad(set_to_none=True)
        out = model(x_cat, x_num)
        loss = gaussian_nll_clamped(out["mu"], out["logvar"], y)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()

        bs = y.size(0)
        n += bs
        loss_sum += loss.item() * bs

        if log_every > 0 and step % log_every == 0:
            print(f"  [step {step}] nll={loss.item():.4f}")

    return {"nll_log1p": loss_sum / max(1, n), "n": float(n)}


def parse_args():
    p = argparse.ArgumentParser("Train STM #2: duration (log1p) heteroscedastic regression")
    p.add_argument("--data-dir", type=str, default="data/model_ready")
    p.add_argument("--epochs", type=int, default=10)
    p.add_argument("--batch-size", type=int, default=4096)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--weight-decay", type=float, default=1e-4)
    p.add_argument("--num-workers", type=int, default=4)
    p.add_argument("--device", type=str, default="cuda")
    p.add_argument("--seed", type=int, default=123)

    # Partial-data controls for debugging
    p.add_argument("--train-fraction", type=float, default=0.10)
    p.add_argument("--max-train-rows", type=int, default=None)

    p.add_argument("--log-every", type=int, default=300)
    p.add_argument("--patience", type=int, default=3)
    return p.parse_args()


def main():
    args = parse_args()
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    device = torch.device(args.device if (args.device == "cpu" or torch.cuda.is_available()) else "cpu")
    print(f"[INFO] Using device: {device}")

    artifacts_path = os.path.join(args.data_dir, "phase3_artifacts.pkl")
    train_path = os.path.join(args.data_dir, "train.parquet")
    val_path = os.path.join(args.data_dir, "val.parquet")
    test_path = os.path.join(args.data_dir, "test.parquet")

    print(f"[INFO] Loading artifacts: {artifacts_path}")
    with open(artifacts_path, "rb") as f:
        artifacts = pickle.load(f)

    category_mappings = artifacts["category_mappings"]
    feature_config = artifacts["feature_config"]

    cfg = STMConfig(
        num_start_stations=len(category_mappings["start_station_id"]),
        num_end_stations=len(category_mappings["end_station_id"]),
        num_member_types=len(category_mappings["member_casual"]),
        num_numeric_features=len(feature_config["numeric_feature_cols"]),
    )
    print(f"[INFO] Config: {feature_config}")

    # Datasets
    train_fraction = None if args.max_train_rows is not None else args.train_fraction
    train_ds = BikeDurationDataset(train_path, feature_config, fraction=train_fraction, max_rows=args.max_train_rows, seed=args.seed)
    val_ds = BikeDurationDataset(val_path, feature_config)
    test_ds = BikeDurationDataset(test_path, feature_config)

    print(f"[INFO] Train n={len(train_ds)} (partial), Val n={len(val_ds)}, Test n={len(test_ds)}")

    pin = (device.type == "cuda")
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True,
                              num_workers=args.num_workers, pin_memory=pin)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False,
                            num_workers=args.num_workers, pin_memory=pin)
    test_loader = DataLoader(test_ds, batch_size=args.batch_size, shuffle=False,
                             num_workers=args.num_workers, pin_memory=pin)

    model = DurationSTM(cfg).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    best_val = float("inf")
    bad_epochs = 0
    ckpt_path = os.path.join(args.data_dir, "stm_duration_best.pt")

    for epoch in range(1, args.epochs + 1):
        print(f"\n===== Epoch {epoch}/{args.epochs} =====")
        tr = train_one_epoch(model, train_loader, optimizer, device, log_every=args.log_every)
        va = eval_metrics(model, val_loader, device)

        print(f"[TRAIN] nll_log1p={tr['nll_log1p']:.4f} (n={int(tr['n'])})")
        print(f"[VAL]   nll_log1p={va['nll_log1p']:.4f}, rmse={va['rmse_min']:.2f} min, "
              f"mae={va['mae_min']:.2f} min, r2={va['r2_min']:.4f}")

        if va["nll_log1p"] < best_val - 1e-4:
            best_val = va["nll_log1p"]
            bad_epochs = 0
            torch.save({"model": model.state_dict(), "cfg": cfg.__dict__}, ckpt_path)
            print("[INFO] Saved best STM checkpoint: stm_duration_best.pt")
        else:
            bad_epochs += 1
            print(f"[INFO] No improvement ({bad_epochs}/{args.patience}).")
            if bad_epochs >= args.patience:
                print("[INFO] Early stopping.")
                break

    if os.path.exists(ckpt_path):
        ckpt = torch.load(ckpt_path, map_location=device)
        model.load_state_dict(ckpt["model"])
        print(f"[INFO] Loaded best checkpoint from: {ckpt_path}")

    te = eval_metrics(model, test_loader, device)
    print("\n===== Final Test Metrics (STM #2: duration) =====")
    print(f"[TEST] nll_log1p={te['nll_log1p']:.4f}, rmse={te['rmse_min']:.2f} min, "
          f"mae={te['mae_min']:.2f} min, r2={te['r2_min']:.4f} (n={int(te['n'])})")


if __name__ == "__main__":
    main()
