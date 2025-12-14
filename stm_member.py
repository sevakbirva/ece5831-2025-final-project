"""
STM #3: Single-task member_casual classification (fast)

Inputs:
  categorical: start_station_id_idx, end_station_id_idx, rideable_type_idx
  numeric: feature_config["numeric_feature_cols"] EXCLUDING any duration-derived columns if desired

Target:
  member_casual_idx

Run:
  python stm_member.py --data-dir data/model_ready --epochs 10 --batch-size 4096 --lr 1e-3 --train-fraction 0.10 --device cuda
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
class BikeMemberDataset(Dataset):
    def __init__(
        self,
        parquet_path: str,
        feature_config: Dict[str, Any],
        fraction: Optional[float] = None,
        max_rows: Optional[int] = None,
        seed: int = 123,
        drop_duration_inputs: bool = False,
    ):
        super().__init__()
        df = pd.read_parquet(parquet_path)

        # Inputs: exclude member_casual_idx from X (it is the target)
        self.cat_cols = ["start_station_id_idx", "end_station_id_idx", "rideable_type_idx"]
        num_cols = list(feature_config["numeric_feature_cols"])

        if drop_duration_inputs:
            num_cols = [c for c in num_cols if "trip_duration" not in c]

        self.num_cols = num_cols
        self.y_col = "member_casual_idx"

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
        self.y = df[self.y_col].to_numpy(dtype=np.int64)

    def __len__(self) -> int:
        return self.x_cat.shape[0]

    def __getitem__(self, i: int) -> Dict[str, torch.Tensor]:
        return {
            "x_cat": torch.from_numpy(self.x_cat[i]),
            "x_num": torch.from_numpy(self.x_num[i]),
            "y": torch.tensor(self.y[i], dtype=torch.long),
        }


# -----------------------------
# Model
# -----------------------------
@dataclass
class STMConfig:
    num_start_stations: int
    num_end_stations: int
    num_ride_types: int
    num_numeric_features: int
    num_member_classes: int

    emb_dim_station: int = 16
    emb_dim_ride: int = 4
    hidden_dim: int = 128
    dropout: float = 0.10


class MemberSTM(nn.Module):
    def __init__(self, cfg: STMConfig):
        super().__init__()
        self.emb_start = nn.Embedding(cfg.num_start_stations, cfg.emb_dim_station)
        self.emb_end = nn.Embedding(cfg.num_end_stations, cfg.emb_dim_station)
        self.emb_ride = nn.Embedding(cfg.num_ride_types, cfg.emb_dim_ride)

        in_dim = 2 * cfg.emb_dim_station + cfg.emb_dim_ride + cfg.num_numeric_features

        self.mlp = nn.Sequential(
            nn.Linear(in_dim, cfg.hidden_dim),
            nn.ReLU(),
            nn.Dropout(cfg.dropout),
            nn.Linear(cfg.hidden_dim, cfg.hidden_dim),
            nn.ReLU(),
            nn.Dropout(cfg.dropout),
            nn.Linear(cfg.hidden_dim, cfg.num_member_classes),
        )

    def forward(self, x_cat: torch.Tensor, x_num: torch.Tensor) -> torch.Tensor:
        s = self.emb_start(x_cat[:, 0])
        e = self.emb_end(x_cat[:, 1])
        r = self.emb_ride(x_cat[:, 2])
        x = torch.cat([s, e, r, x_num], dim=-1)
        return self.mlp(x)


# -----------------------------
# Train / Eval
# -----------------------------
@torch.no_grad()
def evaluate(model: nn.Module, loader: DataLoader, device: torch.device) -> Dict[str, float]:
    model.eval()
    ce = nn.CrossEntropyLoss()
    total, correct, loss_sum = 0, 0, 0.0

    for batch in loader:
        x_cat = batch["x_cat"].to(device)
        x_num = batch["x_num"].to(device)
        y = batch["y"].to(device)

        logits = model(x_cat, x_num)
        loss = ce(logits, y)

        preds = logits.argmax(dim=-1)
        correct += (preds == y).sum().item()
        bs = y.size(0)
        total += bs
        loss_sum += loss.item() * bs

    return {"loss": loss_sum / max(1, total), "acc": correct / max(1, total), "n": float(total)}


def train_one_epoch(model, loader, optimizer, device, log_every: int = 300) -> Dict[str, float]:
    model.train()
    ce = nn.CrossEntropyLoss()
    total, correct, loss_sum = 0, 0, 0.0

    for step, batch in enumerate(loader, start=1):
        x_cat = batch["x_cat"].to(device)
        x_num = batch["x_num"].to(device)
        y = batch["y"].to(device)

        optimizer.zero_grad(set_to_none=True)
        logits = model(x_cat, x_num)
        loss = ce(logits, y)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()

        preds = logits.argmax(dim=-1)
        correct += (preds == y).sum().item()
        bs = y.size(0)
        total += bs
        loss_sum += loss.item() * bs

        if log_every and step % log_every == 0:
            print(f"  [step {step}] loss={loss.item():.4f}")

    return {"loss": loss_sum / max(1, total), "acc": correct / max(1, total), "n": float(total)}


# -----------------------------
# Main
# -----------------------------
def parse_args():
    p = argparse.ArgumentParser("STM #3: member_casual classifier")
    p.add_argument("--data-dir", type=str, default="data/model_ready")
    p.add_argument("--epochs", type=int, default=10)
    p.add_argument("--batch-size", type=int, default=4096)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--weight-decay", type=float, default=1e-4)
    p.add_argument("--num-workers", type=int, default=2)  # avoid your warning
    p.add_argument("--device", type=str, default="cuda")
    p.add_argument("--seed", type=int, default=123)

    p.add_argument("--train-fraction", type=float, default=0.10)
    p.add_argument("--max-train-rows", type=int, default=None)
    p.add_argument("--patience", type=int, default=3)
    p.add_argument("--log-every", type=int, default=400)

    # Optional: make it more “fair” (drops duration-derived inputs)
    p.add_argument("--drop-duration-inputs", action="store_true")

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

    # Determine numeric feature count after optional dropping
    num_cols = list(feature_config["numeric_feature_cols"])
    if args.drop_duration_inputs:
        num_cols = [c for c in num_cols if "trip_duration" not in c]

    cfg = STMConfig(
        num_start_stations=len(category_mappings["start_station_id"]),
        num_end_stations=len(category_mappings["end_station_id"]),
        num_ride_types=len(category_mappings["rideable_type"]),
        num_numeric_features=len(num_cols),
        num_member_classes=len(category_mappings["member_casual"]),
    )

    # Datasets
    train_fraction = None if args.max_train_rows is not None else args.train_fraction
    train_ds = BikeMemberDataset(
        train_path, feature_config,
        fraction=train_fraction, max_rows=args.max_train_rows, seed=args.seed,
        drop_duration_inputs=args.drop_duration_inputs
    )
    val_ds = BikeMemberDataset(val_path, feature_config, drop_duration_inputs=args.drop_duration_inputs)
    test_ds = BikeMemberDataset(test_path, feature_config, drop_duration_inputs=args.drop_duration_inputs)

    print(f"[INFO] Train n={len(train_ds)} (partial), Val n={len(val_ds)}, Test n={len(test_ds)}")

    pin = (device.type == "cuda")
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True,
                              num_workers=args.num_workers, pin_memory=pin)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False,
                            num_workers=args.num_workers, pin_memory=pin)
    test_loader = DataLoader(test_ds, batch_size=args.batch_size, shuffle=False,
                             num_workers=args.num_workers, pin_memory=pin)

    model = MemberSTM(cfg).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    best_val = float("inf")
    bad = 0
    ckpt_path = os.path.join(args.data_dir, "stm_member_best.pt")

    for epoch in range(1, args.epochs + 1):
        print(f"\n===== Epoch {epoch}/{args.epochs} =====")
        tr = train_one_epoch(model, train_loader, opt, device, log_every=args.log_every)
        va = evaluate(model, val_loader, device)

        print(f"[TRAIN] loss={tr['loss']:.4f}, acc={tr['acc']*100:.2f}%")
        print(f"[VAL]   loss={va['loss']:.4f}, acc={va['acc']*100:.2f}%")

        if va["loss"] < best_val - 1e-4:
            best_val = va["loss"]
            bad = 0
            torch.save({"model": model.state_dict(), "cfg": cfg.__dict__}, ckpt_path)
            print("[INFO] Saved best STM checkpoint: stm_member_best.pt")
        else:
            bad += 1
            print(f"[INFO] No improvement ({bad}/{args.patience}).")
            if bad >= args.patience:
                print("[INFO] Early stopping.")
                break

    if os.path.exists(ckpt_path):
        ckpt = torch.load(ckpt_path, map_location=device)
        model.load_state_dict(ckpt["model"])
        print(f"[INFO] Loaded best checkpoint from: {ckpt_path}")

    te = evaluate(model, test_loader, device)
    print("\n===== Final Test Metrics (STM #3: member_casual) =====")
    print(f"[TEST] loss={te['loss']:.4f}, acc={te['acc']*100:.2f}% (n={int(te['n'])})")


if __name__ == "__main__":
    main()
