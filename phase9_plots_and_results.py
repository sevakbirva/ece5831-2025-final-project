"""
Phase 9: Plots and quantitative summaries for HVAE results.

Generates:
  - Rideable-type confusion matrix on test split
  - Anomaly score histogram with p95/p99 thresholds (test)
  - Bar plot of p99 anomaly rate per intent cluster

Assumes:
  - phase3_artifacts.pkl in data/model_ready
  - test.parquet in data/model_ready
  - latent_with_flags_test.parquet in data/case_studies
  - best HVAE checkpoint at checkpoints/hvae_v2/best_model.pt

Run:
  python phase9_plots_and_results.py
"""

import os
import pickle

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader

import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.preprocessing import LabelEncoder

from phase4_hvae_model import HVAEConfig, HierarchicalVAE


# =========================
# 1. Utility: build model
# =========================

def load_artifacts(artifacts_path):
    print(f"[INFO] Loading Phase 3 artifacts from: {artifacts_path}")
    with open(artifacts_path, "rb") as f:
        return pickle.load(f)


def build_model_from_artifacts(artifacts, checkpoint_path, device):
    category_mappings = artifacts["category_mappings"]
    feature_config = artifacts["feature_config"]

    num_start_stations = len(category_mappings["start_station_id"])
    num_end_stations = len(category_mappings["end_station_id"])
    num_ride_types = len(category_mappings["rideable_type"])
    num_member_types = len(category_mappings["member_casual"])
    num_numeric_features = len(feature_config["numeric_feature_cols"])

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
    return model, feature_config, category_mappings


# =========================
# 2. Dataset for test eval
# =========================

class TestClassificationDataset(Dataset):
    """
    Only what we need for rideable-type classification evaluation.

    Expects test.parquet with:
      - start_station_id_idx
      - end_station_id_idx
      - member_casual_idx
      - numeric_feature_cols
      - rideable_type_idx (target)
    """

    def __init__(self, parquet_path, feature_config):
        df = pd.read_parquet(parquet_path)

        self.cat_cols = [
            "start_station_id_idx",
            "end_station_id_idx",
            "member_casual_idx",
        ]
        self.num_cols = feature_config["numeric_feature_cols"]

        missing_cat = [c for c in self.cat_cols if c not in df.columns]
        missing_num = [c for c in self.num_cols if c not in df.columns]
        if missing_cat or missing_num:
            raise ValueError(
                f"Missing columns in {parquet_path}.\n"
                f"  Missing categorical: {missing_cat}\n"
                f"  Missing numeric: {missing_num}"
            )

        if "rideable_type_idx" not in df.columns:
            raise ValueError("Column 'rideable_type_idx' not found in test parquet.")

        self.x_cat = df[self.cat_cols].to_numpy(dtype=np.int64)
        self.x_num = df[self.num_cols].to_numpy(dtype=np.float32)
        self.y = df["rideable_type_idx"].to_numpy(dtype=np.int64)
        self.n = self.x_cat.shape[0]

    def __len__(self):
        return self.n

    def __getitem__(self, idx):
        return {
            "x_cat": torch.from_numpy(self.x_cat[idx]),
            "x_num": torch.from_numpy(self.x_num[idx]),
            "y": torch.tensor(self.y[idx], dtype=torch.long),
        }


# =========================
# 3. Evaluation & plots
# =========================

def evaluate_ride_type(
    model,
    dataset,
    category_mappings,
    device,
    batch_size=4096,
    out_dir="figures",
):
    os.makedirs(out_dir, exist_ok=True)

    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=4,
        pin_memory=(device.type == "cuda"),
    )

    all_y_true = []
    all_y_pred = []

    with torch.no_grad():
        for batch in loader:
            x_cat = batch["x_cat"].to(device)
            x_num = batch["x_num"].to(device)
            y = batch["y"].to(device)

            outputs, _ = model(x_cat, x_num)
            logits = outputs["ride_logits"]
            preds = logits.argmax(dim=-1)

            all_y_true.append(y.cpu().numpy())
            all_y_pred.append(preds.cpu().numpy())

    y_true = np.concatenate(all_y_true, axis=0)
    y_pred = np.concatenate(all_y_pred, axis=0)

    acc = (y_true == y_pred).mean()
    print(f"\n[RESULT] Test rideable_type accuracy: {acc * 100:.2f}%")

    # Build label names from category_mappings
    # idx -> label string
    idx_to_label = {idx: label for label, idx in category_mappings["rideable_type"].items()}
    labels_sorted = sorted(idx_to_label.keys())
    target_names = [idx_to_label[i] for i in labels_sorted]

    cm = confusion_matrix(y_true, y_pred, labels=labels_sorted)
    print("\n[CONFUSION MATRIX (raw counts)]")
    print(cm)
    print("\n[CLASSIFICATION REPORT]")
    print(classification_report(y_true, y_pred, labels=labels_sorted, target_names=target_names))

    # Plot confusion matrix
    fig, ax = plt.subplots(figsize=(5, 4))
    im = ax.imshow(cm, cmap="Blues")

    ax.set_xticks(range(len(labels_sorted)))
    ax.set_yticks(range(len(labels_sorted)))
    ax.set_xticklabels(target_names, rotation=45, ha="right")
    ax.set_yticklabels(target_names)

    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, cm[i, j], ha="center", va="center", fontsize=8)

    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    ax.set_title("Rideable Type – Confusion Matrix (Test)")

    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    fig.tight_layout()
    out_path = os.path.join(out_dir, "fig_rideable_confusion_test.png")
    fig.savefig(out_path, dpi=300)
    plt.close(fig)
    print(f"[INFO] Saved confusion matrix figure to: {out_path}")

    return acc, cm, target_names


def plot_anomaly_histogram(
    latent_flags_path="data/case_studies/latent_with_flags_test.parquet",
    out_dir="figures",
):
    os.makedirs(out_dir, exist_ok=True)
    df = pd.read_parquet(latent_flags_path)

    scores = df["anomaly_score"].values
    p95 = df["anomaly_score"].quantile(0.95)
    p99 = df["anomaly_score"].quantile(0.99)

    fig, ax = plt.subplots(figsize=(6, 4))
    ax.hist(scores, bins=100, alpha=0.7)
    ax.axvline(p95, linestyle="--", label=f"p95={p95:.2f}")
    ax.axvline(p99, linestyle="--", label=f"p99={p99:.2f}")

    ax.set_xlabel("Anomaly score")
    ax.set_ylabel("Trip count")
    ax.set_title("Anomaly score distribution (test)")
    ax.legend()

    fig.tight_layout()
    out_path = os.path.join(out_dir, "fig_anomaly_hist_test.png")
    fig.savefig(out_path, dpi=300)
    plt.close(fig)
    print(f"[INFO] Saved anomaly histogram figure to: {out_path}")

    return p95, p99


def plot_cluster_anomaly_bar(
    latent_flags_path="data/case_studies/latent_with_flags_test.parquet",
    out_dir="figures",
):
    os.makedirs(out_dir, exist_ok=True)
    df = pd.read_parquet(latent_flags_path)

    if "intent_cluster" not in df.columns or "is_anom_p99" not in df.columns:
        raise ValueError("Columns 'intent_cluster' and 'is_anom_p99' missing in latent flags parquet.")

    cluster_counts = (
        df.groupby("intent_cluster")["is_anom_p99"]
        .agg(["sum", "count"])
        .rename(columns={"sum": "n_anom_p99", "count": "n_total"})
    )
    cluster_counts["pct_anom_p99"] = cluster_counts["n_anom_p99"] / cluster_counts["n_total"] * 100.0

    print("\n[Cluster-level anomaly rates (p99 flag)]")
    print(cluster_counts.round(2))

    fig, ax = plt.subplots(figsize=(6, 4))
    ax.bar(cluster_counts.index.astype(str), cluster_counts["pct_anom_p99"])

    ax.set_xlabel("Intent cluster")
    ax.set_ylabel("p99 anomaly rate (%)")
    ax.set_title("p99 anomaly rate by intent cluster (test)")

    fig.tight_layout()
    out_path = os.path.join(out_dir, "fig_cluster_anomaly_rate_p99_test.png")
    fig.savefig(out_path, dpi=300)
    plt.close(fig)
    print(f"[INFO] Saved cluster anomaly rate figure to: {out_path}")

    return cluster_counts


# =========================
# 4. Main
# =========================

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[INFO] Using device: {device}")

    data_dir = "data/model_ready"
    artifacts_path = os.path.join(data_dir, "phase3_artifacts.pkl")
    test_path = os.path.join(data_dir, "test.parquet")
    ckpt_path = "checkpoints/hvae_v2/best_model.pt"

    artifacts = load_artifacts(artifacts_path)
    model, feature_config, category_mappings = build_model_from_artifacts(
        artifacts, ckpt_path, device
    )

    # 1) Rideable-type evaluation + confusion matrix
    print("\n=== Rideable-type classification (test) ===")
    test_dataset = TestClassificationDataset(test_path, feature_config)
    acc, cm, target_names = evaluate_ride_type(
        model, test_dataset, category_mappings, device, batch_size=4096, out_dir="figures"
    )

    # 2) Anomaly score histogram
    print("\n=== Anomaly score distribution (test) ===")
    p95, p99 = plot_anomaly_histogram(
        latent_flags_path="data/case_studies/latent_with_flags_test.parquet",
        out_dir="figures",
    )

    # 3) Cluster-level anomaly bar
    print("\n=== Cluster-level anomaly rates (p99) ===")
    cluster_counts = plot_cluster_anomaly_bar(
        latent_flags_path="data/case_studies/latent_with_flags_test.parquet",
        out_dir="figures",
    )

    print("\n[SUMMARY]")
    print(f"  Test rideable-type accuracy: {acc * 100:.2f}%")
    print(f"  Anomaly score p95 (top 5%): {p95:.4f}")
    print(f"  Anomaly score p99 (top 1%): {p99:.4f}")
    print("  Cluster-level p99 anomaly rates (%):")
    print(cluster_counts['pct_anom_p99'].round(2))


if __name__ == "__main__":
    main()
