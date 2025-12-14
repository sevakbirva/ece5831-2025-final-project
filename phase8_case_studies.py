"""
Phase 8: Case Studies and Anomaly Slices

This script:
- Loads latent_analysis_<split>.parquet (from Phase 7)
- Extracts:
    (1) Global top-k anomalous trips
    (2) Top-k anomalies per intent cluster
    (3) Threshold-based anomaly flags (e.g., top 1% / 0.5%)
- Saves these subsets for inspection and reporting

Usage example:

    python phase8_case_studies.py \
        --input-path data/latent_analysis/latent_analysis_test.parquet \
        --split test \
        --output-dir data/case_studies \
        --top-k-global 200 \
        --top-k-per-cluster 50 \
        --p95-thresh 0.95 \
        --p99-thresh 0.99
"""

import argparse
import os
from typing import List

import numpy as np
import pandas as pd


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Phase 8: Case studies and anomaly slices from HVAE outputs."
    )
    parser.add_argument(
        "--input-path",
        type=str,
        default="data/latent_analysis/latent_analysis_test.parquet",
        help="Path to latent_analysis_<split>.parquet.",
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
        default="data/case_studies",
        help="Directory to save case-study files.",
    )
    parser.add_argument(
        "--top-k-global",
        type=int,
        default=200,
        help="Number of globally most anomalous trips to export.",
    )
    parser.add_argument(
        "--top-k-per-cluster",
        type=int,
        default=50,
        help="Number of top anomalies to export per intent cluster.",
    )
    parser.add_argument(
        "--p95-thresh",
        type=float,
        default=0.95,
        help="Quantile for 'high anomaly' threshold (e.g., 0.95 for top 5%).",
    )
    parser.add_argument(
        "--p99-thresh",
        type=float,
        default=0.99,
        help="Quantile for 'extreme anomaly' threshold (e.g., 0.99 for top 1%).",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    print(f"[INFO] Loading latent+anomaly data from: {args.input_path}")
    df = pd.read_parquet(args.input_path)

    if "anomaly_score" not in df.columns:
        raise ValueError("Column 'anomaly_score' not found in input dataframe.")
    if "intent_cluster" not in df.columns:
        raise ValueError("Column 'intent_cluster' not found in input dataframe.")

    os.makedirs(args.output_dir, exist_ok=True)

    # -------------------------------------------------------------
    # 1. Global top-k anomalies
    # -------------------------------------------------------------
    print(f"[INFO] Selecting global top-{args.top_k_global} anomalies...")
    df_global_topk = df.sort_values("anomaly_score", ascending=False).head(args.top_k_global)

    # Optional: restrict to key columns for readability in case-study table
    core_cols: List[str] = [
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
        "intent_cluster",
    ]
    core_cols = [c for c in core_cols if c in df.columns]

    global_core = df_global_topk[core_cols].copy()

    out_global_parquet = os.path.join(
        args.output_dir, f"global_top{args.top_k_global}_anomalies_{args.split}.parquet"
    )
    out_global_csv = os.path.join(
        args.output_dir, f"global_top{args.top_k_global}_anomalies_{args.split}.csv"
    )
    global_core.to_parquet(out_global_parquet, index=False)
    global_core.to_csv(out_global_csv, index=False)
    print(f"[INFO] Saved global top-{args.top_k_global} anomalies to:")
    print(f"       {out_global_parquet}")
    print(f"       {out_global_csv}")

    # -------------------------------------------------------------
    # 2. Top-k anomalies per intent cluster
    # -------------------------------------------------------------
    print(f"[INFO] Selecting top-{args.top_k_per_cluster} anomalies per cluster...")
    df_sorted = df.sort_values("anomaly_score", ascending=False)

    per_cluster_list = []
    for cluster_id, group in df_sorted.groupby("intent_cluster"):
        top_cluster = group.head(args.top_k_per_cluster).copy()
        top_cluster["cluster_rank"] = np.arange(1, len(top_cluster) + 1)
        per_cluster_list.append(top_cluster)

    df_per_cluster = pd.concat(per_cluster_list, axis=0)
    per_cluster_core = df_per_cluster[core_cols + ["cluster_rank"]].copy()

    out_cluster_parquet = os.path.join(
        args.output_dir,
        f"cluster_top{args.top_k_per_cluster}_anomalies_{args.split}.parquet",
    )
    out_cluster_csv = os.path.join(
        args.output_dir,
        f"cluster_top{args.top_k_per_cluster}_anomalies_{args.split}.csv",
    )
    per_cluster_core.to_parquet(out_cluster_parquet, index=False)
    per_cluster_core.to_csv(out_cluster_csv, index=False)
    print(f"[INFO] Saved per-cluster top-{args.top_k_per_cluster} anomalies to:")
    print(f"       {out_cluster_parquet}")
    print(f"       {out_cluster_csv}")

    # -------------------------------------------------------------
    # 3. Threshold-based anomaly flags (p95 / p99)
    # -------------------------------------------------------------
    print("[INFO] Computing threshold-based anomaly flags...")

    p95_val = df["anomaly_score"].quantile(args.p95_thresh)
    p99_val = df["anomaly_score"].quantile(args.p99_thresh)

    df["is_anom_p95"] = df["anomaly_score"] >= p95_val
    df["is_anom_p99"] = df["anomaly_score"] >= p99_val

    n_total = len(df)
    n_p95 = df["is_anom_p95"].sum()
    n_p99 = df["is_anom_p99"].sum()

    print(f"  [STATS] p95 threshold (top {int((1-args.p95_thresh)*100)}%): {p95_val:.4f}")
    print(f"          Count >= p95: {n_p95} ({n_p95 / n_total * 100:.2f} % of trips)")
    print(f"  [STATS] p99 threshold (top {int((1-args.p99_thresh)*100)}%): {p99_val:.4f}")
    print(f"          Count >= p99: {n_p99} ({n_p99 / n_total * 100:.2f} % of trips)")

    out_flagged_parquet = os.path.join(
        args.output_dir, f"latent_with_flags_{args.split}.parquet"
    )
    df.to_parquet(out_flagged_parquet, index=False)
    print(f"[INFO] Saved latent+flags dataframe to: {out_flagged_parquet}")

    # -------------------------------------------------------------
    # 4. Quick cluster-level counts of flagged anomalies
    # -------------------------------------------------------------
    print("\n[Cluster-level anomaly counts (p99 flag)]")
    if "intent_cluster" in df.columns:
        cluster_flag_counts = (
            df.groupby("intent_cluster")["is_anom_p99"]
            .agg(["sum", "count"])
            .rename(columns={"sum": "n_anom_p99", "count": "n_total"})
        )
        cluster_flag_counts["pct_anom_p99"] = (
            cluster_flag_counts["n_anom_p99"] / cluster_flag_counts["n_total"] * 100.0
        )
        print(cluster_flag_counts.round(2))


if __name__ == "__main__":
    main()
