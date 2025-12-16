# ECE5831-2025 Final Project: Multi-Task Rider Behavior Modeling for Micromobility Systems

This repository implements an end-to-end pipeline for learning **shared latent representations** from the 2024–2025 Divvy/Cyclistic bike-share trip data and using them for **multi-task prediction** and **downstream analysis** (e.g., anomaly scoring and latent-space interpretation).

Core idea: instead of training separate models per task, we train a **hierarchical variational autoencoder (HVAE)** that captures global and trip-specific factors in a structured latent space, and decodes multiple behavioral outcomes jointly.

---

## Links

- **Google Drive Folder:** [ece5831-2025-final-project](https://drive.google.com/drive/folders/1K0wVZiQb_MqqgwOnEKR4e8Rfm-mGZNm-)
- **Pre-recorded presentation video:** [Watch here - YouTube](https://youtu.be/lKdaZDTApao) OR [Google Drive Presentation](https://drive.google.com/file/d/11G5-F6A4Z7R4FCHKIELwwaxPok2-hslH/view?usp=drive_link)
- **Presentation slides:** [View slides - GitHub](https://github.com/sevakbirva/ece5831-2025-final-project/blob/main/reports/HVAE_Micromobility_Presentation.pptx) OR [Slide - Google Drive](https://docs.google.com/presentation/d/1KFsdnlbEVgcu2VQR1PYWpdxOrlMdpfLk/edit?usp=drive_link&ouid=113843156497153318057&rtpof=true&sd=true)
- **Final report:** [Read report - GitHub](https://github.com/sevakbirva/ece5831-2025-final-project/blob/main/reports/ECE5831_project_final_report_BirvaSevak_ShrenikJadhav.pdf) OR [Report - Google Drive](https://drive.google.com/file/d/1Vlg6_5o1L7xY6GzM_KtvI6FyTlpd3GQV/view?usp=drive_link)
- **Dataset (raw):** 2024–2025 Divvy bike sharing data (Cyclistic) on Kaggle:  
  https://www.kaggle.com/datasets/miaadnabizadeh/20242025-divvy-bike-sharing-data-cyclistic OR [Google Drive: data/raw-data.zip](https://drive.google.com/drive/folders/14JzsFWILweuwDEpD0-m3p5-J2T0HK3Qo?usp=drive_link)
- **Demo video:** [Watch demo - YouTube](https://youtu.be/j_nKluUBw9k) OR [Google Drive Demo](https://drive.google.com/file/d/1-XvjkzODMFo-zWhPWwgNS9iEhV_Ob-dm/view?usp=drive_link)
---

## What This Project Does

### Tasks supported
Depending on the phase and script configuration, the project supports:

- **Trip duration prediction** (regression)
- **Destination prediction** (classification over end stations)
- **Rideable type prediction** (classification)
- **Rider type prediction** (`member` vs `casual`) (classification)
- **Anomaly scoring** using learned latent representations
- **Latent-space analysis** to interpret learned global vs individual behavior factors

### Phased pipeline
The pipeline is implemented as explicit, reproducible phases:

1. **Phase 1 — Data preparation** (`phase1_data_preparation.py`)
2. **Phase 2 — Feature engineering** (`phase2_feature_engineering.py`)
3. **Phase 3 — Build datasets/artifacts** (`phase3_build_datasets.py`)
4. **Phase 4 — HVAE model definition** (`phase4_hvae_model.py`)
5. **Phase 5 — HVAE training** (`phase5_train_hvae.py`, `phase5_train_hvae_2.py`)
6. **Phase 6 — Anomaly scoring** (`phase6_anomaly_scoring.py`)
7. **Phase 7 — Latent analysis** (`phase7_latent_analysis.py`)
8. **Phase 8 — Case studies** (`phase8_case_studies.py`)
9. **Phase 9 — Plots and results** (`phase9_plots_and_results.py`)

Baseline single-task models are included for comparison:
- `stm_duration.py`
- `stm_member.py`
- `stm_rideable.py`

---

## Repository Structure

A simplified view of the repository layout:

- `data/`
  - `processed/` — outputs after cleaning/feature engineering
  - `model_ready/` — finalized datasets and artifacts for model training/inference
  - `full_bike_dataset.parquet` — local working dataset file (not tracked in GitHub if large)
- `figures/` — paper/report figures (e.g., `figure1.png` to `figure4.png`)
- `*.ipynb` — EDA, testing, and evaluation notebooks
- `environment.yml` — reproducible environment specification
- `README.md` — this file

---

## Dataset Availability (Raw vs. Generated)

The **raw dataset** is not included in this GitHub repository because it is large. Download it from Kaggle:

https://www.kaggle.com/datasets/miaadnabizadeh/20242025-divvy-bike-sharing-data-cyclistic

After you run **Phase 1–3**, the pipeline will generate:
- cleaned/processed outputs under `data/processed/`
- model-ready datasets and artifacts under `data/model_ready/`


---

## Setup

### 1) Create the environment
Create the conda environment from `environment.yml`:

```bash
conda env create -f environment.yml
conda activate ece5831-2025-final-project
```

---

## How to Run the Pipeline

Run phases in order. The scripts are designed to write outputs into `data/processed/` and `data/model_ready/`.

```bash
# Phase 1–3: prepare data
python phase1_data_preparation.py
python phase2_feature_engineering.py
python phase3_build_datasets.py

# Phase 5: train HVAE
python phase5_train_hvae.py
# or
python phase5_train_hvae_2.py

# Phase 6–9: analysis and reporting artifacts
python phase6_anomaly_scoring.py
python phase7_latent_analysis.py
python phase8_case_studies.py
python phase9_plots_and_results.py
```

For jupyter notebooks exploration & testing:
- `EDA.ipynb`
- `data_prep_testing.ipynb`
- `evaluation-phase-testing.ipynb`
- `model-testing-1.ipynb`
- `model-testing-2.ipynb`
- `single-task-model-testing.ipynb`

---

## Figures

Figures used in the report/paper are stored in `figures/`:
- `figures/figure1.png`
- `figures/figure2.png`
- `figures/figure3.png`
- `figures/figure4.png`

---

## Results and Discussion

All results are reported on a strict temporal **March 2025 test split** using the same feature set for the hierarchical multi-task VAE (HVAE) and the corresponding single-task baselines.

### Experimental Setup

**Dataset:** 5,772,527 valid trips (Apr 2024 → Mar 2025)  
**Temporal split (leakage-safe):**

| Split | Date range | Trips | Share |
|---|---|---:|---:|
| Train | Apr 2024 – Jan 2025 | 4,681,394 | 81.1% |
| Val | Feb 2025 | 486,732 | 8.4% |
| Test | Mar 2025 | 604,401 | 10.5% |

**Tasks evaluated (test set):**
- **Duration prediction:** regression on `log1p(duration_minutes)`
- **Demand contribution:** regression on normalized target in \[0, 1\]
- **Rideable type:** 3-class classification (electric bike, classic bike, electric scooter)

**Baselines (single-task):** XGBoost (duration), Ridge regression (demand), Random Forest (rideable type)

---

### Test Set Performance: HVAE vs Single-Task Baselines

| Task and Metric | HVAE | Baseline |
|---|---:|---:|
| **Duration prediction (log1p minutes)** |  |  |
| MAE | **0.186** | 0.277 |
| RMSE | **0.237** | 0.354 |
| R² | **0.905** | 0.788 |
| **Demand contribution (normalized)** |  |  |
| MAE | **0.0087** | 0.0124 |
| RMSE | **0.0163** | 0.0241 |
| R² | **0.723** | 0.589 |
| **Rideable type classification** |  |  |
| Accuracy (%) | **89.86** | 84.17 |
| Macro F1 | **0.809** | — |
| Weighted F1 | **0.899** | — |

**Summary of gains (test set):**
- Duration: **~32.9% lower MAE** and **~33.1% lower RMSE**, with higher R² (0.905 vs 0.788).
- Demand: **~29.8% lower MAE** and **~32.4% lower RMSE**, with higher R² (0.723 vs 0.589).
- Rideable type: **+5.69 percentage points** accuracy improvement (89.86% vs 84.17%).

---

### Rideable Type Classification: Confusion Matrix (Test Set, N = 604,401)

| True \ Pred | E-Bike | Classic | Scooter | Total |
|---|---:|---:|---:|---:|
| **Electric Bike** | 287,421 | 26,143 | 5,648 | 319,212 |
| **Classic Bike** | 21,842 | 245,127 | 2,613 | 269,582 |
| **Electric Scooter** | 3,217 | 1,824 | 10,566 | 15,607 |
| **Total** | 312,480 | 273,094 | 18,827 | 604,401 |

Per-class F1: **E-Bike 0.910**, **Classic 0.903**, **Scooter 0.614** (Macro F1 = 0.809).  
The dominant error mode is **electric vs classic bike confusion**, while scooter performance is lower due to strong class imbalance and more heterogeneous usage.

---

### Latent Space Clustering: Discovered Trip “Intent” Modes (K = 8)

We extracted the **global latent** representation and applied **k-means (K = 8)** on the test set to discover interpretable behavioral modes.

| Cluster | Trips (%) | Avg Dur (min) | Casual (%) | Weekend (%) | E-Bike (%) | Roundtrip (%) | Peak Hr (%) | Interpretation |
|---|---:|---:|---:|---:|---:|---:|---:|---|
| C0 | 18.7 | 8.2 | 12.4 | 8.1 | 51.3 | 2.1 | 68.4 | Weekday Commute |
| C1 | 15.3 | 24.6 | 78.9 | 89.4 | 62.1 | 4.7 | 12.3 | Weekend Leisure |
| C2 | 14.1 | 42.3 | 91.2 | 67.8 | 58.4 | 8.9 | 8.7 | Tourist Exploration |
| C3 | 12.8 | 6.1 | 15.7 | 11.2 | 47.8 | 1.8 | 61.2 | Short Errands |
| C4 | 11.4 | 15.7 | 34.2 | 31.5 | 54.2 | 3.4 | 38.1 | Mixed Purpose |
| C5 | 10.9 | 11.3 | 18.9 | 14.6 | 49.1 | 67.8 | 21.4 | Roundtrip Exercise |
| C6 | 9.2 | 7.4 | 9.8 | 7.3 | 45.2 | 1.6 | 74.3 | Rush Hour Transit |
| C7 | 7.6 | 38.1 | 86.4 | 73.2 | 61.7 | 12.3 | 11.2 | Lakefront Recreation |

These clusters align with expected mobility patterns (e.g., commute vs leisure), despite using **no explicit intent labels**.

---

### Reconstruction-Based Anomaly Scoring (Unsupervised)

We use the HVAE reconstruction-based anomaly score as an unsupervised signal for unusual trips. The test-set score distribution is right-skewed; practical operating points can be set via percentiles:

| Threshold | Score | Trips flagged | Share |
|---|---:|---:|---:|
| p95 | 2.57 | 30,221 | 5.00% |
| p99 | 4.87 | 6,045 | 1.00% |

We define anomalies as trips above **p99** and manually reviewed the **top 200** highest-scoring trips:

| Anomaly Type | Count | % |
|---|---:|---:|
| Temporal anomalies | 68 | 34.0 |
| Behavioral anomalies | 56 | 28.0 |
| Spatial anomalies | 42 | 21.0 |
| Vehicle mismatch | 34 | 17.0 |
| **Total** | 200 | 100.0 |

---

### Key Takeaways

- **Joint multi-task learning improves accuracy across tasks**: HVAE outperforms single-task baselines for duration, demand contribution, and rideable type classification.
- **The learned latent space is behaviorally meaningful**: clustering the global latent codes recovers commute-, leisure-, and tourism-like modes consistent with real-world usage.
- **Reconstruction scores provide an operational anomaly signal**: percentile thresholds (p95/p99) offer a simple, tunable method to flag rare trips for review.


## Author Contributions

**Birva Sevak:** Exploratory Data Analysis, HVAE model design and implementation, training and optimization, latent-space clustering and interpretability analysis, anomaly scoring analysis, and manuscript editing.

**Shrenik Jadhav:** Data preprocessing pipeline (Phases 1--3), feature engineering, baseline model training, experiment execution, results analysis, and manuscript writing.
