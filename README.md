# ECE5831-2025 Final Project: Multi-Task Rider Behavior Modeling for Micromobility Systems

This repository implements an end-to-end pipeline for learning **shared latent representations** from the 2024–2025 Divvy/Cyclistic bike-share trip data and using them for **multi-task prediction** and **downstream analysis** (e.g., anomaly scoring and latent-space interpretation).

Core idea: instead of training separate models per task, we train a **hierarchical variational autoencoder (HVAE)** that captures global and trip-specific factors in a structured latent space, and decodes multiple behavioral outcomes jointly.

---

## Links

- **Pre-recorded presentation video:** [Watch here](PASTE_LINK_HERE)
- **Presentation slides:** [View slides](PASTE_LINK_HERE)
- **Final report:** [Read report](PASTE_LINK_HERE)
- **Dataset (raw):** 2024–2025 Divvy bike sharing data (Cyclistic) on Kaggle:  
  https://www.kaggle.com/datasets/miaadnabizadeh/20242025-divvy-bike-sharing-data-cyclistic
- **Demo video (pipeline / model inference):** [Watch demo](PASTE_LINK_HERE)


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
