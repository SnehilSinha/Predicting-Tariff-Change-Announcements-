# Predicting Tariff Change Announcements Using Machine Learning & Geopolitical Event Data

Code accompanying the MSc thesis *Predicting Tariff Change Announcements Using Machine Learning
and Geopolitical Event Data* (DTU Compute, in collaboration with Novonesis).

The project forecasts whether a **trade-restricting tariff** will be announced on a trade
corridor within the next **30, 60, or 90 days**, using only open geopolitical event data as
predictive features and the Global Trade Alert register as the source of labels. The problem is
framed as imbalanced binary classification at the **corridor-month** level.

---

## Overview

The pipeline runs as a set of Jupyter notebooks, each covering one stage. The output of one
stage is the input to the next, so the notebooks are intended to be run **in the order below**.

| # | Notebook | Stage | Main inputs | Main outputs |
|---|----------|-------|-------------|--------------|
| 1 | `GDELT_Preprocessing.ipynb` | Extract the GDELT 2.0 event signal from BigQuery, filter to the corridors / trade-relevant CAMEO codes, de-duplicate to root events, and aggregate to corridor-month features | BigQuery `gdelt-bq.gdeltv2.events` | `data/gdelt_features_monthly.parquet`, `data/gdelt_corridor_trade.parquet` |
| 2 | `GDELT_EDA.ipynb` | Exploratory analysis of the GDELT 2.0 signal (volume, families, sparsity, missingness) | GDELT features / BigQuery | figures in `eda_figures/` |
| 3 | `GTA_Label_Construction.ipynb` | Build the prediction labels from the GTA extract (three-criterion funnel → 92 tariff events → 260 corridor-events), join to the features, write the modelling dataset | `data/GTA/interventions (4).csv`, GDELT features | `data/modelling_dataset.parquet` |
| 4 | `GTA_EDA.ipynb` | Exploratory analysis of the GTA labels (coverage, HS headings, announcement–implementation lag, class balance) | GTA extract | figures in `eda_figures/` |
| 5 | `Baseline_Model.ipynb` | Train the class-weighted, elastic-net logistic-regression baseline; rolling-origin backtest vs. trivial floors | `data/modelling_dataset.parquet` | `data/baseline_results.csv`, `data/baseline_backtest.csv` |
| 6 | `Baseline_Evaluation.ipynb` | Evaluate the baseline (metrics, confusion, coefficients, permutation importance, calibration, error analysis) | modelling dataset, baseline outputs | `data/baseline_eval_metrics.csv`, figures |
| 7 | `Advanced_Models.ipynb` | Train and compare the tree ensembles (HistGBM, random forest) against the baseline; SHAP attribution | `data/modelling_dataset.parquet` | `data/advanced_backtest.csv`, `data/advanced_test_comparison.csv`, figures |
| 8 | `POLECAT_Convergent_Validity.ipynb` | Convergent-validity check of the GDELT 2.0 signal against POLECAT (2018–2023) | POLECAT Dataverse files, GDELT features | `data/polecat_convergence_*.csv`, figures |
| 9 | `aggregation_denoising_proof.ipynb` | Supporting analysis: how aggregation to the corridor-month cancels much of the event-level coding error | GDELT features | figures |

All figures used in the thesis are written to `eda_figures/`.

### Repository structure

```
Codes/
├── GDELT_Preprocessing.ipynb        # 1  feature extraction
├── GDELT_EDA.ipynb                  # 2  GDELT EDA
├── GTA_Label_Construction.ipynb     # 3  label construction + join
├── GTA_EDA.ipynb                    # 4  GTA EDA
├── Baseline_Model.ipynb             # 5  baseline training / backtest
├── Baseline_Evaluation.ipynb        # 6  baseline evaluation
├── Advanced_Models.ipynb            # 7  tree ensembles + SHAP
├── POLECAT_Convergent_Validity.ipynb# 8  robustness / convergent validity
├── aggregation_denoising_proof.ipynb# 9  supporting proof
├── data/                            # inputs + derived datasets (see Datasets)
└── eda_figures/                     # generated figures
```

---

## Setup

### Requirements

- **Python 3.10+** and Jupyter (JupyterLab or the classic notebook).
- A **Google Cloud** account with access to BigQuery, only for the GDELT 2.0 extraction
  (notebooks 1–2). Once the derived Parquet files exist, the remaining notebooks run offline.

### Install

```bash
# (recommended) create an isolated environment
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate

pip install \
  pandas numpy scipy statsmodels scikit-learn shap matplotlib \
  pyarrow google-cloud-bigquery jupyter
```

Core libraries used across the notebooks: `pandas`, `numpy`, `scipy` /
`statsmodels` (rank correlations, variance-inflation factors), `scikit-learn`
(`LogisticRegression`, `HistGradientBoostingClassifier`, `RandomForestClassifier`,
`TimeSeriesSplit`, `GridSearchCV`, metrics, calibration), `shap` (feature attribution),
`matplotlib` (figures), `pyarrow` (Parquet), and `google-cloud-bigquery` (GDELT extraction).

> Exact versions are not pinned here. For an exact reproduction, pin the versions actually
> installed (e.g. via `pip freeze > requirements.txt`).

### BigQuery authentication (notebooks 1–2 only)

The GDELT 2.0 events are extracted from the public table `gdelt-bq.gdeltv2.events`. Before
running `GDELT_Preprocessing.ipynb`, authenticate and set a billing project:

```bash
gcloud auth application-default login
export GOOGLE_CLOUD_PROJECT="your-gcp-project-id"   # Windows: set GOOGLE_CLOUD_PROJECT=...
```

Every query is run under explicit cost governance (a dry-run byte estimate and a hard cap on
bytes billed), and results are cached locally as Parquet so the extraction is auditable and
safe to re-run.

### Running

Launch Jupyter and run the notebooks top-to-bottom in the numbered order above:

```bash
jupyter lab
```

A fixed random seed is used throughout the modelling notebooks for reproducibility. If the
derived files in `data/` are already present, you can start directly at notebook 5 to reproduce
the modelling results.

---

## Datasets

Three external data sources are used, each with a distinct role, plus internal company data that
is **not** included in this repository.

### External sources

| Source | Role | Location in repo | Notes |
|--------|------|------------------|-------|
| **GDELT 2.0 Events** | Predictive features (model inputs) | *not stored* — pulled from BigQuery `gdelt-bq.gdeltv2.events` | Global Database of Events, Language, and Tone; covers 18 Feb 2015 onward. Cloud-scale, extracted via notebook 1. |
| **Global Trade Alert (GTA)** | Prediction labels | `data/GTA/interventions (4).csv`, `data/gta_interventions_2107.csv` | Register of trade-policy interventions (~1,605 rows scoped to the study corridors / HS headings). Only "Red" tariff measures become positive labels. |
| **POLECAT** | Robustness / convergent-validity check only | `data/dataverse_files (1)/ngecEvents.DV.YYYY.txt` (2018–2024) | Political Event Classification, Attributes, and Types, from the Harvard Dataverse. Cached per year in `data/_polecat_cache/agg_YYYY.parquet`. Licensed for **non-commercial research use**. |
| **Novonesis internal trade-flow data** | Scope only (which corridors / products) | **excluded — confidential** | Used solely to select the relevant trade corridors and map products to HS headings. Never a model input or the target. |

### Derived datasets (generated by the notebooks)

| File | Description | Shape |
|------|-------------|-------|
| `data/gdelt_features_monthly.parquet` | Monthly corridor-level GDELT 2.0 feature matrix | 3,088 corridor-months × 25 variables |
| `data/gdelt_corridor_trade.parquet` | Trade-relevant GDELT 2.0 events aggregated per corridor | — |
| `data/modelling_dataset.parquet` | Final modelling dataset: 17 features + 3 labels (`y30`, `y60`, `y90`) + tier + corridor/month keys | 3,088 × 28 |
| `data/baseline_results.csv`, `data/baseline_backtest.csv`, `data/baseline_eval_metrics.csv` | Baseline single-split and rolling-origin results | — |
| `data/advanced_backtest.csv`, `data/advanced_test_comparison.csv` | Tree-ensemble comparison results | — |
| `data/polecat_convergence_pooled.csv`, `data/polecat_convergence_within.csv`, `data/polecat_convergence_percorridor.csv` | GDELT 2.0–POLECAT agreement (Spearman) | — |
| `data/corridor_counts.json` | Corridor-level event counts | — |

The corridor set covers 23 country pairs (17 Tier-1 corridors carrying Novonesis exposure, used
for all evaluation and reporting, plus 6 Tier-2 corridors added only to enrich the rare positive
class during training).

> **Confidentiality:** the internal Novonesis trade-flow dataset is not shared. All results in
> the thesis are reproducible from the open sources above; the internal data only fixed which
> corridors and products to study.
