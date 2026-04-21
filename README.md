# XAI Investment Banking

This repository is a dissertation project for explainable investment-banking style modelling.
It combines:

- valuation modelling from fundamentals and news
- risk modelling from market data and forward drawdown labels
- SHAP-based explainability for both workstreams
- reporting, plots, and statistical comparison outputs

The codebase is organised around `data_fetch -> dataset building -> model training -> evaluation -> explainability`.

## What This Repo Produces

There are three main output families:

1. Processed datasets in `data/processed/`
   - joined valuation datasets
   - market-feature merges
   - valuation target datasets
   - risk-ready datasets

2. Experiment artifacts in `experiments/valuation/` and `experiments/risk/`
   - `run_summary.json` for headline metrics and config
   - `predictions.csv` for row-level predictions
   - `history.json` for training curves
   - saved model/preprocessing files such as `.npz` and `xgb_model.json`
   - comparison plots and metrics tables

3. Explainability artifacts in `experiments/*/SHAP/`
   - SHAP value arrays
   - feature-importance CSVs
   - beeswarm and bar plots
   - local explanation tables and PNGs

## Current Repo State

This checkout already contains experiment outputs, but it is also mid-refactor.

- The most complete valuation outputs are in `experiments/valuation/runs/valuation2_artifacts/` and `experiments/valuation/runs/xgb_valuation_artifacts/`.
- The most complete risk outputs are in `experiments/risk/runs/mlp_risk_artifacts/`.
- Many scripts still expect `data/processed/main_dataset.csv`, but that file is not present in this checkout.
  The closest current derived datasets are:
  - `data/processed/main_dataset_expanded_fundamentals.csv`
  - `data/processed/main_dataset_with_market.csv`
  - `data/processed/valuation/main_dataset_valuation_ready.csv`
- Some legacy evaluation modules still reference older paths or missing modules such as `src.models.valuation.valuation`, `final_valuation`, `exp_xgb_valuation`, and `residual_mlp_valuation_artifacts`.
- Some older risk training code still points to `data/nprocessed/risk/...`, but the datasets now written in this repo are under `data/processed/risk/...`.

If you are navigating the repo for development, treat the `valuation2`, `xgb_valuation`, `mlp_risk`, `prepare_datasets`, `plots_risk`, `shap_valuation`, and `shap_risk` paths as the most up-to-date source files.

## High-Level Pipeline

### 1. Fundamentals

- Fetch raw annual statements from Massive into `data/raw/balance-sheets/`, `data/raw/income-statements/`, and `data/raw/cash-flow-statements/`.
- Turn those raw statements into ratio features, raw accounting features, and expanded fundamentals.

### 2. News

- Fetch ticker-level news into `data/raw/news/<TICKER>/`.
- Attach news back onto the main dataset by ticker and year.
- Score each attached news item with FinBERT and write sentiment lists back into the dataset.

### 3. Market Data

- Download or store daily flatfiles in `data/raw/market_data/`.
- Build rolling volatility, return, liquidity, and drawdown features.
- Merge those features back into the main company-year dataset.

### 4. Targets

- Valuation work currently predicts `total_assets` in the main model training scripts.
- A separate valuation-target builder also creates market-cap and enterprise-value style targets in `data/processed/valuation/`.
- Risk work builds forward 1-year maximum drawdown labels and converts them into a model-ready `drawdown_severity`.

### 5. Modelling

- Valuation models live in `src/models/valuation/`.
- Risk models live in `src/models/risk/`.
- Shared neural-network code lives in `src/models/nn/`.

### 6. Reporting and Explainability

- Evaluation scripts write metrics tables, comparison plots, test-set summaries, and statistical test outputs.
- SHAP scripts write per-model explanation artifacts plus cross-model comparison plots.

## Repository Map

This section focuses on the source files and artifact folders that matter.
Generated files such as `__pycache__`, `.DS_Store`, and `xai_investment_banking.egg-info/` are omitted from the descriptions below.


### `data/`

| Path | What is stored there |
| --- | --- |
| `data/raw/tickers/final_tickers.txt` | Ticker universe downloaded from Massive. |
| `data/raw/balance-sheets/`, `income-statements/`, `cash-flow-statements/` | Raw annual statement JSON dumps per ticker. |
| `data/raw/news/<TICKER>/` | Raw news articles and metadata per ticker. |
| `data/raw/market_data/` | Daily market flatfiles plus download summaries and missing-ticker logs. |
| `data/processed/main_dataset_expanded_fundamentals.csv` | Expanded fundamentals dataset with extra accounting features and growth features. |
| `data/processed/main_dataset_with_market.csv` | Main dataset after market features have been merged in. |
| `data/processed/valuation/` | Valuation-target dataset outputs such as market cap and enterprise value targets. |
| `data/processed/risk/` | Risk-preparation outputs such as `main_with_market.csv` and per-split risk datasets. |
| `data/processed/market/` | Intermediate market-feature merge files and drawdown-target merges. |
| `data/processed/final_valuation_artifacts/` | Stored final valuation test predictions and summary JSON from an older valuation path. |

### `src/data_fetch/fundamental/`

| File | Purpose | Main outputs |
| --- | --- | --- |
| `ticker.py` | Downloads the ticker universe from Massive and writes the final ticker list. | `data/raw/tickers/final_tickers.txt` |
| `fetch_data_raw.py` | Fetches raw annual balance sheet, income statement, and cash-flow data for all tickers. | `data/raw/balance-sheets/*.json`, `data/raw/income-statements/*.json`, `data/raw/cash-flow-statements/*.json` |
| `massive_client.py` | Reusable Massive REST client with caching, retries, and pagination. | Cached JSON under `data/raw/` when used |
| `build_training_dataset.py` | Earlier feature-engineering script that builds a ratio-focused dataset from raw statements. | By default `data/processed/final_features_dataset.csv` |
| `expand_main_dataset_fundamentals.py` | Adds a much richer set of raw accounting, per-share, margin, and growth features onto the main dataset. | `data/processed/main_dataset_expanded_fundamentals.csv`, `data/processed/main_dataset_expanded_fundamentals_summary.json`, `data/processed/fundamental/main_dataset_fundamental_expansion_features.csv` |

### `src/data_fetch/news/`

| File | Purpose | Main outputs |
| --- | --- | --- |
| `news_client.py` | Fetches news from Massive for one ticker or a whole ticker list. | `data/raw/news/<TICKER>/news.jsonl`, `data/raw/news/<TICKER>/meta.json` |
| `attach_news_to_tickers.py` | Matches stored news back to dataset rows by ticker and fiscal year. By default it updates the dataset in place. | Adds `news_count`, `news_description`, `news_published_utc`, `news_year` to the dataset |
| `score_news_sentiment_finbert.py` | Scores each attached news item with FinBERT and stores one sentiment score per article as a JSON list string. | Adds `news_sentiment_score` to the dataset |

### `src/data_fetch/market/`

| File | Purpose | Main outputs |
| --- | --- | --- |
| `fetch_market_data.py` | Legacy one-off market-data downloader. The downloaded flatfiles already present in this repo live under `data/raw/market_data/`. | Daily `.csv.gz` market files |
| `build_market_feature_merge.py` | Builds rolling market features for each observation date from daily market bars. | `data/processed/market/market_feature_merge.csv`, `data/processed/market/market_feature_merge_summary.json` |
| `merge_market_feature_into_main.py` | Merges the market-feature table back into the main dataset without overwriting existing columns. | `data/processed/main_dataset_with_market.csv`, `data/processed/main_dataset_with_market_summary.json` |
| `build_future_drawdown.py` | Computes forward 1-year drawdown labels from market bars and merges them into the market-feature dataset. | Updates market-feature dataset with `future_*` target columns, writes `data/processed/market/market_feature_merge_drawdown_summary.json` |
| `build_valuation_target_dataset.py` | Aligns anchor prices and share-count data to create market-cap and enterprise-value style valuation targets. | `data/processed/valuation/main_dataset_valuation_ready.csv`, `data/processed/valuation/main_dataset_valuation_targets.csv`, `data/processed/valuation/main_dataset_valuation_target_summary.json` |

### `src/models/nn/`

These files implement the custom NumPy-based neural-network stack used by the MLP models.

| File | Purpose |
| --- | --- |
| `activation.py` | Activation functions. |
| `layers.py` | Core layer implementations. |
| `losses.py` | Loss functions including Huber loss. |
| `mlp.py` | MLP model definition. |
| `optimizer.py` | Optimisers such as Adam. |
| `sequential.py` | Sequential network utilities. |
| `train.py` | Training loop, early stopping, and learning-rate scheduling logic. |

### `src/models/valuation/`

| File | Purpose | Main outputs |
| --- | --- | --- |
| `xgb_valuation.py` | Main XGBoost valuation benchmark. Predicts `total_assets` from numeric company features. | `experiments/valuation/runs/xgb_valuation_artifacts/predictions.csv`, `run_summary.json`, `xgb_model.json` |
| `valuation2.py` | Main current MLP valuation model. It expands news sentiment aggregates and trains an MLP to predict `total_assets`. | `experiments/valuation/runs/valuation2_artifacts/history.json`, `predictions.csv`, `run_summary.json`, `scalers_and_features.npz` |
| `news_driven_mlp_valuation.py` | More elaborate valuation MLP with identity baseline, engineered news features, optional feature selection, PCA, and production-safety guards. | `experiments/valuation/runs/news_driven_mlp_valuation_artifacts/` |

### `src/models/risk/`

| File | Purpose | Main outputs |
| --- | --- | --- |
| `dataset_utils.py` | Shared dataset loading, normalisation, JSON writing, and observation helpers for risk prep. | Utility only. |
| `build_future_drawdown_targets.py` | Core logic for detecting market-data layout and computing forward drawdown targets from daily bars. | Used by drawdown target builders. |
| `build_risk_dataset.py` | Converts a dataset with `future_1y_max_drawdown` into a model-ready risk dataset with `drawdown_severity`. | By code default `data/nprocessed/risk/...`, but current prepared datasets are under `data/processed/risk/` |
| `prepare_datasets.py` | Current end-to-end risk dataset builder for main, out-of-time, and unseen-ticker splits. This is the safest risk-prep entry point in the repo. | `data/processed/risk/*_with_market.csv`, `*_with_market_targets.csv`, `*_risk_dataset.csv`, plus matching summaries |
| `modeling.py` | Shared risk modelling code, metrics, feature selection, and XGBoost helpers. | Utility only. |
| `mlp_risk.py` | Main current MLP risk model for drawdown severity. | `experiments/risk/runs/mlp_risk_artifacts/predictions.csv`, `feature_profile.csv`, `history.json`, `scalers_and_features.npz`, `model_weights.npz`, `run_summary.json` |
| `xgb_risk.py` | XGBoost benchmark for risk severity. | Expected output: `experiments/risk/runs/xgb_risk_artifacts/` |

### `src/evaluation/`

| File | Purpose | Main outputs |
| --- | --- | --- |
| `metrics.py` | Shared evaluation metrics. | Utility only. |
| `plots.py` | Valuation validation comparison plots and metrics tables. Some references are still legacy. | `experiments/valuation/metrics/*.csv`, `experiments/valuation/plots/*.png` |
| `plots_risk.py` | Risk validation comparison plots and metrics tables. | `experiments/risk/metrics/*.csv`, `experiments/risk/plots/*.png` |
| `evaluate_valuation2_testsets.py` | Evaluates valuation MLP on out-of-time and unseen-ticker splits. | Test prediction CSVs and summary JSON under a chosen output dir |
| `evaluate_xgb_testsets.py` | Evaluates valuation XGBoost on out-of-time and unseen-ticker splits. Some imports still point to legacy helper modules. | `experiments/valuation/runs/xgb_evaluation_artifacts/` |
| `evaluate_final_testsets.py` | Older final-valuation evaluation script using a legacy valuation path. | Final valuation test-set artifacts |
| `evaluate_risk_testsets.py` | Evaluates risk XGBoost on out-of-time and unseen-ticker risk datasets. | `experiments/risk/runs/xgb_risk_evaluation_artifacts/` |
| `evaluate_risk_mlp_testsets.py` | Risk MLP test-set evaluation counterpart. | Risk MLP test-set artifacts |
| `custom_valuation.py` | Huber-delta sweep for the older valuation MLP setup. | `data/processed/
| `pca_report.py` | Produces PCA loadings, explained variance, and scores for the valuation feature pipeline. | `experiments/valuation/runs/pca_report_artifacts/` |
| `stat_tests/valuation.py` | Pairwise statistical comparison for valuation predictions. | `experiments/valuation/runs/stat_tests_validation_artifacts/` |
| `stat_tests/risk.py` | Pairwise statistical comparison for risk predictions. | `experiments/risk/runs/stat_tests_validation_artifacts/` |
| `stat_tests/__main__.py` | Small dispatcher for `valuation` vs `risk` statistical tests. | Delegates to the files above |

### `src/explainability/`

| File | Purpose | Main outputs |
| --- | --- | --- |
| `shap_valuation.py` | Computes SHAP artifacts for valuation models (`valuation2`, `xgb`). | `experiments/valuation/SHAP/<model>/...` and `experiments/valuation/SHAP/valuation_shap_summary.json` |
| `plot_shap_valuation.py` | Builds cross-model valuation SHAP comparison plots plus per-model dependence and local plots. | `experiments/valuation/SHAP/plots/` and subplots inside each model folder |
| `shap_risk.py` | Computes SHAP artifacts for risk models (`mlp`, optionally `xgb`). | `experiments/risk/SHAP/<model>/...` and `experiments/risk/SHAP/risk_shap_summary.json` |
| `plot_shap_risk.py` | Builds cross-model and per-model risk SHAP plots. | `experiments/risk/SHAP/plots/` and per-model subplots |

### `src/scripts/`

| File | Purpose |
| --- | --- |
| `train_risk.py` | Thin wrapper that dispatches to `mlp_risk` or `xgb_risk` via `--model`. |
| `export_top.py` | Writes the top 100 rows of the main dataset to `data/processed/top100_main_dataset.csv`. |

### Other Source Files

| File | Purpose |
| --- | --- |
| `src/models/feature_engineering.py` | Feature selection and PCA helpers used by valuation experiments. |
| `src/demo.py` | Currently empty placeholder. |

## Experiment and Result Folders

### Valuation

| Folder | What it contains |
| --- | --- |
| `experiments/valuation/runs/valuation2_artifacts/` | Main current MLP valuation training artifacts. |
| `experiments/valuation/runs/xgb_valuation_artifacts/` | Main XGBoost valuation benchmark artifacts. |
| `experiments/valuation/runs/xgb_evaluation_artifacts/` | Out-of-time and unseen-ticker valuation test predictions plus summary JSON. |
| `experiments/valuation/runs/valuation2_dynamic_reduce_artifacts/` | Alternative `valuation2` run with a different scheduler setup. |
| `experiments/valuation/runs/valuation_artifacts/` | Older valuation artifact folder retained in the repo. |
| `experiments/valuation/runs/pca_report_artifacts/` | PCA loadings, explained variance, and score tables. |
| `experiments/valuation/runs/stat_tests_validation_artifacts/` | Bootstrap deltas, pairwise tests, calibration tables, and aligned prediction CSVs. |
| `experiments/valuation/metrics/` | Aggregated validation metrics tables. |
| `experiments/valuation/plots/` | Validation comparison PNGs. |
| `experiments/valuation/SHAP/valuation2/` | SHAP outputs for the valuation MLP. |
| `experiments/valuation/SHAP/xgb/` | SHAP outputs for the valuation XGBoost model. |
| `experiments/valuation/SHAP/plots/` | Cross-model SHAP comparison plots. |

### Risk

| Folder | What it contains |
| --- | --- |
| `experiments/risk/runs/mlp_risk_artifacts/` | Main current MLP risk training artifacts. |
| `experiments/risk/runs/stat_tests_validation_artifacts/` | Statistical testing outputs for risk validation predictions. |
| `experiments/risk/metrics/` | Aggregated validation metrics tables for risk models. |
| `experiments/risk/plots/` | Validation performance and error-distribution plots for risk models. |
| `experiments/risk/SHAP/mlp/` | SHAP outputs for the risk MLP. |
| `experiments/risk/SHAP/xgb/` | XGBoost SHAP artifacts retained in the repo. |
| `experiments/risk/SHAP/plots/` | Cross-model SHAP comparison plots. |

## Current Stored Results In This Checkout

These are the clearest currently checked-in experiment summaries:

### Valuation

- `experiments/valuation/runs/valuation2_artifacts/run_summary.json`
  - validation RMSE on log target: `0.3584`
  - validation R2 on log target: `0.9822`
  - validation raw MAPE: `0.2583`
- `experiments/valuation/runs/xgb_valuation_artifacts/run_summary.json`
  - validation RMSE on log target: `0.0638`
  - validation R2 on log target: `0.9994`
  - validation raw MAPE: `0.0310`
- `experiments/valuation/metrics/validation_metrics_table.csv`
  - aggregate comparison table used by the valuation plots
- `experiments/valuation/SHAP/valuation_shap_summary.json`
  - consolidated valuation SHAP metadata for `valuation2` and `xgb`

### Risk

- `experiments/risk/runs/mlp_risk_artifacts/run_summary.json`
  - validation RMSE: `0.1464`
  - validation MAE: `0.1110`
  - validation Spearman: `0.7889`
- `experiments/risk/metrics/validation_metrics_table.csv`
  - aggregate comparison table used by the risk plots
- `experiments/risk/SHAP/risk_shap_summary.json`
  - consolidated SHAP metadata for the risk MLP currently registered in the summary

## How To Read The Output Files

| File type | Meaning |
| --- | --- |
| `run_summary.json` | Headline metrics, training config, feature list, and artifact paths for a model run. |
| `predictions.csv` | Row-level predictions. Usually includes `split`, `y_true`, `y_pred`, and sometimes baseline predictions. |
| `history.json` | Training loss and learning-rate history. |
| `feature_profile.csv` | Per-feature metadata written by risk models. |
| `scalers_and_features.npz` | Saved preprocessing objects and feature ordering for MLP inference. |
| `model_weights.npz` | Saved MLP parameter arrays. |
| `xgb_model.json` | Saved XGBoost booster. |
| `*_summary.json` | Dataset-build or evaluation summary files. |
| `validation_shap_values.npz` | Saved SHAP arrays and metadata. |
| `validation_shap_feature_importance.csv` | Global SHAP importance table. |
| `validation_shap_top_error_local_explanations.csv` | Local explanation table for selected difficult examples. |
| `plots/*.png` | Presentation-ready performance or explainability figures. |
| `metrics/*.csv` | Reporting tables used in the dissertation write-up. |

## Suggested Entry Points

If you want the shortest route into the repo, start here:

- Valuation training: `src/models/valuation/valuation2.py`
- Valuation benchmark: `src/models/valuation/xgb_valuation.py`
- Risk dataset prep: `src/models/risk/prepare_datasets.py`
- Risk training: `src/models/risk/mlp_risk.py`
- Valuation SHAP: `src/explainability/shap_valuation.py`
- Risk SHAP: `src/explainability/shap_risk.py`
- Valuation reporting: `src/evaluation/plots.py`
- Risk reporting: `src/evaluation/plots_risk.py`

## Setup

Typical local setup is:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```



