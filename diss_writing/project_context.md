# Project Context Brief (for ChatGPT)

## 1) Research Question
How can an explainable, news-augmented valuation model predict firm `total_assets` robustly across time and across unseen companies, while remaining grounded in financial accounting structure?

Primary objectives:
- Build a valuation model using fundamentals + news sentiment.
- Improve robustness versus direct prediction by using residual learning around an accounting baseline.
- Evaluate both temporal generalization and unseen-ticker generalization.
- Write a dissertation-ready methodology section that matches the implemented pipeline.

## 2) Dataset Scope
- Main dataset: `data/processed/main_dataset.csv`
- Unit of analysis: one row per company-period (annual financial statement observations with `ticker`, `fiscal_year`, `period_end` metadata).
- Target variable: `total_assets`
- Core structured inputs: financial ratios and raw accounting fields (assets, liabilities, equity, income, cash-flow derived variables).
- News inputs:
  - News attached by ticker-year from `data/raw/news/<TICKER>/news.jsonl` via `src/data_fetch/news/attach_news_to_tickers.py`
  - Per-item sentiment scored using FinBERT (`ProsusAI/finbert`) via `src/data_fetch/news/score_news_sentiment_finbert.py`
  - Sentiment scores stored in `news_sentiment_score` and aggregated into engineered numeric news features during training.

## 3) Model Pipeline (Implementation-Aligned)
Implemented files:
- `src/models/final_valuation.py`
- `src/models/residual_mlp_valuation.py`
- `src/models/news_driven_mlp_valuation.py`
- `src/models/evaluate_final_testsets.py`
- `src/models/xgb_valuation.py`
- Shared helpers: `src/models/valuation.py`, `src/models/feature_engineering.py`

Pipeline steps:
1. Load dataset, build engineered news features, and create accounting identity baseline feature:
   - `identity_base_assets = total_liabilities + total_equity`
2. Apply time-aware split for train/validation (latest fiscal year(s) for validation when possible).
3. Select numeric feature columns, apply optional correlation-based feature selection, and optional PCA (disabled by default in final config).
4. Apply `log1p` transform to selected magnitude-type features and standardize features.
5. Train residual target:
   - `residual = target_scale(total_assets) - target_scale(identity_base_assets)`
6. Fit regularized MLP:
   - Huber loss, Adam optimizer, dropout, L1/L2 penalties
   - Early stopping + ReduceLROnPlateau scheduler
7. Reconstruct predictions:
   - `pred_target = baseline + predicted_residual`
   - Apply output clipping and tiny-asset regime safeguard (use baseline for very small-asset cases).
8. Save artifacts:
   - training history, selected features, scalers, predictions CSV, run summary JSON.

## 4) Evaluation Protocol
Validation and comparison:
- Internal validation uses time-aware split.
- External strict test settings use `src/models/evaluate_final_testsets.py`:
  - Out-of-time test set
  - Unseen-ticker test set

Baselines:
- Accounting identity baseline (`total_liabilities + total_equity`) is always evaluated.

Metrics reported:
- RMSE (log and raw)
- R² (log and raw)
- MAPE
- Non-zero MAPE
- SMAPE
- Tail-risk diagnostics (high relative-error rates and quantiles)

Additional diagnostics:
- Feature selection and PCA stats
- Tiny-asset regime counts
- Prediction-level outputs for auditability

## 5) What I Want ChatGPT To Help With Next
I want help writing my dissertation methodology section.

Please help with:
- Writing dissertation-quality text for each methodology subsection.
- Keeping all content strictly aligned with the implemented pipeline above.
- Producing both:
  - clean narrative paragraphs, and
  - Overleaf-ready LaTeX subsections.
- Highlighting assumptions/placeholders when data is missing instead of inventing details.

Writing preferences:
- Formal academic tone.
- Clear justification of each methodological choice.
- Explicit linkage between methodology, implementation, and evaluation design.
