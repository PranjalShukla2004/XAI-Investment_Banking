# Handoff Summary

## 1) Project goal
- Build an explainable investment-banking modeling pipeline with two parallel tracks:
- a valuation track that now targets real valuation labels rather than only accounting size
- a risk track that predicts future 1-year drawdown severity

## 2) Current state
- The repo now contains working valuation, risk, statistical-testing, and explainability pipelines.
- The valuation codebase has two distinct layers:
- legacy / benchmark accounting-size models around `total_assets`
- the current real-valuation pipeline around `market_cap_target`
- The risk pipeline is no longer paused. Both XGBoost and MLP risk models are implemented and runnable.
- SHAP explainability now exists for both valuation and risk.

## 3) What is implemented

### Valuation data and target building
- Fundamental expansion script:
- `src/data_fetch/fundamental/expand_main_dataset_fundamentals.py`
- It expands `data/processed/main_dataset.csv` into:
- `data/processed/main_dataset_expanded_fundamentals.csv`
- It adds EBITDA, gross profit, SG&A, depreciation, retained earnings, per-share features, growth features, cash-flow fields, and missingness flags.

- Valuation target builder:
- `src/data_fetch/market/build_valuation_target_dataset.py`
- It builds:
- `data/processed/valuation/main_dataset_valuation_ready.csv`
- `data/processed/valuation/main_dataset_valuation_targets.csv`
- Current target columns include:
- `market_cap_target`
- `log_market_cap_target`
- `enterprise_value_target`
- `enterprise_value_target_clipped`
- `log_enterprise_value_target`

### Valuation models
- Baseline direct MLP:
- `src/models/valuation/valuation.py`
- Target:
- `total_assets`
- Dataset:
- `data/processed/main_dataset.csv`

- Benchmark XGB:
- `src/models/valuation/xgb_valuation.py`
- Target:
- `total_assets`
- Important note:
- this is not apples-to-apples with the current real-valuation `final_valuation.py`

- Residual valuation MLP:
- `src/models/valuation/residual_mlp_valuation.py`
- Uses accounting identity residualization around `total_liabilities + total_equity`

- Anchor-only residual ablation:
- `src/models/valuation/anchor_residual_valuation.py`
- Baseline is used only as an anchor
- Accounting leakage features are excluded from `X`
- Writes:
- `experiments/valuation/runs/anchor_residual_valuation_artifacts/`

- Scheduler experimentation runner:
- `src/models/valuation/valuation2.py`
- Uses the same simple MLP stack as `valuation.py`
- Writes to:
- `experiments/valuation/runs/valuation2_artifacts/`
- Supports:
- `IncreaseLRonPlateau`
- `DynamicLRonPlateau`

- Current primary real-valuation MLP:
- `src/models/valuation/final_valuation.py`
- Dataset:
- `data/processed/valuation/main_dataset_valuation_ready.csv`
- Target:
- `market_cap_target`
- Current training mode:
- `log_multiple`
- Current default denominator:
- `revenue`
- Reconstruction:
- `exp(predicted_log_multiple) * revenue`
- It filters:
- `valuation_target_usable == True`
- It excludes target-derived leakage columns before feature selection.
- It uses:
- engineered news features
- correlation-based feature selection
- optional PCA hook (implemented but disabled by default)
- `increase_on_plateau` scheduler by default
- Current default artifact directory:
- `experiments/valuation/runs/final_valuation_artifacts/`

### Shared neural-network training
- `src/models/nn/train.py`
- Added scheduler support for:
- `reduce_on_plateau`
- `increase_on_plateau`
- `dynamic_on_plateau`
- Tracks:
- LR reductions
- LR increases
- scheduler switches
- scheduler mode history

### Valuation evaluation and statistical testing
- Evaluation entrypoints:
- `src/evaluation/evaluate_xgb_testsets.py`
- `src/evaluation/evaluate_valuation2_testsets.py`
- `src/evaluation/evaluate_residual_testsets.py`
- `src/evaluation/evaluate_final_testsets.py`

- Statistical testing entrypoint:
- `src/evaluation/stat_tests/__main__.py`
- Main implementations:
- `src/evaluation/stat_tests/valuation.py`
- `src/evaluation/stat_tests/risk.py`
- It computes:
- aligned prediction tables
- headline metrics
- paired bootstrap confidence intervals
- pairwise bootstrap deltas
- paired Wilcoxon and paired t-tests with correction
- Mincer-Zarnowitz calibration
- forecast-bias tests
- size-bucket Kruskal / Dunn tables
- Current saved artifacts:
- `experiments/valuation/runs/stat_tests_validation_artifacts/`
- `experiments/risk/runs/stat_tests_validation_artifacts/`

### Valuation explainability
- Existing valuation SHAP runner:
- `src/explainability/shap_valuation.py`
- It now explains:
- `valuation2`
- `xgb`
- It no longer uses the residual valuation model by default.
- Existing valuation plot generator:
- `src/explainability/plot_shap_valuation.py`
- Current saved valuation SHAP artifacts:
- `experiments/valuation/SHAP/`

### Risk pipeline
- Shared risk modeling utilities:
- `src/models/risk/modeling.py`
- Canonical current default data path resolution:
- prefers `data/processed/risk/main_risk_dataset.csv`
- otherwise falls back to `data/nprocessed/risk/risk_dataset.csv`

- XGB risk model:
- `src/models/risk/xgb_risk.py`
- Artifact directory:
- `experiments/risk/runs/xgb_risk_artifacts/`

- MLP risk model:
- `src/models/risk/mlp_risk.py`
- Artifact directory:
- `experiments/risk/runs/mlp_risk_artifacts/`

- Risk dataset preparation:
- `src/models/risk/prepare_datasets.py`
- This builds processed split-specific risk datasets from valuation splits if needed.
- Important note:
- the user already has a legacy usable risk dataset at:
- `data/nprocessed/risk/risk_dataset.csv`
- The MLP and XGB risk trainers already work directly with that legacy path.

- Risk evaluation:
- `src/evaluation/evaluate_risk_testsets.py`
- `src/evaluation/evaluate_risk_mlp_testsets.py`

- Train script:
- `src/scripts/train_risk.py`
- Dispatches between:
- `xgb`
- `mlp`

### Risk explainability
- Risk SHAP runner:
- `src/explainability/shap_risk.py`
- Risk SHAP plot generator:
- `src/explainability/plot_shap_risk.py`
- Current saved risk SHAP artifacts:
- `experiments/risk/SHAP/`
- Includes:
- per-model SHAP summaries
- beeswarm plots
- importance bars
- dependence plots
- local explanation plots
- comparison plots at the root `plots/` level

## 4) Key current datasets
- Legacy valuation base dataset:
- `data/processed/main_dataset.csv`

- Expanded valuation-feature dataset:
- `data/processed/main_dataset_expanded_fundamentals.csv`

- Current valuation-ready dataset:
- `data/processed/valuation/main_dataset_valuation_ready.csv`

- Current legacy risk dataset used in practice:
- `data/nprocessed/risk/risk_dataset.csv`

## 5) Current important artifact directories
- Valuation baseline artifacts:
- `experiments/valuation/runs/valuation_artifacts/`

- Real-valuation MLP artifacts:
- `experiments/valuation/runs/final_valuation_artifacts/`

- Scheduler experiment artifacts:
- `experiments/valuation/runs/valuation2_artifacts/`
- `experiments/valuation/runs/valuation2_increase_artifacts/`
- `experiments/valuation/runs/valuation2_dynamic_reduce_artifacts/`
- `experiments/valuation/runs/valuation2_evaluation_artifacts/`

- Anchor residual ablation:
- `experiments/valuation/runs/anchor_residual_valuation_artifacts/`

- XGB held-out evaluation:
- `experiments/valuation/runs/xgb_evaluation_artifacts/`

- Valuation statistical tests:
- `experiments/valuation/runs/stat_tests_validation_artifacts/`

- Valuation SHAP:
- `experiments/valuation/SHAP/`

- Risk XGB:
- `experiments/risk/runs/xgb_risk_artifacts/`

- Risk MLP:
- `experiments/risk/runs/mlp_risk_artifacts/`

- Risk SHAP:
- `experiments/risk/SHAP/`
- Risk statistical tests:
- `experiments/risk/runs/stat_tests_validation_artifacts/`

## 6) Current model interpretation
- `valuation.py` is still a simple accounting-size baseline on `total_assets`.
- `final_valuation.py` is the main real-valuation pipeline and should be treated as the current valuation model of record.
- `xgb_valuation.py` remains useful as a benchmark, but its saved metrics are on a different target unless explicitly retrained on the valuation-ready dataset.
- Risk is a continuous regression task on `drawdown_severity`, not a classification task by default.

## 7) Known caveats
- The saved `xgb_valuation` benchmark is on `total_assets`, so direct metric comparison against current `final_valuation.py` is not apples-to-apples.
- The valuation `stat_tests` module was built against saved valuation prediction artifacts and baseline conventions from the older valuation stack. If it is used for the newer real-valuation models, verify column compatibility first.
- The risk `stat_tests` module now compares aligned validation predictions from `mlp_risk_artifacts` and `xgb_risk_artifacts`, and includes the saved `y_pred_baseline` as the current-drawdown baseline by default.
- Full split-specific processed risk datasets are optional and can be generated, but most current risk work has been run successfully from `data/nprocessed/risk/risk_dataset.csv`.
- Risk ROC-AUC / F1 style evaluation has not yet been added. It is valid only after defining binary event labels from the continuous drawdown target.

## 8) Commands to run

### Build valuation-ready data
```bash
python -m src.data_fetch.fundamental.expand_main_dataset_fundamentals
python -m src.data_fetch.market.build_valuation_target_dataset
```

### Train valuation models
```bash
python -m src.models.valuation.valuation
python -m src.models.valuation.valuation2
python -m src.models.valuation.anchor_residual_valuation
python -m src.models.valuation.final_valuation
```

### Run valuation held-out evaluation
```bash
/opt/miniconda3/envs/diss/bin/python -m src.evaluation.evaluate_xgb_testsets
/opt/miniconda3/envs/diss/bin/python -m src.evaluation.evaluate_valuation2_testsets
```

### Run valuation statistics / explainability
```bash
python -m src.evaluation.stat_tests
python -m src.evaluation.stat_tests valuation
python -m src.evaluation.stat_tests risk
python -m src.explainability.shap_valuation
python -m src.explainability.plot_shap_valuation
```

### Train risk models
```bash
python -m src.models.risk.xgb_risk --data data/nprocessed/risk/risk_dataset.csv
python -m src.models.risk.mlp_risk --data data/nprocessed/risk/risk_dataset.csv
python src/scripts/train_risk.py --model mlp --data data/nprocessed/risk/risk_dataset.csv
```

### Run risk explainability
```bash
python -m src.evaluation.stat_tests.risk
python -m src.explainability.shap_risk --out-dir experiments/risk/SHAP
python -m src.explainability.plot_shap_risk --shap-dir experiments/risk/SHAP
```

## 9) Key files to read first
- `src/models/valuation/final_valuation.py`
- `src/data_fetch/fundamental/expand_main_dataset_fundamentals.py`
- `src/data_fetch/market/build_valuation_target_dataset.py`
- `src/evaluation/stat_tests/valuation.py`
- `src/evaluation/stat_tests/risk.py`
- `src/models/risk/modeling.py`
- `src/models/risk/mlp_risk.py`
- `src/models/risk/xgb_risk.py`
- `src/explainability/shap_risk.py`
- `src/explainability/plot_shap_risk.py`

## 10) Recommended next-step orientation
- If continuing valuation work:
- treat `final_valuation.py` as the main model
- treat `valuation.py` and `xgb_valuation.py` as older baselines / benchmarks
- be careful not to compare `total_assets` metrics directly with `market_cap_target` metrics

- If continuing risk work:
- keep using `data/nprocessed/risk/risk_dataset.csv` unless split-specific processed datasets are explicitly needed
- SHAP and plots are already implemented for both risk MLP and risk XGB
- classification metrics such as ROC-AUC or F1 should only be added after explicitly thresholding the continuous drawdown target
