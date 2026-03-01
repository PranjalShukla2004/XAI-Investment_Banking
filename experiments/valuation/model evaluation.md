# Model Evaluation (Valuation)

## 1. Objective and Scope
This document defines a rigorous and reproducible evaluation framework for valuation models in this project.

The target variable is `total_assets`.
The main objective is to evaluate predictive accuracy and robustness of the following model families:

- MLP (Direct): MLP predicting `total_assets` directly.
- MLP (Residual): MLP predicting residuals around accounting identity baseline.
- XGBoost: tree-based baseline model.
- News-Driven Residual MLP: residual MLP augmented with engineered sentiment features.

The framework is designed for dissertation use, so it emphasizes:

- strict temporal splitting,
- strong baselines,
- multiple complementary metrics,
- statistical confidence (not only point estimates),
- subgroup error analysis,
- ablation and robustness testing,
- explicit validity limitations.

## 2. Current Result Snapshot
The latest comparison table from `experiments/valuation/metrics/validation_metrics_table.csv` is:

| Model | Val MAPE (raw) | Val RMSE (log) | Val RMSE (raw) | Val R2 (raw) | Val R2 (log) |
|---|---:|---:|---:|---:|---:|
| MLP (Direct) | 0.156677 | 0.270701 | 39,793,923,859.79 | 0.923460 | 0.989824 |
| XGBoost | 0.030601 | 0.064823 | 23,510,837,248.00 | 0.973283 | 0.999416 |
| MLP (Residual) | 0.014184 | 0.068728 | 21,320,162,724.35 | 0.978030 | 0.999344 |

Additional identity-baseline metrics from `data/processed/residual_mlp_valuation_artifacts/run_summary.json`:

- Baseline formula: `total_liabilities + total_equity`
- Val MAPE (raw): `0.008460`
- Val RMSE (raw): `202,252,288.0`
- Val R2 (raw): `0.999998`

Interpretation:

- Residual MLP dramatically outperforms direct MLP.
- Residual MLP improves over XGBoost on raw-space metrics in this run.
- The accounting identity baseline is exceptionally strong for this target and must be treated as a primary benchmark.

## 3. Fixed Evaluation Protocol
### 3.1 Data Split Strategy
Use one fixed split protocol for all models:

- Primary split: time-aware split by `fiscal_year`.
- Validation period: latest year (or latest two years when latest-year rows are below threshold, as implemented).
- No random reshuffling between model families.

Rationale:

- Avoids look-ahead bias.
- Ensures comparable train/validation distributions across models.

### 3.2 Leakage Controls
- Feature computation must not use validation labels.
- If `period_end`-based logic is used, enforce that engineered features are derivable at prediction time.
- For sentiment fields, avoid future leakage from post-period records if the timestamp source is not event time.

### 3.3 Consistent Preprocessing
All models should use consistent handling of:

- missing values,
- infinities/outliers,
- optional log transforms,
- sample weighting,
- clipping strategy in prediction space.

Any model-specific preprocessing should be explicitly documented and justified.

### 3.4 Reproducibility Controls
For each experiment run, store:

- config,
- random seed,
- split stats,
- feature list,
- scalers,
- full metric dictionary,
- predictions file.

Use fixed seed values and log all non-default hyperparameters.

## 4. Baselines and Model Set
### 4.1 Baselines (must include)
1. Mean baseline in target-transform space.
2. Accounting identity baseline:
   `total_assets_hat = total_liabilities + total_equity`.

### 4.2 Learning Models
1. MLP (Direct).
2. XGBoost.
3. MLP (Residual):
   - Base estimate `b = total_liabilities + total_equity`.
   - Residual target `r = y - b` (or in log-space when `log_target=True`).
   - Final prediction `y_hat = b + r_hat`.
4. News-Driven Residual MLP:
   - Same residual target setup,
   - plus engineered numeric sentiment features.

## 5. Metric Suite (Primary and Secondary)
### 5.1 Primary Metric
Primary metric for business-facing interpretation:

- `MAPE_raw = mean(|y_hat - y| / max(y, eps))`

Reason:

- Percentage error is easier to interpret for valuation impact and cross-scale comparability.

### 5.2 Secondary Metrics
Report all of the following:

- `RMSE_raw = sqrt(mean((y_hat - y)^2))`
- `R2_raw = 1 - SSE/SST`
- `RMSE_log` and `R2_log` (in transformed space)
- Factor interpretation for log RMSE: `exp(RMSE_log)`

Why both spaces:

- Log-space reflects multiplicative consistency and stabilizes heavy tails.
- Raw-space reflects real monetary error magnitude.

## 6. Statistical Confidence and Significance
Point estimates alone are insufficient.

### 6.1 Bootstrap Confidence Intervals
For each model and each key metric (especially MAPE_raw):

- Resample validation rows with replacement.
- Recommended bootstrap iterations: 1,000 to 5,000.
- Compute 95% CI from percentile bounds (2.5%, 97.5%).

Report format example:

- `Val MAPE_raw = 0.0142 [95% CI: 0.0128, 0.0159]`.

### 6.2 Paired Significance Tests
When comparing two models A and B:

- Compute row-wise absolute percentage errors on identical validation rows.
- Test paired differences with Wilcoxon signed-rank test.
- Report p-value and effect size (median error difference).

This guards against over-interpreting small point-estimate differences.

## 7. Error Analysis by Subgroup
Global metrics can hide failure modes.

### 7.1 Size-Stratified Evaluation
Bucket validation samples into size quantiles using true `total_assets`.

For each bucket report:

- count,
- MAPE,
- median relative error,
- RMSE,
- mean true,
- mean predicted.

This is essential because earlier diagnostics showed direct MLP underperforming heavily in small-firm regimes.

### 7.2 Time-Stratified Evaluation
Evaluate by fiscal year in validation period:

- metric drift across years,
- potential regime sensitivity.

### 7.3 Optional Domain Stratification
If sector/industry metadata is available:

- report per-sector metrics,
- identify where residual or sentiment features help/hurt.

## 8. Ablation Study Design
Ablation should isolate what causes improvement.

### 8.1 Core Ablations
Run at least:

1. Direct MLP vs Residual MLP.
2. Residual MLP without news vs with news features.
3. Residual MLP with and without sample weighting.
4. Residual MLP with and without quantile clipping.
5. Huber delta sweep (e.g., 0.05, 0.1, 0.2, 0.5, 1.0).

### 8.2 Reporting Ablation Results
Use a single table with:

- configuration ID,
- changed factor,
- primary metric,
- secondary metrics,
- relative delta vs control configuration.

Interpret only changes that are consistent across seeds and statistically supported.

## 9. Robustness and Stability Tests
### 9.1 Multi-Seed Stability
Repeat each final candidate model with multiple seeds (recommended 5 to 10).

Report:

- mean and std of key metrics,
- min/max performance,
- seed sensitivity ranking.

### 9.2 Distribution Robustness
Check for instability under:

- outlier trimming changes,
- alternate clipping quantiles,
- mild feature perturbations.

A robust model should not collapse under small preprocessing changes.

## 10. Plot Requirements for Dissertation
The current generated plots in `experiments/valuation/plots` are suitable foundations:

- `validation_performance_comparison.png`
- `validation_relative_error_quantiles.png`
- `validation_relative_error_cdf.png`

Recommended final figure set:

1. Core metric bar chart (MAPE, RMSE, R2 across models).
2. Relative-error quantile chart (p50, p75, p90, p95).
3. Relative-error CDF plot.
4. Predicted vs true scatter (validation only).
5. Residual distribution histogram.
6. Size-bucket error bar chart.

All figures should include:

- axis labels with units,
- consistent color mapping by model,
- concise captions with interpretation.

## 11. Dissertation Narrative (Findings Template)
A defensible narrative aligned with current numbers:

- Direct MLP underperforms tree and residual strategies.
- Residualization around accounting identity is the dominant performance driver.
- News sentiment features must be justified by ablation; they should not be assumed helpful by default.
- Identity baseline itself is very strong, implying the valuation target is strongly constrained by accounting structure.

Example statement:

"Replacing direct target learning with residual learning around `total_liabilities + total_equity` reduced validation MAPE from 15.67% to 1.42%, indicating that most predictable structure is captured by accounting identity and residual modeling is a better inductive bias for this target."

## 12. Threats to Validity
Include a dedicated subsection covering:

1. Target-definition dependency:
   `total_assets` is algebraically tied to liabilities and equity, making baseline unusually strong.
2. Metric dependence:
   model ranking can differ between log-space and raw-space metrics.
3. Data quality and timestamp ambiguity:
   sentiment timestamps may not represent true publication time.
4. External validity:
   results may not transfer to targets with weaker accounting identities.
5. Split sensitivity:
   performance may vary under alternative temporal boundaries.

## 13. Recommended Final Reporting Table Set
For dissertation clarity, include these tables:

1. Main model comparison table.
2. Bootstrap CI table for primary metric.
3. Paired significance table (model vs model).
4. Size-bucket diagnostics table.
5. Ablation table.
6. Multi-seed robustness table.

## 14. Minimum Checklist Before Final Write-Up
- Use one fixed split across all models.
- Include accounting identity baseline in all comparisons.
- Report both raw and log metrics.
- Provide confidence intervals and paired tests.
- Provide subgroup error analysis.
- Provide ablations for every claimed improvement.
- Provide multi-seed robustness.
- Include threats-to-validity discussion.

## 15. File References (Current Artifacts)
- Metrics table: `experiments/valuation/metrics/validation_metrics_table.csv`
- Error quantiles table: `experiments/valuation/metrics/validation_relative_error_quantiles.csv`
- Residual MLP summary: `data/processed/residual_mlp_valuation_artifacts/run_summary.json`
- XGBoost summary: `data/processed/xgb_valuation_artifacts/run_summary.json`
- Direct MLP summary: `data/processed/valuation_artifacts/run_summary.json`

