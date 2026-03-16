# Handoff Summary

## 1) Project goal (what you're building)
- Build an explainable investment-banking modelling pipeline with:
- a valuation model that predicts firm value / `total_assets` from fundamentals + news features
- a risk model that will predict upcoming 1-year drawdown using market data for the same ticker universe

## 2) Current task (what you were doing just before context ended)
- The current priority is now statistical testing for the valuation-model results.
- The goal is to produce dissertation-ready statistical evidence for model performance, not just descriptive metrics.
- The main comparison targets are:
- `residual_mlp` as the primary model
- `xgb` as the benchmark model
- the accounting-identity baseline where relevant
- The immediate work should focus on testing whether observed performance differences are statistically meaningful on the valuation test sets.
- The risk-model pipeline remains paused.

## 3) What is already done
- Valuation codebase was refactored into:
- `src/models/valuation/`
- `src/evaluation/`

- Run artifacts were moved to:
- `experiments/valuation/runs/`

- Existing valuation training/evaluation modules now include:
- `src/models/valuation/xgb_valuation.py`
- `src/models/valuation/exp_xgb_valuation.py`
- `src/models/valuation/residual_mlp_valuation.py`
- `src/models/valuation/news_driven_mlp_valuation.py`
- `src/models/valuation/final_valuation.py`
- `src/evaluation/evaluate_xgb_testsets.py`
- `src/evaluation/evaluate_xgb_groupcv_testsets.py`
- `src/evaluation/evaluate_residual_testsets.py`
- `src/evaluation/evaluate_final_testsets.py`
- `src/evaluation/cross_validate_news_driven_mlp.py`

- A dedicated XGB evaluator was created:
- `src/evaluation/evaluate_xgb_testsets.py`
- It retrains XGB separately for:
- out-of-time test set
- unseen-ticker test set
- It avoids using the actual test set for early stopping by creating an internal train-only validation split.
- It writes:
- predictions CSVs for both test sets
- `xgb_test_set_evaluation_summary.json`

- A separate GroupCV XGB training module was created:
- `src/models/valuation/exp_xgb_groupcv_valuation.py`
- It adds:
- `GroupKFold` by ticker
- optional ticker-frequency weighting
- optional asset weighting
- ratio-focused feature mode (`ratio_plus_news`)
- CV result export

- A separate GroupCV XGB test evaluator was created:
- `src/evaluation/evaluate_xgb_groupcv_testsets.py`
- It writes:
- predictions for each test split
- per-ticker error tables
- feature shift tables using PSI and KS
- model and CV artifacts for each split

- The plain XGB evaluator smoke run completed successfully and produced:
- `out_of_time rmse_log=0.057427`
- `unseen_ticker rmse_log=0.608905`
- Interpretation already established:
- out-of-time performance is strong
- unseen-ticker generalization is poor

- The GroupCV evaluator also ran successfully from a tooling perspective, but model quality degraded substantially.

## 4) How we have to fix it / pending
- The current priority is valuation-model statistical testing, not the risk-model pipeline.
- Treat `residual_mlp` as the main valuation pipeline.
- Treat `xgb` as a comparison / benchmark model only.

- Recommended statistical tests for the valuation dissertation workstream:
- paired bootstrap confidence intervals for key metrics such as RMSE, RMSE-log, MAPE, non-zero MAPE, sMAPE, and R-squared on the out-of-time and unseen-ticker test sets
- paired Wilcoxon signed-rank tests on per-observation error differences between models, using absolute error, absolute log error, and absolute percentage error as the main loss views
- paired t-tests only as a secondary robustness check if the paired error-difference distribution looks approximately symmetric; do not rely on the t-test alone because valuation errors are likely skewed
- Mincer-Zarnowitz style regression of `y_true` on `y_pred`, with joint testing of intercept = 0 and slope = 1, to assess calibration / forecast efficiency in a finance-friendly way
- one-sample bias tests on signed residuals to test whether mean or median forecast error is statistically different from zero
- Kruskal-Wallis tests across firm-size buckets on relative error measures, followed by a post-hoc procedure such as Dunn-Holm if the global test is significant
- multiple-comparison correction, preferably Holm or Benjamini-Hochberg, when many pairwise tests or subgroup tests are reported

- Tests that are likely beneficial for this use case:
- use bootstrap confidence intervals as the main uncertainty quantification layer because they work well for non-normal error metrics
- use Wilcoxon signed-rank as the main pairwise significance test between `residual_mlp`, `xgb`, and the baseline because predictions are paired at the same observation level
- use Mincer-Zarnowitz regression because it is dissertation-friendly and interpretable in a valuation / forecasting context
- use subgroup tests by firm-size bucket because valuation error behavior often varies strongly with scale

- Tests that should be optional rather than central:
- Diebold-Mariano can be considered for the out-of-time split if predictions are evaluated as a true ordered forecast sequence, but it is not the best primary test for the unseen-ticker split
- normality tests on residuals should not be a core result because with large samples they are rarely informative for model comparison

- Secondary valuation clean-up pending:
- remove obsolete / experimental valuation files under `src/models/valuation/`
- remove their paired evaluation / test scripts under `src/evaluation/`
- keep the focus on the baseline valuation pipeline files rather than maintaining duplicate experimental paths in parallel

- Risk-model work remains pending, but it should stay paused for now:
- do not resume risk-model work until the user explicitly asks for it later

## 5) Important constraints
- Do not break the current module layout under:
- `src/models/valuation/`
- `src/evaluation/`

- Keep valuation run outputs under:
- `experiments/valuation/runs/`

- Keep SHAP outputs under:
- `experiments/valuation/SHAP/`

- Preserve the current CLI pattern:
- `python -m src.models.valuation.<module>`
- `python -m src.evaluation.<module>`

- Do not silently move evaluation scripts back into `src/models/`.

- Avoid touching unrelated user changes in the worktree.

- Use separate runnable files for new experiments rather than rewriting the current baseline files in-place.

- For the next workstream, the risk model should be added cleanly as a parallel pipeline, not by entangling it with the valuation scripts.

## 6) Key files to read first
- [handoff.md](/Users/burphh/Desktop/Dissertation/XAI-Investment_Banking/handoff.md)
- [xgb_valuation.py](/Users/burphh/Desktop/Dissertation/XAI-Investment_Banking/src/models/valuation/xgb_valuation.py)
- [residual_mlp_valuation.py](/Users/burphh/Desktop/Dissertation/XAI-Investment_Banking/src/models/valuation/residual_mlp_valuation.py)
- [news_driven_mlp_valuation.py](/Users/burphh/Desktop/Dissertation/XAI-Investment_Banking/src/models/valuation/news_driven_mlp_valuation.py)
- [final_valuation.py](/Users/burphh/Desktop/Dissertation/XAI-Investment_Banking/src/models/valuation/final_valuation.py)
- [evaluate_xgb_testsets.py](/Users/burphh/Desktop/Dissertation/XAI-Investment_Banking/src/evaluation/evaluate_xgb_testsets.py)
- [evaluate_residual_testsets.py](/Users/burphh/Desktop/Dissertation/XAI-Investment_Banking/src/evaluation/evaluate_residual_testsets.py)
- [evaluate_final_testsets.py](/Users/burphh/Desktop/Dissertation/XAI-Investment_Banking/src/evaluation/evaluate_final_testsets.py)
- [metrics.py](/Users/burphh/Desktop/Dissertation/XAI-Investment_Banking/src/evaluation/metrics.py)

- Clean-up instruction to remember:
- remove experimental / duplicate files in `src/models/valuation/` and their corresponding evaluation / test files in `src/evaluation/` once the baseline pipeline is stabilized

## 7) Commands to run
- Main residual-MLP training run:
```bash
conda run -n diss python -m src.models.valuation.residual_mlp_valuation
```

- Main residual-MLP test evaluation:
```bash
conda run -n diss python -m src.evaluation.evaluate_residual_testsets \
  --out-dir experiments/valuation/runs/residual_mlp_testset_evaluation
```

- XGB benchmark training run:
```bash
conda run -n diss python -m src.models.valuation.xgb_valuation
```

- XGB benchmark test evaluation:
```bash
conda run -n diss python -m src.evaluation.evaluate_xgb_testsets \
  --out-dir experiments/valuation/runs/xgb_testset_evaluation
```

- Final valuation-model test evaluation:
```bash
conda run -n diss python -m src.evaluation.evaluate_final_testsets \
  --out-dir experiments/valuation/runs/final_model_testset_evaluation
```

- For statistical testing, use the prediction CSVs and summary JSONs produced by the evaluation scripts above as the input tables.

## 8) Known errors / logs
- The main pipeline is now `residual_mlp`, and current validation performance is strong:
- `val_rmse_log=0.067394 | val_r2_raw=0.978031 | val_mape_raw=0.013648`

- XGB remains useful as a benchmark / comparison model only:
- its SHAP output is valid for comparison, but it is not the main interpretation target

- Current gap:
- the codebase reports descriptive evaluation metrics, but it does not yet provide dissertation-ready statistical significance tests or confidence intervals for model comparison
- the next workstream should fill that gap using the saved per-observation prediction outputs

## 9) Next exact step
1. Keep the workstream on valuation-model statistical testing first.
2. Generate or refresh the prediction tables for:
- `residual_mlp`
- `xgb`
- the final valuation model if needed
- both out-of-time and unseen-ticker test sets
3. Build aligned paired comparison tables at the observation level so each model is compared on the same rows.
4. Implement and report the following dissertation-facing tests:
- paired bootstrap confidence intervals for headline metrics
- paired Wilcoxon signed-rank tests for model-vs-model error differences
- Mincer-Zarnowitz regression with joint coefficient tests
- bias tests on signed residuals
- firm-size bucket robustness tests with post-hoc correction if needed
5. Save final statistical outputs as clean tables that can be moved into the dissertation results chapter.
6. After that, clean up the valuation module scope:
- keep `xgb_valuation.py`, `residual_mlp_valuation.py`, and `news_driven_mlp_valuation.py` as the core valuation files
- remove obsolete experimental files in `src/models/valuation/`
- remove matching evaluation / test files in `src/evaluation/`
7. Do not move on to risk-model work automatically.
   Only when the user explicitly asks later should the workstream move on to the risk model.
