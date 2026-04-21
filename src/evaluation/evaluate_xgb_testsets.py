from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import xgboost as xgb

from src.models.valuation.exp_xgb_valuation import (
    ExpXGBValuationConfig,
    _deduplicate_entity_period_rows,
    _ensure_dir,
    _jsonable,
    _make_asset_weights_from_log_target,
    _mape,
    _r2,
    _rmse,
    _select_feature_columns,
)


def _normalize_tickers(series: pd.Series) -> pd.Series:
    out = series.astype(str).str.upper().str.strip()
    return out.replace({"": pd.NA, "NAN": pd.NA, "NONE": pd.NA, "<NA>": pd.NA, "NULL": pd.NA})


def _target_to_model_scale(df: pd.DataFrame, cfg: ExpXGBValuationConfig) -> np.ndarray:
    y = (
        pd.to_numeric(df[cfg.target_col], errors="coerce")
        .replace([np.inf, -np.inf], np.nan)
        .fillna(0.0)
        .to_numpy(dtype=np.float32)
    )
    if cfg.log_target:
        y = np.log1p(np.clip(y, a_min=0.0, a_max=None))
    return y.reshape(-1, 1)


def _features_from_cols(df: pd.DataFrame, feature_cols: List[str]) -> np.ndarray:
    return (
        df.reindex(columns=feature_cols)
        .replace([np.inf, -np.inf], np.nan)
        .fillna(0.0)
        .to_numpy(dtype=np.float32)
    )


def _inner_train_val_indices(train_df: pd.DataFrame, seed: int, val_ratio: float = 0.1) -> tuple[np.ndarray, np.ndarray, str]:
    n_rows = len(train_df)
    if n_rows < 2:
        idx = np.arange(n_rows)
        return idx, idx, "single_row"

    n_val_target = max(1, int(round(n_rows * float(val_ratio))))
    if "ticker" in train_df.columns:
        tickers = _normalize_tickers(train_df["ticker"])
        valid_tickers = tickers.loc[tickers.notna()]
        uniq = valid_tickers.unique()
        if uniq.size > 1:
            rng = np.random.default_rng(seed)
            shuffled = np.array(uniq, dtype=object)
            rng.shuffle(shuffled)
            counts = valid_tickers.value_counts().to_dict()
            chosen: List[str] = []
            rows_accum = 0
            for t in shuffled:
                ts = str(t)
                chosen.append(ts)
                rows_accum += int(counts.get(ts, 0))
                if rows_accum >= n_val_target and len(chosen) < uniq.size:
                    break
            val_mask = tickers.isin(chosen).fillna(False).to_numpy()
            if np.any(val_mask) and np.any(~val_mask):
                val_idx = np.flatnonzero(val_mask)
                fit_idx = np.flatnonzero(~val_mask)
                return fit_idx, val_idx, "group_ticker"

    idx = np.arange(n_rows)
    rng = np.random.default_rng(seed)
    rng.shuffle(idx)
    val_idx = idx[:n_val_target]
    fit_idx = idx[n_val_target:]
    if fit_idx.size == 0:
        fit_idx = val_idx
    return fit_idx, val_idx, "row"


def _train_and_eval_xgb(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    cfg: ExpXGBValuationConfig,
) -> Dict[str, object]:
    feature_cols = _select_feature_columns(train_df, cfg.target_col, cfg.time_col)
    X_train = _features_from_cols(train_df, feature_cols)
    X_test = _features_from_cols(test_df, feature_cols)
    y_train = _target_to_model_scale(train_df, cfg)
    y_test = _target_to_model_scale(test_df, cfg)

    fit_idx, val_idx, inner_split_mode = _inner_train_val_indices(train_df, seed=cfg.random_seed, val_ratio=0.1)

    X_fit = X_train[fit_idx]
    y_fit = y_train[fit_idx].reshape(-1).astype(np.float32)
    X_val_in = X_train[val_idx]
    y_val_in = y_train[val_idx].reshape(-1).astype(np.float32)

    w_fit = None
    if cfg.use_sample_weights:
        w_full = _make_asset_weights_from_log_target(y_train, power=cfg.weight_power, w_min=cfg.w_min, w_max=cfg.w_max)
        w_fit = w_full[fit_idx]

    dfit = xgb.DMatrix(X_fit, label=y_fit, weight=w_fit)
    dval_in = xgb.DMatrix(X_val_in, label=y_val_in)
    dtest = xgb.DMatrix(X_test, label=y_test.reshape(-1).astype(np.float32))
    dtrain_full = xgb.DMatrix(X_train, label=y_train.reshape(-1).astype(np.float32))

    params = {
        "objective": "reg:squarederror",
        "eval_metric": "rmse",
        "tree_method": cfg.tree_method,
        "max_depth": cfg.max_depth,
        "eta": cfg.eta,
        "subsample": cfg.subsample,
        "colsample_bytree": cfg.colsample_bytree,
        "lambda": cfg.reg_lambda,
        "alpha": cfg.reg_alpha,
        "min_child_weight": cfg.min_child_weight,
        "gamma": cfg.gamma,
        "seed": cfg.random_seed,
        "nthread": cfg.nthread,
    }

    booster = xgb.train(
        params=params,
        dtrain=dfit,
        num_boost_round=cfg.num_boost_round,
        evals=[(dfit, "fit"), (dval_in, "inner_val")],
        early_stopping_rounds=cfg.early_stopping_rounds,
        verbose_eval=False,
    )
    best_iteration = int(getattr(booster, "best_iteration", booster.num_boosted_rounds() - 1))
    it_range = (0, best_iteration + 1)

    yhat_train_log_preclip = booster.predict(dtrain_full, iteration_range=it_range).reshape(-1, 1)
    yhat_test_log_preclip = booster.predict(dtest, iteration_range=it_range).reshape(-1, 1)

    if cfg.log_target:
        clip_lo = float(np.log1p(max(float(cfg.min_target_raw), 0.0)))
    else:
        clip_lo = float(max(float(cfg.min_target_raw), 0.0))
    clip_hi = float(np.max(y_train))
    if cfg.use_quantile_clip:
        clip_hi = float(np.quantile(y_train, cfg.clip_q_hi))

    yhat_train_log = np.clip(yhat_train_log_preclip, clip_lo, clip_hi)
    yhat_test_log = np.clip(yhat_test_log_preclip, clip_lo, clip_hi)

    if cfg.log_target:
        y_train_raw = np.expm1(y_train)
        y_test_raw = np.expm1(y_test)
        yhat_train_raw = np.expm1(yhat_train_log)
        yhat_test_raw = np.expm1(yhat_test_log)
    else:
        y_train_raw = y_train
        y_test_raw = y_test
        yhat_train_raw = yhat_train_log
        yhat_test_raw = yhat_test_log

    y_mean_log = float(np.mean(y_train))
    yhat_base_test_log = np.full_like(y_test, y_mean_log)
    if cfg.log_target:
        yhat_base_test_raw = np.expm1(yhat_base_test_log)
    else:
        yhat_base_test_raw = yhat_base_test_log

    metrics = {
        "rows_test": int(len(test_df)),
        "unique_tickers_test": int(_normalize_tickers(test_df["ticker"]).nunique()) if "ticker" in test_df.columns else None,
        "feature_count": int(len(feature_cols)),
        "inner_split_mode": inner_split_mode,
        "best_iteration": best_iteration,
        "rmse_log_preclip": float(_rmse(y_test, yhat_test_log_preclip)),
        "r2_log_preclip": float(_r2(y_test, yhat_test_log_preclip)),
        "rmse_log": float(_rmse(y_test, yhat_test_log)),
        "r2_log": float(_r2(y_test, yhat_test_log)),
        "rmse_raw": float(_rmse(y_test_raw, yhat_test_raw)),
        "r2_raw": float(_r2(y_test_raw, yhat_test_raw)),
        "mape_raw": float(_mape(y_test_raw, yhat_test_raw)),
        "baseline_rmse_log": float(_rmse(y_test, yhat_base_test_log)),
        "baseline_r2_log": float(_r2(y_test, yhat_base_test_log)),
        "baseline_rmse_raw": float(_rmse(y_test_raw, yhat_base_test_raw)),
        "baseline_r2_raw": float(_r2(y_test_raw, yhat_base_test_raw)),
        "baseline_mape_raw": float(_mape(y_test_raw, yhat_base_test_raw)),
        "train_rmse_log_preclip": float(_rmse(y_train, yhat_train_log_preclip)),
        "train_rmse_log": float(_rmse(y_train, yhat_train_log)),
    }

    preds = test_df.copy()
    preds["y_true"] = y_test_raw.reshape(-1)
    preds["y_pred"] = yhat_test_raw.reshape(-1)
    preds["y_pred_preclip"] = (np.expm1(yhat_test_log_preclip).reshape(-1) if cfg.log_target else yhat_test_log_preclip.reshape(-1))
    preds["y_pred_baseline"] = yhat_base_test_raw.reshape(-1)
    return {
        "metrics": metrics,
        "predictions": preds,
        "feature_names": feature_cols,
        "clip": {"use_quantile_clip": bool(cfg.use_quantile_clip), "q_hi": float(cfg.clip_q_hi), "min_target_raw": float(cfg.min_target_raw), "clip_lo": clip_lo, "clip_hi": clip_hi},
    }


def _prepare_df(df: pd.DataFrame, cfg: ExpXGBValuationConfig) -> tuple[pd.DataFrame, Dict[str, object]]:
    if cfg.enable_deduplicate_entity_period:
        return _deduplicate_entity_period_rows(df, cfg.dedup_ticker_col, cfg.dedup_period_col, cfg.time_col)
    return df.copy(), {"enabled": False, "applied": False, "rows_before": int(len(df)), "rows_after": int(len(df)), "rows_dropped": 0}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--main", type=str, default="data/processed/main_dataset.csv")
    ap.add_argument("--test-out-of-time", type=str, default="data/processed/test_dataset_out_of_time.csv")
    ap.add_argument("--test-unseen-tickers", type=str, default="data/processed/test_dataset_unseen_tickers.csv")
    ap.add_argument("--out-dir", type=str, default="experiments/valuation/runs/xgb_evaluation_artifacts")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--use-sample-weights", action="store_true")
    args = ap.parse_args()

    cfg = ExpXGBValuationConfig()
    cfg.data_path = Path(args.main)
    cfg.random_seed = int(args.seed)
    cfg.use_sample_weights = bool(args.use_sample_weights)

    main_df = pd.read_csv(args.main)
    oot_df = pd.read_csv(args.test_out_of_time)
    unseen_df = pd.read_csv(args.test_unseen_tickers)

    main_prepped, main_dedup = _prepare_df(main_df, cfg)
    oot_prepped, oot_dedup = _prepare_df(oot_df, cfg)
    unseen_prepped, unseen_dedup = _prepare_df(unseen_df, cfg)

    out_dir = Path(args.out_dir)
    _ensure_dir(out_dir)

    oot_years = sorted(pd.to_numeric(oot_prepped["fiscal_year"], errors="coerce").dropna().astype(int).unique().tolist()) if "fiscal_year" in oot_prepped.columns else []
    train_oot = main_prepped[~main_prepped["fiscal_year"].isin(oot_years)].copy() if "fiscal_year" in main_prepped.columns else main_prepped.copy()
    res_oot = _train_and_eval_xgb(train_oot, oot_prepped, cfg)
    oot_preds_path = out_dir / "xgb_test_out_of_time_predictions.csv"
    res_oot["predictions"].to_csv(oot_preds_path, index=False)

    unseen_tickers = set(_normalize_tickers(unseen_prepped["ticker"]).dropna()) if "ticker" in unseen_prepped.columns else set()
    train_unseen = (
        main_prepped[~_normalize_tickers(main_prepped["ticker"]).isin(unseen_tickers)].copy()
        if "ticker" in main_prepped.columns and unseen_tickers
        else main_prepped.copy()
    )
    res_unseen = _train_and_eval_xgb(train_unseen, unseen_prepped, cfg)
    unseen_preds_path = out_dir / "xgb_test_unseen_tickers_predictions.csv"
    res_unseen["predictions"].to_csv(unseen_preds_path, index=False)

    summary = {
        "config": {k: _jsonable(v) for k, v in cfg.__dict__.items()},
        "inputs": {
            "main_dataset": args.main,
            "test_out_of_time": args.test_out_of_time,
            "test_unseen_tickers": args.test_unseen_tickers,
        },
        "dedup": {
            "main": main_dedup,
            "out_of_time": oot_dedup,
            "unseen_tickers": unseen_dedup,
        },
        "out_of_time": {
            "years": oot_years,
            "train_rows": int(len(train_oot)),
            "test_metrics": res_oot["metrics"],
            "clip": res_oot["clip"],
            "predictions_path": str(oot_preds_path),
        },
        "unseen_tickers": {
            "heldout_tickers": int(len(unseen_tickers)),
            "train_rows": int(len(train_unseen)),
            "test_metrics": res_unseen["metrics"],
            "clip": res_unseen["clip"],
            "predictions_path": str(unseen_preds_path),
        },
    }

    summary_path = out_dir / "xgb_test_set_evaluation_summary.json"
    with summary_path.open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    print("Saved:")
    print(f"- {summary_path}")
    print(f"- {oot_preds_path}")
    print(f"- {unseen_preds_path}")
    print("\nXGB test metrics:")
    print(
        f"out_of_time rmse_log={summary['out_of_time']['test_metrics']['rmse_log']:.6f} | "
        f"rmse_log_preclip={summary['out_of_time']['test_metrics']['rmse_log_preclip']:.6f} | "
        f"r2_raw={summary['out_of_time']['test_metrics']['r2_raw']:.6f}"
    )
    print(
        f"unseen_ticker rmse_log={summary['unseen_tickers']['test_metrics']['rmse_log']:.6f} | "
        f"rmse_log_preclip={summary['unseen_tickers']['test_metrics']['rmse_log_preclip']:.6f} | "
        f"r2_raw={summary['unseen_tickers']['test_metrics']['r2_raw']:.6f}"
    )


if __name__ == "__main__":
    main()
