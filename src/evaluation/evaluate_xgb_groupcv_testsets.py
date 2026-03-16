from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd
import xgboost as xgb

from src.models.valuation.exp_xgb_groupcv_valuation import (
    GroupCVXGBValuationConfig,
    features_from_cols,
    normalize_tickers,
    target_to_model_scale,
    train_groupcv_xgb,
)
from src.models.valuation.exp_xgb_valuation import (
    _deduplicate_entity_period_rows,
    _ensure_dir,
    _jsonable,
    _mape,
    _r2,
    _rmse,
)


def _prepare_df(df: pd.DataFrame, cfg: GroupCVXGBValuationConfig) -> tuple[pd.DataFrame, Dict[str, object]]:
    if cfg.enable_deduplicate_entity_period:
        return _deduplicate_entity_period_rows(df, cfg.dedup_ticker_col, cfg.dedup_period_col, cfg.time_col)
    return (
        df.copy(),
        {
            "enabled": False,
            "applied": False,
            "rows_before": int(len(df)),
            "rows_after": int(len(df)),
            "rows_dropped": 0,
        },
    )


def _ks_statistic(train_vals: np.ndarray, test_vals: np.ndarray) -> float:
    a = np.sort(train_vals.astype(np.float64))
    b = np.sort(test_vals.astype(np.float64))
    if a.size == 0 or b.size == 0:
        return float("nan")
    grid = np.concatenate([a, b])
    cdf_a = np.searchsorted(a, grid, side="right") / float(a.size)
    cdf_b = np.searchsorted(b, grid, side="right") / float(b.size)
    return float(np.max(np.abs(cdf_a - cdf_b)))


def _psi(train_vals: np.ndarray, test_vals: np.ndarray, bins: int = 10, eps: float = 1e-6) -> float:
    if train_vals.size == 0 or test_vals.size == 0:
        return float("nan")
    qs = np.linspace(0.0, 1.0, bins + 1)
    edges = np.quantile(train_vals, qs)
    edges = np.unique(edges)
    if edges.size < 3:
        return 0.0
    edges[0] = -np.inf
    edges[-1] = np.inf

    train_hist, _ = np.histogram(train_vals, bins=edges)
    test_hist, _ = np.histogram(test_vals, bins=edges)
    if train_hist.sum() == 0 or test_hist.sum() == 0:
        return float("nan")

    train_pct = np.clip(train_hist / float(train_hist.sum()), eps, None)
    test_pct = np.clip(test_hist / float(test_hist.sum()), eps, None)
    return float(np.sum((test_pct - train_pct) * np.log(test_pct / train_pct)))


def build_feature_shift_table(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    feature_cols: List[str],
) -> pd.DataFrame:
    rows: List[Dict[str, float]] = []
    for c in feature_cols:
        tr = pd.to_numeric(train_df[c], errors="coerce").replace([np.inf, -np.inf], np.nan)
        te = pd.to_numeric(test_df[c], errors="coerce").replace([np.inf, -np.inf], np.nan)
        tr_vals = tr.dropna().to_numpy(dtype=np.float64)
        te_vals = te.dropna().to_numpy(dtype=np.float64)
        rows.append(
            {
                "feature": c,
                "train_missing_rate": float(tr.isna().mean()),
                "test_missing_rate": float(te.isna().mean()),
                "train_mean": float(np.mean(tr_vals)) if tr_vals.size > 0 else float("nan"),
                "test_mean": float(np.mean(te_vals)) if te_vals.size > 0 else float("nan"),
                "train_std": float(np.std(tr_vals)) if tr_vals.size > 0 else float("nan"),
                "test_std": float(np.std(te_vals)) if te_vals.size > 0 else float("nan"),
                "ks_stat": _ks_statistic(tr_vals, te_vals),
                "psi": _psi(tr_vals, te_vals),
            }
        )
    out = pd.DataFrame(rows)
    if out.empty:
        return out
    return out.sort_values(["psi", "ks_stat"], ascending=[False, False], kind="mergesort").reset_index(drop=True)


def build_per_ticker_table(
    test_df: pd.DataFrame,
    y_true_log: np.ndarray,
    y_pred_log: np.ndarray,
    y_true_raw: np.ndarray,
    y_pred_raw: np.ndarray,
) -> pd.DataFrame:
    if "ticker" not in test_df.columns:
        return pd.DataFrame()
    tick = normalize_tickers(test_df["ticker"]).to_numpy()
    data = pd.DataFrame(
        {
            "ticker": tick,
            "y_true_log": y_true_log.reshape(-1),
            "y_pred_log": y_pred_log.reshape(-1),
            "y_true_raw": y_true_raw.reshape(-1),
            "y_pred_raw": y_pred_raw.reshape(-1),
        }
    )
    rows: List[Dict[str, float]] = []
    for t, g in data.groupby("ticker", sort=True):
        yt_log = g["y_true_log"].to_numpy(dtype=np.float64)
        yp_log = g["y_pred_log"].to_numpy(dtype=np.float64)
        yt_raw = g["y_true_raw"].to_numpy(dtype=np.float64)
        yp_raw = g["y_pred_raw"].to_numpy(dtype=np.float64)
        rows.append(
            {
                "ticker": str(t),
                "rows": int(len(g)),
                "rmse_log": float(np.sqrt(np.mean((yt_log - yp_log) ** 2))),
                "rmse_raw": float(np.sqrt(np.mean((yt_raw - yp_raw) ** 2))),
                "mape_raw": float(_mape(yt_raw.reshape(-1, 1), yp_raw.reshape(-1, 1))),
                "mean_true_raw": float(np.mean(yt_raw)),
                "mean_pred_raw": float(np.mean(yp_raw)),
            }
        )
    if not rows:
        return pd.DataFrame()
    return pd.DataFrame(rows).sort_values("rmse_log", ascending=False, kind="mergesort").reset_index(drop=True)


def _run_one_split(
    name: str,
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    cfg: GroupCVXGBValuationConfig,
    out_dir: Path,
) -> Dict[str, object]:
    split_dir = out_dir / name
    _ensure_dir(split_dir)

    train_info = train_groupcv_xgb(train_df, cfg)
    booster: xgb.Booster = train_info["booster"]  # type: ignore[assignment]
    feature_cols: List[str] = train_info["feature_names"]  # type: ignore[assignment]
    best_num_boost_round = int(train_info["metrics"]["best_num_boost_round"])  # type: ignore[index]
    clip_lo = float(train_info["clip_lo"])
    clip_hi = float(train_info["clip_hi"])

    X_test = features_from_cols(test_df, feature_cols)
    y_test = target_to_model_scale(test_df, cfg)
    dtest = xgb.DMatrix(X_test, label=y_test.reshape(-1).astype(np.float32))
    y_pred_log_preclip = booster.predict(dtest, iteration_range=(0, best_num_boost_round)).reshape(-1, 1)
    y_pred_log = np.clip(y_pred_log_preclip, clip_lo, clip_hi)

    if cfg.log_target:
        y_true_raw = np.expm1(y_test)
        y_pred_raw = np.expm1(y_pred_log)
        y_pred_raw_preclip = np.expm1(y_pred_log_preclip)
    else:
        y_true_raw = y_test
        y_pred_raw = y_pred_log
        y_pred_raw_preclip = y_pred_log_preclip

    y_mean_log = float(np.mean(target_to_model_scale(train_df, cfg)))
    y_pred_baseline_log = np.full_like(y_test, y_mean_log)
    y_pred_baseline_raw = np.expm1(y_pred_baseline_log) if cfg.log_target else y_pred_baseline_log

    metrics = {
        "rows_test": int(len(test_df)),
        "unique_tickers_test": int(normalize_tickers(test_df["ticker"]).nunique()) if "ticker" in test_df.columns else None,
        "feature_count": int(len(feature_cols)),
        "rmse_log_preclip": float(_rmse(y_test, y_pred_log_preclip)),
        "rmse_log": float(_rmse(y_test, y_pred_log)),
        "r2_log_preclip": float(_r2(y_test, y_pred_log_preclip)),
        "r2_log": float(_r2(y_test, y_pred_log)),
        "rmse_raw": float(_rmse(y_true_raw, y_pred_raw)),
        "r2_raw": float(_r2(y_true_raw, y_pred_raw)),
        "mape_raw": float(_mape(y_true_raw, y_pred_raw)),
        "baseline_rmse_log": float(_rmse(y_test, y_pred_baseline_log)),
        "baseline_rmse_raw": float(_rmse(y_true_raw, y_pred_baseline_raw)),
        "train_cv_rmse_log_mean": float(train_info["metrics"]["cv_rmse_log_mean"]),  # type: ignore[index]
        "train_cv_rmse_log_std": float(train_info["metrics"]["cv_rmse_log_std"]),  # type: ignore[index]
        "train_rmse_log": float(train_info["metrics"]["train_rmse_log"]),  # type: ignore[index]
        "train_rmse_log_preclip": float(train_info["metrics"]["train_rmse_log_preclip"]),  # type: ignore[index]
        "best_num_boost_round": int(train_info["metrics"]["best_num_boost_round"]),  # type: ignore[index]
    }

    preds = test_df.copy()
    preds["y_true"] = y_true_raw.reshape(-1)
    preds["y_pred"] = y_pred_raw.reshape(-1)
    preds["y_pred_preclip"] = y_pred_raw_preclip.reshape(-1)
    preds["y_pred_baseline"] = y_pred_baseline_raw.reshape(-1)

    per_ticker = build_per_ticker_table(test_df, y_test, y_pred_log, y_true_raw, y_pred_raw)
    shift_table = build_feature_shift_table(train_df, test_df, feature_cols)
    cv_df: pd.DataFrame = train_info["cv_results"]  # type: ignore[assignment]

    pred_path = split_dir / f"{name}_predictions.csv"
    per_ticker_path = split_dir / f"{name}_per_ticker_metrics.csv"
    shift_path = split_dir / f"{name}_feature_shift.csv"
    cv_path = split_dir / f"{name}_cv_results.csv"
    model_path = split_dir / f"{name}_model.json"
    train_pred_path = split_dir / f"{name}_train_predictions.csv"

    preds.to_csv(pred_path, index=False)
    per_ticker.to_csv(per_ticker_path, index=False)
    shift_table.to_csv(shift_path, index=False)
    cv_df.to_csv(cv_path, index=False)
    booster.save_model(str(model_path))
    train_preds: pd.DataFrame = train_info["train_predictions"]  # type: ignore[assignment]
    train_preds.to_csv(train_pred_path, index=False)

    per_ticker_summary = None
    if not per_ticker.empty:
        rmse_log_vals = per_ticker["rmse_log"].to_numpy(dtype=np.float64)
        p90 = float(np.quantile(rmse_log_vals, 0.9))
        worst_decile = per_ticker[per_ticker["rmse_log"] >= p90]["rmse_log"].to_numpy(dtype=np.float64)
        per_ticker_summary = {
            "tickers": int(len(per_ticker)),
            "median_rmse_log": float(np.median(rmse_log_vals)),
            "p90_rmse_log": p90,
            "worst_decile_mean_rmse_log": float(np.mean(worst_decile)) if worst_decile.size > 0 else float("nan"),
            "worst_ticker_rmse_log": float(np.max(rmse_log_vals)),
        }

    shift_summary = None
    if not shift_table.empty:
        top_psi = shift_table.sort_values("psi", ascending=False, kind="mergesort").head(10)
        top_ks = shift_table.sort_values("ks_stat", ascending=False, kind="mergesort").head(10)
        shift_summary = {
            "top_psi_features": top_psi[["feature", "psi"]].to_dict(orient="records"),
            "top_ks_features": top_ks[["feature", "ks_stat"]].to_dict(orient="records"),
        }

    return {
        "metrics": metrics,
        "train_metrics": train_info["metrics"],
        "best_params": train_info["best_params"],
        "feature_names": feature_cols,
        "clip": {
            "clip_lo": clip_lo,
            "clip_hi": clip_hi,
            "use_quantile_clip": bool(cfg.use_quantile_clip),
            "clip_q_hi": float(cfg.clip_q_hi),
        },
        "per_ticker_summary": per_ticker_summary,
        "shift_summary": shift_summary,
        "artifacts": {
            "predictions": str(pred_path),
            "per_ticker_metrics": str(per_ticker_path),
            "feature_shift": str(shift_path),
            "cv_results": str(cv_path),
            "model": str(model_path),
            "train_predictions": str(train_pred_path),
        },
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--main", type=str, default="data/processed/main_dataset.csv")
    ap.add_argument("--test-out-of-time", type=str, default="data/processed/test_dataset_out_of_time.csv")
    ap.add_argument("--test-unseen-tickers", type=str, default="data/processed/test_dataset_unseen_tickers.csv")
    ap.add_argument("--out-dir", type=str, default="experiments/valuation/runs/exp_xgb_groupcv_testset_evaluation")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--feature-set", type=str, choices=["all_numeric", "ratio_plus_news"], default="ratio_plus_news")
    ap.add_argument("--cv-splits", type=int, default=5)
    ap.add_argument("--max-candidates", type=int, default=8)
    ap.add_argument("--num-boost-round", type=int, default=4000)
    ap.add_argument("--early-stopping-rounds", type=int, default=80)
    ap.add_argument("--disable-ticker-frequency-weights", action="store_true")
    ap.add_argument("--use-asset-weights", action="store_true")
    args = ap.parse_args()

    cfg = GroupCVXGBValuationConfig()
    cfg.data_path = Path(args.main)
    cfg.random_seed = int(args.seed)
    cfg.feature_set = str(args.feature_set)
    cfg.cv_splits = int(args.cv_splits)
    cfg.max_candidates = int(args.max_candidates)
    cfg.num_boost_round = int(args.num_boost_round)
    cfg.early_stopping_rounds = int(args.early_stopping_rounds)
    cfg.use_ticker_frequency_weights = not bool(args.disable_ticker_frequency_weights)
    cfg.use_asset_weights = bool(args.use_asset_weights)

    main_df = pd.read_csv(args.main)
    oot_df = pd.read_csv(args.test_out_of_time)
    unseen_df = pd.read_csv(args.test_unseen_tickers)

    main_prepped, main_dedup = _prepare_df(main_df, cfg)
    oot_prepped, oot_dedup = _prepare_df(oot_df, cfg)
    unseen_prepped, unseen_dedup = _prepare_df(unseen_df, cfg)

    out_dir = Path(args.out_dir)
    _ensure_dir(out_dir)

    oot_years: List[int] = []
    if "fiscal_year" in oot_prepped.columns and "fiscal_year" in main_prepped.columns:
        oot_years = sorted(pd.to_numeric(oot_prepped["fiscal_year"], errors="coerce").dropna().astype(int).unique().tolist())
    train_oot = main_prepped[~main_prepped["fiscal_year"].isin(oot_years)].copy() if oot_years else main_prepped.copy()

    unseen_tickers = set(normalize_tickers(unseen_prepped["ticker"])) if "ticker" in unseen_prepped.columns else set()
    if "ticker" in main_prepped.columns and unseen_tickers:
        train_unseen = main_prepped[~normalize_tickers(main_prepped["ticker"]).isin(unseen_tickers)].copy()
    else:
        train_unseen = main_prepped.copy()

    res_oot = _run_one_split("out_of_time", train_oot, oot_prepped, cfg, out_dir)
    res_unseen = _run_one_split("unseen_tickers", train_unseen, unseen_prepped, cfg, out_dir)

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
            **res_oot,
        },
        "unseen_tickers": {
            "heldout_tickers": int(len(unseen_tickers)),
            "train_rows": int(len(train_unseen)),
            **res_unseen,
        },
    }

    summary_path = out_dir / "xgb_groupcv_testset_evaluation_summary.json"
    with summary_path.open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    print("Saved:")
    print(f"- {summary_path}")
    print(f"- {res_oot['artifacts']['predictions']}")
    print(f"- {res_unseen['artifacts']['predictions']}")
    print(f"- {res_unseen['artifacts']['per_ticker_metrics']}")
    print(f"- {res_unseen['artifacts']['feature_shift']}")
    print("\nGroupCV XGB test metrics:")
    print(
        f"out_of_time rmse_log={summary['out_of_time']['metrics']['rmse_log']:.6f} | "
        f"rmse_log_preclip={summary['out_of_time']['metrics']['rmse_log_preclip']:.6f} | "
        f"r2_raw={summary['out_of_time']['metrics']['r2_raw']:.6f}"
    )
    print(
        f"unseen_ticker rmse_log={summary['unseen_tickers']['metrics']['rmse_log']:.6f} | "
        f"rmse_log_preclip={summary['unseen_tickers']['metrics']['rmse_log_preclip']:.6f} | "
        f"r2_raw={summary['unseen_tickers']['metrics']['r2_raw']:.6f}"
    )


if __name__ == "__main__":
    main()
