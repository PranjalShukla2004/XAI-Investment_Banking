from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd

from src.models.nn.losses import HuberLoss
from src.models.nn.mlp import MLP
from src.models.nn.optimizer import Adam
from src.models.nn.train import TrainConfig, fit
from src.models.valuation.valuation import (
    _ensure_dir,
    _fit_standard_scaler,
    _make_asset_weights_from_log_target,
    _mape,
    _r2,
    _rmse,
    _select_feature_columns,
    _select_log1p_features,
    _to_jsonable,
    _transform_standard_scaler,
    build_xy,
)
from src.models.valuation.valuation2 import Valuation2Config


def _normalize_tickers(series: pd.Series) -> pd.Series:
    out = series.astype(str).str.upper().str.strip()
    return out.replace({"": pd.NA, "NAN": pd.NA, "NONE": pd.NA, "<NA>": pd.NA, "NULL": pd.NA})


def _mape_nonzero(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    y_true = np.asarray(y_true, dtype=np.float64).reshape(-1)
    y_pred = np.asarray(y_pred, dtype=np.float64).reshape(-1)
    mask = y_true > 0
    if not np.any(mask):
        return float("nan")
    return float(np.mean(np.abs(y_pred[mask] - y_true[mask]) / np.maximum(y_true[mask], 1e-8)))


def _smape(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    y_true = np.asarray(y_true, dtype=np.float64).reshape(-1)
    y_pred = np.asarray(y_pred, dtype=np.float64).reshape(-1)
    denom = np.maximum((np.abs(y_true) + np.abs(y_pred)) / 2.0, 1e-8)
    return float(np.mean(np.abs(y_pred - y_true) / denom))


def _inner_train_val_indices(train_df: pd.DataFrame, seed: int, val_ratio: float = 0.1) -> tuple[np.ndarray, np.ndarray, str]:
    n_rows = len(train_df)
    if n_rows < 2:
        idx = np.arange(n_rows)
        return idx, idx, "single_row"

    n_val_target = max(1, int(round(n_rows * float(val_ratio))))
    if "ticker" in train_df.columns:
        tickers = _normalize_tickers(train_df["ticker"])
        valid_tickers = tickers.loc[tickers.notna()]
        unique_tickers = valid_tickers.unique()
        if unique_tickers.size > 1:
            rng = np.random.default_rng(seed)
            shuffled = np.array(unique_tickers, dtype=object)
            rng.shuffle(shuffled)
            counts = valid_tickers.value_counts().to_dict()
            chosen: List[str] = []
            rows_accum = 0
            for ticker in shuffled:
                ticker_str = str(ticker)
                chosen.append(ticker_str)
                rows_accum += int(counts.get(ticker_str, 0))
                if rows_accum >= n_val_target and len(chosen) < unique_tickers.size:
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


def _train_and_eval_valuation2(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    cfg: Valuation2Config,
) -> Dict[str, object]:
    feature_names = _select_feature_columns(train_df, cfg.target_col)
    log1p_feature_names = _select_log1p_features(
        train_df=train_df,
        feature_cols=feature_names,
        enabled=cfg.use_log1p_feature_transform,
    )

    X_train, y_train, feature_names = build_xy(
        train_df,
        cfg,
        feature_cols=feature_names,
        log1p_features=log1p_feature_names,
    )
    X_test, y_test, _ = build_xy(
        test_df,
        cfg,
        feature_cols=feature_names,
        log1p_features=log1p_feature_names,
    )

    x_scaler = _fit_standard_scaler(X_train)
    X_train_s = _transform_standard_scaler(X_train, x_scaler)
    X_test_s = _transform_standard_scaler(X_test, x_scaler)

    y_scaler = _fit_standard_scaler(y_train)
    y_train_s = _transform_standard_scaler(y_train, y_scaler)

    model = MLP.from_dims(
        input_dim=int(X_train_s.shape[1]),
        hidden_dims=cfg.hidden_dims,
        output_dim=1,
        activation=cfg.activation.lower(),
        dropout=float(cfg.dropout),
        init=cfg.init,
        weight_scale=float(cfg.weight_scale),
        l2=float(cfg.l2_in_layers),
        l1=float(cfg.l1_in_layers),
    )

    loss_fn = HuberLoss(delta=cfg.huber_delta)
    opt = Adam(lr=cfg.lr, weight_decay=cfg.weight_decay)
    train_cfg = TrainConfig(
        epochs=cfg.epochs,
        batch_size=cfg.batch_size,
        seed=cfg.random_seed,
        print_every=cfg.print_every,
        early_stopping=cfg.early_stopping,
        patience=cfg.patience,
        min_delta=cfg.min_delta,
        restore_best=cfg.restore_best,
        lr_scheduler=cfg.lr_scheduler,
        lr_factor=cfg.lr_factor,
        lr_increase_factor=cfg.lr_increase_factor,
        lr_patience=cfg.lr_patience,
        lr_min=cfg.lr_min,
        lr_max=cfg.lr_max,
        lr_threshold=cfg.lr_threshold,
        lr_cooldown=cfg.lr_cooldown,
        dynamic_lr_start_mode=cfg.dynamic_lr_start_mode,
        dynamic_lr_switch_patience=cfg.dynamic_lr_switch_patience,
    )

    fit_idx, val_idx, inner_split_mode = _inner_train_val_indices(train_df, seed=cfg.random_seed, val_ratio=0.1)
    X_fit, y_fit = X_train_s[fit_idx], y_train_s[fit_idx]
    X_val_in, y_val_in = X_train_s[val_idx], y_train_s[val_idx]

    w_train = _make_asset_weights_from_log_target(y_train[fit_idx], power=0.25, w_min=0.5, w_max=2.0)
    history = fit(
        model=model,
        criterion=loss_fn,
        optimizer=opt,
        X_train=X_fit,
        y_train=y_fit,
        X_val=X_val_in,
        y_val=y_val_in,
        cfg=train_cfg,
        metric_fn=None,
        w_train=w_train,
        w_val=None,
    )

    model.eval()
    yhat_train_s = model.forward(X_train_s, training=False)
    yhat_test_s = model.forward(X_test_s, training=False)

    yhat_train_model_preclip = yhat_train_s * y_scaler["sigma"] + y_scaler["mu"]
    yhat_test_model_preclip = yhat_test_s * y_scaler["sigma"] + y_scaler["mu"]

    if cfg.use_quantile_clip:
        clip_lo = float(np.quantile(y_train, cfg.clip_q_lo))
        clip_hi = float(np.quantile(y_train, cfg.clip_q_hi))
    else:
        clip_lo = float(np.min(y_train)) - float(cfg.clip_margin)
        clip_hi = float(np.max(y_train)) + float(cfg.clip_margin)

    yhat_train_model = np.clip(yhat_train_model_preclip, clip_lo, clip_hi)
    yhat_test_model = np.clip(yhat_test_model_preclip, clip_lo, clip_hi)

    if cfg.log_target:
        y_train_raw = np.expm1(y_train)
        y_test_raw = np.expm1(y_test)
        yhat_train_raw = np.expm1(yhat_train_model)
        yhat_test_raw = np.expm1(yhat_test_model)
        yhat_test_raw_preclip = np.expm1(yhat_test_model_preclip)
    else:
        y_train_raw = y_train
        y_test_raw = y_test
        yhat_train_raw = yhat_train_model
        yhat_test_raw = yhat_test_model
        yhat_test_raw_preclip = yhat_test_model_preclip

    y_mean_model = float(np.mean(y_train))
    yhat_base_test_model = np.full_like(y_test, y_mean_model)
    yhat_base_test_raw = np.expm1(yhat_base_test_model) if cfg.log_target else yhat_base_test_model

    metrics = {
        "rows_test": int(len(test_df)),
        "unique_tickers_test": int(_normalize_tickers(test_df["ticker"]).nunique()) if "ticker" in test_df.columns else None,
        "feature_count": int(len(feature_names)),
        "log1p_feature_count": int(len(log1p_feature_names)),
        "inner_split_mode": inner_split_mode,
        "rmse_log_preclip": float(_rmse(y_test, yhat_test_model_preclip)),
        "r2_log_preclip": float(_r2(y_test, yhat_test_model_preclip)),
        "rmse_log": float(_rmse(y_test, yhat_test_model)),
        "r2_log": float(_r2(y_test, yhat_test_model)),
        "rmse_raw": float(_rmse(y_test_raw, yhat_test_raw)),
        "r2_raw": float(_r2(y_test_raw, yhat_test_raw)),
        "mape_raw": float(_mape(y_test_raw, yhat_test_raw)),
        "mape_nonzero_raw": float(_mape_nonzero(y_test_raw, yhat_test_raw)),
        "smape_raw": float(_smape(y_test_raw, yhat_test_raw)),
        "baseline_rmse_log": float(_rmse(y_test, yhat_base_test_model)),
        "baseline_r2_log": float(_r2(y_test, yhat_base_test_model)),
        "baseline_rmse_raw": float(_rmse(y_test_raw, yhat_base_test_raw)),
        "baseline_r2_raw": float(_r2(y_test_raw, yhat_base_test_raw)),
        "baseline_mape_raw": float(_mape(y_test_raw, yhat_base_test_raw)),
        "baseline_mape_nonzero_raw": float(_mape_nonzero(y_test_raw, yhat_base_test_raw)),
        "baseline_smape_raw": float(_smape(y_test_raw, yhat_base_test_raw)),
        "train_rmse_log_preclip": float(_rmse(y_train, yhat_train_model_preclip)),
        "train_rmse_log": float(_rmse(y_train, yhat_train_model)),
        "train_rmse_raw": float(_rmse(y_train_raw, yhat_train_raw)),
    }

    training = {
        "best_epoch_inner": _to_jsonable(history.get("best_epoch")),
        "best_val_loss_inner": _to_jsonable(history.get("best_val_loss")),
        "stopped_early_inner": _to_jsonable(history.get("stopped_early")),
        "final_lr": float(history.get("lr", [float(cfg.lr)])[-1]) if history.get("lr") else float(cfg.lr),
        "lr_reductions": int(history.get("lr_reductions", 0)),
        "lr_increases": int(history.get("lr_increases", 0)),
        "lr_scheduler_switches": int(history.get("lr_scheduler_switches", 0)),
        "lr_scheduler_mode_last": history.get("lr_scheduler_mode", [None])[-1] if history.get("lr_scheduler_mode") else None,
    }

    preds = test_df.copy()
    preds["y_true"] = y_test_raw.reshape(-1)
    preds["y_pred"] = yhat_test_raw.reshape(-1)
    preds["y_pred_preclip"] = yhat_test_raw_preclip.reshape(-1)
    preds["y_pred_baseline"] = yhat_base_test_raw.reshape(-1)

    return {
        "metrics": metrics,
        "training": training,
        "predictions": preds,
        "feature_names": feature_names,
        "log1p_features": log1p_feature_names,
        "clip": {
            "use_quantile_clip": bool(cfg.use_quantile_clip),
            "q_lo": float(cfg.clip_q_lo),
            "q_hi": float(cfg.clip_q_hi),
            "clip_lo": float(clip_lo),
            "clip_hi": float(clip_hi),
            "clip_margin": float(cfg.clip_margin),
        },
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--main", type=str, default="data/processed/main_dataset.csv")
    ap.add_argument("--test-out-of-time", type=str, default="data/processed/test_dataset_out_of_time.csv")
    ap.add_argument("--test-unseen-tickers", type=str, default="data/processed/test_dataset_unseen_tickers.csv")
    ap.add_argument("--out-dir", type=str, default="experiments/valuation/runs/valuation2_evaluation_artifacts")
    ap.add_argument("--epochs", type=int, default=200)
    ap.add_argument("--batch-size", type=int, default=128)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument(
        "--hidden-dims",
        type=int,
        nargs="+",
        default=None,
        help="Override MLP hidden layer sizes, e.g. --hidden-dims 128 64",
    )
    args = ap.parse_args()

    cfg = Valuation2Config()
    cfg.data_path = Path(args.main)
    cfg.epochs = int(args.epochs)
    cfg.batch_size = int(args.batch_size)
    cfg.random_seed = int(args.seed)
    cfg.print_every = 10
    if args.hidden_dims is not None:
        if len(args.hidden_dims) == 0:
            raise SystemExit("--hidden-dims provided but no values supplied.")
        cfg.hidden_dims = tuple(int(v) for v in args.hidden_dims)

    main_df = pd.read_csv(args.main)
    oot_df = pd.read_csv(args.test_out_of_time)
    unseen_df = pd.read_csv(args.test_unseen_tickers)

    out_dir = Path(args.out_dir)
    _ensure_dir(out_dir)

    oot_years = sorted(pd.to_numeric(oot_df["fiscal_year"], errors="coerce").dropna().astype(int).unique().tolist()) if "fiscal_year" in oot_df.columns else []
    train_oot = main_df[~main_df["fiscal_year"].isin(oot_years)].copy() if "fiscal_year" in main_df.columns else main_df.copy()
    res_oot = _train_and_eval_valuation2(train_oot, oot_df, cfg)
    oot_preds_path = out_dir / "valuation2_test_out_of_time_predictions.csv"
    res_oot["predictions"].to_csv(oot_preds_path, index=False)

    unseen_tickers = set(_normalize_tickers(unseen_df["ticker"]).dropna()) if "ticker" in unseen_df.columns else set()
    train_unseen = (
        main_df[~_normalize_tickers(main_df["ticker"]).isin(unseen_tickers)].copy()
        if "ticker" in main_df.columns and unseen_tickers
        else main_df.copy()
    )
    res_unseen = _train_and_eval_valuation2(train_unseen, unseen_df, cfg)
    unseen_preds_path = out_dir / "valuation2_test_unseen_tickers_predictions.csv"
    res_unseen["predictions"].to_csv(unseen_preds_path, index=False)

    summary = {
        "config": {
            "main_dataset": args.main,
            "test_out_of_time": args.test_out_of_time,
            "test_unseen_tickers": args.test_unseen_tickers,
            "epochs": cfg.epochs,
            "batch_size": cfg.batch_size,
            "seed": cfg.random_seed,
            "target_col": cfg.target_col,
            "log_target": bool(cfg.log_target),
            "hidden_dims": list(cfg.hidden_dims),
            "dropout": float(cfg.dropout),
            "l1_in_layers": float(cfg.l1_in_layers),
            "l2_in_layers": float(cfg.l2_in_layers),
            "lr_scheduler": str(cfg.lr_scheduler),
            "lr_factor": float(cfg.lr_factor),
            "lr_increase_factor": float(cfg.lr_increase_factor),
            "lr_patience": int(cfg.lr_patience),
            "lr_min": float(cfg.lr_min),
            "lr_max": float(cfg.lr_max),
            "dynamic_lr_start_mode": str(cfg.dynamic_lr_start_mode),
            "dynamic_lr_switch_patience": int(cfg.dynamic_lr_switch_patience),
        },
        "out_of_time": {
            "years": oot_years,
            "train_rows": int(len(train_oot)),
            "feature_names": res_oot["feature_names"],
            "log1p_features": res_oot["log1p_features"],
            "clip": res_oot["clip"],
            "training": res_oot["training"],
            "test_metrics": res_oot["metrics"],
            "predictions_path": str(oot_preds_path),
        },
        "unseen_tickers": {
            "heldout_tickers": int(len(unseen_tickers)),
            "train_rows": int(len(train_unseen)),
            "feature_names": res_unseen["feature_names"],
            "log1p_features": res_unseen["log1p_features"],
            "clip": res_unseen["clip"],
            "training": res_unseen["training"],
            "test_metrics": res_unseen["metrics"],
            "predictions_path": str(unseen_preds_path),
        },
    }

    summary_path = out_dir / "valuation2_test_set_evaluation_summary.json"
    with summary_path.open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    print("Saved:")
    print(f"- {summary_path}")
    print(f"- {oot_preds_path}")
    print(f"- {unseen_preds_path}")
    print("\nValuation2 test metrics:")
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
