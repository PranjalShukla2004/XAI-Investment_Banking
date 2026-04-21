from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd

from src.models.feature_engineering import fit_pca, select_features_by_correlation, transform_pca
from src.models.nn.losses import HuberLoss
from src.models.nn.mlp import MLP
from src.models.nn.optimizer import Adam
from src.models.nn.train import TrainConfig, fit
from src.models.valuation.residual_mlp_valuation import ResidualMLPValuationConfig, _build_identity_base_feature
from src.models.valuation.valuation import (
    _ensure_dir,
    _fit_standard_scaler,
    _make_asset_weights_from_log_target,
    _mape,
    _r2,
    _rmse,
    _select_feature_columns,
    _select_log1p_features,
    _transform_standard_scaler,
    build_xy,
)


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


def _train_and_eval_residual(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    cfg: ResidualMLPValuationConfig,
) -> Dict[str, object]:
    train_df = _build_identity_base_feature(
        train_df,
        liabilities_col=cfg.liabilities_col,
        equity_col=cfg.equity_col,
        out_col=cfg.base_feature_col,
    )
    test_df = _build_identity_base_feature(
        test_df,
        liabilities_col=cfg.liabilities_col,
        equity_col=cfg.equity_col,
        out_col=cfg.base_feature_col,
    )

    raw_feature_names = _select_feature_columns(train_df, cfg.target_col)
    if cfg.enable_feature_selection:
        feature_names, _ = select_features_by_correlation(
            train_df=train_df,
            feature_cols=raw_feature_names,
            target_col=cfg.target_col,
            log_target=cfg.log_target,
            min_abs_target_corr=float(cfg.min_abs_target_corr),
            max_features=cfg.max_features_by_target_corr,
            max_inter_feature_corr=float(cfg.max_inter_feature_corr),
            min_features=int(cfg.min_features_after_selection),
        )
    else:
        feature_names = list(raw_feature_names)

    log1p_feature_names = _select_log1p_features(
        train_df=train_df,
        feature_cols=feature_names,
        enabled=cfg.use_log1p_feature_transform,
    )

    X_train, y_train, feature_names = build_xy(
        train_df,
        cfg,  # type: ignore[arg-type]
        feature_cols=feature_names,
        log1p_features=log1p_feature_names,
    )
    X_test, y_test, _ = build_xy(
        test_df,
        cfg,  # type: ignore[arg-type]
        feature_cols=feature_names,
        log1p_features=log1p_feature_names,
    )

    base_train_raw = np.clip(
        pd.to_numeric(train_df[cfg.base_feature_col], errors="coerce").fillna(0.0).to_numpy(dtype=np.float32),
        a_min=0.0,
        a_max=None,
    )
    base_test_raw = np.clip(
        pd.to_numeric(test_df[cfg.base_feature_col], errors="coerce").fillna(0.0).to_numpy(dtype=np.float32),
        a_min=0.0,
        a_max=None,
    )

    if cfg.log_target:
        base_train_target_scale = np.log1p(base_train_raw).reshape(-1, 1)
        base_test_target_scale = np.log1p(base_test_raw).reshape(-1, 1)
    else:
        base_train_target_scale = base_train_raw.reshape(-1, 1)
        base_test_target_scale = base_test_raw.reshape(-1, 1)

    y_train_residual = y_train - base_train_target_scale

    x_scaler = _fit_standard_scaler(X_train)
    X_train_s = _transform_standard_scaler(X_train, x_scaler)
    X_test_s = _transform_standard_scaler(X_test, x_scaler)
    if cfg.enable_pca:
        pca_model = fit_pca(
            X_train_s,
            explained_variance=float(cfg.pca_explained_variance),
            max_components=cfg.pca_max_components,
        )
        X_train_model = transform_pca(X_train_s, pca_model)
        X_test_model = transform_pca(X_test_s, pca_model)
    else:
        X_train_model = X_train_s
        X_test_model = X_test_s

    y_res_scaler = _fit_standard_scaler(y_train_residual)
    y_train_res_s = _transform_standard_scaler(y_train_residual, y_res_scaler)

    model = MLP.from_dims(
        input_dim=int(X_train_model.shape[1]),
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
        lr_patience=cfg.lr_patience,
        lr_min=cfg.lr_min,
        lr_threshold=cfg.lr_threshold,
        lr_cooldown=cfg.lr_cooldown,
    )

    # For early stopping behavior we carve a small internal validation slice from train only.
    n_train = X_train_model.shape[0]
    n_val_inner = max(1, int(round(n_train * 0.1)))
    idx = np.arange(n_train)
    rng = np.random.default_rng(cfg.random_seed)
    rng.shuffle(idx)
    val_idx = idx[:n_val_inner]
    fit_idx = idx[n_val_inner:]
    if fit_idx.size == 0:
        fit_idx = val_idx

    X_fit, y_fit = X_train_model[fit_idx], y_train_res_s[fit_idx]
    X_val_in, y_val_in = X_train_model[val_idx], y_train_res_s[val_idx]

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
    yhat_test_res_s = model.forward(X_test_model, training=False)
    yhat_test_residual = yhat_test_res_s * y_res_scaler["sigma"] + y_res_scaler["mu"]
    yhat_test_target_scale = base_test_target_scale + yhat_test_residual

    if cfg.use_quantile_clip:
        clip_lo = float(np.quantile(y_train, cfg.clip_q_lo))
        clip_hi = float(np.quantile(y_train, cfg.clip_q_hi))
    else:
        train_min = float(np.min(y_train))
        train_max = float(np.max(y_train))
        clip_lo = train_min - float(cfg.clip_margin)
        clip_hi = train_max + float(cfg.clip_margin)
    yhat_test_target_scale = np.clip(yhat_test_target_scale, clip_lo, clip_hi)

    if cfg.log_target:
        y_test_raw = np.expm1(y_test)
        yhat_test_raw = np.expm1(yhat_test_target_scale)
        yhat_base_test_raw = np.expm1(base_test_target_scale)
    else:
        y_test_raw = y_test
        yhat_test_raw = yhat_test_target_scale
        yhat_base_test_raw = base_test_target_scale

    metrics = {
        "rows_test": int(len(test_df)),
        "unique_tickers_test": int(_normalize_tickers(test_df["ticker"]).nunique()) if "ticker" in test_df.columns else None,
        "raw_feature_count": int(len(raw_feature_names)),
        "selected_feature_count": int(len(feature_names)),
        "model_input_dim": int(X_train_model.shape[1]),
        "zeros_in_y_true": int(np.sum(y_test_raw.reshape(-1) <= 0.0)),
        "rmse_log": float(_rmse(y_test, yhat_test_target_scale)),
        "r2_log": float(_r2(y_test, yhat_test_target_scale)),
        "rmse_raw": float(_rmse(y_test_raw, yhat_test_raw)),
        "r2_raw": float(_r2(y_test_raw, yhat_test_raw)),
        "mape_raw": float(_mape(y_test_raw, yhat_test_raw)),
        "mape_nonzero_raw": float(_mape_nonzero(y_test_raw, yhat_test_raw)),
        "smape_raw": float(_smape(y_test_raw, yhat_test_raw)),
        "baseline_rmse_log": float(_rmse(y_test, base_test_target_scale)),
        "baseline_r2_log": float(_r2(y_test, base_test_target_scale)),
        "baseline_rmse_raw": float(_rmse(y_test_raw, yhat_base_test_raw)),
        "baseline_r2_raw": float(_r2(y_test_raw, yhat_base_test_raw)),
        "baseline_mape_raw": float(_mape(y_test_raw, yhat_base_test_raw)),
        "baseline_mape_nonzero_raw": float(_mape_nonzero(y_test_raw, yhat_base_test_raw)),
        "baseline_smape_raw": float(_smape(y_test_raw, yhat_base_test_raw)),
        "best_epoch_inner": history.get("best_epoch"),
        "best_val_loss_inner": history.get("best_val_loss"),
        "stopped_early_inner": history.get("stopped_early"),
    }

    preds = test_df.copy()
    preds["y_true"] = y_test_raw.reshape(-1)
    preds["y_pred"] = yhat_test_raw.reshape(-1)
    preds["y_pred_baseline"] = yhat_base_test_raw.reshape(-1)

    return {"metrics": metrics, "predictions": preds, "feature_names": feature_names, "log1p_features": log1p_feature_names}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--main", type=str, default="data/processed/main_dataset.csv")
    ap.add_argument("--test-out-of-time", type=str, default="data/processed/test_dataset_out_of_time.csv")
    ap.add_argument("--test-unseen-tickers", type=str, default="data/processed/test_dataset_unseen_tickers.csv")
    ap.add_argument("--out-dir", type=str, default="data/processed/residual_mlp_valuation_artifacts")
    ap.add_argument("--epochs", type=int, default=200)
    ap.add_argument("--batch-size", type=int, default=128)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    main_df = pd.read_csv(args.main)
    oot_df = pd.read_csv(args.test_out_of_time)
    unseen_df = pd.read_csv(args.test_unseen_tickers)

    cfg = ResidualMLPValuationConfig()
    cfg.epochs = int(args.epochs)
    cfg.batch_size = int(args.batch_size)
    cfg.random_seed = int(args.seed)
    cfg.print_every = 10

    out_dir = Path(args.out_dir)
    _ensure_dir(out_dir)

    # Out-of-time: train on years not present in OOT test.
    oot_years = sorted(pd.to_numeric(oot_df["fiscal_year"], errors="coerce").dropna().astype(int).unique().tolist())
    train_oot = main_df[~main_df["fiscal_year"].isin(oot_years)].copy()
    res_oot = _train_and_eval_residual(train_oot, oot_df, cfg)
    oot_preds_path = out_dir / "test_out_of_time_predictions.csv"
    res_oot["predictions"].to_csv(oot_preds_path, index=False)

    # Unseen tickers: train on non-heldout tickers.
    unseen_tickers = set(_normalize_tickers(unseen_df["ticker"]).dropna())
    train_unseen = main_df[~_normalize_tickers(main_df["ticker"]).isin(unseen_tickers)].copy()
    res_unseen = _train_and_eval_residual(train_unseen, unseen_df, cfg)
    unseen_preds_path = out_dir / "test_unseen_tickers_predictions.csv"
    res_unseen["predictions"].to_csv(unseen_preds_path, index=False)

    summary = {
        "config": {
            "main_dataset": args.main,
            "test_out_of_time": args.test_out_of_time,
            "test_unseen_tickers": args.test_unseen_tickers,
            "epochs": cfg.epochs,
            "batch_size": cfg.batch_size,
            "seed": cfg.random_seed,
        },
        "out_of_time": {
            "years": oot_years,
            "train_rows": int(len(train_oot)),
            "test_metrics": res_oot["metrics"],
            "predictions_path": str(oot_preds_path),
        },
        "unseen_tickers": {
            "heldout_tickers": int(len(unseen_tickers)),
            "train_rows": int(len(train_unseen)),
            "test_metrics": res_unseen["metrics"],
            "predictions_path": str(unseen_preds_path),
        },
    }

    summary_path = out_dir / "test_set_evaluation_summary.json"
    with summary_path.open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    print("Saved:")
    print(f"- {summary_path}")
    print(f"- {oot_preds_path}")
    print(f"- {unseen_preds_path}")
    print("\nTest metrics:")
    print(f"out_of_time  mape={summary['out_of_time']['test_metrics']['mape_raw']:.6f} | "
          f"mape_nonzero={summary['out_of_time']['test_metrics']['mape_nonzero_raw']:.6f} | "
          f"smape={summary['out_of_time']['test_metrics']['smape_raw']:.6f} | "
          f"rmse_log={summary['out_of_time']['test_metrics']['rmse_log']:.6f} | "
          f"r2_raw={summary['out_of_time']['test_metrics']['r2_raw']:.6f}")
    print(f"unseen_ticker mape={summary['unseen_tickers']['test_metrics']['mape_raw']:.6f} | "
          f"mape_nonzero={summary['unseen_tickers']['test_metrics']['mape_nonzero_raw']:.6f} | "
          f"smape={summary['unseen_tickers']['test_metrics']['smape_raw']:.6f} | "
          f"rmse_log={summary['unseen_tickers']['test_metrics']['rmse_log']:.6f} | "
          f"r2_raw={summary['unseen_tickers']['test_metrics']['r2_raw']:.6f}")


if __name__ == "__main__":
    main()
