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
from src.models.valuation.final_valuation import (
    FinalValuationConfig,
    _build_identity_base_feature,
    _build_news_features,
    _mape_nonzero,
    _smape,
    _tail_risk_rates,
)
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


def _prepare_with_news(df: pd.DataFrame, cfg: FinalValuationConfig) -> tuple[pd.DataFrame, Dict[str, object]]:
    news_df, news_stats = _build_news_features(df, cfg.news_score_col, cfg.news_text_col)
    out_df = _build_identity_base_feature(
        news_df,
        liabilities_col=cfg.liabilities_col,
        equity_col=cfg.equity_col,
        out_col=cfg.base_feature_col,
    )
    return out_df, news_stats


def _inner_train_val_indices(
    train_df: pd.DataFrame,
    eligible_mask: np.ndarray,
    seed: int,
    val_ratio: float = 0.1,
) -> tuple[np.ndarray, np.ndarray, str]:
    eligible_idx = np.flatnonzero(eligible_mask)
    if eligible_idx.size == 0:
        all_idx = np.arange(len(train_df))
        return all_idx, all_idx, "row_fallback_all"
    if eligible_idx.size == 1:
        return eligible_idx, eligible_idx, "single_row"

    n_val_target = max(1, int(round(eligible_idx.size * float(val_ratio))))
    if "ticker" in train_df.columns:
        ticker_sub = _normalize_tickers(train_df.iloc[eligible_idx]["ticker"])
        valid_mask_sub = ticker_sub.notna().to_numpy()
        valid_tickers = ticker_sub.loc[ticker_sub.notna()]
        unique_tickers = valid_tickers.unique()
        if unique_tickers.size > 1:
            rng = np.random.default_rng(seed)
            tickers_shuffled = np.array(unique_tickers, dtype=object)
            rng.shuffle(tickers_shuffled)
            ticker_counts = valid_tickers.value_counts().to_dict()

            chosen_tickers: List[str] = []
            rows_accum = 0
            for t in tickers_shuffled:
                ts = str(t)
                chosen_tickers.append(ts)
                rows_accum += int(ticker_counts.get(ts, 0))
                if rows_accum >= n_val_target and len(chosen_tickers) < unique_tickers.size:
                    break

            val_mask_sub = valid_mask_sub & ticker_sub.isin(chosen_tickers).to_numpy()
            if np.any(val_mask_sub) and np.any(~val_mask_sub):
                val_idx = eligible_idx[val_mask_sub]
                fit_idx = eligible_idx[~val_mask_sub]
                return fit_idx, val_idx, "group_ticker"

    idx = eligible_idx.copy()
    rng = np.random.default_rng(seed)
    rng.shuffle(idx)
    val_idx = idx[:n_val_target]
    fit_idx = idx[n_val_target:]
    if fit_idx.size == 0:
        fit_idx = val_idx
    return fit_idx, val_idx, "row"


def _train_and_eval_final(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    cfg: FinalValuationConfig,
) -> Dict[str, object]:
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

    model_train_mask = np.ones(base_train_raw.shape[0], dtype=bool)
    if cfg.enable_tiny_asset_regime:
        tiny_train_rows = int(np.sum(base_train_raw <= float(cfg.tiny_asset_threshold_raw)))
        tiny_test_mask = base_test_raw <= float(cfg.tiny_asset_threshold_raw)
    else:
        tiny_train_rows = 0
        tiny_test_mask = np.zeros(base_test_raw.shape[0], dtype=bool)

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

    fit_idx, val_idx, inner_split_mode = _inner_train_val_indices(
        train_df=train_df,
        eligible_mask=model_train_mask,
        seed=cfg.random_seed,
        val_ratio=0.1,
    )

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

    y_train_clip_ref = y_train
    if cfg.use_quantile_clip:
        clip_hi = float(np.quantile(y_train_clip_ref, cfg.clip_q_hi))
    else:
        train_max = float(np.max(y_train_clip_ref))
        clip_hi = train_max + float(cfg.clip_margin)

    if cfg.log_target:
        clip_floor = float(np.log1p(max(float(cfg.min_target_raw), 0.0)))
    else:
        clip_floor = float(max(float(cfg.min_target_raw), 0.0))
    clip_lo = clip_floor

    yhat_test_target_scale = np.clip(yhat_test_target_scale, clip_lo, clip_hi)
    if cfg.enable_tiny_asset_regime:
        yhat_test_target_scale[tiny_test_mask] = base_test_target_scale[tiny_test_mask]

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
        "inner_split_mode": inner_split_mode,
        "rows_train_model": int(np.sum(model_train_mask)),
        "rows_train_tiny": int(tiny_train_rows),
        "rows_test_tiny_regime": int(np.sum(tiny_test_mask)),
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
        "tail_risk": _tail_risk_rates(y_test_raw, yhat_test_raw),
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
    ap.add_argument("--out-dir", type=str, default="data/processed/final_valuation_artifacts")
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

    cfg = FinalValuationConfig()
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

    main_prepped, main_news_stats = _prepare_with_news(main_df, cfg)
    oot_prepped, oot_news_stats = _prepare_with_news(oot_df, cfg)
    unseen_prepped, unseen_news_stats = _prepare_with_news(unseen_df, cfg)

    out_dir = Path(args.out_dir)
    _ensure_dir(out_dir)

    oot_years = sorted(pd.to_numeric(oot_prepped["fiscal_year"], errors="coerce").dropna().astype(int).unique().tolist())
    train_oot = main_prepped[~main_prepped["fiscal_year"].isin(oot_years)].copy()
    res_oot = _train_and_eval_final(train_oot, oot_prepped, cfg)
    oot_preds_path = out_dir / "final_test_out_of_time_predictions.csv"
    res_oot["predictions"].to_csv(oot_preds_path, index=False)

    unseen_tickers = set(_normalize_tickers(unseen_prepped["ticker"]).dropna())
    train_unseen = main_prepped[~_normalize_tickers(main_prepped["ticker"]).isin(unseen_tickers)].copy()
    res_unseen = _train_and_eval_final(train_unseen, unseen_prepped, cfg)
    unseen_preds_path = out_dir / "final_test_unseen_tickers_predictions.csv"
    res_unseen["predictions"].to_csv(unseen_preds_path, index=False)

    summary = {
        "config": {
            "main_dataset": args.main,
            "test_out_of_time": args.test_out_of_time,
            "test_unseen_tickers": args.test_unseen_tickers,
            "epochs": cfg.epochs,
            "batch_size": cfg.batch_size,
            "seed": cfg.random_seed,
            "hidden_dims": list(cfg.hidden_dims),
            "dropout": float(cfg.dropout),
            "l1_in_layers": float(cfg.l1_in_layers),
            "l2_in_layers": float(cfg.l2_in_layers),
            "enable_feature_selection": bool(cfg.enable_feature_selection),
            "enable_pca": bool(cfg.enable_pca),
            "enable_tiny_asset_regime": bool(cfg.enable_tiny_asset_regime),
            "tiny_asset_threshold_raw": float(cfg.tiny_asset_threshold_raw),
        },
        "news_stats": {
            "main": main_news_stats,
            "out_of_time": oot_news_stats,
            "unseen_tickers": unseen_news_stats,
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

    summary_path = out_dir / "final_test_set_evaluation_summary.json"
    with summary_path.open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    print("Saved:")
    print(f"- {summary_path}")
    print(f"- {oot_preds_path}")
    print(f"- {unseen_preds_path}")
    print("\nFinal model test metrics:")
    print(
        f"out_of_time  mape_nonzero={summary['out_of_time']['test_metrics']['mape_nonzero_raw']:.6f} | "
        f"smape={summary['out_of_time']['test_metrics']['smape_raw']:.6f} | "
        f"rmse_log={summary['out_of_time']['test_metrics']['rmse_log']:.6f} | "
        f"r2_raw={summary['out_of_time']['test_metrics']['r2_raw']:.6f}"
    )
    print(
        f"unseen_ticker mape_nonzero={summary['unseen_tickers']['test_metrics']['mape_nonzero_raw']:.6f} | "
        f"smape={summary['unseen_tickers']['test_metrics']['smape_raw']:.6f} | "
        f"rmse_log={summary['unseen_tickers']['test_metrics']['rmse_log']:.6f} | "
        f"r2_raw={summary['unseen_tickers']['test_metrics']['r2_raw']:.6f}"
    )


if __name__ == "__main__":
    main()
