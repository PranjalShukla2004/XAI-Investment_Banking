from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, Iterable, Iterator, List, Tuple

import numpy as np
import pandas as pd

from src.models.feature_engineering import fit_pca, select_features_by_correlation, transform_pca
from src.models.nn.losses import HuberLoss
from src.models.nn.mlp import MLP
from src.models.nn.optimizer import Adam
from src.models.nn.train import TrainConfig, fit
from src.models.valuation.news_driven_mlp_valuation import (
    NewsDrivenMLPValuationConfig,
    _build_identity_base_feature,
    _build_news_features,
)
from src.models.valuation.production_safe_inference import (
    ProductionSafetyConfig,
    apply_production_safety,
    fit_production_safety_stats,
    mape_nonzero,
    smape,
    tail_risk_rates,
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


def _meta_cols(d: pd.DataFrame) -> Dict[str, np.ndarray]:
    out: Dict[str, np.ndarray] = {}
    for c in ["ticker", "fiscal_year", "period_end", "timeframe"]:
        if c in d.columns:
            out[c] = d[c].to_numpy()
    return out


def _normalize_groups(series: pd.Series) -> np.ndarray:
    return series.astype(str).str.upper().str.strip().to_numpy(dtype=object)


def _iter_group_kfold(
    groups: np.ndarray,
    n_splits: int,
    seed: int,
) -> Iterator[Tuple[int, np.ndarray, np.ndarray, List[str]]]:
    uniq = np.unique(groups)
    if uniq.size < 2:
        raise ValueError("Need at least 2 unique groups for group k-fold.")

    k = int(max(2, min(int(n_splits), int(uniq.size))))
    rng = np.random.default_rng(seed)
    shuffled = uniq.copy()
    rng.shuffle(shuffled)
    buckets = np.array_split(shuffled, k)

    for i, val_groups in enumerate(buckets, start=1):
        val_mask = np.isin(groups, val_groups)
        val_idx = np.where(val_mask)[0]
        train_idx = np.where(~val_mask)[0]
        if train_idx.size == 0 or val_idx.size == 0:
            continue
        groups_list = [str(x) for x in val_groups.tolist()]
        yield i, train_idx, val_idx, groups_list


def _iter_row_kfold(
    n_rows: int,
    n_splits: int,
    seed: int,
) -> Iterator[Tuple[int, np.ndarray, np.ndarray, List[str]]]:
    if n_rows < 2:
        raise ValueError("Need at least 2 rows for row k-fold.")

    k = int(max(2, min(int(n_splits), int(n_rows))))
    rng = np.random.default_rng(seed)
    idx = np.arange(n_rows)
    rng.shuffle(idx)
    buckets = np.array_split(idx, k)

    for i, val_idx in enumerate(buckets, start=1):
        train_idx = np.setdiff1d(idx, val_idx, assume_unique=False)
        if train_idx.size == 0 or val_idx.size == 0:
            continue
        yield i, train_idx, val_idx, []


def _iter_time_kfold(
    years: np.ndarray,
    n_splits: int,
) -> Iterator[Tuple[int, np.ndarray, np.ndarray, List[str]]]:
    year_series = pd.to_numeric(pd.Series(years), errors="coerce")
    if year_series.notna().sum() == 0:
        raise ValueError("No valid fiscal_year values for time k-fold.")
    years_int = year_series.fillna(-1).astype(np.int64).to_numpy()
    uniq_years = np.unique(years_int)
    if uniq_years.size < 2:
        raise ValueError("Need at least 2 unique fiscal years for time k-fold.")

    k = int(max(2, min(int(n_splits), int(uniq_years.size))))
    year_buckets = np.array_split(np.sort(uniq_years), k)
    row_idx = np.arange(years_int.shape[0])

    for i, val_years in enumerate(year_buckets, start=1):
        val_mask = np.isin(years_int, val_years)
        val_idx = row_idx[val_mask]
        train_idx = row_idx[~val_mask]
        if train_idx.size == 0 or val_idx.size == 0:
            continue
        years_list = [str(int(y)) for y in val_years.tolist()]
        yield i, train_idx, val_idx, years_list


def _train_eval_fold(
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    cfg: NewsDrivenMLPValuationConfig,
) -> Dict[str, object]:
    raw_feature_names = _select_feature_columns(train_df, cfg.target_col)
    if cfg.enable_feature_selection:
        feature_names, feature_selection_stats = select_features_by_correlation(
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
        feature_selection_stats = {
            "selection_applied": False,
            "raw_feature_count": int(len(raw_feature_names)),
            "selected_feature_count": int(len(feature_names)),
        }

    log1p_feature_names = _select_log1p_features(
        train_df=train_df,
        feature_cols=feature_names,
        enabled=cfg.use_log1p_feature_transform,
    )
    used_news_features = [c for c in feature_names if c.startswith("news_")]

    X_train, y_train, feature_names = build_xy(
        train_df,
        cfg,  # type: ignore[arg-type]
        feature_cols=feature_names,
        log1p_features=log1p_feature_names,
    )
    X_val, y_val, _ = build_xy(
        val_df,
        cfg,  # type: ignore[arg-type]
        feature_cols=feature_names,
        log1p_features=log1p_feature_names,
    )

    base_train_raw = np.clip(
        pd.to_numeric(train_df[cfg.base_feature_col], errors="coerce").fillna(0.0).to_numpy(dtype=np.float32),
        a_min=0.0,
        a_max=None,
    )
    base_val_raw = np.clip(
        pd.to_numeric(val_df[cfg.base_feature_col], errors="coerce").fillna(0.0).to_numpy(dtype=np.float32),
        a_min=0.0,
        a_max=None,
    )

    if cfg.log_target:
        base_train_target_scale = np.log1p(base_train_raw).reshape(-1, 1)
        base_val_target_scale = np.log1p(base_val_raw).reshape(-1, 1)
    else:
        base_train_target_scale = base_train_raw.reshape(-1, 1)
        base_val_target_scale = base_val_raw.reshape(-1, 1)

    y_train_residual = y_train - base_train_target_scale
    y_val_residual = y_val - base_val_target_scale

    x_scaler = _fit_standard_scaler(X_train)
    X_train_s = _transform_standard_scaler(X_train, x_scaler)
    X_val_s = _transform_standard_scaler(X_val, x_scaler)
    if cfg.enable_pca:
        pca_model = fit_pca(
            X_train_s,
            explained_variance=float(cfg.pca_explained_variance),
            max_components=cfg.pca_max_components,
        )
        X_train_model = transform_pca(X_train_s, pca_model)
        X_val_model = transform_pca(X_val_s, pca_model)
        pca_stats = {
            "enabled": True,
            "n_features_in": int(pca_model["n_features_in"]),
            "n_components": int(pca_model["n_components"]),
            "explained_variance_ratio_cum": float(pca_model["explained_variance_ratio_cum"]),
        }
    else:
        X_train_model = X_train_s
        X_val_model = X_val_s
        pca_stats = {
            "enabled": False,
            "n_features_in": int(X_train_s.shape[1]),
            "n_components": int(X_train_s.shape[1]),
            "explained_variance_ratio_cum": 1.0,
        }

    y_res_scaler = _fit_standard_scaler(y_train_residual)
    y_train_res_s = _transform_standard_scaler(y_train_residual, y_res_scaler)
    y_val_res_s = _transform_standard_scaler(y_val_residual, y_res_scaler)

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

    w_train = _make_asset_weights_from_log_target(y_train, power=0.25, w_min=0.5, w_max=2.0)
    history = fit(
        model=model,
        criterion=loss_fn,
        optimizer=opt,
        X_train=X_train_model,
        y_train=y_train_res_s,
        X_val=X_val_model,
        y_val=y_val_res_s,
        cfg=train_cfg,
        metric_fn=None,
        w_train=w_train,
        w_val=None,
    )

    model.eval()
    yhat_train_res_s = model.forward(X_train_model, training=False)
    yhat_val_res_s = model.forward(X_val_model, training=False)

    yhat_train_residual = yhat_train_res_s * y_res_scaler["sigma"] + y_res_scaler["mu"]
    yhat_val_residual = yhat_val_res_s * y_res_scaler["sigma"] + y_res_scaler["mu"]

    yhat_train_target_scale = base_train_target_scale + yhat_train_residual
    yhat_val_target_scale = base_val_target_scale + yhat_val_residual

    if cfg.log_target:
        y_train_raw = np.expm1(y_train)
        y_val_raw = np.expm1(y_val)
        yhat_train_model_raw = np.expm1(yhat_train_target_scale)
        yhat_val_model_raw = np.expm1(yhat_val_target_scale)
        yhat_base_train_raw = np.expm1(base_train_target_scale)
        yhat_base_val_raw = np.expm1(base_val_target_scale)
    else:
        y_train_raw = y_train
        y_val_raw = y_val
        yhat_train_model_raw = yhat_train_target_scale
        yhat_val_model_raw = yhat_val_target_scale
        yhat_base_train_raw = base_train_target_scale
        yhat_base_val_raw = base_val_target_scale

    safety_cfg = ProductionSafetyConfig(
        enable_residual_clip=bool(cfg.enable_residual_clip),
        residual_q_lo=float(cfg.residual_q_lo),
        residual_q_hi=float(cfg.residual_q_hi),
        min_target_raw=float(cfg.min_target_raw),
        max_target_q_hi=float(cfg.max_target_q_hi),
        max_target_cap_multiplier=float(cfg.max_target_cap_multiplier),
        enable_baseline_fallback=bool(cfg.enable_baseline_fallback),
        fallback_rel_dev_threshold=float(cfg.fallback_rel_dev_threshold),
        fallback_min_base_abs=float(cfg.fallback_min_base_abs),
        fallback_ratio_low=float(cfg.fallback_ratio_low),
        fallback_ratio_high=float(cfg.fallback_ratio_high),
        enable_ood_fallback=bool(cfg.enable_ood_fallback),
        ood_zscore_threshold=float(cfg.ood_zscore_threshold),
    )
    safety_stats = fit_production_safety_stats(y_train_raw, yhat_base_train_raw, safety_cfg)

    train_feature_zmax = np.max(np.abs(X_train_s), axis=1)
    val_feature_zmax = np.max(np.abs(X_val_s), axis=1)
    safety_train = apply_production_safety(
        y_model_raw=yhat_train_model_raw,
        y_baseline_raw=yhat_base_train_raw,
        cfg=safety_cfg,
        stats=safety_stats,
        feature_zscore_max=train_feature_zmax,
    )
    safety_val = apply_production_safety(
        y_model_raw=yhat_val_model_raw,
        y_baseline_raw=yhat_base_val_raw,
        cfg=safety_cfg,
        stats=safety_stats,
        feature_zscore_max=val_feature_zmax,
    )

    yhat_train_raw = np.asarray(safety_train["y_pred_final"], dtype=np.float64).reshape(-1, 1)
    yhat_val_raw = np.asarray(safety_val["y_pred_final"], dtype=np.float64).reshape(-1, 1)
    yhat_val_after_safety_raw = np.asarray(safety_val["y_pred_after_clip"], dtype=np.float64).reshape(-1, 1)
    used_fallback_val = np.asarray(safety_val["fallback_any"], dtype=bool).reshape(-1, 1)

    if cfg.log_target:
        yhat_train_log_final = np.log1p(np.clip(yhat_train_raw, a_min=0.0, a_max=None))
        yhat_val_log_final = np.log1p(np.clip(yhat_val_raw, a_min=0.0, a_max=None))
    else:
        yhat_train_log_final = yhat_train_raw
        yhat_val_log_final = yhat_val_raw

    train_rmse_log = _rmse(y_train, yhat_train_log_final)
    val_rmse_log = _rmse(y_val, yhat_val_log_final)
    train_r2_log = _r2(y_train, yhat_train_log_final)
    val_r2_log = _r2(y_val, yhat_val_log_final)

    metrics = {
        "train_rows": int(len(train_df)),
        "val_rows": int(len(val_df)),
        "raw_feature_count": int(len(raw_feature_names)),
        "feature_count": int(len(feature_names)),
        "model_input_dim": int(X_train_model.shape[1]),
        "news_feature_count": int(len(used_news_features)),
        "train_rmse_log": float(train_rmse_log),
        "val_rmse_log": float(val_rmse_log),
        "train_r2_log": float(train_r2_log),
        "val_r2_log": float(val_r2_log),
        "train_rmse_raw": float(_rmse(y_train_raw, yhat_train_raw)),
        "val_rmse_raw": float(_rmse(y_val_raw, yhat_val_raw)),
        "train_r2_raw": float(_r2(y_train_raw, yhat_train_raw)),
        "val_r2_raw": float(_r2(y_val_raw, yhat_val_raw)),
        "val_mape_raw": float(_mape(y_val_raw, yhat_val_raw)),
        "val_mape_nonzero_raw": float(mape_nonzero(y_val_raw, yhat_val_raw)),
        "val_smape_raw": float(smape(y_val_raw, yhat_val_raw)),
        "baseline_val_rmse_log": float(_rmse(y_val, base_val_target_scale)),
        "baseline_val_r2_log": float(_r2(y_val, base_val_target_scale)),
        "baseline_val_rmse_raw": float(_rmse(y_val_raw, yhat_base_val_raw)),
        "baseline_val_r2_raw": float(_r2(y_val_raw, yhat_base_val_raw)),
        "baseline_val_mape_raw": float(_mape(y_val_raw, yhat_base_val_raw)),
        "baseline_val_mape_nonzero_raw": float(mape_nonzero(y_val_raw, yhat_base_val_raw)),
        "baseline_val_smape_raw": float(smape(y_val_raw, yhat_base_val_raw)),
        "val_fallback_rate": float(safety_val["fallback_rate"]),
        "val_fallback_rel_rate": float(safety_val["fallback_rel_rate"]),
        "val_fallback_ratio_rate": float(safety_val["fallback_ratio_rate"]),
        "val_fallback_ood_rate": float(safety_val["fallback_ood_rate"]),
        "generalization_gap_rmse_log": float(val_rmse_log - train_rmse_log),
        "generalization_gap_r2_log": float(train_r2_log - val_r2_log),
        "best_epoch": float(history.get("best_epoch")) if history.get("best_epoch") is not None else np.nan,
        "best_val_loss": float(history.get("best_val_loss")) if history.get("best_val_loss") is not None else np.nan,
        "stopped_early": float(bool(history.get("stopped_early"))),
    }
    metrics.update({f"val_tail_{k}": float(v) for k, v in tail_risk_rates(y_val_raw, yhat_val_raw).items()})

    val_meta = _meta_cols(val_df)
    pred_rows: List[dict] = []
    for i in range(len(val_df)):
        row = {
            "y_true": float(y_val_raw[i, 0]),
            "y_pred": float(yhat_val_raw[i, 0]),
            "y_pred_model": float(yhat_val_model_raw[i, 0]),
            "y_pred_after_safety_clip": float(yhat_val_after_safety_raw[i, 0]),
            "y_pred_baseline": float(yhat_base_val_raw[i, 0]),
            "used_baseline_fallback": int(used_fallback_val[i, 0]),
            "feature_zscore_max": float(val_feature_zmax[i]),
        }
        for k, arr in val_meta.items():
            row[k] = arr[i]
        pred_rows.append(row)

    return {
        "metrics": metrics,
        "predictions_val": pd.DataFrame(pred_rows),
        "safety_stats": safety_stats,
        "feature_selection": feature_selection_stats,
        "pca": pca_stats,
    }


def _aggregate_metrics(rows: Iterable[Dict[str, float]]) -> Dict[str, Dict[str, float]]:
    df = pd.DataFrame(list(rows))
    out: Dict[str, Dict[str, float]] = {}
    if df.empty:
        return out

    for col in df.columns:
        s = pd.to_numeric(df[col], errors="coerce")
        if s.notna().sum() == 0:
            continue
        out[col] = {
            "mean": float(s.mean()),
            "std": float(s.std(ddof=0)),
            "min": float(s.min()),
            "max": float(s.max()),
        }
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", type=str, default="data/processed/main_dataset.csv")
    ap.add_argument("--out-dir", type=str, default="data/processed/news_driven_mlp_valuation_artifacts")
    ap.add_argument("--n-splits", type=int, default=5)
    ap.add_argument("--split-mode", type=str, choices=("group_ticker", "row", "time"), default="group_ticker")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--epochs", type=int, default=300)
    ap.add_argument("--batch-size", type=int, default=128)
    ap.add_argument("--print-every", type=int, default=0)
    ap.add_argument("--disable-feature-selection", action="store_true")
    ap.add_argument("--min-abs-target-corr", type=float, default=None)
    ap.add_argument("--max-features-by-target-corr", type=int, default=None)
    ap.add_argument("--max-inter-feature-corr", type=float, default=None)
    ap.add_argument("--min-features-after-selection", type=int, default=None)
    ap.add_argument("--enable-pca", action="store_true")
    ap.add_argument("--pca-explained-variance", type=float, default=None)
    ap.add_argument("--pca-max-components", type=int, default=None)
    args = ap.parse_args()

    cfg = NewsDrivenMLPValuationConfig()
    cfg.data_path = Path(args.data)
    cfg.out_dir = Path(args.out_dir)
    cfg.epochs = int(args.epochs)
    cfg.batch_size = int(args.batch_size)
    cfg.random_seed = int(args.seed)
    cfg.print_every = int(args.print_every)
    if args.disable_feature_selection:
        cfg.enable_feature_selection = False
    if args.min_abs_target_corr is not None:
        cfg.min_abs_target_corr = float(args.min_abs_target_corr)
    if args.max_features_by_target_corr is not None:
        cfg.max_features_by_target_corr = int(args.max_features_by_target_corr)
    if args.max_inter_feature_corr is not None:
        cfg.max_inter_feature_corr = float(args.max_inter_feature_corr)
    if args.min_features_after_selection is not None:
        cfg.min_features_after_selection = int(args.min_features_after_selection)
    if args.enable_pca:
        cfg.enable_pca = True
    if args.pca_explained_variance is not None:
        cfg.pca_explained_variance = float(args.pca_explained_variance)
    if args.pca_max_components is not None:
        cfg.pca_max_components = int(args.pca_max_components)

    if not cfg.data_path.exists():
        raise SystemExit(f"Dataset not found at: {cfg.data_path.resolve()}")

    _ensure_dir(cfg.out_dir)

    raw_df = pd.read_csv(cfg.data_path)
    news_df, news_stats = _build_news_features(raw_df, cfg.news_score_col, cfg.news_text_col)
    df = _build_identity_base_feature(
        news_df,
        liabilities_col=cfg.liabilities_col,
        equity_col=cfg.equity_col,
        out_col=cfg.base_feature_col,
    )

    if args.split_mode == "group_ticker":
        if "ticker" not in df.columns:
            raise SystemExit("split-mode=group_ticker requires a 'ticker' column.")
        folds = list(_iter_group_kfold(_normalize_groups(df["ticker"]), args.n_splits, args.seed))
    elif args.split_mode == "time":
        if cfg.time_col not in df.columns:
            raise SystemExit(f"split-mode=time requires column '{cfg.time_col}'.")
        folds = list(_iter_time_kfold(df[cfg.time_col].to_numpy(), args.n_splits))
    else:
        folds = list(_iter_row_kfold(len(df), args.n_splits, args.seed))

    if not folds:
        raise SystemExit("No valid folds were generated.")

    fold_rows: List[dict] = []
    oof_parts: List[pd.DataFrame] = []

    print(
        f"[cv] mode={args.split_mode} | folds={len(folds)} | rows={len(df):,} | "
        f"text_items={news_stats['total_text_items']:,} | score_items={news_stats['total_score_items']:,}"
    )

    for fold_id, train_idx, val_idx, heldout in folds:
        train_df = df.iloc[train_idx].copy()
        val_df = df.iloc[val_idx].copy()
        if train_df.empty or val_df.empty:
            continue

        print(f"[cv] fold={fold_id} train={len(train_df):,} val={len(val_df):,}")
        fold_res = _train_eval_fold(train_df, val_df, cfg)

        m = dict(fold_res["metrics"])
        m["fold"] = int(fold_id)
        m["heldout_groups_n"] = int(len(heldout))
        m["selection_applied"] = bool(cfg.enable_feature_selection)
        m["pca_enabled"] = bool(cfg.enable_pca)
        fold_rows.append(m)

        preds = fold_res["predictions_val"].copy()
        preds["fold"] = int(fold_id)
        oof_parts.append(preds)

    fold_df = pd.DataFrame(fold_rows)
    if fold_df.empty:
        raise SystemExit("All folds failed; no metrics were produced.")

    oof_df = pd.concat(oof_parts, axis=0, ignore_index=True) if oof_parts else pd.DataFrame()
    fold_metrics_path = cfg.out_dir / f"cv_{args.split_mode}_fold_metrics.csv"
    oof_preds_path = cfg.out_dir / f"cv_{args.split_mode}_oof_predictions.csv"
    summary_path = cfg.out_dir / f"cv_{args.split_mode}_summary.json"
    fold_df.to_csv(fold_metrics_path, index=False)
    oof_df.to_csv(oof_preds_path, index=False)

    summary = {
        "data_path": str(cfg.data_path),
        "split_mode": args.split_mode,
        "requested_splits": int(args.n_splits),
        "actual_splits": int(len(fold_df)),
        "rows_total": int(len(df)),
        "news_stats": news_stats,
        "config": {
            "hidden_dims": list(cfg.hidden_dims),
            "dropout": float(cfg.dropout),
            "l1_in_layers": float(cfg.l1_in_layers),
            "l2_in_layers": float(cfg.l2_in_layers),
            "weight_decay": float(cfg.weight_decay),
            "enable_feature_selection": bool(cfg.enable_feature_selection),
            "min_abs_target_corr": float(cfg.min_abs_target_corr),
            "max_features_by_target_corr": int(cfg.max_features_by_target_corr) if cfg.max_features_by_target_corr is not None else None,
            "max_inter_feature_corr": float(cfg.max_inter_feature_corr),
            "min_features_after_selection": int(cfg.min_features_after_selection),
            "enable_pca": bool(cfg.enable_pca),
            "pca_explained_variance": float(cfg.pca_explained_variance),
            "pca_max_components": int(cfg.pca_max_components) if cfg.pca_max_components is not None else None,
            "epochs": int(cfg.epochs),
            "batch_size": int(cfg.batch_size),
            "seed": int(cfg.random_seed),
            "log_target": bool(cfg.log_target),
            "target_col": cfg.target_col,
        },
        "aggregate_metrics": _aggregate_metrics(fold_rows),
    }
    with summary_path.open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    print("Saved:")
    print(f"- {fold_metrics_path}")
    print(f"- {oof_preds_path}")
    print(f"- {summary_path}")
    print(
        "CV mean metrics: "
        f"val_rmse_log={float(fold_df['val_rmse_log'].mean()):.6f} | "
        f"val_mape_nonzero_raw={float(fold_df['val_mape_nonzero_raw'].mean()):.6f} | "
        f"val_smape_raw={float(fold_df['val_smape_raw'].mean()):.6f} | "
        f"val_r2_raw={float(fold_df['val_r2_raw'].mean()):.6f}"
    )


if __name__ == "__main__":
    main()
