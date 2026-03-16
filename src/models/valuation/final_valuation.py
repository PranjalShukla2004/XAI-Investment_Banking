from __future__ import annotations

import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np
import pandas as pd

from src.models.feature_engineering import fit_pca, select_features_by_correlation, transform_pca
from src.models.valuation.news_driven_mlp_valuation import (
    NEWS_FEATURE_COLUMNS,
    _build_identity_base_feature,
    _build_news_features,
    _meta_cols,
)
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
    _time_aware_split,
    _to_jsonable,
    _transform_standard_scaler,
    build_xy,
)


@dataclass
class FinalValuationConfig:
    # Data
    data_path: Path = Path("data/processed/main_dataset.csv")
    out_dir: Path = Path("experiments/valuation/runs/final_valuation_artifacts")

    # Target
    target_col: str = "total_assets"
    log_target: bool = True

    # Residual identity baseline
    liabilities_col: str = "total_liabilities"
    equity_col: str = "total_equity"
    base_feature_col: str = "identity_base_assets"

    # Splitting
    time_col: str = "fiscal_year"
    min_val_rows: int = 20
    random_seed: int = 42
    val_ratio_fallback: float = 0.2

    # Training
    epochs: int = 500
    batch_size: int = 128
    lr: float = 5e-4
    weight_decay: float = 1e-5
    print_every: int = 1
    early_stopping: bool = True
    patience: int = 40
    min_delta: float = 1e-4
    restore_best: bool = True
    lr_scheduler: str = "reduce_on_plateau"
    lr_factor: float = 0.5
    lr_patience: int = 20
    lr_min: float = 1e-6
    lr_threshold: float = 5e-5
    lr_cooldown: int = 5
    huber_delta: float = 0.1

    # Feature transform and regularization controls (anti-overfitting)
    use_log1p_feature_transform: bool = True
    enable_feature_selection: bool = True
    min_abs_target_corr: float = 0.01
    max_features_by_target_corr: int | None = 200
    max_inter_feature_corr: float = 0.98
    min_features_after_selection: int = 20
    enable_pca: bool = False
    pca_explained_variance: float = 0.95
    pca_max_components: int | None = 64

    # Output clipping only (no fallback model)
    use_quantile_clip: bool = True
    clip_q_lo: float = 0.005
    clip_q_hi: float = 0.995
    clip_margin: float = 0.2
    min_target_raw: float = 0.0
    # Separate tiny-asset regime to avoid extreme tiny-target error blowups.
    enable_tiny_asset_regime: bool = True
    tiny_asset_threshold_raw: float = 5_000_000.0

    # Model
    hidden_dims: Tuple[int, ...] = (64, 15)
    dropout: float = 0.2
    activation: str = "relu"
    init: str = "he"
    weight_scale: float = 0.01
    l2_in_layers: float = 5e-6
    l1_in_layers: float = 1e-7

    # News columns
    news_score_col: str = "news_sentiment_score"
    news_text_col: str = "news_description"


def _mape_nonzero(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    yt = np.asarray(y_true, dtype=np.float64).reshape(-1)
    yp = np.asarray(y_pred, dtype=np.float64).reshape(-1)
    mask = yt > 0.0
    if not np.any(mask):
        return float("nan")
    return float(np.mean(np.abs(yp[mask] - yt[mask]) / np.maximum(yt[mask], 1e-8)))


def _smape(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    yt = np.asarray(y_true, dtype=np.float64).reshape(-1)
    yp = np.asarray(y_pred, dtype=np.float64).reshape(-1)
    denom = np.maximum((np.abs(yt) + np.abs(yp)) / 2.0, 1e-8)
    return float(np.mean(np.abs(yp - yt) / denom))


def _tail_risk_rates(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    yt = np.asarray(y_true, dtype=np.float64).reshape(-1)
    yp = np.asarray(y_pred, dtype=np.float64).reshape(-1)
    rel = np.abs(yp - yt) / np.maximum(yt, 1e-8)
    return {
        "rate_rel_err_gt_1_0": float(np.mean(rel > 1.0)),
        "rate_rel_err_gt_5_0": float(np.mean(rel > 5.0)),
        "rate_rel_err_gt_10_0": float(np.mean(rel > 10.0)),
        "rel_err_p50": float(np.quantile(rel, 0.50)),
        "rel_err_p90": float(np.quantile(rel, 0.90)),
        "rel_err_p95": float(np.quantile(rel, 0.95)),
        "rel_err_p99": float(np.quantile(rel, 0.99)),
    }


def _count_mlp_capacity(input_dim: int, hidden_dims: Tuple[int, ...], output_dim: int = 1) -> Dict[str, Any]:
    dims = [int(input_dim), *[int(h) for h in hidden_dims], int(output_dim)]
    trainable_params = 0
    for i in range(len(dims) - 1):
        trainable_params += dims[i] * dims[i + 1] + dims[i + 1]  # W + b

    return {
        "layer_dims": dims,
        "input_features": int(dims[0]),
        "hidden_layers": int(len(hidden_dims)),
        "hidden_neurons_total": int(sum(hidden_dims)),
        "output_neurons": int(output_dim),
        "total_neurons_hidden_plus_output": int(sum(hidden_dims) + output_dim),
        "trainable_parameters": int(trainable_params),
        "learnables": int(trainable_params),
    }


def main() -> None:
    cfg = FinalValuationConfig()

    env_path = os.getenv("VAL_DATA_PATH")
    if env_path:
        cfg.data_path = Path(env_path)
    env_out_dir = os.getenv("VAL_OUT_DIR")
    if env_out_dir:
        cfg.out_dir = Path(env_out_dir)
    env_hidden_dims = os.getenv("VAL_HIDDEN_DIMS")
    if env_hidden_dims:
        parsed_dims = tuple(int(tok.strip()) for tok in env_hidden_dims.split(",") if tok.strip())
        if not parsed_dims:
            raise SystemExit("VAL_HIDDEN_DIMS is set but empty; expected comma-separated integers like '128,64'")
        cfg.hidden_dims = parsed_dims

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

    train_df, val_df = _time_aware_split(
        df=df,
        time_col=cfg.time_col,
        min_val_rows=cfg.min_val_rows,
        seed=cfg.random_seed,
        val_ratio_fallback=cfg.val_ratio_fallback,
    )

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
    used_news_features = [c for c in NEWS_FEATURE_COLUMNS if c in feature_names]
    has_identity_base_feature = cfg.base_feature_col in feature_names

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

    # Residual learning around accounting identity.
    y_train_residual = y_train - base_train_target_scale
    y_val_residual = y_val - base_val_target_scale

    model_train_mask = np.ones(base_train_raw.shape[0], dtype=bool)
    model_val_mask = np.ones(base_val_raw.shape[0], dtype=bool)
    if cfg.enable_tiny_asset_regime:
        tiny_train_mask = base_train_raw <= float(cfg.tiny_asset_threshold_raw)
        tiny_val_mask = base_val_raw <= float(cfg.tiny_asset_threshold_raw)
    else:
        tiny_train_mask = np.zeros(base_train_raw.shape[0], dtype=bool)
        tiny_val_mask = np.zeros(base_val_raw.shape[0], dtype=bool)

    x_scaler = _fit_standard_scaler(X_train)
    X_train_s = _transform_standard_scaler(X_train, x_scaler)
    X_val_s = _transform_standard_scaler(X_val, x_scaler)

    pca_model = None
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

    X_val_for_training = X_val_model
    y_val_for_training = y_val_res_s

    w_train_full = _make_asset_weights_from_log_target(y_train, power=0.25, w_min=0.5, w_max=2.0)
    w_train = w_train_full
    history = fit(
        model=model,
        criterion=loss_fn,
        optimizer=opt,
        X_train=X_train_model,
        y_train=y_train_res_s,
        X_val=X_val_for_training,
        y_val=y_val_for_training,
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

    yhat_train_target_scale = np.clip(yhat_train_target_scale, clip_lo, clip_hi)
    yhat_val_target_scale = np.clip(yhat_val_target_scale, clip_lo, clip_hi)

    if cfg.enable_tiny_asset_regime:
        yhat_train_target_scale[tiny_train_mask] = base_train_target_scale[tiny_train_mask]
        yhat_val_target_scale[tiny_val_mask] = base_val_target_scale[tiny_val_mask]

    if cfg.log_target:
        y_train_orig = np.expm1(y_train)
        y_val_orig = np.expm1(y_val)
        yhat_train = np.expm1(yhat_train_target_scale)
        yhat_val = np.expm1(yhat_val_target_scale)
        yhat_base_train = np.expm1(base_train_target_scale)
        yhat_base_val = np.expm1(base_val_target_scale)
    else:
        y_train_orig = y_train
        y_val_orig = y_val
        yhat_train = yhat_train_target_scale
        yhat_val = yhat_val_target_scale
        yhat_base_train = base_train_target_scale
        yhat_base_val = base_val_target_scale

    hist_path = cfg.out_dir / "history.json"
    with open(hist_path, "w", encoding="utf-8") as f:
        json.dump(history, f, indent=2)

    if pca_model is None:
        pca_mu = np.array([], dtype=np.float32)
        pca_components = np.empty((0, 0), dtype=np.float32)
        pca_explained_ratio = np.array([], dtype=np.float32)
    else:
        pca_mu = np.asarray(pca_model["mu"], dtype=np.float32)
        pca_components = np.asarray(pca_model["components"], dtype=np.float32)
        pca_explained_ratio = np.asarray(pca_model["explained_variance_ratio"], dtype=np.float32)

    np.savez(
        cfg.out_dir / "scalers_and_features.npz",
        x_mu=x_scaler["mu"],
        x_sigma=x_scaler["sigma"],
        y_res_mu=y_res_scaler["mu"],
        y_res_sigma=y_res_scaler["sigma"],
        feature_names_raw=np.array(raw_feature_names, dtype=object),
        feature_names=np.array(feature_names, dtype=object),
        feature_log1p_names=np.array(log1p_feature_names, dtype=object),
        news_feature_names=np.array(used_news_features, dtype=object),
        target_col=np.array([cfg.target_col], dtype=object),
        log_target=np.array([cfg.log_target], dtype=object),
        base_feature_col=np.array([cfg.base_feature_col], dtype=object),
        liabilities_col=np.array([cfg.liabilities_col], dtype=object),
        equity_col=np.array([cfg.equity_col], dtype=object),
        use_quantile_clip=np.array([cfg.use_quantile_clip], dtype=object),
        clip_q_lo=np.array([cfg.clip_q_lo], dtype=np.float32),
        clip_q_hi=np.array([cfg.clip_q_hi], dtype=np.float32),
        clip_lo=np.array([clip_lo], dtype=np.float32),
        clip_hi=np.array([clip_hi], dtype=np.float32),
        clip_margin=np.array([cfg.clip_margin], dtype=np.float32),
        min_target_raw=np.array([cfg.min_target_raw], dtype=np.float32),
        enable_feature_selection=np.array([cfg.enable_feature_selection], dtype=object),
        min_abs_target_corr=np.array([cfg.min_abs_target_corr], dtype=np.float32),
        max_features_by_target_corr=np.array([cfg.max_features_by_target_corr], dtype=object),
        max_inter_feature_corr=np.array([cfg.max_inter_feature_corr], dtype=np.float32),
        min_features_after_selection=np.array([cfg.min_features_after_selection], dtype=np.int32),
        enable_pca=np.array([cfg.enable_pca], dtype=object),
        pca_explained_variance=np.array([cfg.pca_explained_variance], dtype=np.float32),
        pca_max_components=np.array([cfg.pca_max_components], dtype=object),
        pca_n_features_in=np.array([pca_stats["n_features_in"]], dtype=np.int32),
        pca_n_components=np.array([pca_stats["n_components"]], dtype=np.int32),
        pca_explained_variance_ratio_cum=np.array([pca_stats["explained_variance_ratio_cum"]], dtype=np.float32),
        pca_mu=pca_mu,
        pca_components=pca_components,
        pca_explained_variance_ratio=pca_explained_ratio,
    )

    train_meta = _meta_cols(train_df)
    val_meta = _meta_cols(val_df)

    rows: List[dict] = []
    for i in range(len(train_df)):
        row = {
            "split": "train",
            "y_true": float(y_train_orig[i, 0]),
            "y_pred": float(yhat_train[i, 0]),
            "y_pred_baseline": float(yhat_base_train[i, 0]),
        }
        for k, arr in train_meta.items():
            row[k] = arr[i]
        rows.append(row)
    for i in range(len(val_df)):
        row = {
            "split": "val",
            "y_true": float(y_val_orig[i, 0]),
            "y_pred": float(yhat_val[i, 0]),
            "y_pred_baseline": float(yhat_base_val[i, 0]),
        }
        for k, arr in val_meta.items():
            row[k] = arr[i]
        rows.append(row)

    preds_df = pd.DataFrame(rows)
    preds_path = cfg.out_dir / "predictions.csv"
    preds_df.to_csv(preds_path, index=False)

    train_rmse_log = _rmse(y_train, yhat_train_target_scale)
    val_rmse_log = _rmse(y_val, yhat_val_target_scale)
    train_factor = float(np.exp(train_rmse_log))
    val_factor = float(np.exp(val_rmse_log))

    r2_train_log = _r2(y_train, yhat_train_target_scale)
    r2_val_log = _r2(y_val, yhat_val_target_scale)
    r2_train_raw = _r2(y_train_orig, yhat_train)
    r2_val_raw = _r2(y_val_orig, yhat_val)
    val_mape_raw = _mape(y_val_orig, yhat_val)
    val_mape_nonzero_raw = _mape_nonzero(y_val_orig, yhat_val)
    val_smape_raw = _smape(y_val_orig, yhat_val)
    train_rmse_raw = _rmse(y_train_orig, yhat_train)
    val_rmse_raw = _rmse(y_val_orig, yhat_val)
    val_tail_risk = _tail_risk_rates(y_val_orig, yhat_val)

    baseline_rmse_log = _rmse(y_val, base_val_target_scale)
    baseline_factor = float(np.exp(baseline_rmse_log))
    baseline_mape_raw = _mape(y_val_orig, yhat_base_val)
    baseline_mape_nonzero_raw = _mape_nonzero(y_val_orig, yhat_base_val)
    baseline_smape_raw = _smape(y_val_orig, yhat_base_val)
    baseline_rmse_raw = _rmse(y_val_orig, yhat_base_val)
    baseline_r2_raw = _r2(y_val_orig, yhat_base_val)
    baseline_r2_log = _r2(y_val, base_val_target_scale)
    baseline_tail_risk = _tail_risk_rates(y_val_orig, yhat_base_val)

    capacity = _count_mlp_capacity(
        input_dim=int(X_train_model.shape[1]),
        hidden_dims=cfg.hidden_dims,
        output_dim=1,
    )
    data_points = {
        "rows_total": int(len(df)),
        "train_rows": int(len(train_df)),
        "val_rows": int(len(val_df)),
    }

    run_summary = {
        "target_col": cfg.target_col,
        "log_target": cfg.log_target,
        "years": sorted(df[cfg.time_col].dropna().unique().tolist()) if cfg.time_col in df.columns else None,
        "data_points": data_points,
        "model_capacity": capacity,
        "news": {
            "score_column": cfg.news_score_col,
            "text_column": cfg.news_text_col,
            "feature_columns_used": used_news_features,
            "stats": news_stats,
        },
        "feature_engineering": {
            "selection": feature_selection_stats,
            "pca": pca_stats,
        },
        "tiny_asset_regime": {
            "enabled": bool(cfg.enable_tiny_asset_regime),
            "threshold_raw": float(cfg.tiny_asset_threshold_raw),
            "train_rows_tiny": int(np.sum(tiny_train_mask)),
            "train_rows_model": int(np.sum(model_train_mask)),
            "val_rows_tiny": int(np.sum(tiny_val_mask)),
            "val_rows_model": int(np.sum(model_val_mask)),
        },
        "identity_baseline": {
            "formula": f"{cfg.liabilities_col} + {cfg.equity_col}",
            "base_feature_col": cfg.base_feature_col,
            "included_in_features": bool(has_identity_base_feature),
            "val_rmse_log": float(baseline_rmse_log),
            "val_factor": float(baseline_factor),
            "val_rmse_raw": float(baseline_rmse_raw),
            "val_mape_raw": float(baseline_mape_raw),
            "val_mape_nonzero_raw": float(baseline_mape_nonzero_raw),
            "val_smape_raw": float(baseline_smape_raw),
            "val_r2_raw": float(baseline_r2_raw),
            "val_r2_log": float(baseline_r2_log),
            "val_tail_risk": baseline_tail_risk,
        },
        "clip": {
            "use_quantile_clip": bool(cfg.use_quantile_clip),
            "q_lo": float(cfg.clip_q_lo),
            "q_hi": float(cfg.clip_q_hi),
            "lo": float(clip_lo),
            "hi": float(clip_hi),
            "margin": float(cfg.clip_margin),
            "min_target_raw": float(cfg.min_target_raw),
        },
        "config": {
            "hidden_dims": list(cfg.hidden_dims),
            "dropout": float(cfg.dropout),
            "lr": float(cfg.lr),
            "weight_decay": float(cfg.weight_decay),
            "loss_fn": "HuberLoss",
            "huber_delta": float(cfg.huber_delta),
            "l1_in_layers": float(cfg.l1_in_layers),
            "l2_in_layers": float(cfg.l2_in_layers),
            "lr_scheduler": str(cfg.lr_scheduler),
            "lr_factor": float(cfg.lr_factor),
            "lr_patience": int(cfg.lr_patience),
            "lr_min": float(cfg.lr_min),
            "lr_threshold": float(cfg.lr_threshold),
            "lr_cooldown": int(cfg.lr_cooldown),
            "batch_size": int(cfg.batch_size),
            "epochs_max": int(train_cfg.epochs),
            "early_stopping": bool(getattr(train_cfg, "early_stopping", False)),
            "patience": int(getattr(train_cfg, "patience", 0)),
            "min_delta": float(getattr(train_cfg, "min_delta", 0.0)),
            "restore_best": bool(getattr(train_cfg, "restore_best", False)),
            "log1p_feature_transform": bool(cfg.use_log1p_feature_transform),
            "log1p_feature_count": int(len(log1p_feature_names)),
            "log1p_features": list(log1p_feature_names),
            "raw_feature_count": int(len(raw_feature_names)),
            "feature_count": int(len(feature_names)),
            "model_input_dim": int(X_train_model.shape[1]),
            "enable_feature_selection": bool(cfg.enable_feature_selection),
            "min_abs_target_corr": float(cfg.min_abs_target_corr),
            "max_features_by_target_corr": int(cfg.max_features_by_target_corr) if cfg.max_features_by_target_corr is not None else None,
            "max_inter_feature_corr": float(cfg.max_inter_feature_corr),
            "min_features_after_selection": int(cfg.min_features_after_selection),
            "enable_pca": bool(cfg.enable_pca),
            "pca_explained_variance": float(cfg.pca_explained_variance),
            "pca_max_components": int(cfg.pca_max_components) if cfg.pca_max_components is not None else None,
            "enable_tiny_asset_regime": bool(cfg.enable_tiny_asset_regime),
            "tiny_asset_threshold_raw": float(cfg.tiny_asset_threshold_raw),
        },
        "training": {
            "best_epoch": _to_jsonable(history.get("best_epoch")),
            "best_val_loss": _to_jsonable(history.get("best_val_loss")),
            "stopped_early": _to_jsonable(history.get("stopped_early")),
            "final_lr": float(history.get("lr", [float(cfg.lr)])[-1]) if history.get("lr") else float(cfg.lr),
            "lr_reductions": int(history.get("lr_reductions", 0)),
        },
        "metrics": {
            "train_rmse_log": float(train_rmse_log),
            "train_factor": float(train_factor),
            "val_rmse_log": float(val_rmse_log),
            "val_factor": float(val_factor),
            "train_r2_log": float(r2_train_log),
            "val_r2_log": float(r2_val_log),
            "train_r2_raw": float(r2_train_raw),
            "val_r2_raw": float(r2_val_raw),
            "val_mape_raw": float(val_mape_raw),
            "val_mape_nonzero_raw": float(val_mape_nonzero_raw),
            "val_smape_raw": float(val_smape_raw),
            "val_tail_risk": val_tail_risk,
            "train_rmse_raw": float(train_rmse_raw),
            "val_rmse_raw": float(val_rmse_raw),
        },
    }

    summary_path = cfg.out_dir / "run_summary.json"
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(run_summary, f, indent=2)

    print("=== Done (Final Valuation: News + Residual MLP, No Fallback) ===")
    print(f"Target: {cfg.target_col} (log_target={cfg.log_target})")
    print(f"Data points: total={data_points['rows_total']} | train={data_points['train_rows']} | val={data_points['val_rows']}")
    print(
        "Model capacity: "
        f"input_dim={capacity['input_features']} | hidden={list(cfg.hidden_dims)} | "
        f"hidden_neurons={capacity['hidden_neurons_total']} | "
        f"trainable_params={capacity['trainable_parameters']}"
    )
    print(
        "Feature selection: "
        f"enabled={cfg.enable_feature_selection} | raw={len(raw_feature_names)} | selected={len(feature_names)}"
    )
    print(
        "PCA: "
        f"enabled={cfg.enable_pca} | in_dim={int(X_train_s.shape[1])} | model_dim={int(X_train_model.shape[1])} | "
        f"explained={float(pca_stats['explained_variance_ratio_cum']):.4f}"
    )
    print(f"News features used: {len(used_news_features)}")
    print(f"Identity baseline included as feature: {has_identity_base_feature}")
    print(
        "Tiny-asset regime: "
        f"enabled={cfg.enable_tiny_asset_regime} | threshold_raw={cfg.tiny_asset_threshold_raw:.0f} | "
        f"train_tiny={int(np.sum(tiny_train_mask))} | val_tiny={int(np.sum(tiny_val_mask))}"
    )
    print(f"Val MAPE (raw): {val_mape_raw:.6f}")
    print(f"Val MAPE (nonzero): {val_mape_nonzero_raw:.6f}")
    print(f"Val sMAPE (raw): {val_smape_raw:.6f}")
    print(f"Val RMSE (log): {val_rmse_log:.6f} (~x{val_factor:.3f})")
    print(f"Baseline Val RMSE (log): {baseline_rmse_log:.6f} (~x{baseline_factor:.3f})")
    print(f"Saved: {hist_path}")
    print(f"Saved: {preds_path}")
    print(f"Saved: {summary_path}")
    print(f"Saved: {cfg.out_dir / 'scalers_and_features.npz'}")


if __name__ == "__main__":
    main()
