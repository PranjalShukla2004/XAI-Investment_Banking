from __future__ import annotations

import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd

from .nn.losses import HuberLoss
from .nn.mlp import MLP
from .nn.optimizer import Adam
from .nn.train import TrainConfig, fit
from .valuation import (
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
class ResidualMLPValuationConfig:
    # Data
    data_path: Path = Path("data/processed/main_dataset.csv")
    out_dir: Path = Path("data/processed/residual_mlp_valuation_artifacts")

    # Target
    target_col: str = "total_assets"
    log_target: bool = True

    # Baseline identity columns
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

    # Prediction clipping in log-space
    use_quantile_clip: bool = True
    clip_q_lo: float = 0.005
    clip_q_hi: float = 0.995
    clip_margin: float = 0.2

    # Feature transform
    use_log1p_feature_transform: bool = True

    # Model
    hidden_dims: Tuple[int, ...] = (256, 128)
    dropout: float = 0.1
    activation: str = "relu"
    init: str = "he"
    weight_scale: float = 0.01
    l2_in_layers: float = 1e-6


def _require_column(df: pd.DataFrame, col: str) -> None:
    if col not in df.columns:
        cols = list(df.columns)[:80]
        raise ValueError(
            f"Required column '{col}' not found. "
            f"Available columns include: {cols}{'...' if len(df.columns) > 80 else ''}"
        )


def _build_identity_base_feature(
    df: pd.DataFrame,
    liabilities_col: str,
    equity_col: str,
    out_col: str,
) -> pd.DataFrame:
    _require_column(df, liabilities_col)
    _require_column(df, equity_col)

    out = df.copy()
    liab = pd.to_numeric(out[liabilities_col], errors="coerce").fillna(0.0)
    eq = pd.to_numeric(out[equity_col], errors="coerce").fillna(0.0)
    # Keep base non-negative for stable log-space residual target.
    out[out_col] = np.clip((liab + eq).to_numpy(dtype=np.float64), a_min=0.0, a_max=None)
    return out


def _meta_cols(d: pd.DataFrame) -> Dict[str, np.ndarray]:
    out: Dict[str, np.ndarray] = {}
    for c in ["ticker", "fiscal_year", "period_end", "timeframe"]:
        if c in d.columns:
            out[c] = d[c].to_numpy()
    return out


def main() -> None:
    cfg = ResidualMLPValuationConfig()

    env_path = os.getenv("VAL_DATA_PATH")
    if env_path:
        cfg.data_path = Path(env_path)

    if not cfg.data_path.exists():
        raise SystemExit(f"Dataset not found at: {cfg.data_path.resolve()}")

    _ensure_dir(cfg.out_dir)

    raw_df = pd.read_csv(cfg.data_path)
    df = _build_identity_base_feature(
        raw_df,
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

    feature_names = _select_feature_columns(train_df, cfg.target_col)
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

    # Residual target around accounting identity baseline.
    y_train_residual = y_train - base_train_target_scale
    y_val_residual = y_val - base_val_target_scale

    x_scaler = _fit_standard_scaler(X_train)
    X_train_s = _transform_standard_scaler(X_train, x_scaler)
    X_val_s = _transform_standard_scaler(X_val, x_scaler)

    y_res_scaler = _fit_standard_scaler(y_train_residual)
    y_train_res_s = _transform_standard_scaler(y_train_residual, y_res_scaler)
    y_val_res_s = _transform_standard_scaler(y_val_residual, y_res_scaler)

    model = MLP.from_dims(
        input_dim=int(X_train_s.shape[1]),
        hidden_dims=cfg.hidden_dims,
        output_dim=1,
        activation=cfg.activation.lower(),
        dropout=float(cfg.dropout),
        init=cfg.init,
        weight_scale=float(cfg.weight_scale),
        l2=float(cfg.l2_in_layers),
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

    # Keep weighting by original asset scale.
    w_train = _make_asset_weights_from_log_target(y_train, power=0.25, w_min=0.5, w_max=2.0)
    history = fit(
        model=model,
        criterion=loss_fn,
        optimizer=opt,
        X_train=X_train_s,
        y_train=y_train_res_s,
        X_val=X_val_s,
        y_val=y_val_res_s,
        cfg=train_cfg,
        metric_fn=None,
        w_train=w_train,
        w_val=None,
    )

    model.eval()
    yhat_train_res_s = model.forward(X_train_s, training=False)
    yhat_val_res_s = model.forward(X_val_s, training=False)

    yhat_train_residual = yhat_train_res_s * y_res_scaler["sigma"] + y_res_scaler["mu"]
    yhat_val_residual = yhat_val_res_s * y_res_scaler["sigma"] + y_res_scaler["mu"]

    yhat_train_target_scale = base_train_target_scale + yhat_train_residual
    yhat_val_target_scale = base_val_target_scale + yhat_val_residual

    if cfg.use_quantile_clip:
        clip_lo = float(np.quantile(y_train, cfg.clip_q_lo))
        clip_hi = float(np.quantile(y_train, cfg.clip_q_hi))
    else:
        train_min = float(np.min(y_train))
        train_max = float(np.max(y_train))
        clip_lo = train_min - float(cfg.clip_margin)
        clip_hi = train_max + float(cfg.clip_margin)

    yhat_train_target_scale_clip = np.clip(yhat_train_target_scale, clip_lo, clip_hi)
    yhat_val_target_scale_clip = np.clip(yhat_val_target_scale, clip_lo, clip_hi)

    if cfg.log_target:
        y_train_orig = np.expm1(y_train)
        y_val_orig = np.expm1(y_val)
        yhat_train = np.expm1(yhat_train_target_scale_clip)
        yhat_val = np.expm1(yhat_val_target_scale_clip)
        yhat_base_train = np.expm1(base_train_target_scale)
        yhat_base_val = np.expm1(base_val_target_scale)
    else:
        y_train_orig = y_train
        y_val_orig = y_val
        yhat_train = yhat_train_target_scale_clip
        yhat_val = yhat_val_target_scale_clip
        yhat_base_train = base_train_target_scale
        yhat_base_val = base_val_target_scale

    hist_path = cfg.out_dir / "history.json"
    with open(hist_path, "w") as f:
        json.dump(history, f, indent=2)

    np.savez(
        cfg.out_dir / "scalers_and_features.npz",
        x_mu=x_scaler["mu"],
        x_sigma=x_scaler["sigma"],
        y_res_mu=y_res_scaler["mu"],
        y_res_sigma=y_res_scaler["sigma"],
        feature_names=np.array(feature_names, dtype=object),
        feature_log1p_names=np.array(log1p_feature_names, dtype=object),
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
    )

    train_meta = _meta_cols(train_df)
    val_meta = _meta_cols(val_df)

    rows: List[dict] = []
    for i in range(len(train_df)):
        r = {
            "split": "train",
            "y_true": float(y_train_orig[i, 0]),
            "y_pred": float(yhat_train[i, 0]),
            "y_pred_baseline": float(yhat_base_train[i, 0]),
        }
        for k, arr in train_meta.items():
            r[k] = arr[i]
        rows.append(r)

    for i in range(len(val_df)):
        r = {
            "split": "val",
            "y_true": float(y_val_orig[i, 0]),
            "y_pred": float(yhat_val[i, 0]),
            "y_pred_baseline": float(yhat_base_val[i, 0]),
        }
        for k, arr in val_meta.items():
            r[k] = arr[i]
        rows.append(r)

    preds_df = pd.DataFrame(rows)
    preds_path = cfg.out_dir / "predictions.csv"
    preds_df.to_csv(preds_path, index=False)

    train_rmse_log = _rmse(y_train, yhat_train_target_scale_clip)
    val_rmse_log = _rmse(y_val, yhat_val_target_scale_clip)
    train_factor = float(np.exp(train_rmse_log))
    val_factor = float(np.exp(val_rmse_log))

    r2_train_log = _r2(y_train, yhat_train_target_scale_clip)
    r2_val_log = _r2(y_val, yhat_val_target_scale_clip)
    r2_train_raw = _r2(y_train_orig, yhat_train)
    r2_val_raw = _r2(y_val_orig, yhat_val)
    val_mape_raw = _mape(y_val_orig, yhat_val)
    train_rmse_raw = _rmse(y_train_orig, yhat_train)
    val_rmse_raw = _rmse(y_val_orig, yhat_val)

    # Identity baseline metrics for direct comparison.
    baseline_rmse_log = _rmse(y_val, base_val_target_scale)
    baseline_factor = float(np.exp(baseline_rmse_log))
    baseline_mape_raw = _mape(y_val_orig, yhat_base_val)
    baseline_rmse_raw = _rmse(y_val_orig, yhat_base_val)
    baseline_r2_raw = _r2(y_val_orig, yhat_base_val)
    baseline_r2_log = _r2(y_val, base_val_target_scale)

    val_only = preds_df[preds_df["split"] == "val"].copy()
    val_only["abs_err"] = np.abs(val_only["y_pred"] - val_only["y_true"])
    val_only["rel_err"] = val_only["abs_err"] / np.maximum(val_only["y_true"], 1e-8)

    bucket_table = None
    try:
        val_only["size_bucket"] = pd.qcut(val_only["y_true"], q=5, duplicates="drop")
        bucket_table = val_only.groupby("size_bucket").agg(
            n=("y_true", "size"),
            mape=("rel_err", "mean"),
            median_rel=("rel_err", "median"),
            rmse=("abs_err", lambda x: float(np.sqrt(np.mean(x**2)))),
            mean_true=("y_true", "mean"),
            mean_pred=("y_pred", "mean"),
            mean_baseline=("y_pred_baseline", "mean"),
        )
    except Exception:
        bucket_table = None

    run_summary = {
        "target_col": cfg.target_col,
        "log_target": cfg.log_target,
        "train_rows": int(len(train_df)),
        "val_rows": int(len(val_df)),
        "years": sorted(df[cfg.time_col].dropna().unique().tolist()) if cfg.time_col in df.columns else None,
        "identity_baseline": {
            "formula": f"{cfg.liabilities_col} + {cfg.equity_col}",
            "base_feature_col": cfg.base_feature_col,
            "val_rmse_log": float(baseline_rmse_log),
            "val_factor": float(baseline_factor),
            "val_rmse_raw": float(baseline_rmse_raw),
            "val_mape_raw": float(baseline_mape_raw),
            "val_r2_raw": float(baseline_r2_raw),
            "val_r2_log": float(baseline_r2_log),
        },
        "clip": {
            "use_quantile_clip": bool(cfg.use_quantile_clip),
            "q_lo": float(cfg.clip_q_lo),
            "q_hi": float(cfg.clip_q_hi),
            "lo": float(clip_lo),
            "hi": float(clip_hi),
            "margin": float(cfg.clip_margin),
        },
        "config": {
            "hidden_dims": list(cfg.hidden_dims),
            "dropout": float(cfg.dropout),
            "lr": float(cfg.lr),
            "weight_decay": float(cfg.weight_decay),
            "loss_fn": "HuberLoss",
            "huber_delta": float(cfg.huber_delta),
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
            "feature_count": int(len(feature_names)),
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
            "train_rmse_raw": float(train_rmse_raw),
            "val_rmse_raw": float(val_rmse_raw),
        },
    }

    summary_path = cfg.out_dir / "run_summary.json"
    with open(summary_path, "w") as f:
        json.dump(run_summary, f, indent=2)

    print("=== Done (Residual MLP Valuation) ===")
    print(f"Target: {cfg.target_col} (log_target={cfg.log_target})")
    print(f"Residual baseline: {cfg.liabilities_col} + {cfg.equity_col}")
    print(f"Train rows: {len(train_df)} | Val rows: {len(val_df)}")
    if cfg.time_col in df.columns:
        yrs = sorted(df[cfg.time_col].dropna().unique().tolist())
        print(f"Years in data: {yrs}")
    if cfg.use_quantile_clip:
        print(f"Log clip quantiles: [{cfg.clip_q_lo}, {cfg.clip_q_hi}] => [{clip_lo:.3f}, {clip_hi:.3f}]")
    else:
        print(f"Log clip range: [{clip_lo:.3f}, {clip_hi:.3f}] (margin={cfg.clip_margin})")

    print(f"Identity baseline Val MAPE (raw): {baseline_mape_raw:.6f}")
    print(f"Residual MLP  Val MAPE (raw): {val_mape_raw:.6f}")
    print(f"Identity baseline Val RMSE (log): {baseline_rmse_log:.6f} (~x{baseline_factor:.3f})")
    print(f"Residual MLP  Val RMSE (log): {val_rmse_log:.6f} (~x{val_factor:.3f})")

    if bucket_table is not None:
        print("\nVal performance by size bucket (quintiles):")
        print(bucket_table)

    print(f"Saved: {hist_path}")
    print(f"Saved: {preds_path}")
    print(f"Saved: {summary_path}")
    print(f"Saved: {cfg.out_dir / 'scalers_and_features.npz'}")


if __name__ == "__main__":
    main()
