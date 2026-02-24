# src/models/valuation.py
from __future__ import annotations

import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from .nn.mlp import MLP
from .nn.losses import HuberLoss
from .nn.optimizer import Adam
from .nn.train import fit, TrainConfig


@dataclass
class ValuationConfig:
    # Data
    data_path: Path = Path("data/processed/main_dataset.csv")
    out_dir: Path = Path("data/processed/valuation_artifacts")

    # Target
    target_col: str = "total_assets"
    log_target: bool = True

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
    # Keep ReduceLROnPlateau enabled for valuation training
    lr_scheduler: str = "reduce_on_plateau"
    lr_factor: float = 0.5
    lr_patience: int = 20
    lr_min: float = 1e-6
    lr_threshold: float = 5e-5
    lr_cooldown: int = 5
    huber_delta: float = 0.5

    # Prediction clipping in log-space
    use_quantile_clip: bool = True
    clip_q_lo: float = 0.005
    clip_q_hi: float = 0.995
    clip_margin: float = 0.2  # fallback when quantile clipping is disabled

    # Feature transform
    use_log1p_feature_transform: bool = True

    # Model
    hidden_dims: Tuple[int, ...] = (256, 128)
    dropout: float = 0.2
    activation: str = "relu"
    init: str = "he"
    weight_scale: float = 0.01
    l2_in_layers: float = 1e-6


def _ensure_dir(p: Path) -> None:
    p.mkdir(parents=True, exist_ok=True)


def _select_feature_columns(df: pd.DataFrame, target_col: str) -> List[str]:
    exclude = {
        target_col,
        "ticker",
        "cik",
        "period_end",
        "timeframe",
        "has_income_statement",
        "has_cash_flow",
        "fiscal_year",
    }
    feats: List[str] = []
    for c in df.columns:
        if c in exclude:
            continue
        if pd.api.types.is_numeric_dtype(df[c]):
            feats.append(c)
    return feats


def _time_aware_split(
    df: pd.DataFrame,
    time_col: str,
    min_val_rows: int,
    seed: int,
    val_ratio_fallback: float,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    rng = np.random.default_rng(seed)

    if time_col in df.columns and pd.api.types.is_numeric_dtype(df[time_col]):
        years = np.sort(df[time_col].dropna().unique())
        if len(years) >= 2:
            last_year = years[-1]
            val_df = df[df[time_col] == last_year].copy()

            if len(val_df) < min_val_rows and len(years) >= 3:
                last2 = years[-2:]
                val_df = df[df[time_col].isin(last2)].copy()

            train_df = df.drop(val_df.index).copy()
            if len(train_df) > 0 and len(val_df) > 0:
                return train_df, val_df

    idx = np.arange(len(df))
    rng.shuffle(idx)
    n_val = max(1, int(len(df) * val_ratio_fallback))
    val_idx = idx[:n_val]
    train_idx = idx[n_val:]
    return df.iloc[train_idx].copy(), df.iloc[val_idx].copy()


def _fit_standard_scaler(X: np.ndarray) -> Dict[str, np.ndarray]:
    mu = X.mean(axis=0)
    sigma = X.std(axis=0)
    sigma = np.where(sigma < 1e-12, 1.0, sigma)
    return {"mu": mu, "sigma": sigma}


def _transform_standard_scaler(X: np.ndarray, scaler: Dict[str, np.ndarray]) -> np.ndarray:
    return (X - scaler["mu"]) / scaler["sigma"]


def _select_log1p_features(
    train_df: pd.DataFrame,
    feature_cols: List[str],
    enabled: bool = True,
) -> List[str]:
    if not enabled:
        return []

    include_tokens = (
        "revenue",
        "asset",
        "liabilit",
        "equity",
        "cash",
        "debt",
        "income",
        "cfo",
        "capex",
        "fcf",
    )
    exclude_tokens = ("ratio", "_to_", "margin", "coverage", "roe", "roa")

    picked: List[str] = []
    for c in feature_cols:
        lc = c.lower()
        if any(t in lc for t in exclude_tokens):
            continue
        if not any(t in lc for t in include_tokens):
            continue

        s = (
            pd.to_numeric(train_df[c], errors="coerce")
            .replace([np.inf, -np.inf], np.nan)
            .dropna()
        )
        if s.empty:
            continue

        if float(s.min()) >= 0.0:
            picked.append(c)
    return picked


def build_xy(
    df: pd.DataFrame,
    cfg: ValuationConfig,
    feature_cols: Optional[List[str]] = None,
    log1p_features: Optional[List[str]] = None,
) -> Tuple[np.ndarray, np.ndarray, List[str]]:
    if cfg.target_col not in df.columns:
        cols = list(df.columns)
        preview = cols[:60]
        raise ValueError(
            f"target_col='{cfg.target_col}' not found. "
            f"Available columns include: {preview}{'...' if len(cols) > 60 else ''}"
        )

    if feature_cols is None:
        feature_cols = _select_feature_columns(df, cfg.target_col)
    if not feature_cols:
        raise ValueError("No numeric feature columns found after exclusions.")

    missing = [c for c in feature_cols if c not in df.columns]
    if missing:
        raise ValueError(f"Missing expected feature columns: {missing[:10]}")

    X_df = (
        df[feature_cols]
        .replace([np.inf, -np.inf], np.nan)
        .fillna(0.0)
        .copy()
    )

    if log1p_features:
        for c in log1p_features:
            if c in X_df.columns:
                v = X_df[c].to_numpy(dtype=np.float64, copy=False)
                X_df[c] = np.log1p(np.clip(v, a_min=0.0, a_max=None))

    X = X_df.to_numpy(dtype=np.float32)

    y = (
        df[cfg.target_col]
        .replace([np.inf, -np.inf], np.nan)
        .fillna(0.0)
        .to_numpy(dtype=np.float32)
    )

    if cfg.log_target:
        y = np.clip(y, a_min=0.0, a_max=None)
        y = np.log1p(y)

    y = y.reshape(-1, 1)
    return X, y, feature_cols


def _rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(np.sqrt(np.mean((y_true - y_pred) ** 2)))


def _r2(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    ss_res = float(np.sum((y_true - y_pred) ** 2))
    ss_tot = float(np.sum((y_true - np.mean(y_true)) ** 2))
    return 1.0 - ss_res / (ss_tot + 1e-12)


def _mape(y_true: np.ndarray, y_pred: np.ndarray, eps: float = 1e-8) -> float:
    y_true = np.maximum(y_true, eps)
    return float(np.mean(np.abs((y_true - y_pred) / y_true)))

def _make_asset_weights_from_log_target(
    y_log: np.ndarray,
    power: float = 0.25,      # milder than 0.5
    w_min: float = 0.5,
    w_max: float = 2.0,
    eps: float = 1.0,
) -> np.ndarray:
    """
    y_log: log1p(assets) shape (N,1)
    weights ~ 1/(assets^power), normalized, then clipped to [w_min, w_max]
    """
    y_raw = np.expm1(y_log).reshape(-1)
    y_raw = np.maximum(y_raw, eps)

    w = 1.0 / (y_raw ** power)
    w = w / (np.mean(w) + 1e-12)
    w = np.clip(w, w_min, w_max)
    return w.astype(np.float32)

def _to_jsonable(x):
    if isinstance(x, (np.integer, np.floating)):
        return x.item()
    if isinstance(x, (np.ndarray,)):
        return x.tolist()
    return x



def main() -> None:
    cfg = ValuationConfig()

    env_path = os.getenv("VAL_DATA_PATH")
    if env_path:
        cfg.data_path = Path(env_path)

    if not cfg.data_path.exists():
        raise SystemExit(f"Dataset not found at: {cfg.data_path.resolve()}")

    _ensure_dir(cfg.out_dir)

    df = pd.read_csv(cfg.data_path)

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
        cfg,
        feature_cols=feature_names,
        log1p_features=log1p_feature_names,
    )
    X_val, y_val, _ = build_xy(
        val_df,
        cfg,
        feature_cols=feature_names,
        log1p_features=log1p_feature_names,
    )

    x_scaler = _fit_standard_scaler(X_train)
    X_train_s = _transform_standard_scaler(X_train, x_scaler)
    X_val_s = _transform_standard_scaler(X_val, x_scaler)

    y_scaler = _fit_standard_scaler(y_train)
    y_train_s = _transform_standard_scaler(y_train, y_scaler)
    y_val_s = _transform_standard_scaler(y_val, y_scaler)

    input_dim = int(X_train_s.shape[1])
    model = MLP.from_dims(
        input_dim=input_dim,
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



    w_train = _make_asset_weights_from_log_target(y_train, power=0.25, w_min=0.5, w_max=2.0)
    # optional: also weight val loss the same way (I recommend starting with None)
    # w_val = _make_asset_weights_from_log_target(y_val)
    w_val = None

    history = fit(
        model=model,
        criterion=loss_fn,
        optimizer=opt,
        X_train=X_train_s,
        y_train=y_train_s,
        X_val=X_val_s,
        y_val=y_val_s,
        cfg=train_cfg,
        metric_fn=None,
        w_train=w_train,
        w_val=w_val,
    )

    model.eval()
    yhat_train_s = model.forward(X_train_s, training=False)
    yhat_val_s = model.forward(X_val_s, training=False)

    # Unscale to log space
    yhat_train_log = yhat_train_s * y_scaler["sigma"] + y_scaler["mu"]
    yhat_val_log = yhat_val_s * y_scaler["sigma"] + y_scaler["mu"]

    train_log_min = float(np.min(y_train))  # y_train is log1p(target)
    train_log_max = float(np.max(y_train))
    if cfg.use_quantile_clip:
        clip_lo = float(np.quantile(y_train, cfg.clip_q_lo))
        clip_hi = float(np.quantile(y_train, cfg.clip_q_hi))
    else:
        clip_lo = train_log_min - float(cfg.clip_margin)
        clip_hi = train_log_max + float(cfg.clip_margin)

    yhat_train_log_clip = np.clip(yhat_train_log, clip_lo, clip_hi)
    yhat_val_log_clip = np.clip(yhat_val_log, clip_lo, clip_hi)

    if cfg.log_target:
        yhat_train = np.expm1(yhat_train_log_clip)
        yhat_val = np.expm1(yhat_val_log_clip)
        y_train_orig = np.expm1(y_train)
        y_val_orig = np.expm1(y_val)
    else:
        yhat_train = yhat_train_log
        yhat_val = yhat_val_log
        y_train_orig = y_train
        y_val_orig = y_val

    # Save history
    hist_path = cfg.out_dir / "history.json"
    with open(hist_path, "w") as f:
        json.dump(history, f, indent=2)

    # Save scalers + feature names
    np.savez(
        cfg.out_dir / "scalers_and_features.npz",
        x_mu=x_scaler["mu"],
        x_sigma=x_scaler["sigma"],
        y_mu=y_scaler["mu"],
        y_sigma=y_scaler["sigma"],
        feature_names=np.array(feature_names, dtype=object),
        feature_log1p_names=np.array(log1p_feature_names, dtype=object),
        target_col=np.array([cfg.target_col], dtype=object),
        log_target=np.array([cfg.log_target], dtype=object),
        use_quantile_clip=np.array([cfg.use_quantile_clip], dtype=object),
        clip_q_lo=np.array([cfg.clip_q_lo], dtype=np.float32),
        clip_q_hi=np.array([cfg.clip_q_hi], dtype=np.float32),
        clip_lo=np.array([clip_lo], dtype=np.float32),
        clip_hi=np.array([clip_hi], dtype=np.float32),
        clip_margin=np.array([cfg.clip_margin], dtype=np.float32),
    )

    # Save predictions
    def _meta_cols(d: pd.DataFrame) -> Dict[str, np.ndarray]:
        out: Dict[str, np.ndarray] = {}
        for c in ["ticker", "fiscal_year", "period_end", "timeframe"]:
            if c in d.columns:
                out[c] = d[c].to_numpy()
        return out

    train_meta = _meta_cols(train_df)
    val_meta = _meta_cols(val_df)

    rows: List[dict] = []
    for i in range(len(train_df)):
        r = {"split": "train", "y_true": float(y_train_orig[i, 0]), "y_pred": float(yhat_train[i, 0])}
        for k, arr in train_meta.items():
            r[k] = arr[i]
        rows.append(r)

    for i in range(len(val_df)):
        r = {"split": "val", "y_true": float(y_val_orig[i, 0]), "y_pred": float(yhat_val[i, 0])}
        for k, arr in val_meta.items():
            r[k] = arr[i]
        rows.append(r)

    preds_df = pd.DataFrame(rows)
    preds_path = cfg.out_dir / "predictions.csv"
    preds_df.to_csv(preds_path, index=False)

    # Metrics (log + raw + baseline)
    train_rmse_log = _rmse(y_train, yhat_train_log_clip)
    val_rmse_log = _rmse(y_val, yhat_val_log_clip)
    train_factor = float(np.exp(train_rmse_log))
    val_factor = float(np.exp(val_rmse_log))

    r2_train_log = _r2(y_train, yhat_train_log_clip)
    r2_val_log = _r2(y_val, yhat_val_log_clip)
    r2_train_raw = _r2(y_train_orig, yhat_train)
    r2_val_raw = _r2(y_val_orig, yhat_val)

    y_mean_log = float(np.mean(y_train))
    yhat_base_val_log = np.full_like(y_val, y_mean_log)
    baseline_rmse_log = _rmse(y_val, yhat_base_val_log)
    baseline_factor = float(np.exp(baseline_rmse_log))

    train_rmse_raw = _rmse(y_train_orig, yhat_train)
    val_rmse_raw = _rmse(y_val_orig, yhat_val)

    val_mape_raw = _mape(y_val_orig, yhat_val)

    # Bucket diagnostics (val split)
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
        )
    except Exception:
        bucket_table = None

    run_summary = {
    "target_col": cfg.target_col,
    "log_target": cfg.log_target,
    "train_rows": int(len(train_df)),
    "val_rows": int(len(val_df)),
    "years": sorted(df[cfg.time_col].dropna().unique().tolist()) if cfg.time_col in df.columns else None,
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
    },
    "training": {
        "best_epoch": _to_jsonable(history.get("best_epoch")),
        "best_val_loss": _to_jsonable(history.get("best_val_loss")),
        "stopped_early": _to_jsonable(history.get("stopped_early")),
        "final_lr": float(history.get("lr", [float(cfg.lr)])[-1]) if history.get("lr") else float(cfg.lr),
        "lr_reductions": int(history.get("lr_reductions", 0)),
    },
    "metrics": {
        "baseline_rmse_log": float(baseline_rmse_log),
        "baseline_factor": float(baseline_factor),
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
    print(f"Saved: {summary_path}")


    print("=== Done ===")
    print(f"Target: {cfg.target_col} (log_target={cfg.log_target})")
    print(f"Train rows: {len(train_df)} | Val rows: {len(val_df)}")
    if cfg.time_col in df.columns:
        yrs = sorted(df[cfg.time_col].dropna().unique().tolist())
        print(f"Years in data: {yrs}")

    if cfg.use_quantile_clip:
        print(
            f"Log clip quantiles: [{cfg.clip_q_lo}, {cfg.clip_q_hi}] => "
            f"[{clip_lo:.3f}, {clip_hi:.3f}]"
        )
    else:
        print(f"Log clip range: [{clip_lo:.3f}, {clip_hi:.3f}] (margin={cfg.clip_margin})")

    print(f"log1p-transformed feature count: {len(log1p_feature_names)}")

    print(f"Baseline RMSE (log): {baseline_rmse_log:.4f}  (~×{baseline_factor:.2f})")
    print(f"Train RMSE  (log): {train_rmse_log:.4f}  (~×{train_factor:.2f})")
    print(f"Val   RMSE  (log): {val_rmse_log:.4f}  (~×{val_factor:.2f})")

    print(f"Train R2 (log): {r2_train_log:.4f} | Val R2 (log): {r2_val_log:.4f}")
    print(f"Train R2 (raw): {r2_train_raw:.4f} | Val R2 (raw): {r2_val_raw:.4f}")

    print(f"Val MAPE (raw): {val_mape_raw:.4f}")
    print(f"Train RMSE (raw): {train_rmse_raw:,.4f}")
    print(f"Val   RMSE (raw): {val_rmse_raw:,.4f}")

    if bucket_table is not None:
        print("\nVal performance by size bucket (quintiles):")
        print(bucket_table)

    print(f"Saved: {hist_path}")
    print(f"Saved: {preds_path}")
    print(f"Saved: {cfg.out_dir / 'scalers_and_features.npz'}")


if __name__ == "__main__":
    main()
