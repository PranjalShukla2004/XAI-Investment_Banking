# src/models/valuation/valuation2.py
from __future__ import annotations

import ast
import json
import os
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from src.models.nn.losses import HuberLoss
from src.models.nn.mlp import MLP
from src.models.nn.optimizer import Adam
from src.models.nn.train import TrainConfig, fit
from src.models.valuation.valuation import (
    ValuationConfig,
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
class Valuation2Config(ValuationConfig):
    out_dir: Path = Path("experiments/valuation/runs/valuation2_artifacts")
    lr_scheduler: str = "dynamic_on_plateau"
    lr_increase_factor: float = 1.5
    lr_max: float = 5e-3
    dynamic_lr_start_mode: str = "increase"
    dynamic_lr_switch_patience: int = 2


def _parse_sentiment_list(value: object) -> list[float]:
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return []
    if isinstance(value, (list, tuple, np.ndarray)):
        raw_values = list(value)
    else:
        text = str(value).strip()
        if not text:
            return []
        try:
            parsed = ast.literal_eval(text)
        except (SyntaxError, ValueError):
            return []
        if isinstance(parsed, (list, tuple, np.ndarray)):
            raw_values = list(parsed)
        else:
            return []

    out: list[float] = []
    for item in raw_values:
        try:
            num = float(item)
        except (TypeError, ValueError):
            continue
        if np.isfinite(num):
            out.append(num)
    return out


def _expand_news_sentiment_features(df: pd.DataFrame) -> pd.DataFrame:
    if "news_sentiment_score" not in df.columns:
        return df

    out = df.copy()
    parsed = out["news_sentiment_score"].apply(_parse_sentiment_list)

    def _safe_stat(values: list[float], fn, default: float = 0.0) -> float:
        if not values:
            return default
        arr = np.asarray(values, dtype=np.float64)
        result = fn(arr)
        return float(result) if np.isfinite(result) else default

    out["news_sentiment_mean"] = parsed.apply(lambda vals: _safe_stat(vals, np.mean))
    out["news_sentiment_std"] = parsed.apply(lambda vals: _safe_stat(vals, np.std))
    out["news_sentiment_min"] = parsed.apply(lambda vals: _safe_stat(vals, np.min))
    out["news_sentiment_max"] = parsed.apply(lambda vals: _safe_stat(vals, np.max))
    out["news_sentiment_last"] = parsed.apply(lambda vals: float(vals[-1]) if vals else 0.0)
    out["news_sentiment_sum"] = parsed.apply(lambda vals: _safe_stat(vals, np.sum))
    out["news_sentiment_pos_share"] = parsed.apply(
        lambda vals: float(np.mean(np.asarray(vals, dtype=np.float64) > 0.0)) if vals else 0.0
    )
    out["news_sentiment_neg_share"] = parsed.apply(
        lambda vals: float(np.mean(np.asarray(vals, dtype=np.float64) < 0.0)) if vals else 0.0
    )
    out["news_sentiment_nonzero_share"] = parsed.apply(
        lambda vals: float(np.mean(np.asarray(vals, dtype=np.float64) != 0.0)) if vals else 0.0
    )
    out["has_news_sentiment"] = parsed.apply(lambda vals: float(len(vals) > 0))
    return out


def _meta_cols(d: pd.DataFrame) -> dict[str, np.ndarray]:
    out: dict[str, np.ndarray] = {}
    for c in ["ticker", "fiscal_year", "period_end", "timeframe"]:
        if c in d.columns:
            out[c] = d[c].to_numpy()
    return out


def main() -> None:
    cfg = Valuation2Config()

    env_path = os.getenv("VAL2_DATA_PATH") or os.getenv("VAL_DATA_PATH")
    if env_path:
        cfg.data_path = Path(env_path)

    env_out_dir = os.getenv("VAL2_OUT_DIR")
    if env_out_dir:
        cfg.out_dir = Path(env_out_dir)

    env_scheduler = os.getenv("VAL2_LR_SCHEDULER")
    if env_scheduler:
        cfg.lr_scheduler = env_scheduler.strip()

    env_increase_factor = os.getenv("VAL2_LR_INCREASE_FACTOR")
    if env_increase_factor:
        cfg.lr_increase_factor = float(env_increase_factor)

    env_lr_max = os.getenv("VAL2_LR_MAX")
    if env_lr_max:
        cfg.lr_max = float(env_lr_max)

    env_start_mode = os.getenv("VAL2_DYNAMIC_LR_START_MODE")
    if env_start_mode:
        cfg.dynamic_lr_start_mode = env_start_mode.strip()

    env_switch_patience = os.getenv("VAL2_DYNAMIC_LR_SWITCH_PATIENCE")
    if env_switch_patience:
        cfg.dynamic_lr_switch_patience = int(env_switch_patience)

    env_hidden_dims = os.getenv("VAL2_HIDDEN_DIMS")
    if env_hidden_dims:
        parsed_dims = tuple(int(tok.strip()) for tok in env_hidden_dims.split(",") if tok.strip())
        if not parsed_dims:
            raise SystemExit("VAL2_HIDDEN_DIMS is set but empty; expected comma-separated integers like '256,128'")
        cfg.hidden_dims = parsed_dims

    if not cfg.data_path.exists():
        raise SystemExit(f"Dataset not found at: {cfg.data_path.resolve()}")

    _ensure_dir(cfg.out_dir)

    df = pd.read_csv(cfg.data_path)
    df = _expand_news_sentiment_features(df)

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

    w_train = _make_asset_weights_from_log_target(y_train, power=0.25, w_min=0.5, w_max=2.0)
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
        w_val=None,
    )

    model.eval()
    yhat_train_s = model.forward(X_train_s, training=False)
    yhat_val_s = model.forward(X_val_s, training=False)

    yhat_train_log = yhat_train_s * y_scaler["sigma"] + y_scaler["mu"]
    yhat_val_log = yhat_val_s * y_scaler["sigma"] + y_scaler["mu"]

    train_log_min = float(np.min(y_train))
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

    hist_path = cfg.out_dir / "history.json"
    hist_path.write_text(json.dumps(history, indent=2), encoding="utf-8")

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

    train_meta = _meta_cols(train_df)
    val_meta = _meta_cols(val_df)
    rows: list[dict[str, object]] = []
    for i in range(len(train_df)):
        row = {"split": "train", "y_true": float(y_train_orig[i, 0]), "y_pred": float(yhat_train[i, 0])}
        for k, arr in train_meta.items():
            row[k] = arr[i]
        rows.append(row)
    for i in range(len(val_df)):
        row = {"split": "val", "y_true": float(y_val_orig[i, 0]), "y_pred": float(yhat_val[i, 0])}
        for k, arr in val_meta.items():
            row[k] = arr[i]
        rows.append(row)

    preds_df = pd.DataFrame(rows)
    preds_path = cfg.out_dir / "predictions.csv"
    preds_df.to_csv(preds_path, index=False)

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
            "l1_in_layers": float(cfg.l1_in_layers),
            "l2_in_layers": float(cfg.l2_in_layers),
            "lr_scheduler": str(cfg.lr_scheduler),
            "lr_factor": float(cfg.lr_factor),
            "lr_increase_factor": float(cfg.lr_increase_factor),
            "lr_patience": int(cfg.lr_patience),
            "lr_min": float(cfg.lr_min),
            "lr_max": float(cfg.lr_max),
            "lr_threshold": float(cfg.lr_threshold),
            "lr_cooldown": int(cfg.lr_cooldown),
            "dynamic_lr_start_mode": str(cfg.dynamic_lr_start_mode),
            "dynamic_lr_switch_patience": int(cfg.dynamic_lr_switch_patience),
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
            "lr_increases": int(history.get("lr_increases", 0)),
            "lr_scheduler_switches": int(history.get("lr_scheduler_switches", 0)),
            "lr_scheduler_mode_last": history.get("lr_scheduler_mode", [None])[-1] if history.get("lr_scheduler_mode") else None,
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
    summary_path.write_text(json.dumps(run_summary, indent=2), encoding="utf-8")

    print("=== Done (valuation2) ===")
    print(f"Target: {cfg.target_col} (log_target={cfg.log_target})")
    print(f"Train rows: {len(train_df)} | Val rows: {len(val_df)}")
    print(f"LR scheduler: {cfg.lr_scheduler}")
    print(
        "Scheduler activity: "
        f"reductions={history.get('lr_reductions', 0)} | "
        f"increases={history.get('lr_increases', 0)} | "
        f"switches={history.get('lr_scheduler_switches', 0)}"
    )
    print(f"Val RMSE (log): {val_rmse_log:.4f} (~x{val_factor:.2f})")
    print(f"Val MAPE (raw): {val_mape_raw:.4f}")
    print(f"Saved: {hist_path}")
    print(f"Saved: {preds_path}")
    print(f"Saved: {summary_path}")
    print(f"Saved: {cfg.out_dir / 'scalers_and_features.npz'}")


if __name__ == "__main__":
    main()
