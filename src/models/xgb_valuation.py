from __future__ import annotations

import json
import os
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import xgboost as xgb


@dataclass
class XGBValuationConfig:
    # Data
    data_path: Path = Path("data/processed/main_dataset.csv")
    out_dir: Path = Path("data/processed/xgb_valuation_artifacts")

    # Target
    target_col: str = "total_assets"
    log_target: bool = True

    # Splitting
    time_col: str = "fiscal_year"
    min_val_rows: int = 20
    random_seed: int = 42
    val_ratio_fallback: float = 0.2

    # Sample weights (gently emphasize smaller firms)
    use_sample_weights: bool = True
    weight_power: float = 0.25
    w_min: float = 0.5
    w_max: float = 2.0

    # Quantile clipping (winsorization) in log-space for stability
    use_quantile_clip: bool = True
    clip_q_lo: float = 0.005
    clip_q_hi: float = 0.995

    # XGBoost (native API)
    num_boost_round: int = 10000
    early_stopping_rounds: int = 50
    tree_method: str = "hist"
    max_depth: int = 6
    eta: float = 0.03
    subsample: float = 0.9
    colsample_bytree: float = 0.9
    reg_lambda: float = 1.0
    reg_alpha: float = 0.0
    min_child_weight: float = 1.0
    gamma: float = 0.0
    nthread: int = -1

    verbose_eval: int = 50


def _ensure_dir(p: Path) -> None:
    p.mkdir(parents=True, exist_ok=True)


def _select_feature_columns(df: pd.DataFrame, target_col: str, time_col: str) -> List[str]:
    exclude = {
        target_col,
        "ticker",
        "cik",
        "period_end",
        "timeframe",
        "has_income_statement",
        "has_cash_flow",
        time_col,
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


def build_xy(df: pd.DataFrame, cfg: XGBValuationConfig) -> Tuple[np.ndarray, np.ndarray, List[str]]:
    if cfg.target_col not in df.columns:
        cols = list(df.columns)
        preview = cols[:60]
        raise ValueError(
            f"target_col='{cfg.target_col}' not found. "
            f"Available columns include: {preview}{'...' if len(cols) > 60 else ''}"
        )

    feature_cols = _select_feature_columns(df, cfg.target_col, cfg.time_col)
    if not feature_cols:
        raise ValueError("No numeric feature columns found after exclusions.")

    X = (
        df[feature_cols]
        .replace([np.inf, -np.inf], np.nan)
        .fillna(0.0)
        .to_numpy(dtype=np.float32)
    )

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
    power: float,
    w_min: float,
    w_max: float,
    eps: float = 1.0,
) -> np.ndarray:
    y_raw = np.expm1(y_log).reshape(-1)
    y_raw = np.maximum(y_raw, eps)
    w = 1.0 / (y_raw ** power)
    w = w / (np.mean(w) + 1e-12)
    w = np.clip(w, w_min, w_max)
    return w.astype(np.float32)


def _meta_cols(d: pd.DataFrame) -> Dict[str, np.ndarray]:
    out: Dict[str, np.ndarray] = {}
    for c in ["ticker", "fiscal_year", "period_end", "timeframe"]:
        if c in d.columns:
            out[c] = d[c].to_numpy()
    return out

def _jsonable(x):
    if isinstance(x, Path):
        return str(x)
    if isinstance(x, (np.integer, np.floating)):
        return x.item()
    if isinstance(x, np.ndarray):
        return x.tolist()
    return x


def main() -> None:
    cfg = XGBValuationConfig()

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

    X_train, y_train, feature_names = build_xy(train_df, cfg)
    X_val, y_val, _ = build_xy(val_df, cfg)

    y_train_1d = y_train.reshape(-1).astype(np.float32)
    y_val_1d = y_val.reshape(-1).astype(np.float32)

    w_train = None
    if cfg.use_sample_weights:
        w_train = _make_asset_weights_from_log_target(
            y_train, power=cfg.weight_power, w_min=cfg.w_min, w_max=cfg.w_max
        )

    dtrain = xgb.DMatrix(X_train, label=y_train_1d, weight=w_train)
    dval = xgb.DMatrix(X_val, label=y_val_1d)

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
        dtrain=dtrain,
        num_boost_round=cfg.num_boost_round,
        evals=[(dtrain, "train"), (dval, "val")],
        early_stopping_rounds=cfg.early_stopping_rounds,
        verbose_eval=cfg.verbose_eval,
    )

    best_iteration = int(getattr(booster, "best_iteration", booster.num_boosted_rounds() - 1))
    it_range = (0, best_iteration + 1)

    yhat_train_log = booster.predict(dtrain, iteration_range=it_range).reshape(-1, 1)
    yhat_val_log = booster.predict(dval, iteration_range=it_range).reshape(-1, 1)

    clip_lo = float(np.min(y_train))
    clip_hi = float(np.max(y_train))
    if cfg.use_quantile_clip:
        clip_lo = float(np.quantile(y_train, cfg.clip_q_lo))
        clip_hi = float(np.quantile(y_train, cfg.clip_q_hi))
        yhat_train_log = np.clip(yhat_train_log, clip_lo, clip_hi)
        yhat_val_log = np.clip(yhat_val_log, clip_lo, clip_hi)

    if cfg.log_target:
        y_train_raw = np.expm1(y_train)
        y_val_raw = np.expm1(y_val)
        yhat_train_raw = np.expm1(yhat_train_log)
        yhat_val_raw = np.expm1(yhat_val_log)
    else:
        y_train_raw = y_train
        y_val_raw = y_val
        yhat_train_raw = yhat_train_log
        yhat_val_raw = yhat_val_log

    y_mean_log = float(np.mean(y_train))
    yhat_base_val_log = np.full_like(y_val, y_mean_log)
    baseline_rmse_log = _rmse(y_val, yhat_base_val_log)
    baseline_factor = float(np.exp(baseline_rmse_log))

    train_rmse_log = _rmse(y_train, yhat_train_log)
    val_rmse_log = _rmse(y_val, yhat_val_log)
    train_factor = float(np.exp(train_rmse_log))
    val_factor = float(np.exp(val_rmse_log))

    r2_train_log = _r2(y_train, yhat_train_log)
    r2_val_log = _r2(y_val, yhat_val_log)

    train_rmse_raw = _rmse(y_train_raw, yhat_train_raw)
    val_rmse_raw = _rmse(y_val_raw, yhat_val_raw)
    r2_train_raw = _r2(y_train_raw, yhat_train_raw)
    r2_val_raw = _r2(y_val_raw, yhat_val_raw)

    val_mape_raw = _mape(y_val_raw, yhat_val_raw)

    train_meta = _meta_cols(train_df)
    val_meta = _meta_cols(val_df)

    rows: List[dict] = []
    for i in range(len(train_df)):
        r = {"split": "train", "y_true": float(y_train_raw[i, 0]), "y_pred": float(yhat_train_raw[i, 0])}
        for k, arr in train_meta.items():
            r[k] = arr[i]
        rows.append(r)

    for i in range(len(val_df)):
        r = {"split": "val", "y_true": float(y_val_raw[i, 0]), "y_pred": float(yhat_val_raw[i, 0])}
        for k, arr in val_meta.items():
            r[k] = arr[i]
        rows.append(r)

    preds_df = pd.DataFrame(rows)
    preds_path = cfg.out_dir / "predictions.csv"
    preds_df.to_csv(preds_path, index=False)

    val_only = preds_df[preds_df["split"] == "val"].copy()
    val_only["abs_err"] = np.abs(val_only["y_pred"] - val_only["y_true"])
    val_only["rel_err"] = val_only["abs_err"] / np.maximum(val_only["y_true"], 1e-8)

    bucket_table = None
    try:
        val_only["size_bucket"] = pd.qcut(val_only["y_true"], q=5, duplicates="drop")
        bucket_table = val_only.groupby("size_bucket", observed=False).agg(
            n=("y_true", "size"),
            mape=("rel_err", "mean"),
            median_rel=("rel_err", "median"),
            rmse=("abs_err", lambda x: float(np.sqrt(np.mean(x**2)))),
            mean_true=("y_true", "mean"),
            mean_pred=("y_pred", "mean"),
        )
    except Exception:
        bucket_table = None

    model_path = cfg.out_dir / "xgb_model.json"
    booster.save_model(str(model_path))

    summary = {
        "config": {k: _jsonable(v) for k, v in asdict(cfg).items()},
        "feature_names": feature_names,
        "target_col": cfg.target_col,
        "log_target": cfg.log_target,
        "train_rows": int(len(train_df)),
        "val_rows": int(len(val_df)),
        "years": sorted(df[cfg.time_col].dropna().unique().tolist()) if cfg.time_col in df.columns else None,
        "best_iteration": best_iteration,
        "clip": {
            "use_quantile_clip": cfg.use_quantile_clip,
            "q_lo": cfg.clip_q_lo,
            "q_hi": cfg.clip_q_hi,
            "clip_lo": clip_lo,
            "clip_hi": clip_hi,
        },
        "metrics": {
            "baseline_rmse_log": baseline_rmse_log,
            "baseline_factor": baseline_factor,
            "train_rmse_log": train_rmse_log,
            "train_factor": train_factor,
            "val_rmse_log": val_rmse_log,
            "val_factor": val_factor,
            "train_r2_log": r2_train_log,
            "val_r2_log": r2_val_log,
            "train_rmse_raw": train_rmse_raw,
            "val_rmse_raw": val_rmse_raw,
            "train_r2_raw": r2_train_raw,
            "val_r2_raw": r2_val_raw,
            "val_mape_raw": val_mape_raw,
        },
    }
    summary_path = cfg.out_dir / "run_summary.json"
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)

    print("=== Done (XGBoost Baseline, native API) ===")
    print(f"Target: {cfg.target_col} (log_target={cfg.log_target})")
    print(f"Train rows: {len(train_df)} | Val rows: {len(val_df)}")
    if cfg.time_col in df.columns:
        yrs = sorted(df[cfg.time_col].dropna().unique().tolist())
        print(f"Years in data: {yrs}")
    print(f"Best iteration (early stopping): {best_iteration}")

    if cfg.use_quantile_clip:
        print(f"Log clip quantiles: [{cfg.clip_q_lo}, {cfg.clip_q_hi}] => [{clip_lo:.3f}, {clip_hi:.3f}]")

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

    print(f"Saved: {preds_path}")
    print(f"Saved: {summary_path}")
    print(f"Saved: {model_path}")


if __name__ == "__main__":
    main()

