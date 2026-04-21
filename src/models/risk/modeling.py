from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import xgboost as xgb
from scipy import stats

DEFAULT_MAIN_RISK_DATASET = Path("data/processed/risk/main_risk_dataset.csv")
LEGACY_MAIN_RISK_DATASET = Path("data/nprocessed/risk/risk_dataset.csv")
DEFAULT_OUT_DIR = Path("experiments/risk/runs/xgb_risk_artifacts")

IDENTIFIER_COLUMNS = {
    "ticker",
    "cik",
    "fiscal_year",
    "period_end",
    "timeframe",
}
TARGET_AND_AUX_COLUMNS = {
    "future_1y_max_drawdown",
    "drawdown_severity",
    "target_status",
    "target_usable",
    "target_issue_codes",
}


def _default_risk_dataset_path() -> Path:
    if DEFAULT_MAIN_RISK_DATASET.exists():
        return DEFAULT_MAIN_RISK_DATASET
    return LEGACY_MAIN_RISK_DATASET


@dataclass
class RiskXGBConfig:
    data_path: Path = _default_risk_dataset_path()
    out_dir: Path = DEFAULT_OUT_DIR
    target_col: str = "drawdown_severity"
    raw_target_col: str = "future_1y_max_drawdown"
    time_col: str = "fiscal_year"
    ticker_col: str = "ticker"
    period_end_col: str = "period_end"
    baseline_feature_col: str = "drawdown_126d"
    val_ratio_fallback: float = 0.15
    min_val_rows: int = 250
    num_boost_round: int = 4000
    early_stopping_rounds: int = 100
    tree_method: str = "hist"
    max_depth: int = 5
    eta: float = 0.03
    subsample: float = 0.9
    colsample_bytree: float = 0.9
    reg_lambda: float = 1.0
    reg_alpha: float = 0.0
    min_child_weight: float = 2.0
    gamma: float = 0.0
    nthread: int = -1
    random_seed: int = 42
    use_sample_weights: bool = True
    sample_weight_power: float = 1.0
    sample_weight_min: float = 1.0
    sample_weight_max: float = 2.0
    top_fraction: float = 0.1
    event_thresholds: tuple[float, ...] = (0.3, 0.5)


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def jsonable(value: object) -> object:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, (np.integer, np.floating)):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, pd.Timestamp):
        return value.strftime("%Y-%m-%d")
    if isinstance(value, tuple):
        return list(value)
    return value


def write_json(path: Path, payload: object) -> None:
    ensure_dir(path.parent)
    path.write_text(json.dumps(payload, default=jsonable, indent=2), encoding="utf-8")


def _normalize_tickers(series: pd.Series) -> pd.Series:
    out = series.astype(str).str.upper().str.strip()
    return out.replace({"": pd.NA, "NAN": pd.NA, "NONE": pd.NA, "<NA>": pd.NA, "NULL": pd.NA})


def load_risk_dataset(path: Path, cfg: RiskXGBConfig) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(
            f"Risk dataset not found: {path}. "
            "If you have not prepared split-specific risk datasets yet, run "
            "`python -m src.models.risk.prepare_datasets` first."
        )

    df = pd.read_csv(path)
    if cfg.target_col not in df.columns:
        raise ValueError(f"Risk dataset {path} is missing target column {cfg.target_col!r}.")

    if cfg.ticker_col in df.columns:
        df[cfg.ticker_col] = _normalize_tickers(df[cfg.ticker_col])
    if cfg.period_end_col in df.columns:
        df[cfg.period_end_col] = pd.to_datetime(df[cfg.period_end_col], errors="coerce").dt.strftime("%Y-%m-%d")
    if cfg.time_col in df.columns:
        df[cfg.time_col] = pd.to_numeric(df[cfg.time_col], errors="coerce").astype("Int64")

    df[cfg.target_col] = pd.to_numeric(df[cfg.target_col], errors="coerce").clip(lower=0.0, upper=1.0)
    if cfg.raw_target_col in df.columns:
        df[cfg.raw_target_col] = pd.to_numeric(df[cfg.raw_target_col], errors="coerce")

    if "target_usable" in df.columns:
        df["target_usable"] = df["target_usable"].fillna(False).astype(bool)

    df = df.dropna(subset=[cfg.target_col]).copy()
    if cfg.ticker_col in df.columns and cfg.period_end_col in df.columns:
        df = df.dropna(subset=[cfg.ticker_col, cfg.period_end_col]).copy()
        sort_cols: list[str] = []
        if cfg.time_col in df.columns:
            sort_cols.append(cfg.time_col)
        sort_cols.extend([cfg.ticker_col, cfg.period_end_col])
        df = df.sort_values(sort_cols, kind="stable")
    elif cfg.time_col in df.columns:
        df = df.sort_values([cfg.time_col], kind="stable")
    return df.reset_index(drop=True)


def _as_numeric_series(series: pd.Series) -> pd.Series:
    if pd.api.types.is_bool_dtype(series):
        return series.astype(float)
    return pd.to_numeric(series, errors="coerce")


def select_feature_columns(df: pd.DataFrame, cfg: RiskXGBConfig) -> tuple[list[str], pd.DataFrame]:
    excluded = set(IDENTIFIER_COLUMNS) | set(TARGET_AND_AUX_COLUMNS)
    rows: list[dict[str, object]] = []
    feature_cols: list[str] = []

    for col in df.columns:
        if col in excluded or col.startswith("future_"):
            continue
        numeric = _as_numeric_series(df[col])
        non_null = int(numeric.notna().sum())
        nunique = int(numeric.dropna().nunique())
        row = {
            "feature": col,
            "non_null_rows": non_null,
            "missing_rate": float(1.0 - (non_null / max(len(df), 1))),
            "nunique_non_null": nunique,
        }
        if non_null < 2 or nunique <= 1:
            row["selected"] = False
            rows.append(row)
            continue
        feature_cols.append(col)
        row["selected"] = True
        rows.append(row)

    return feature_cols, pd.DataFrame(rows).sort_values(["selected", "feature"], ascending=[False, True])


def _build_matrix(df: pd.DataFrame, feature_cols: list[str]) -> np.ndarray:
    if not feature_cols:
        raise ValueError("No usable feature columns were found for the risk model.")
    cols = []
    for col in feature_cols:
        cols.append(_as_numeric_series(df[col]).to_numpy(dtype=np.float32))
    return np.column_stack(cols).astype(np.float32, copy=False)


def _target_array(df: pd.DataFrame, target_col: str) -> np.ndarray:
    return pd.to_numeric(df[target_col], errors="coerce").to_numpy(dtype=np.float32)


def fit_standard_scaler(X: np.ndarray) -> dict[str, np.ndarray]:
    arr = np.asarray(X, dtype=np.float64)
    mu = np.nanmean(arr, axis=0)
    sigma = np.nanstd(arr, axis=0)
    mu = np.where(np.isfinite(mu), mu, 0.0)
    sigma = np.where((~np.isfinite(sigma)) | (np.asarray(sigma) < 1e-12), 1.0, sigma)
    return {"mu": np.asarray(mu), "sigma": np.asarray(sigma)}


def transform_standard_scaler(X: np.ndarray, scaler: dict[str, np.ndarray]) -> np.ndarray:
    arr = np.asarray(X, dtype=np.float64)
    arr = np.where(np.isfinite(arr), arr, scaler["mu"])
    return (arr - scaler["mu"]) / scaler["sigma"]


def _clip_predictions(preds: np.ndarray) -> np.ndarray:
    return np.clip(np.asarray(preds, dtype=np.float64).reshape(-1), a_min=0.0, a_max=1.0)


def _safe_corr(method: str, a: np.ndarray, b: np.ndarray) -> float:
    a = np.asarray(a, dtype=np.float64).reshape(-1)
    b = np.asarray(b, dtype=np.float64).reshape(-1)
    if a.size < 2 or np.allclose(np.nanstd(a), 0.0) or np.allclose(np.nanstd(b), 0.0):
        return float("nan")
    if method == "pearson":
        corr = stats.pearsonr(a, b).statistic
    elif method == "spearman":
        corr = stats.spearmanr(a, b, nan_policy="omit").statistic
    else:
        raise ValueError(f"Unknown correlation method: {method}")
    return float(corr) if corr is not None else float("nan")


def _r2(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    yt = np.asarray(y_true, dtype=np.float64)
    yp = np.asarray(y_pred, dtype=np.float64)
    ss_res = float(np.sum((yt - yp) ** 2))
    ss_tot = float(np.sum((yt - np.mean(yt)) ** 2))
    if ss_tot <= 0.0:
        return float("nan")
    return float(1.0 - ss_res / ss_tot)


def _threshold_tag(value: float) -> str:
    return f"{value:.2f}".replace(".", "p")


def _event_metrics(
    y_true: np.ndarray,
    y_score: np.ndarray,
    threshold: float,
    top_fraction: float,
) -> dict[str, float]:
    threshold = float(threshold)
    positives = np.asarray(y_true, dtype=np.float64) >= threshold
    n_rows = int(positives.size)
    k = max(1, int(round(n_rows * float(top_fraction))))
    order = np.argsort(-np.asarray(y_score, dtype=np.float64), kind="stable")
    top_idx = order[:k]
    top_positives = positives[top_idx]
    n_positive = int(np.sum(positives))

    precision = float(np.mean(top_positives))
    recall = float(np.sum(top_positives) / n_positive) if n_positive > 0 else float("nan")
    prevalence = float(np.mean(positives))
    lift = float(precision / prevalence) if prevalence > 0.0 else float("nan")
    return {
        "event_rate": prevalence,
        "top_precision": precision,
        "top_recall": recall,
        "top_lift": lift,
    }


def risk_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    cfg: RiskXGBConfig,
) -> dict[str, float]:
    yt = np.asarray(y_true, dtype=np.float64).reshape(-1)
    yp = _clip_predictions(y_pred)
    abs_err = np.abs(yp - yt)

    metrics: dict[str, float] = {
        "rows": float(yt.size),
        "target_mean": float(np.mean(yt)),
        "target_median": float(np.median(yt)),
        "prediction_mean": float(np.mean(yp)),
        "prediction_median": float(np.median(yp)),
        "mae": float(np.mean(abs_err)),
        "rmse": float(np.sqrt(np.mean((yp - yt) ** 2))),
        "r2": float(_r2(yt, yp)),
        "spearman": _safe_corr("spearman", yt, yp),
        "pearson": _safe_corr("pearson", yt, yp),
        "mean_abs_error": float(np.mean(abs_err)),
        "p90_abs_error": float(np.quantile(abs_err, 0.90)),
        "p95_abs_error": float(np.quantile(abs_err, 0.95)),
    }

    k = max(1, int(round(len(yt) * float(cfg.top_fraction))))
    top_idx = np.argsort(-yp, kind="stable")[:k]
    metrics["top_bucket_fraction"] = float(cfg.top_fraction)
    metrics["top_bucket_true_mean"] = float(np.mean(yt[top_idx]))
    metrics["top_bucket_true_median"] = float(np.median(yt[top_idx]))

    for threshold in cfg.event_thresholds:
        tag = _threshold_tag(float(threshold))
        event_stats = _event_metrics(yt, yp, threshold=float(threshold), top_fraction=float(cfg.top_fraction))
        metrics[f"event_{tag}_rate"] = float(event_stats["event_rate"])
        metrics[f"event_{tag}_top_precision"] = float(event_stats["top_precision"])
        metrics[f"event_{tag}_top_recall"] = float(event_stats["top_recall"])
        metrics[f"event_{tag}_top_lift"] = float(event_stats["top_lift"])

    return metrics


def _baseline_predictions(
    df: pd.DataFrame,
    train_target_mean: float,
    cfg: RiskXGBConfig,
) -> tuple[np.ndarray, dict[str, object]]:
    if cfg.baseline_feature_col in df.columns:
        baseline = -pd.to_numeric(df[cfg.baseline_feature_col], errors="coerce").clip(upper=0.0)
        baseline = baseline.clip(lower=0.0, upper=1.0)
        baseline = baseline.fillna(float(train_target_mean)).to_numpy(dtype=np.float64)
        baseline_source = f"clip(-{cfg.baseline_feature_col}, 0, 1)"
    else:
        baseline = np.full(len(df), float(train_target_mean), dtype=np.float64)
        baseline_source = "train_target_mean"
    return baseline, {"baseline_name": "current_drawdown_baseline", "baseline_source": baseline_source}


def _make_sample_weights(y: np.ndarray, cfg: RiskXGBConfig) -> np.ndarray | None:
    if not cfg.use_sample_weights:
        return None
    clipped = np.clip(np.asarray(y, dtype=np.float64), a_min=0.0, a_max=1.0)
    weights = 1.0 + np.power(clipped, float(cfg.sample_weight_power))
    weights = np.clip(weights, a_min=float(cfg.sample_weight_min), a_max=float(cfg.sample_weight_max))
    return weights.astype(np.float32)


def _inner_train_val_indices(train_df: pd.DataFrame, cfg: RiskXGBConfig) -> tuple[np.ndarray, np.ndarray, str]:
    n_rows = len(train_df)
    if n_rows < 2:
        idx = np.arange(n_rows)
        return idx, idx, "single_row"

    target_val_rows = min(
        max(int(round(n_rows * float(cfg.val_ratio_fallback))), int(cfg.min_val_rows)),
        n_rows - 1,
    )

    if cfg.time_col in train_df.columns:
        year_series = pd.to_numeric(train_df[cfg.time_col], errors="coerce")
        counts = year_series.dropna().astype(int).value_counts().sort_index()
        if counts.size > 1:
            chosen_years: list[int] = []
            chosen_rows = 0
            for year, count in counts.sort_index(ascending=False).items():
                if chosen_rows + int(count) >= n_rows:
                    continue
                chosen_years.append(int(year))
                chosen_rows += int(count)
                if chosen_rows >= target_val_rows:
                    break
            val_mask = year_series.isin(chosen_years).fillna(False).to_numpy()
            if np.any(val_mask) and np.any(~val_mask):
                return np.flatnonzero(~val_mask), np.flatnonzero(val_mask), "time_latest_years"

    if cfg.ticker_col in train_df.columns:
        tickers = _normalize_tickers(train_df[cfg.ticker_col])
        valid_tickers = tickers.loc[tickers.notna()]
        uniq = valid_tickers.unique()
        if uniq.size > 1:
            rng = np.random.default_rng(cfg.random_seed)
            shuffled = np.array(uniq, dtype=object)
            rng.shuffle(shuffled)
            counts = valid_tickers.value_counts().to_dict()
            chosen: list[str] = []
            rows_accum = 0
            for ticker in shuffled:
                ts = str(ticker)
                ticker_count = int(counts.get(ts, 0))
                if rows_accum + ticker_count >= n_rows:
                    continue
                chosen.append(ts)
                rows_accum += ticker_count
                if rows_accum >= target_val_rows:
                    break
            val_mask = tickers.isin(chosen).fillna(False).to_numpy()
            if np.any(val_mask) and np.any(~val_mask):
                return np.flatnonzero(~val_mask), np.flatnonzero(val_mask), "group_ticker"

    idx = np.arange(n_rows)
    rng = np.random.default_rng(cfg.random_seed)
    rng.shuffle(idx)
    val_idx = idx[:target_val_rows]
    fit_idx = idx[target_val_rows:]
    if fit_idx.size == 0:
        fit_idx = val_idx
    return fit_idx, val_idx, "row"


def _xgb_params(cfg: RiskXGBConfig) -> dict[str, object]:
    return {
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


def _fit_booster(train_df: pd.DataFrame, cfg: RiskXGBConfig) -> dict[str, object]:
    feature_cols, feature_profile = select_feature_columns(train_df, cfg)
    X_train = _build_matrix(train_df, feature_cols)
    y_train = _target_array(train_df, cfg.target_col)

    fit_idx, val_idx, inner_split_mode = _inner_train_val_indices(train_df, cfg)
    X_fit = X_train[fit_idx]
    y_fit = y_train[fit_idx]
    X_val = X_train[val_idx]
    y_val = y_train[val_idx]

    dfit = xgb.DMatrix(X_fit, label=y_fit, weight=_make_sample_weights(y_fit, cfg), feature_names=feature_cols)
    dval = xgb.DMatrix(X_val, label=y_val, feature_names=feature_cols)

    booster = xgb.train(
        params=_xgb_params(cfg),
        dtrain=dfit,
        num_boost_round=cfg.num_boost_round,
        evals=[(dfit, "fit"), (dval, "inner_val")],
        early_stopping_rounds=cfg.early_stopping_rounds,
        verbose_eval=False,
    )
    best_iteration = int(getattr(booster, "best_iteration", booster.num_boosted_rounds() - 1))
    return {
        "booster": booster,
        "best_iteration": best_iteration,
        "feature_cols": feature_cols,
        "feature_profile": feature_profile,
        "inner_split_mode": inner_split_mode,
        "fit_idx": fit_idx,
        "val_idx": val_idx,
        "train_target_mean": float(np.mean(y_train)),
    }


def _predict_with_booster(
    booster: xgb.Booster,
    df: pd.DataFrame,
    feature_cols: list[str],
    best_iteration: int,
) -> np.ndarray:
    X = _build_matrix(df, feature_cols)
    dmat = xgb.DMatrix(X, feature_names=feature_cols)
    preds = booster.predict(dmat, iteration_range=(0, best_iteration + 1))
    return _clip_predictions(preds)


def _prediction_frame(
    df: pd.DataFrame,
    y_pred: np.ndarray,
    y_pred_baseline: np.ndarray,
    cfg: RiskXGBConfig,
    split: str,
) -> pd.DataFrame:
    preds = df.copy()
    preds["split"] = split
    preds["y_true"] = pd.to_numeric(df[cfg.target_col], errors="coerce").to_numpy(dtype=np.float64)
    preds["y_pred"] = _clip_predictions(y_pred)
    preds["y_pred_baseline"] = _clip_predictions(y_pred_baseline)
    if cfg.raw_target_col in df.columns:
        preds["y_true_raw_drawdown"] = pd.to_numeric(df[cfg.raw_target_col], errors="coerce").to_numpy(dtype=np.float64)
    return preds


def _add_prefixed_metrics(prefix: str, metrics: dict[str, float], out: dict[str, object]) -> None:
    for key, value in metrics.items():
        out[f"{prefix}_{key}"] = float(value)


def train_risk_xgb(df: pd.DataFrame, cfg: RiskXGBConfig) -> dict[str, object]:
    fitted = _fit_booster(df, cfg)
    train_target_mean = float(fitted["train_target_mean"])
    train_pred = _predict_with_booster(fitted["booster"], df, fitted["feature_cols"], fitted["best_iteration"])
    baseline_pred, baseline_info = _baseline_predictions(df, train_target_mean, cfg)

    fit_idx = np.asarray(fitted["fit_idx"], dtype=np.int64)
    val_idx = np.asarray(fitted["val_idx"], dtype=np.int64)
    y_true = _target_array(df, cfg.target_col)

    train_metrics = risk_metrics(y_true[fit_idx], train_pred[fit_idx], cfg)
    val_metrics = risk_metrics(y_true[val_idx], train_pred[val_idx], cfg)
    baseline_train_metrics = risk_metrics(y_true[fit_idx], baseline_pred[fit_idx], cfg)
    baseline_val_metrics = risk_metrics(y_true[val_idx], baseline_pred[val_idx], cfg)

    preds_fit = _prediction_frame(df.iloc[fit_idx].copy(), train_pred[fit_idx], baseline_pred[fit_idx], cfg, split="train")
    preds_val = _prediction_frame(df.iloc[val_idx].copy(), train_pred[val_idx], baseline_pred[val_idx], cfg, split="val")
    predictions = pd.concat([preds_fit, preds_val], ignore_index=True)

    metrics: dict[str, object] = {
        "rows_total": int(len(df)),
        "rows_train": int(fit_idx.size),
        "rows_val": int(val_idx.size),
        "unique_tickers_total": int(df[cfg.ticker_col].nunique()) if cfg.ticker_col in df.columns else None,
        "feature_count": int(len(fitted["feature_cols"])),
        "best_iteration": int(fitted["best_iteration"]),
        "inner_split_mode": str(fitted["inner_split_mode"]),
    }
    _add_prefixed_metrics("train", train_metrics, metrics)
    _add_prefixed_metrics("val", val_metrics, metrics)
    _add_prefixed_metrics("baseline_train", baseline_train_metrics, metrics)
    _add_prefixed_metrics("baseline_val", baseline_val_metrics, metrics)

    return {
        "metrics": metrics,
        "predictions": predictions,
        "feature_cols": fitted["feature_cols"],
        "feature_profile": fitted["feature_profile"],
        "baseline": baseline_info,
        "booster": fitted["booster"],
        "best_iteration": fitted["best_iteration"],
    }


def train_and_eval_risk_xgb(train_df: pd.DataFrame, test_df: pd.DataFrame, cfg: RiskXGBConfig) -> dict[str, object]:
    fitted = _fit_booster(train_df, cfg)
    train_target_mean = float(fitted["train_target_mean"])
    test_pred = _predict_with_booster(fitted["booster"], test_df, fitted["feature_cols"], fitted["best_iteration"])
    baseline_pred, baseline_info = _baseline_predictions(test_df, train_target_mean, cfg)
    y_true = _target_array(test_df, cfg.target_col)
    test_metrics = risk_metrics(y_true, test_pred, cfg)
    baseline_metrics = risk_metrics(y_true, baseline_pred, cfg)

    metrics: dict[str, object] = {
        "rows_test": int(len(test_df)),
        "unique_tickers_test": int(test_df[cfg.ticker_col].nunique()) if cfg.ticker_col in test_df.columns else None,
        "feature_count": int(len(fitted["feature_cols"])),
        "best_iteration": int(fitted["best_iteration"]),
        "inner_split_mode": str(fitted["inner_split_mode"]),
    }
    _add_prefixed_metrics("test", test_metrics, metrics)
    _add_prefixed_metrics("baseline_test", baseline_metrics, metrics)

    predictions = _prediction_frame(test_df, test_pred, baseline_pred, cfg, split="test")
    return {
        "metrics": metrics,
        "predictions": predictions,
        "feature_cols": fitted["feature_cols"],
        "feature_profile": fitted["feature_profile"],
        "baseline": baseline_info,
        "booster": fitted["booster"],
        "best_iteration": fitted["best_iteration"],
    }


def summary_payload(
    cfg: RiskXGBConfig,
    metrics: dict[str, object],
    feature_cols: list[str],
    baseline: dict[str, object],
) -> dict[str, object]:
    return {
        "config": asdict(cfg),
        "feature_names": list(feature_cols),
        "baseline": baseline,
        "metrics": metrics,
    }
