from __future__ import annotations

import argparse
import itertools
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Dict, Iterable, Sequence

import numpy as np
import pandas as pd
from scipy import stats


DEFAULT_MODEL_PATHS: Dict[str, Path] = {
    "xgb": Path("experiments/valuation/runs/xgb_valuation_artifacts/predictions.csv"),
    "valuation2": Path("experiments/valuation/runs/residual_mlp_valuation_artifacts/predictions.csv"),
}
DEFAULT_KEY_COLS = ("ticker", "fiscal_year", "period_end", "timeframe")
DEFAULT_OUT_DIR = Path("experiments/valuation/runs/stat_tests_validation_artifacts")
IDENTITY_BASELINE_NAME = "identity_baseline"
BOOTSTRAP_ALPHA = 0.05
PAIRWISE_ALPHA = 0.05


@dataclass(frozen=True)
class PredictionSource:
    name: str
    path: Path


def _clip_nonnegative(arr: np.ndarray) -> np.ndarray:
    return np.clip(np.asarray(arr, dtype=np.float64).reshape(-1), a_min=0.0, a_max=None)


def _safe_divide(num: np.ndarray, denom: np.ndarray, eps: float = 1e-8) -> np.ndarray:
    return np.asarray(num, dtype=np.float64) / np.maximum(np.asarray(denom, dtype=np.float64), eps)


def _rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    diff = np.asarray(y_pred, dtype=np.float64) - np.asarray(y_true, dtype=np.float64)
    return float(np.sqrt(np.mean(diff**2)))


def _r2(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    yt = np.asarray(y_true, dtype=np.float64)
    yp = np.asarray(y_pred, dtype=np.float64)
    ss_res = float(np.sum((yt - yp) ** 2))
    ss_tot = float(np.sum((yt - np.mean(yt)) ** 2))
    if ss_tot <= 0.0:
        return float("nan")
    return float(1.0 - ss_res / ss_tot)


def _mape(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    yt = _clip_nonnegative(y_true)
    yp = _clip_nonnegative(y_pred)
    return float(np.mean(_safe_divide(np.abs(yp - yt), yt)))


def _mape_nonzero(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    yt = _clip_nonnegative(y_true)
    yp = _clip_nonnegative(y_pred)
    mask = yt > 0.0
    if not np.any(mask):
        return float("nan")
    return float(np.mean(np.abs(yp[mask] - yt[mask]) / np.maximum(yt[mask], 1e-8)))


def _smape(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    yt = _clip_nonnegative(y_true)
    yp = _clip_nonnegative(y_pred)
    denom = np.maximum((np.abs(yt) + np.abs(yp)) / 2.0, 1e-8)
    return float(np.mean(np.abs(yp - yt) / denom))


def _log1p_arr(arr: np.ndarray) -> np.ndarray:
    return np.log1p(_clip_nonnegative(arr))


def _metric_functions() -> Dict[str, Callable[[np.ndarray, np.ndarray], float]]:
    return {
        "rmse_raw": _rmse,
        "rmse_log": lambda y_true, y_pred: _rmse(_log1p_arr(y_true), _log1p_arr(y_pred)),
        "mape_raw": _mape,
        "mape_nonzero_raw": _mape_nonzero,
        "smape_raw": _smape,
        "r2_raw": _r2,
        "r2_log": lambda y_true, y_pred: _r2(_log1p_arr(y_true), _log1p_arr(y_pred)),
    }


def _loss_views(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, np.ndarray]:
    yt = _clip_nonnegative(y_true)
    yp = _clip_nonnegative(y_pred)
    return {
        "abs_error": np.abs(yp - yt),
        "abs_log_error": np.abs(_log1p_arr(yp) - _log1p_arr(yt)),
        "abs_percentage_error": _safe_divide(np.abs(yp - yt), yt),
    }


def _normalize_ticker(series: pd.Series) -> pd.Series:
    out = series.astype(str).str.upper().str.strip()
    return out.replace({"": pd.NA, "NAN": pd.NA, "NONE": pd.NA, "<NA>": pd.NA, "NULL": pd.NA, "nan": pd.NA})


def _normalize_key_column(col: str, series: pd.Series) -> pd.Series:
    if col == "ticker":
        return _normalize_ticker(series)
    if col == "fiscal_year":
        return pd.to_numeric(series, errors="coerce").astype("Int64")
    return series.astype(str).str.strip().replace({"nan": pd.NA, "None": pd.NA, "<NA>": pd.NA})


def _parse_prediction_specs(specs: Sequence[str]) -> list[PredictionSource]:
    if not specs:
        return [PredictionSource(name=name, path=path) for name, path in DEFAULT_MODEL_PATHS.items()]

    parsed: list[PredictionSource] = []
    for spec in specs:
        if "=" not in spec:
            raise ValueError(f"Invalid --prediction-spec {spec!r}. Expected NAME=PATH.")
        name, raw_path = spec.split("=", 1)
        clean_name = name.strip()
        if not clean_name:
            raise ValueError(f"Invalid --prediction-spec {spec!r}. Empty model name.")
        parsed.append(PredictionSource(name=clean_name, path=Path(raw_path.strip())))
    return parsed


def _load_prediction_frame(path: Path, split: str) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Missing predictions file: {path}")

    df = pd.read_csv(path)
    required = {"y_true", "y_pred"}
    missing = sorted(required.difference(df.columns))
    if missing:
        raise ValueError(f"Predictions file {path} is missing required columns: {missing}")

    out = df.copy()
    if split != "all" and "split" in out.columns:
        out = out[out["split"] == split].copy()
    if out.empty:
        raise ValueError(f"Predictions file {path} has no rows after applying split={split!r}.")

    out["y_true"] = pd.to_numeric(out["y_true"], errors="coerce")
    out["y_pred"] = pd.to_numeric(out["y_pred"], errors="coerce")
    if "y_pred_baseline" in out.columns:
        out["y_pred_baseline"] = pd.to_numeric(out["y_pred_baseline"], errors="coerce")

    out = out.dropna(subset=["y_true", "y_pred"]).copy()
    return out


def _detect_common_key_cols(frames: Dict[str, pd.DataFrame], preferred: Sequence[str]) -> list[str]:
    key_cols = [col for col in preferred if all(col in df.columns for df in frames.values())]
    if not key_cols:
        raise ValueError(
            "Could not find common alignment keys across prediction files. "
            f"Tried: {list(preferred)}"
        )
    return key_cols


def _prepare_frames(
    sources: Sequence[PredictionSource],
    split: str,
) -> tuple[Dict[str, pd.DataFrame], list[str], dict]:
    raw_frames: Dict[str, pd.DataFrame] = {}
    for source in sources:
        raw_frames[source.name] = _load_prediction_frame(source.path, split=split)

    key_cols = _detect_common_key_cols(raw_frames, DEFAULT_KEY_COLS)

    prepared: Dict[str, pd.DataFrame] = {}
    manifest: dict[str, dict[str, object]] = {}
    for source in sources:
        df = raw_frames[source.name].copy()
        for col in key_cols:
            df[col] = _normalize_key_column(col, df[col])
        dupes = int(df.duplicated(subset=key_cols).sum())
        if dupes:
            raise ValueError(
                f"Predictions file {source.path} has {dupes} duplicate rows on the alignment keys {key_cols}."
            )
        prepared[source.name] = df
        manifest[source.name] = {
            "path": str(source.path),
            "rows_after_split_filter": int(len(df)),
        }
    return prepared, key_cols, manifest


def _validate_same_targets(base: pd.Series, other: pd.Series, model_name: str) -> None:
    base_arr = pd.to_numeric(base, errors="coerce").to_numpy(dtype=np.float64)
    other_arr = pd.to_numeric(other, errors="coerce").to_numpy(dtype=np.float64)
    if base_arr.shape != other_arr.shape or not np.allclose(base_arr, other_arr, rtol=0.0, atol=1e-6, equal_nan=True):
        raise ValueError(f"Target mismatch detected while aligning predictions for model {model_name}.")


def _build_aligned_predictions(
    frames: Dict[str, pd.DataFrame],
    key_cols: Sequence[str],
    baseline_source_model: str,
    include_baseline: bool,
) -> tuple[pd.DataFrame, list[str]]:
    model_names = list(frames.keys())
    first_model = model_names[0]

    aligned = frames[first_model][list(key_cols) + ["y_true", "y_pred"]].copy()
    aligned = aligned.rename(columns={"y_pred": first_model})

    for model_name in model_names[1:]:
        part = frames[model_name][list(key_cols) + ["y_true", "y_pred"]].copy()
        part = part.rename(columns={"y_true": f"y_true__{model_name}", "y_pred": model_name})
        aligned = aligned.merge(part, on=list(key_cols), how="inner", validate="one_to_one")
        _validate_same_targets(aligned["y_true"], aligned[f"y_true__{model_name}"], model_name)
        aligned = aligned.drop(columns=[f"y_true__{model_name}"])

    comparison_models = list(model_names)
    if include_baseline:
        if baseline_source_model not in frames:
            raise ValueError(
                f"Baseline source model {baseline_source_model!r} was not loaded. "
                f"Available models: {model_names}"
            )
        source_df = frames[baseline_source_model]
        if "y_pred_baseline" not in source_df.columns:
            raise ValueError(
                f"Baseline source model {baseline_source_model!r} has no y_pred_baseline column in its predictions."
            )
        baseline_part = source_df[list(key_cols) + ["y_pred_baseline"]].copy()
        baseline_part = baseline_part.rename(columns={"y_pred_baseline": IDENTITY_BASELINE_NAME})
        aligned = aligned.merge(baseline_part, on=list(key_cols), how="inner", validate="one_to_one")
        comparison_models.append(IDENTITY_BASELINE_NAME)

    aligned = aligned.sort_values(list(key_cols)).reset_index(drop=True)
    aligned["y_true"] = pd.to_numeric(aligned["y_true"], errors="coerce")
    for model_name in comparison_models:
        aligned[model_name] = pd.to_numeric(aligned[model_name], errors="coerce")
    aligned = aligned.dropna(subset=["y_true", *comparison_models]).reset_index(drop=True)
    return aligned, comparison_models


def _compute_metric_table(aligned: pd.DataFrame, model_names: Sequence[str]) -> pd.DataFrame:
    metric_fns = _metric_functions()
    y_true = aligned["y_true"].to_numpy(dtype=np.float64)
    rows: list[dict[str, object]] = []
    for model_name in model_names:
        y_pred = aligned[model_name].to_numpy(dtype=np.float64)
        row: dict[str, object] = {
            "model": model_name,
            "rows": int(len(y_true)),
            "nonzero_rows": int(np.sum(_clip_nonnegative(y_true) > 0.0)),
        }
        for metric_name, fn in metric_fns.items():
            row[metric_name] = float(fn(y_true, y_pred))
        rows.append(row)
    return pd.DataFrame(rows)


def _bootstrap_metric_tables(
    aligned: pd.DataFrame,
    model_names: Sequence[str],
    iterations: int,
    seed: int,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    metric_fns = _metric_functions()
    y_true = aligned["y_true"].to_numpy(dtype=np.float64)
    predictions = {name: aligned[name].to_numpy(dtype=np.float64) for name in model_names}
    rng = np.random.default_rng(seed)

    samples: dict[tuple[str, str], np.ndarray] = {}
    for model_name in model_names:
        for metric_name in metric_fns:
            samples[(model_name, metric_name)] = np.empty(iterations, dtype=np.float64)

    for idx in range(iterations):
        draw = rng.integers(0, len(y_true), size=len(y_true))
        y_boot = y_true[draw]
        for model_name in model_names:
            pred_boot = predictions[model_name][draw]
            for metric_name, fn in metric_fns.items():
                samples[(model_name, metric_name)][idx] = fn(y_boot, pred_boot)

    metric_rows: list[dict[str, object]] = []
    for model_name in model_names:
        for metric_name, fn in metric_fns.items():
            point = fn(y_true, predictions[model_name])
            dist = samples[(model_name, metric_name)]
            metric_rows.append(
                {
                    "model": model_name,
                    "metric": metric_name,
                    "point_estimate": float(point),
                    "bootstrap_mean": float(np.nanmean(dist)),
                    "bootstrap_std": float(np.nanstd(dist, ddof=1)),
                    "ci_low": float(np.nanquantile(dist, BOOTSTRAP_ALPHA / 2.0)),
                    "ci_high": float(np.nanquantile(dist, 1.0 - BOOTSTRAP_ALPHA / 2.0)),
                    "iterations": int(iterations),
                }
            )

    delta_rows: list[dict[str, object]] = []
    for model_a, model_b in itertools.combinations(model_names, 2):
        for metric_name, fn in metric_fns.items():
            point_delta = float(fn(y_true, predictions[model_a]) - fn(y_true, predictions[model_b]))
            dist = samples[(model_a, metric_name)] - samples[(model_b, metric_name)]
            delta_rows.append(
                {
                    "model_a": model_a,
                    "model_b": model_b,
                    "metric": metric_name,
                    "point_delta_a_minus_b": point_delta,
                    "ci_low": float(np.nanquantile(dist, BOOTSTRAP_ALPHA / 2.0)),
                    "ci_high": float(np.nanquantile(dist, 1.0 - BOOTSTRAP_ALPHA / 2.0)),
                    "iterations": int(iterations),
                }
            )

    return pd.DataFrame(metric_rows), pd.DataFrame(delta_rows)


def _holm_adjust(p_values: Iterable[float]) -> list[float]:
    arr = np.asarray(list(p_values), dtype=np.float64)
    out = np.full(arr.shape, np.nan, dtype=np.float64)
    mask = np.isfinite(arr)
    if not np.any(mask):
        return out.tolist()

    valid = arr[mask]
    order = np.argsort(valid)
    ranked = valid[order]
    m = ranked.size
    adjusted = np.empty(m, dtype=np.float64)
    running = 0.0
    for i, p in enumerate(ranked):
        candidate = (m - i) * p
        running = max(running, candidate)
        adjusted[i] = min(running, 1.0)
    restored = np.empty(m, dtype=np.float64)
    restored[order] = adjusted
    out[mask] = restored
    return out.tolist()


def _bh_adjust(p_values: Iterable[float]) -> list[float]:
    arr = np.asarray(list(p_values), dtype=np.float64)
    out = np.full(arr.shape, np.nan, dtype=np.float64)
    mask = np.isfinite(arr)
    if not np.any(mask):
        return out.tolist()

    valid = arr[mask]
    order = np.argsort(valid)
    ranked = valid[order]
    m = ranked.size
    adjusted = np.empty(m, dtype=np.float64)
    running = 1.0
    for i in range(m - 1, -1, -1):
        rank = i + 1
        candidate = ranked[i] * m / rank
        running = min(running, candidate)
        adjusted[i] = min(running, 1.0)
    restored = np.empty(m, dtype=np.float64)
    restored[order] = adjusted
    out[mask] = restored
    return out.tolist()


def _wilcoxon_signed(diff: np.ndarray) -> tuple[float, float]:
    delta = np.asarray(diff, dtype=np.float64)
    if delta.size == 0 or np.allclose(delta, 0.0):
        return 0.0, 1.0
    stat, p_value = stats.wilcoxon(delta, zero_method="pratt", alternative="two-sided", method="auto")
    return float(stat), float(p_value)


def _paired_t_test(a: np.ndarray, b: np.ndarray) -> tuple[float, float]:
    diff = np.asarray(a, dtype=np.float64) - np.asarray(b, dtype=np.float64)
    if diff.size == 0 or np.allclose(diff, 0.0):
        return 0.0, 1.0
    stat, p_value = stats.ttest_rel(a, b, nan_policy="omit")
    return float(stat), float(p_value)


def _pairwise_significance_table(aligned: pd.DataFrame, model_names: Sequence[str]) -> pd.DataFrame:
    y_true = aligned["y_true"].to_numpy(dtype=np.float64)
    loss_cache = {model_name: _loss_views(y_true, aligned[model_name].to_numpy(dtype=np.float64)) for model_name in model_names}

    rows: list[dict[str, object]] = []
    for model_a, model_b in itertools.combinations(model_names, 2):
        for loss_name in ("abs_error", "abs_log_error", "abs_percentage_error"):
            loss_a = loss_cache[model_a][loss_name]
            loss_b = loss_cache[model_b][loss_name]
            diff = loss_a - loss_b
            wilcoxon_stat, wilcoxon_p = _wilcoxon_signed(diff)
            t_stat, t_p = _paired_t_test(loss_a, loss_b)
            skewness = float(stats.skew(diff, bias=False)) if diff.size >= 3 and not np.allclose(np.std(diff), 0.0) else float("nan")
            rows.append(
                {
                    "model_a": model_a,
                    "model_b": model_b,
                    "loss_view": loss_name,
                    "rows": int(diff.size),
                    "mean_loss_a": float(np.mean(loss_a)),
                    "mean_loss_b": float(np.mean(loss_b)),
                    "mean_diff_a_minus_b": float(np.mean(diff)),
                    "median_diff_a_minus_b": float(np.median(diff)),
                    "win_rate_a_lt_b": float(np.mean(loss_a < loss_b)),
                    "tie_rate": float(np.mean(np.isclose(loss_a, loss_b))),
                    "diff_skewness": skewness,
                    "approx_symmetric_diff": bool(np.isfinite(skewness) and abs(skewness) <= 1.0),
                    "wilcoxon_statistic": wilcoxon_stat,
                    "wilcoxon_p_value": wilcoxon_p,
                    "paired_t_statistic": t_stat,
                    "paired_t_p_value": t_p,
                }
            )

    out = pd.DataFrame(rows)
    if out.empty:
        return out

    out["wilcoxon_p_holm"] = _holm_adjust(out["wilcoxon_p_value"].tolist())
    out["wilcoxon_p_bh"] = _bh_adjust(out["wilcoxon_p_value"].tolist())
    out["paired_t_p_holm"] = _holm_adjust(out["paired_t_p_value"].tolist())
    out["paired_t_p_bh"] = _bh_adjust(out["paired_t_p_value"].tolist())
    out["wilcoxon_significant_holm_5pct"] = out["wilcoxon_p_holm"] < PAIRWISE_ALPHA
    out["paired_t_significant_holm_5pct"] = out["paired_t_p_holm"] < PAIRWISE_ALPHA
    return out


def _mincer_zarnowitz_row(y_true: np.ndarray, y_pred: np.ndarray, model_name: str) -> dict[str, object]:
    yt = np.asarray(y_true, dtype=np.float64)
    yp = np.asarray(y_pred, dtype=np.float64)
    x = np.column_stack([np.ones(yt.shape[0], dtype=np.float64), yp])
    xtx_inv = np.linalg.pinv(x.T @ x)
    beta = xtx_inv @ x.T @ yt
    fitted = x @ beta
    resid = yt - fitted
    n_obs = int(yt.shape[0])
    n_params = int(x.shape[1])
    dof = max(n_obs - n_params, 1)
    sigma2 = float(np.sum(resid**2) / dof)
    cov_beta = sigma2 * xtx_inv
    std_errors = np.sqrt(np.clip(np.diag(cov_beta), a_min=0.0, a_max=None))

    restrictions = np.eye(2, dtype=np.float64)
    target = np.array([0.0, 1.0], dtype=np.float64)
    delta = restrictions @ beta - target
    wald_cov = restrictions @ cov_beta @ restrictions.T
    wald_stat = float(delta.T @ np.linalg.pinv(wald_cov) @ delta)
    f_stat = float(wald_stat / restrictions.shape[0])
    joint_p = float(stats.f.sf(f_stat, restrictions.shape[0], dof))

    return {
        "model": model_name,
        "rows": n_obs,
        "intercept": float(beta[0]),
        "intercept_se": float(std_errors[0]),
        "slope": float(beta[1]),
        "slope_se": float(std_errors[1]),
        "r2": float(_r2(yt, fitted)),
        "joint_f_statistic_h0_intercept_0_slope_1": f_stat,
        "joint_p_value": joint_p,
    }


def _mincer_zarnowitz_table(aligned: pd.DataFrame, model_names: Sequence[str]) -> pd.DataFrame:
    y_true = aligned["y_true"].to_numpy(dtype=np.float64)
    rows = [_mincer_zarnowitz_row(y_true, aligned[model_name].to_numpy(dtype=np.float64), model_name) for model_name in model_names]
    out = pd.DataFrame(rows)
    if out.empty:
        return out
    out["joint_p_holm"] = _holm_adjust(out["joint_p_value"].tolist())
    out["joint_p_bh"] = _bh_adjust(out["joint_p_value"].tolist())
    return out


def _forecast_bias_table(aligned: pd.DataFrame, model_names: Sequence[str]) -> pd.DataFrame:
    y_true = aligned["y_true"].to_numpy(dtype=np.float64)
    rows: list[dict[str, object]] = []
    for model_name in model_names:
        error = aligned[model_name].to_numpy(dtype=np.float64) - y_true
        if error.size == 0 or np.allclose(error, 0.0):
            mean_t = 0.0
            mean_p = 1.0
            median_w = 0.0
            median_p = 1.0
        else:
            mean_t, mean_p = stats.ttest_1samp(error, popmean=0.0, nan_policy="omit")
            median_w, median_p = _wilcoxon_signed(error)
        rows.append(
            {
                "model": model_name,
                "rows": int(error.size),
                "mean_error": float(np.mean(error)),
                "median_error": float(np.median(error)),
                "mean_abs_error": float(np.mean(np.abs(error))),
                "mean_bias_t_statistic": float(mean_t),
                "mean_bias_p_value": float(mean_p),
                "median_bias_wilcoxon_statistic": float(median_w),
                "median_bias_p_value": float(median_p),
            }
        )

    out = pd.DataFrame(rows)
    if out.empty:
        return out
    out["mean_bias_p_holm"] = _holm_adjust(out["mean_bias_p_value"].tolist())
    out["mean_bias_p_bh"] = _bh_adjust(out["mean_bias_p_value"].tolist())
    out["median_bias_p_holm"] = _holm_adjust(out["median_bias_p_value"].tolist())
    out["median_bias_p_bh"] = _bh_adjust(out["median_bias_p_value"].tolist())
    return out


def _size_bucket_summary(aligned: pd.DataFrame, model_names: Sequence[str], n_buckets: int) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    bucket_df = aligned.copy()
    bucket_df["size_bucket"] = pd.qcut(bucket_df["y_true"], q=n_buckets, duplicates="drop")
    bucket_df["size_bucket_label"] = bucket_df["size_bucket"].astype(str)

    summary_rows: list[dict[str, object]] = []
    kruskal_rows: list[dict[str, object]] = []
    dunn_tables: list[pd.DataFrame] = []

    for model_name in model_names:
        ape = _loss_views(bucket_df["y_true"].to_numpy(dtype=np.float64), bucket_df[model_name].to_numpy(dtype=np.float64))["abs_percentage_error"]
        abs_err = np.abs(bucket_df[model_name].to_numpy(dtype=np.float64) - bucket_df["y_true"].to_numpy(dtype=np.float64))
        model_df = bucket_df.copy()
        model_df["ape"] = ape
        model_df["abs_err"] = abs_err

        grouped = model_df.groupby("size_bucket_label", observed=False)
        for bucket_label, group in grouped:
            summary_rows.append(
                {
                    "model": model_name,
                    "size_bucket": bucket_label,
                    "rows": int(len(group)),
                    "mean_true": float(np.mean(group["y_true"])),
                    "mean_pred": float(np.mean(group[model_name])),
                    "rmse_raw": float(np.sqrt(np.mean(group["abs_err"] ** 2))),
                    "mape_raw": float(np.mean(group["ape"])),
                    "median_ape": float(np.median(group["ape"])),
                }
            )

        groups = [group["ape"].to_numpy(dtype=np.float64) for _, group in grouped if len(group) > 0]
        bucket_names = [bucket_label for bucket_label, group in grouped if len(group) > 0]
        if len(groups) >= 2:
            h_stat, p_value = stats.kruskal(*groups, nan_policy="omit")
        else:
            h_stat, p_value = 0.0, 1.0

        kruskal_rows.append(
            {
                "model": model_name,
                "n_buckets": int(len(groups)),
                "kruskal_h_statistic": float(h_stat),
                "kruskal_p_value": float(p_value),
            }
        )

        if len(groups) >= 2 and float(p_value) < PAIRWISE_ALPHA:
            dunn_table = _dunn_posthoc(groups=groups, group_names=bucket_names)
            dunn_table.insert(0, "model", model_name)
            dunn_tables.append(dunn_table)

    kruskal_df = pd.DataFrame(kruskal_rows)
    if not kruskal_df.empty:
        kruskal_df["kruskal_p_holm"] = _holm_adjust(kruskal_df["kruskal_p_value"].tolist())
        kruskal_df["kruskal_p_bh"] = _bh_adjust(kruskal_df["kruskal_p_value"].tolist())

    dunn_df = pd.concat(dunn_tables, ignore_index=True) if dunn_tables else pd.DataFrame(
        columns=["model", "group_a", "group_b", "z_statistic", "p_value", "p_holm", "p_bh"]
    )
    return pd.DataFrame(summary_rows), kruskal_df, dunn_df


def _dunn_posthoc(groups: Sequence[np.ndarray], group_names: Sequence[str]) -> pd.DataFrame:
    concatenated = np.concatenate([np.asarray(group, dtype=np.float64) for group in groups])
    labels = np.concatenate(
        [np.full(np.asarray(group, dtype=np.float64).shape[0], group_name, dtype=object) for group, group_name in zip(groups, group_names)]
    )
    ranks = stats.rankdata(concatenated, method="average")
    n_total = int(concatenated.size)
    _, tie_counts = np.unique(concatenated, return_counts=True)
    if n_total <= 1:
        tie_correction = 1.0
    else:
        tie_correction = 1.0 - float(np.sum(tie_counts**3 - tie_counts) / (n_total**3 - n_total))
    base_var = (n_total * (n_total + 1.0) / 12.0) * max(tie_correction, 1e-12)

    group_rank_means: dict[str, float] = {}
    group_sizes: dict[str, int] = {}
    for group_name in group_names:
        mask = labels == group_name
        group_rank_means[group_name] = float(np.mean(ranks[mask]))
        group_sizes[group_name] = int(np.sum(mask))

    rows: list[dict[str, object]] = []
    for group_a, group_b in itertools.combinations(group_names, 2):
        se = float(np.sqrt(base_var * (1.0 / group_sizes[group_a] + 1.0 / group_sizes[group_b])))
        z_stat = float((group_rank_means[group_a] - group_rank_means[group_b]) / max(se, 1e-12))
        p_value = float(2.0 * stats.norm.sf(abs(z_stat)))
        rows.append(
            {
                "group_a": group_a,
                "group_b": group_b,
                "z_statistic": z_stat,
                "p_value": p_value,
            }
        )

    out = pd.DataFrame(rows)
    if out.empty:
        return out
    out["p_holm"] = _holm_adjust(out["p_value"].tolist())
    out["p_bh"] = _bh_adjust(out["p_value"].tolist())
    return out


def _write_outputs(
    out_dir: Path,
    aligned: pd.DataFrame,
    metric_table: pd.DataFrame,
    bootstrap_table: pd.DataFrame,
    bootstrap_delta_table: pd.DataFrame,
    pairwise_table: pd.DataFrame,
    mz_table: pd.DataFrame,
    bias_table: pd.DataFrame,
    bucket_summary: pd.DataFrame,
    bucket_kruskal: pd.DataFrame,
    bucket_dunn: pd.DataFrame,
    manifest: dict,
) -> dict[str, str]:
    out_dir.mkdir(parents=True, exist_ok=True)

    outputs = {
        "aligned_predictions": out_dir / "aligned_predictions.csv",
        "headline_metrics": out_dir / "headline_metrics.csv",
        "bootstrap_confidence_intervals": out_dir / "bootstrap_confidence_intervals.csv",
        "bootstrap_pairwise_deltas": out_dir / "bootstrap_pairwise_deltas.csv",
        "pairwise_significance_tests": out_dir / "pairwise_significance_tests.csv",
        "mincer_zarnowitz": out_dir / "mincer_zarnowitz_calibration.csv",
        "forecast_bias": out_dir / "forecast_bias_tests.csv",
        "size_bucket_summary": out_dir / "size_bucket_summary.csv",
        "size_bucket_kruskal": out_dir / "size_bucket_kruskal.csv",
        "size_bucket_dunn_holm": out_dir / "size_bucket_dunn_holm.csv",
        "analysis_manifest": out_dir / "analysis_manifest.json",
    }

    aligned.to_csv(outputs["aligned_predictions"], index=False)
    metric_table.to_csv(outputs["headline_metrics"], index=False)
    bootstrap_table.to_csv(outputs["bootstrap_confidence_intervals"], index=False)
    bootstrap_delta_table.to_csv(outputs["bootstrap_pairwise_deltas"], index=False)
    pairwise_table.to_csv(outputs["pairwise_significance_tests"], index=False)
    mz_table.to_csv(outputs["mincer_zarnowitz"], index=False)
    bias_table.to_csv(outputs["forecast_bias"], index=False)
    bucket_summary.to_csv(outputs["size_bucket_summary"], index=False)
    bucket_kruskal.to_csv(outputs["size_bucket_kruskal"], index=False)
    bucket_dunn.to_csv(outputs["size_bucket_dunn_holm"], index=False)

    with outputs["analysis_manifest"].open("w", encoding="utf-8") as fh:
        json.dump(manifest, fh, indent=2)

    return {name: str(path) for name, path in outputs.items()}


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run valuation-model statistical tests on aligned prediction artifacts."
    )
    parser.add_argument(
        "--prediction-spec",
        action="append",
        default=[],
        help="Optional NAME=PATH override. Repeat for multiple models.",
    )
    parser.add_argument(
        "--split",
        type=str,
        default="val",
        help="Row split to analyze when predictions contain a split column. Use 'all' to keep every row.",
    )
    parser.add_argument(
        "--out-dir",
        type=str,
        default=str(DEFAULT_OUT_DIR),
        help="Directory for the generated statistical tables.",
    )
    parser.add_argument(
        "--bootstrap-iterations",
        type=int,
        default=2000,
        help="Number of paired bootstrap resamples used for confidence intervals.",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--baseline-source-model",
        type=str,
        default="valuation2",
        help="Loaded model whose y_pred_baseline column should be used as the identity baseline when --include-baseline is set.",
    )
    parser.add_argument(
        "--include-baseline",
        action="store_true",
        help="Include the legacy identity baseline from a loaded model's y_pred_baseline column.",
    )
    parser.add_argument(
        "--exclude-baseline",
        action="store_true",
        help=argparse.SUPPRESS,
    )
    parser.add_argument(
        "--size-buckets",
        type=int,
        default=5,
        help="Number of quantile buckets for the firm-size robustness check.",
    )
    args = parser.parse_args()

    include_baseline = bool(args.include_baseline) and not bool(args.exclude_baseline)

    sources = _parse_prediction_specs(args.prediction_spec)
    frames, key_cols, source_manifest = _prepare_frames(sources=sources, split=args.split)
    aligned, model_names = _build_aligned_predictions(
        frames=frames,
        key_cols=key_cols,
        baseline_source_model=args.baseline_source_model,
        include_baseline=include_baseline,
    )

    metric_table = _compute_metric_table(aligned=aligned, model_names=model_names)
    bootstrap_table, bootstrap_delta_table = _bootstrap_metric_tables(
        aligned=aligned,
        model_names=model_names,
        iterations=int(args.bootstrap_iterations),
        seed=int(args.seed),
    )
    pairwise_table = _pairwise_significance_table(aligned=aligned, model_names=model_names)
    mz_table = _mincer_zarnowitz_table(aligned=aligned, model_names=model_names)
    bias_table = _forecast_bias_table(aligned=aligned, model_names=model_names)
    bucket_summary, bucket_kruskal, bucket_dunn = _size_bucket_summary(
        aligned=aligned,
        model_names=model_names,
        n_buckets=int(args.size_buckets),
    )

    manifest = {
        "split": args.split,
        "bootstrap_iterations": int(args.bootstrap_iterations),
        "seed": int(args.seed),
        "size_buckets": int(args.size_buckets),
        "key_columns": list(key_cols),
        "loaded_sources": source_manifest,
        "comparison_models": list(model_names),
        "include_baseline": bool(include_baseline),
        "aligned_rows": int(len(aligned)),
    }
    output_paths = _write_outputs(
        out_dir=Path(args.out_dir),
        aligned=aligned,
        metric_table=metric_table,
        bootstrap_table=bootstrap_table,
        bootstrap_delta_table=bootstrap_delta_table,
        pairwise_table=pairwise_table,
        mz_table=mz_table,
        bias_table=bias_table,
        bucket_summary=bucket_summary,
        bucket_kruskal=bucket_kruskal,
        bucket_dunn=bucket_dunn,
        manifest=manifest,
    )

    print("Saved statistical test outputs:")
    for name, path in output_paths.items():
        print(f"- {name}: {path}")


if __name__ == "__main__":
    main()
