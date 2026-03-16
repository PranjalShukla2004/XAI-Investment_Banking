from __future__ import annotations

from typing import Dict, List, Sequence, Tuple

import numpy as np
import pandas as pd


def _to_numeric_matrix(df: pd.DataFrame, feature_cols: Sequence[str]) -> pd.DataFrame:
    if not feature_cols:
        return pd.DataFrame(index=df.index)
    return (
        df[list(feature_cols)]
        .apply(pd.to_numeric, errors="coerce")
        .replace([np.inf, -np.inf], np.nan)
        .fillna(0.0)
    )


def _target_series(df: pd.DataFrame, target_col: str, log_target: bool) -> np.ndarray:
    y = pd.to_numeric(df[target_col], errors="coerce").replace([np.inf, -np.inf], np.nan).fillna(0.0).to_numpy(dtype=np.float64)
    if log_target:
        y = np.log1p(np.clip(y, a_min=0.0, a_max=None))
    return y


def _corr_with_target(X_df: pd.DataFrame, y: np.ndarray) -> Dict[str, float]:
    out: Dict[str, float] = {}
    if X_df.empty:
        return out

    y = np.asarray(y, dtype=np.float64).reshape(-1)
    y_std = float(np.std(y))
    if y_std < 1e-12:
        return {c: 0.0 for c in X_df.columns}

    y_center = y - np.mean(y)
    for c in X_df.columns:
        x = X_df[c].to_numpy(dtype=np.float64, copy=False)
        x_std = float(np.std(x))
        if x_std < 1e-12:
            out[c] = 0.0
            continue
        x_center = x - np.mean(x)
        corr = float(np.mean(x_center * y_center) / (x_std * y_std + 1e-12))
        if not np.isfinite(corr):
            corr = 0.0
        out[c] = corr
    return out


def select_features_by_correlation(
    train_df: pd.DataFrame,
    feature_cols: Sequence[str],
    target_col: str,
    log_target: bool,
    min_abs_target_corr: float = 0.0,
    max_features: int | None = None,
    max_inter_feature_corr: float = 1.0,
    min_features: int = 10,
) -> Tuple[List[str], Dict[str, object]]:
    if not feature_cols:
        return [], {
            "raw_feature_count": 0,
            "selected_feature_count": 0,
            "selection_applied": False,
        }

    X_df = _to_numeric_matrix(train_df, feature_cols)
    y = _target_series(train_df, target_col=target_col, log_target=log_target)
    corr_map = _corr_with_target(X_df, y)

    ranked = sorted(feature_cols, key=lambda c: abs(float(corr_map.get(c, 0.0))), reverse=True)
    threshold = float(max(min_abs_target_corr, 0.0))
    candidates = [c for c in ranked if abs(float(corr_map.get(c, 0.0))) >= threshold]

    floor_n = int(max(1, min(min_features, len(ranked))))
    if len(candidates) < floor_n:
        for c in ranked:
            if c not in candidates:
                candidates.append(c)
            if len(candidates) >= floor_n:
                break

    if max_features is not None and int(max_features) > 0 and len(candidates) > int(max_features):
        candidates = candidates[: int(max_features)]

    selected = list(candidates)
    inter_th = float(max_inter_feature_corr)
    if inter_th < 1.0 and len(selected) > 1:
        corr_df = X_df[selected].corr().abs().fillna(0.0)
        keep: List[str] = []
        for c in selected:
            if not keep:
                keep.append(c)
                continue
            max_corr = float(corr_df.loc[c, keep].max())
            if max_corr < inter_th:
                keep.append(c)

        if len(keep) < floor_n:
            for c in selected:
                if c not in keep:
                    keep.append(c)
                if len(keep) >= floor_n:
                    break
        selected = keep

    stats: Dict[str, object] = {
        "selection_applied": True,
        "raw_feature_count": int(len(feature_cols)),
        "selected_feature_count": int(len(selected)),
        "min_abs_target_corr": float(threshold),
        "max_features": int(max_features) if max_features is not None else None,
        "max_inter_feature_corr": float(inter_th),
        "min_features_floor": int(floor_n),
        "mean_abs_target_corr_selected": float(np.mean([abs(float(corr_map.get(c, 0.0))) for c in selected])) if selected else 0.0,
        "top_abs_target_corr_features": [
            {"feature": c, "abs_corr": float(abs(corr_map.get(c, 0.0)))}
            for c in ranked[:20]
        ],
        "dropped_feature_count": int(len(feature_cols) - len(selected)),
    }
    return selected, stats


def fit_pca(
    X_train: np.ndarray,
    explained_variance: float = 0.95,
    max_components: int | None = None,
) -> Dict[str, np.ndarray | int | float]:
    X = np.asarray(X_train, dtype=np.float64)
    if X.ndim != 2:
        raise ValueError(f"X_train must be 2D, got shape={X.shape}")
    n, d = X.shape
    if n == 0 or d == 0:
        raise ValueError("X_train must have at least one row and one column.")

    mu = np.mean(X, axis=0, keepdims=True)
    Xc = X - mu
    U, s, vt = np.linalg.svd(Xc, full_matrices=False)
    var = (s**2) / float(max(n - 1, 1))
    total_var = float(np.sum(var))

    if total_var <= 1e-16:
        k = int(1 if max_components is None else max(1, min(int(max_components), d)))
        comps = np.eye(d, k, dtype=np.float64)
        ratio = np.zeros((k,), dtype=np.float64)
    else:
        ratio_all = var / total_var
        target = float(np.clip(explained_variance, 0.0, 1.0))
        k = int(np.searchsorted(np.cumsum(ratio_all), target, side="left") + 1)
        if max_components is not None and int(max_components) > 0:
            k = min(k, int(max_components))
        k = max(1, min(k, vt.shape[0]))
        comps = vt[:k, :].T
        ratio = ratio_all[:k]

    return {
        "mu": mu.reshape(-1).astype(np.float32),
        "components": comps.astype(np.float32),
        "explained_variance_ratio": ratio.astype(np.float32),
        "explained_variance_ratio_cum": float(np.sum(ratio)),
        "n_components": int(comps.shape[1]),
        "n_features_in": int(d),
    }


def transform_pca(X: np.ndarray, pca_model: Dict[str, np.ndarray | int | float]) -> np.ndarray:
    Xv = np.asarray(X, dtype=np.float64)
    mu = np.asarray(pca_model["mu"], dtype=np.float64).reshape(1, -1)
    comps = np.asarray(pca_model["components"], dtype=np.float64)
    if Xv.ndim != 2:
        raise ValueError(f"X must be 2D, got shape={Xv.shape}")
    if Xv.shape[1] != mu.shape[1]:
        raise ValueError(f"X feature dim {Xv.shape[1]} does not match PCA mu dim {mu.shape[1]}")
    return ((Xv - mu) @ comps).astype(np.float32)
