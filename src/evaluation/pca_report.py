from __future__ import annotations

import argparse
import ast
import csv
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List

import numpy as np
import pandas as pd

from src.models.feature_engineering import fit_pca, select_features_by_correlation, transform_pca
from src.models.valuation.valuation import (
    _ensure_dir,
    _fit_standard_scaler,
    _select_feature_columns,
    _select_log1p_features,
    _time_aware_split,
    _transform_standard_scaler,
    build_xy,
)


NEWS_FEATURE_COLUMNS = (
    "news_items_n",
    "news_items_from_scores_n",
    "news_items_from_text_n",
    "news_sent_mean",
    "news_sent_min",
    "news_sent_max",
    "news_sent_std",
    "news_sent_sum",
    "news_sent_abs_mean",
    "news_sent_pos_share",
    "news_sent_neg_share",
    "news_sent_text_coverage",
)


@dataclass
class PCAReportConfig:
    data_path: Path = Path("data/processed/main_dataset.csv")
    out_dir: Path = Path("experiments/valuation/runs/pca_report_artifacts")

    pipeline: str = "final"

    target_col: str = "total_assets"
    log_target: bool = True
    time_col: str = "fiscal_year"
    min_val_rows: int = 20
    random_seed: int = 42
    val_ratio_fallback: float = 0.2

    liabilities_col: str = "total_liabilities"
    equity_col: str = "total_equity"
    base_feature_col: str = "identity_base_assets"

    news_score_col: str = "news_sentiment_score"
    news_text_col: str = "news_description"

    use_log1p_feature_transform: bool = True
    enable_feature_selection: bool = True
    min_abs_target_corr: float = 0.01
    max_features_by_target_corr: int | None = 200
    max_inter_feature_corr: float = 0.98
    min_features_after_selection: int = 20

    pca_explained_variance: float = 0.95
    pca_max_components: int | None = 64
    top_k_loadings: int = 10


def _parse_list_like(value: Any) -> List[Any]:
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return []

    text = str(value).strip()
    if not text:
        return []

    for loader in (json.loads, ast.literal_eval):
        try:
            parsed = loader(text)
        except Exception:
            continue
        if isinstance(parsed, list):
            return list(parsed)

    return []


def _parse_news_scores(value: Any) -> List[float]:
    items = _parse_list_like(value)
    if items:
        out: List[float] = []
        for item in items:
            try:
                out.append(float(item))
            except Exception:
                continue
        return out

    text = str(value).strip()
    if not text:
        return []
    try:
        return [float(text)]
    except Exception:
        return []


def _parse_news_texts(value: Any) -> List[str]:
    items = _parse_list_like(value)
    if items:
        out: List[str] = []
        for item in items:
            s = str(item).strip()
            if s:
                out.append(s)
        return out

    text = str(value).strip()
    if not text:
        return []

    try:
        fields = next(csv.reader([text], skipinitialspace=True))
    except Exception:
        fields = text.split(",")

    out = [field.strip().strip('"').strip("'") for field in fields]
    return [x for x in out if x]


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
    out[out_col] = np.clip((liab + eq).to_numpy(dtype=np.float64), a_min=0.0, a_max=None)
    return out


def _build_news_features(
    df: pd.DataFrame,
    score_col: str,
    text_col: str,
) -> tuple[pd.DataFrame, Dict[str, Any]]:
    if score_col not in df.columns:
        raise ValueError(
            f"Required column '{score_col}' not found. "
            "Run the FinBERT scoring step first to create per-news sentiment scores."
        )
    if text_col not in df.columns:
        raise ValueError(f"Required column '{text_col}' not found.")

    score_lists = [_parse_news_scores(v) for v in df[score_col].tolist()]
    text_lists = [_parse_news_texts(v) for v in df[text_col].tolist()]

    feature_values: Dict[str, List[float]] = {k: [] for k in NEWS_FEATURE_COLUMNS}
    mismatch_count = 0

    for scores, texts in zip(score_lists, text_lists):
        score_n = len(scores)
        text_n = len(texts)
        if score_n > 0 and text_n > 0 and score_n != text_n:
            mismatch_count += 1

        if score_n > 0:
            arr = np.asarray(scores, dtype=np.float32)
            sent_mean = float(np.mean(arr))
            sent_min = float(np.min(arr))
            sent_max = float(np.max(arr))
            sent_std = float(np.std(arr))
            sent_sum = float(np.sum(arr))
            sent_abs_mean = float(np.mean(np.abs(arr)))
            sent_pos_share = float(np.mean(arr > 0))
            sent_neg_share = float(np.mean(arr < 0))
        else:
            sent_mean = 0.0
            sent_min = 0.0
            sent_max = 0.0
            sent_std = 0.0
            sent_sum = 0.0
            sent_abs_mean = 0.0
            sent_pos_share = 0.0
            sent_neg_share = 0.0

        items_n = float(score_n if score_n > 0 else text_n)
        coverage = float(score_n / text_n) if text_n > 0 else (1.0 if score_n > 0 else 0.0)

        feature_values["news_items_n"].append(items_n)
        feature_values["news_items_from_scores_n"].append(float(score_n))
        feature_values["news_items_from_text_n"].append(float(text_n))
        feature_values["news_sent_mean"].append(sent_mean)
        feature_values["news_sent_min"].append(sent_min)
        feature_values["news_sent_max"].append(sent_max)
        feature_values["news_sent_std"].append(sent_std)
        feature_values["news_sent_sum"].append(sent_sum)
        feature_values["news_sent_abs_mean"].append(sent_abs_mean)
        feature_values["news_sent_pos_share"].append(sent_pos_share)
        feature_values["news_sent_neg_share"].append(sent_neg_share)
        feature_values["news_sent_text_coverage"].append(coverage)

    out_df = df.copy()
    for col, vals in feature_values.items():
        out_df[col] = np.asarray(vals, dtype=np.float32)

    text_counts = np.asarray([len(x) for x in text_lists], dtype=np.int32)
    score_counts = np.asarray([len(x) for x in score_lists], dtype=np.int32)

    stats = {
        "rows_total": int(len(out_df)),
        "rows_with_text_items": int(np.sum(text_counts > 0)),
        "rows_with_score_items": int(np.sum(score_counts > 0)),
        "rows_without_text_and_scores": int(np.sum((text_counts == 0) & (score_counts == 0))),
        "rows_with_text_score_length_mismatch": int(mismatch_count),
        "total_text_items": int(np.sum(text_counts)),
        "total_score_items": int(np.sum(score_counts)),
        "engineered_columns": list(NEWS_FEATURE_COLUMNS),
    }
    return out_df, stats


def _jsonable(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, (np.integer, np.floating)):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, pd.Timestamp):
        return value.isoformat()
    return value


def _meta_frame(df: pd.DataFrame) -> pd.DataFrame:
    cols = [c for c in ("ticker", "fiscal_year", "period_end", "timeframe") if c in df.columns]
    if not cols:
        return pd.DataFrame(index=df.index)
    return df.loc[:, cols].copy()


def _prepare_dataset(raw_df: pd.DataFrame, cfg: PCAReportConfig) -> tuple[pd.DataFrame, Dict[str, Any]]:
    prep_stats: Dict[str, Any] = {"pipeline": cfg.pipeline}
    if cfg.pipeline == "final":
        df_with_news, news_stats = _build_news_features(raw_df, cfg.news_score_col, cfg.news_text_col)
        prep_stats["news"] = news_stats
    elif cfg.pipeline == "residual":
        df_with_news = raw_df.copy()
        prep_stats["news"] = {"applied": False}
    else:
        raise ValueError(f"Unsupported pipeline='{cfg.pipeline}'. Expected 'final' or 'residual'.")

    df = _build_identity_base_feature(
        df_with_news,
        liabilities_col=cfg.liabilities_col,
        equity_col=cfg.equity_col,
        out_col=cfg.base_feature_col,
    )
    return df, prep_stats


def _explained_variance_table(pca_model: Dict[str, Any]) -> pd.DataFrame:
    ratio = np.asarray(pca_model["explained_variance_ratio"], dtype=np.float64).reshape(-1)
    cum = np.cumsum(ratio)
    return pd.DataFrame(
        {
            "component": np.arange(1, ratio.size + 1, dtype=np.int32),
            "explained_variance_ratio": ratio.astype(np.float32),
            "explained_variance_ratio_cumulative": cum.astype(np.float32),
        }
    )


def _loadings_table(pca_model: Dict[str, Any], feature_names: List[str]) -> pd.DataFrame:
    components = np.asarray(pca_model["components"], dtype=np.float64)
    rows: List[Dict[str, Any]] = []
    for comp_idx in range(components.shape[1]):
        weights = components[:, comp_idx]
        order = np.argsort(np.abs(weights))[::-1]
        for rank_idx, feat_idx in enumerate(order, start=1):
            rows.append(
                {
                    "component": int(comp_idx + 1),
                    "feature": str(feature_names[int(feat_idx)]),
                    "loading": float(weights[int(feat_idx)]),
                    "abs_loading": float(abs(weights[int(feat_idx)])),
                    "rank_abs_loading": int(rank_idx),
                }
            )
    return pd.DataFrame(rows)


def _top_loadings_by_component(loadings_df: pd.DataFrame, top_k: int) -> Dict[str, List[Dict[str, Any]]]:
    out: Dict[str, List[Dict[str, Any]]] = {}
    for component, group in loadings_df.groupby("component", sort=True):
        top = (
            group.sort_values("rank_abs_loading", kind="stable")
            .head(int(top_k))
            .loc[:, ["feature", "loading", "abs_loading"]]
        )
        out[f"PC{int(component)}"] = [
            {
                "feature": str(row["feature"]),
                "loading": float(row["loading"]),
                "abs_loading": float(row["abs_loading"]),
            }
            for _, row in top.iterrows()
        ]
    return out


def _scores_table(
    df: pd.DataFrame,
    split_name: str,
    pca_scores: np.ndarray,
    target_raw: np.ndarray,
    target_model_scale: np.ndarray,
) -> pd.DataFrame:
    meta = _meta_frame(df)
    out = meta.copy()
    out["split"] = split_name
    out["target_raw"] = target_raw.reshape(-1).astype(np.float32)
    out["target_model_scale"] = target_model_scale.reshape(-1).astype(np.float32)
    for comp_idx in range(pca_scores.shape[1]):
        out[f"PC{comp_idx + 1}"] = pca_scores[:, comp_idx].astype(np.float32)
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description="Generate PCA diagnostics aligned with the valuation preprocessing pipeline.")
    ap.add_argument("--data", type=str, default="data/processed/main_dataset.csv")
    ap.add_argument("--out-dir", type=str, default="experiments/valuation/runs/pca_report_artifacts")
    ap.add_argument("--pipeline", type=str, choices=["final", "residual"], default="final")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--explained-variance", type=float, default=0.95)
    ap.add_argument("--max-components", type=int, default=64)
    ap.add_argument("--top-k-loadings", type=int, default=10)
    ap.add_argument("--disable-feature-selection", action="store_true")
    ap.add_argument("--disable-log1p-feature-transform", action="store_true")
    args = ap.parse_args()

    cfg = PCAReportConfig()
    cfg.data_path = Path(args.data)
    cfg.out_dir = Path(args.out_dir)
    cfg.pipeline = str(args.pipeline)
    cfg.random_seed = int(args.seed)
    cfg.pca_explained_variance = float(args.explained_variance)
    cfg.pca_max_components = int(args.max_components) if int(args.max_components) > 0 else None
    cfg.top_k_loadings = int(args.top_k_loadings)
    cfg.enable_feature_selection = not bool(args.disable_feature_selection)
    cfg.use_log1p_feature_transform = not bool(args.disable_log1p_feature_transform)

    if cfg.top_k_loadings < 1:
        raise SystemExit("--top-k-loadings must be >= 1")
    if not cfg.data_path.exists():
        raise SystemExit(f"Dataset not found at: {cfg.data_path.resolve()}")

    raw_df = pd.read_csv(cfg.data_path)
    df, prep_stats = _prepare_dataset(raw_df, cfg)

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

    x_scaler = _fit_standard_scaler(X_train)
    X_train_s = _transform_standard_scaler(X_train, x_scaler)
    X_val_s = _transform_standard_scaler(X_val, x_scaler)

    pca_model = fit_pca(
        X_train_s,
        explained_variance=float(cfg.pca_explained_variance),
        max_components=cfg.pca_max_components,
    )
    X_train_pca = transform_pca(X_train_s, pca_model)
    X_val_pca = transform_pca(X_val_s, pca_model)

    explained_df = _explained_variance_table(pca_model)
    loadings_df = _loadings_table(pca_model, feature_names)

    y_train_raw = pd.to_numeric(train_df[cfg.target_col], errors="coerce").fillna(0.0).to_numpy(dtype=np.float32)
    y_val_raw = pd.to_numeric(val_df[cfg.target_col], errors="coerce").fillna(0.0).to_numpy(dtype=np.float32)

    train_scores_df = _scores_table(
        train_df,
        split_name="train",
        pca_scores=X_train_pca,
        target_raw=y_train_raw,
        target_model_scale=y_train.reshape(-1),
    )
    val_scores_df = _scores_table(
        val_df,
        split_name="val",
        pca_scores=X_val_pca,
        target_raw=y_val_raw,
        target_model_scale=y_val.reshape(-1),
    )

    used_news_features = [c for c in NEWS_FEATURE_COLUMNS if c in feature_names]
    top_loadings = _top_loadings_by_component(loadings_df, cfg.top_k_loadings)
    split_years = sorted(
        pd.to_numeric(val_df[cfg.time_col], errors="coerce").dropna().astype(int).unique().tolist()
    ) if cfg.time_col in val_df.columns else []

    summary = {
        "config": {
            "data_path": cfg.data_path,
            "out_dir": cfg.out_dir,
            "pipeline": cfg.pipeline,
            "seed": cfg.random_seed,
            "explained_variance_target": cfg.pca_explained_variance,
            "max_components": cfg.pca_max_components,
            "feature_selection_enabled": cfg.enable_feature_selection,
            "log1p_feature_transform_enabled": cfg.use_log1p_feature_transform,
            "top_k_loadings": cfg.top_k_loadings,
        },
        "data_points": {
            "rows_total": int(len(df)),
            "rows_train": int(len(train_df)),
            "rows_val": int(len(val_df)),
            "val_years": split_years,
        },
        "features": {
            "raw_feature_count": int(len(raw_feature_names)),
            "selected_feature_count": int(len(feature_names)),
            "log1p_feature_count": int(len(log1p_feature_names)),
            "used_news_feature_count": int(len(used_news_features)),
            "used_news_features": used_news_features,
            "has_identity_base_feature": bool(cfg.base_feature_col in feature_names),
            "selected_feature_names": feature_names,
            "log1p_feature_names": log1p_feature_names,
            "feature_selection": feature_selection_stats,
        },
        "pca": {
            "n_features_in": int(pca_model["n_features_in"]),
            "n_components": int(pca_model["n_components"]),
            "explained_variance_ratio_cumulative": float(pca_model["explained_variance_ratio_cum"]),
            "top_loadings_by_component": top_loadings,
        },
        "preprocessing": prep_stats,
        "artifacts": {
            "explained_variance_csv": cfg.out_dir / "pca_explained_variance.csv",
            "component_loadings_csv": cfg.out_dir / "pca_component_loadings.csv",
            "train_scores_csv": cfg.out_dir / "pca_scores_train.csv",
            "val_scores_csv": cfg.out_dir / "pca_scores_val.csv",
            "summary_json": cfg.out_dir / "pca_summary.json",
        },
    }

    _ensure_dir(cfg.out_dir)
    explained_path = cfg.out_dir / "pca_explained_variance.csv"
    loadings_path = cfg.out_dir / "pca_component_loadings.csv"
    train_scores_path = cfg.out_dir / "pca_scores_train.csv"
    val_scores_path = cfg.out_dir / "pca_scores_val.csv"
    summary_path = cfg.out_dir / "pca_summary.json"

    explained_df.to_csv(explained_path, index=False)
    loadings_df.to_csv(loadings_path, index=False)
    train_scores_df.to_csv(train_scores_path, index=False)
    val_scores_df.to_csv(val_scores_path, index=False)
    with summary_path.open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, default=_jsonable)

    print("Saved:")
    print(f"- {summary_path}")
    print(f"- {explained_path}")
    print(f"- {loadings_path}")
    print(f"- {train_scores_path}")
    print(f"- {val_scores_path}")
    print(
        f"PCA summary: pipeline={cfg.pipeline} | raw_features={len(raw_feature_names)} | "
        f"selected={len(feature_names)} | components={int(pca_model['n_components'])} | "
        f"explained={float(pca_model['explained_variance_ratio_cum']):.4f}"
    )


if __name__ == "__main__":
    main()
