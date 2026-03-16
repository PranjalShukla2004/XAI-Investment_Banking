from __future__ import annotations

import ast
import csv
import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np
import pandas as pd

from src.models.feature_engineering import fit_pca, select_features_by_correlation, transform_pca
from src.models.nn.losses import HuberLoss
from src.models.nn.mlp import MLP
from src.models.nn.optimizer import Adam
from src.models.nn.train import TrainConfig, fit
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
    _time_aware_split,
    _to_jsonable,
    _transform_standard_scaler,
    build_xy,
)


@dataclass
class NewsDrivenMLPValuationConfig:
    # Data
    data_path: Path = Path("data/processed/main_dataset.csv")
    out_dir: Path = Path("experiments/valuation/runs/news_driven_mlp_valuation_artifacts")

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

    # Production-safe inference guards.
    enable_residual_clip: bool = True
    residual_q_lo: float = 0.005
    residual_q_hi: float = 0.995
    min_target_raw: float = 0.0
    max_target_q_hi: float = 0.995
    max_target_cap_multiplier: float = 1.25

    # Safety fallback to identity baseline
    enable_baseline_fallback: bool = True
    fallback_rel_dev_threshold: float = 1.0
    fallback_min_base_abs: float = 1.0
    fallback_ratio_low: float = 0.25
    fallback_ratio_high: float = 4.0
    enable_ood_fallback: bool = True
    ood_zscore_threshold: float = 8.0

    # Feature transform
    use_log1p_feature_transform: bool = True
    enable_feature_selection: bool = True
    min_abs_target_corr: float = 0.01
    max_features_by_target_corr: int | None = 200
    max_inter_feature_corr: float = 0.98
    min_features_after_selection: int = 20
    enable_pca: bool = False
    pca_explained_variance: float = 0.95
    pca_max_components: int | None = 64

    # Model
    hidden_dims: Tuple[int, ...] = (128, 64)
    dropout: float = 0.2
    activation: str = "relu"
    init: str = "he"
    weight_scale: float = 0.01
    l2_in_layers: float = 5e-6
    l1_in_layers: float = 1e-7

    # News columns
    news_score_col: str = "news_sentiment_score"
    news_text_col: str = "news_description"


NEWS_FEATURE_COLUMNS: Tuple[str, ...] = (
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

    out = [f.strip().strip('"').strip("'") for f in fields]
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
) -> Tuple[pd.DataFrame, Dict[str, Any]]:
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


def _meta_cols(d: pd.DataFrame) -> Dict[str, np.ndarray]:
    out: Dict[str, np.ndarray] = {}
    for c in ["ticker", "fiscal_year", "period_end", "timeframe"]:
        if c in d.columns:
            out[c] = d[c].to_numpy()
    return out


def main() -> None:
    cfg = NewsDrivenMLPValuationConfig()

    env_path = os.getenv("VAL_DATA_PATH")
    if env_path:
        cfg.data_path = Path(env_path)

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

    print(
        "[news] "
        f"text_rows={news_stats['rows_with_text_items']:,} | "
        f"score_rows={news_stats['rows_with_score_items']:,} | "
        f"mismatches={news_stats['rows_with_text_score_length_mismatch']:,} | "
        f"text_items={news_stats['total_text_items']:,} | "
        f"score_items={news_stats['total_score_items']:,}"
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

    y_train_residual = y_train - base_train_target_scale
    y_val_residual = y_val - base_val_target_scale

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
        y_train_orig = np.expm1(y_train)
        y_val_orig = np.expm1(y_val)
        yhat_train_model = np.expm1(yhat_train_target_scale)
        yhat_val_model = np.expm1(yhat_val_target_scale)
        yhat_base_train = np.expm1(base_train_target_scale)
        yhat_base_val = np.expm1(base_val_target_scale)
    else:
        y_train_orig = y_train
        y_val_orig = y_val
        yhat_train_model = yhat_train_target_scale
        yhat_val_model = yhat_val_target_scale
        yhat_base_train = base_train_target_scale
        yhat_base_val = base_val_target_scale

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
    safety_stats = fit_production_safety_stats(y_train_orig, yhat_base_train, safety_cfg)

    train_feature_zmax = np.max(np.abs(X_train_s), axis=1)
    val_feature_zmax = np.max(np.abs(X_val_s), axis=1)

    safety_train = apply_production_safety(
        y_model_raw=yhat_train_model,
        y_baseline_raw=yhat_base_train,
        cfg=safety_cfg,
        stats=safety_stats,
        feature_zscore_max=train_feature_zmax,
    )
    safety_val = apply_production_safety(
        y_model_raw=yhat_val_model,
        y_baseline_raw=yhat_base_val,
        cfg=safety_cfg,
        stats=safety_stats,
        feature_zscore_max=val_feature_zmax,
    )

    yhat_train = np.asarray(safety_train["y_pred_final"], dtype=np.float64).reshape(-1, 1)
    yhat_val = np.asarray(safety_val["y_pred_final"], dtype=np.float64).reshape(-1, 1)
    yhat_train_after_clip = np.asarray(safety_train["y_pred_after_clip"], dtype=np.float64).reshape(-1, 1)
    yhat_val_after_clip = np.asarray(safety_val["y_pred_after_clip"], dtype=np.float64).reshape(-1, 1)
    use_base_train = np.asarray(safety_train["fallback_any"], dtype=bool).reshape(-1, 1)
    use_base_val = np.asarray(safety_val["fallback_any"], dtype=bool).reshape(-1, 1)
    fallback_rel_train = np.asarray(safety_train["fallback_rel"], dtype=bool).reshape(-1, 1)
    fallback_rel_val = np.asarray(safety_val["fallback_rel"], dtype=bool).reshape(-1, 1)
    fallback_ratio_train = np.asarray(safety_train["fallback_ratio"], dtype=bool).reshape(-1, 1)
    fallback_ratio_val = np.asarray(safety_val["fallback_ratio"], dtype=bool).reshape(-1, 1)
    fallback_ood_train = np.asarray(safety_train["fallback_ood"], dtype=bool).reshape(-1, 1)
    fallback_ood_val = np.asarray(safety_val["fallback_ood"], dtype=bool).reshape(-1, 1)

    if cfg.log_target:
        yhat_train_target_scale_final = np.log1p(np.clip(yhat_train, a_min=0.0, a_max=None))
        yhat_val_target_scale_final = np.log1p(np.clip(yhat_val, a_min=0.0, a_max=None))
    else:
        yhat_train_target_scale_final = yhat_train
        yhat_val_target_scale_final = yhat_val

    hist_path = cfg.out_dir / "history.json"
    with open(hist_path, "w") as f:
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
        production_safe_inference=np.array([True], dtype=object),
        enable_residual_clip=np.array([cfg.enable_residual_clip], dtype=object),
        residual_q_lo=np.array([cfg.residual_q_lo], dtype=np.float32),
        residual_q_hi=np.array([cfg.residual_q_hi], dtype=np.float32),
        residual_clip_lo=np.array([safety_stats["residual_clip_lo"]], dtype=np.float32),
        residual_clip_hi=np.array([safety_stats["residual_clip_hi"]], dtype=np.float32),
        min_target_raw=np.array([cfg.min_target_raw], dtype=np.float32),
        max_target_q_hi=np.array([cfg.max_target_q_hi], dtype=np.float32),
        max_target_cap_multiplier=np.array([cfg.max_target_cap_multiplier], dtype=np.float32),
        target_raw_cap_hi=np.array([safety_stats["target_raw_cap_hi"]], dtype=np.float32),
        enable_baseline_fallback=np.array([cfg.enable_baseline_fallback], dtype=object),
        fallback_rel_dev_threshold=np.array([cfg.fallback_rel_dev_threshold], dtype=np.float32),
        fallback_min_base_abs=np.array([cfg.fallback_min_base_abs], dtype=np.float32),
        fallback_ratio_low=np.array([cfg.fallback_ratio_low], dtype=np.float32),
        fallback_ratio_high=np.array([cfg.fallback_ratio_high], dtype=np.float32),
        enable_ood_fallback=np.array([cfg.enable_ood_fallback], dtype=object),
        ood_zscore_threshold=np.array([cfg.ood_zscore_threshold], dtype=np.float32),
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
            "y_pred_model": float(yhat_train_model[i, 0]),
            "y_pred_after_safety_clip": float(yhat_train_after_clip[i, 0]),
            "y_pred_baseline": float(yhat_base_train[i, 0]),
            "used_baseline_fallback": int(use_base_train[i, 0]),
            "fallback_rel": int(fallback_rel_train[i, 0]),
            "fallback_ratio": int(fallback_ratio_train[i, 0]),
            "fallback_ood": int(fallback_ood_train[i, 0]),
            "feature_zscore_max": float(train_feature_zmax[i]),
        }
        for k, arr in train_meta.items():
            row[k] = arr[i]
        rows.append(row)
    for i in range(len(val_df)):
        row = {
            "split": "val",
            "y_true": float(y_val_orig[i, 0]),
            "y_pred": float(yhat_val[i, 0]),
            "y_pred_model": float(yhat_val_model[i, 0]),
            "y_pred_after_safety_clip": float(yhat_val_after_clip[i, 0]),
            "y_pred_baseline": float(yhat_base_val[i, 0]),
            "used_baseline_fallback": int(use_base_val[i, 0]),
            "fallback_rel": int(fallback_rel_val[i, 0]),
            "fallback_ratio": int(fallback_ratio_val[i, 0]),
            "fallback_ood": int(fallback_ood_val[i, 0]),
            "feature_zscore_max": float(val_feature_zmax[i]),
        }
        for k, arr in val_meta.items():
            row[k] = arr[i]
        rows.append(row)

    preds_df = pd.DataFrame(rows)
    preds_path = cfg.out_dir / "predictions.csv"
    preds_df.to_csv(preds_path, index=False)

    train_rmse_log = _rmse(y_train, yhat_train_target_scale_final)
    val_rmse_log = _rmse(y_val, yhat_val_target_scale_final)
    train_factor = float(np.exp(train_rmse_log))
    val_factor = float(np.exp(val_rmse_log))

    r2_train_log = _r2(y_train, yhat_train_target_scale_final)
    r2_val_log = _r2(y_val, yhat_val_target_scale_final)
    r2_train_raw = _r2(y_train_orig, yhat_train)
    r2_val_raw = _r2(y_val_orig, yhat_val)
    val_mape_raw = _mape(y_val_orig, yhat_val)
    val_mape_nonzero_raw = mape_nonzero(y_val_orig, yhat_val)
    val_smape_raw = smape(y_val_orig, yhat_val)
    train_rmse_raw = _rmse(y_train_orig, yhat_train)
    val_rmse_raw = _rmse(y_val_orig, yhat_val)

    baseline_rmse_log = _rmse(y_val, base_val_target_scale)
    baseline_factor = float(np.exp(baseline_rmse_log))
    baseline_mape_raw = _mape(y_val_orig, yhat_base_val)
    baseline_mape_nonzero_raw = mape_nonzero(y_val_orig, yhat_base_val)
    baseline_smape_raw = smape(y_val_orig, yhat_base_val)
    baseline_rmse_raw = _rmse(y_val_orig, yhat_base_val)
    baseline_r2_raw = _r2(y_val_orig, yhat_base_val)
    baseline_r2_log = _r2(y_val, base_val_target_scale)
    val_tail_risk = tail_risk_rates(y_val_orig, yhat_val)
    baseline_tail_risk = tail_risk_rates(y_val_orig, yhat_base_val)

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
        },
        "safety": {
            "production_safe_inference": True,
            "config": {
                "enable_residual_clip": bool(cfg.enable_residual_clip),
                "residual_q_lo": float(cfg.residual_q_lo),
                "residual_q_hi": float(cfg.residual_q_hi),
                "min_target_raw": float(cfg.min_target_raw),
                "max_target_q_hi": float(cfg.max_target_q_hi),
                "max_target_cap_multiplier": float(cfg.max_target_cap_multiplier),
                "enable_baseline_fallback": bool(cfg.enable_baseline_fallback),
                "fallback_rel_dev_threshold": float(cfg.fallback_rel_dev_threshold),
                "fallback_min_base_abs": float(cfg.fallback_min_base_abs),
                "fallback_ratio_low": float(cfg.fallback_ratio_low),
                "fallback_ratio_high": float(cfg.fallback_ratio_high),
                "enable_ood_fallback": bool(cfg.enable_ood_fallback),
                "ood_zscore_threshold": float(cfg.ood_zscore_threshold),
            },
            "fitted_stats": {
                "residual_clip_lo": float(safety_stats["residual_clip_lo"]),
                "residual_clip_hi": float(safety_stats["residual_clip_hi"]),
                "target_raw_cap_hi": float(safety_stats["target_raw_cap_hi"]),
            },
            "rates": {
                "train_fallback_rate": float(safety_train["fallback_rate"]),
                "train_fallback_rel_rate": float(safety_train["fallback_rel_rate"]),
                "train_fallback_ratio_rate": float(safety_train["fallback_ratio_rate"]),
                "train_fallback_ood_rate": float(safety_train["fallback_ood_rate"]),
                "val_fallback_rate": float(safety_val["fallback_rate"]),
                "val_fallback_rel_rate": float(safety_val["fallback_rel_rate"]),
                "val_fallback_ratio_rate": float(safety_val["fallback_ratio_rate"]),
                "val_fallback_ood_rate": float(safety_val["fallback_ood_rate"]),
            },
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
            "enable_baseline_fallback": bool(cfg.enable_baseline_fallback),
            "fallback_rel_dev_threshold": float(cfg.fallback_rel_dev_threshold),
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
            "baseline_val_tail_risk": baseline_tail_risk,
            "train_rmse_raw": float(train_rmse_raw),
            "val_rmse_raw": float(val_rmse_raw),
        },
    }

    summary_path = cfg.out_dir / "run_summary.json"
    with open(summary_path, "w") as f:
        json.dump(run_summary, f, indent=2)

    print("=== Done (News-Driven Residual MLP Valuation) ===")
    print(f"Target: {cfg.target_col} (log_target={cfg.log_target})")
    print(f"Residual baseline: {cfg.liabilities_col} + {cfg.equity_col}")
    print(f"Train rows: {len(train_df)} | Val rows: {len(val_df)}")
    print(f"News features used: {len(used_news_features)}")
    print(f"News feature names: {used_news_features}")
    print(f"Identity baseline included as feature: {has_identity_base_feature}")
    print(
        "Feature selection: "
        f"enabled={cfg.enable_feature_selection} | raw={len(raw_feature_names)} | selected={len(feature_names)}"
    )
    print(
        "PCA: "
        f"enabled={cfg.enable_pca} | in_dim={int(X_train_s.shape[1])} | model_dim={int(X_train_model.shape[1])} | "
        f"explained={float(pca_stats['explained_variance_ratio_cum']):.4f}"
    )
    print(
        "Safety: "
        f"residual_clip={cfg.enable_residual_clip} [{safety_stats['residual_clip_lo']:.3f}, {safety_stats['residual_clip_hi']:.3f}] | "
        f"target_cap_hi={safety_stats['target_raw_cap_hi']:.2f} | "
        f"fallback={cfg.enable_baseline_fallback} (val={float(safety_val['fallback_rate']):.4f}, "
        f"rel={float(safety_val['fallback_rel_rate']):.4f}, ratio={float(safety_val['fallback_ratio_rate']):.4f}, "
        f"ood={float(safety_val['fallback_ood_rate']):.4f})"
    )
    if cfg.time_col in df.columns:
        yrs = sorted(df[cfg.time_col].dropna().unique().tolist())
        print(f"Years in data: {yrs}")
    print(f"log1p-transformed feature count: {len(log1p_feature_names)}")
    print(f"Identity baseline Val MAPE (raw): {baseline_mape_raw:.6f}")
    print(f"Identity baseline Val MAPE (nonzero): {baseline_mape_nonzero_raw:.6f}")
    print(f"News+Residual MLP Val MAPE (raw): {val_mape_raw:.6f}")
    print(f"News+Residual MLP Val MAPE (nonzero): {val_mape_nonzero_raw:.6f}")
    print(f"News+Residual MLP Val sMAPE (raw): {val_smape_raw:.6f}")
    print(f"Identity baseline Val RMSE (log): {baseline_rmse_log:.6f} (~x{baseline_factor:.3f})")
    print(f"News+Residual MLP Val RMSE (log): {val_rmse_log:.6f} (~x{val_factor:.3f})")

    if bucket_table is not None:
        print("\nVal performance by size bucket (quintiles):")
        print(bucket_table)

    print(f"Saved: {hist_path}")
    print(f"Saved: {preds_path}")
    print(f"Saved: {summary_path}")
    print(f"Saved: {cfg.out_dir / 'scalers_and_features.npz'}")


if __name__ == "__main__":
    main()
