from __future__ import annotations

import argparse
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, List, Sequence

import numpy as np
import pandas as pd
import xgboost as xgb

try:
    import shap
except ImportError:  # pragma: no cover - optional dependency
    shap = None

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
from src.models.valuation.valuation2 import Valuation2Config
from src.models.valuation.xgb_valuation import (
    XGBValuationConfig,
    _make_asset_weights_from_log_target as _make_xgb_asset_weights,
    _time_aware_split as _xgb_time_aware_split,
    build_xy as build_xgb_xy,
)

DIRECT_MLP_ATTRIBUTION_SCALE = "log1p(total_assets)"
VALID_MODELS = ("valuation2", "xgb")


@dataclass
class ValuationSHAPConfig:
    data_path: Path = Path("data/processed/main_dataset.csv")
    out_dir: Path = Path("experiments/valuation/SHAP")
    random_seed: int = 42
    models: tuple[str, ...] = VALID_MODELS
    max_display: int = 15
    top_error_rows: int = 50
    top_k_local_features: int = 5
    save_plots: bool = True
    mlp_background_size: int = 64
    mlp_explain_rows: int = 128
    mlp_kernel_nsamples: int = 128


def _import_pyplot():
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    return plt


def _jsonable(x: Any) -> Any:
    if isinstance(x, Path):
        return str(x)
    if isinstance(x, (np.integer, np.floating)):
        return x.item()
    if isinstance(x, np.ndarray):
        return x.tolist()
    return x


def _feature_frame_from_df(df: pd.DataFrame, feature_names: Sequence[str]) -> pd.DataFrame:
    return (
        df.reindex(columns=list(feature_names))
        .replace([np.inf, -np.inf], np.nan)
        .fillna(0.0)
        .astype(np.float32)
        .reset_index(drop=True)
    )


def _meta_cols(df: pd.DataFrame) -> List[str]:
    return [c for c in ["ticker", "fiscal_year", "period_end", "timeframe"] if c in df.columns]


def _sample_indices(n_rows: int, max_rows: int, seed: int) -> np.ndarray:
    if max_rows <= 0 or n_rows <= max_rows:
        return np.arange(n_rows, dtype=np.int64)

    rng = np.random.default_rng(seed)
    return np.sort(rng.choice(n_rows, size=max_rows, replace=False).astype(np.int64))


def _build_feature_importance_table(
    feature_names: Sequence[str],
    feature_df: pd.DataFrame,
    shap_values: np.ndarray,
) -> pd.DataFrame:
    rows: List[Dict[str, Any]] = []
    values = np.asarray(shap_values, dtype=np.float64)

    for idx, feature in enumerate(feature_names):
        feature_vals = pd.to_numeric(feature_df[feature], errors="coerce").to_numpy(dtype=np.float64)
        rows.append(
            {
                "rank": int(idx + 1),
                "feature": str(feature),
                "mean_abs_shap": float(np.mean(np.abs(values[:, idx]))),
                "mean_shap": float(np.mean(values[:, idx])),
                "std_abs_shap": float(np.std(np.abs(values[:, idx]))),
                "mean_feature_value": float(np.mean(feature_vals)),
                "std_feature_value": float(np.std(feature_vals)),
            }
        )

    out = pd.DataFrame(rows).sort_values(
        "mean_abs_shap",
        ascending=False,
        kind="mergesort",
    )
    out["rank"] = np.arange(1, len(out) + 1)
    return out.reset_index(drop=True)


def _build_local_explanations_table(
    predictions: pd.DataFrame,
    feature_df: pd.DataFrame,
    feature_names: Sequence[str],
    shap_values: np.ndarray,
    base_values: np.ndarray,
    top_rows: int,
    top_k_features: int,
) -> pd.DataFrame:
    if predictions.empty:
        return pd.DataFrame()

    ordered_idx = (
        predictions["abs_error_log"]
        .to_numpy(dtype=np.float64)
        .argsort()[::-1][: max(1, int(top_rows))]
    )

    rows: List[Dict[str, Any]] = []
    for rank, row_idx in enumerate(ordered_idx, start=1):
        baseline_value = (
            float(predictions.iloc[row_idx]["y_pred_baseline"])
            if "y_pred_baseline" in predictions.columns
            else float("nan")
        )
        row = {
            "rank_by_abs_error_log": int(rank),
            "base_value_log": float(base_values[row_idx]),
            "y_true_log": float(predictions.iloc[row_idx]["y_true_log"]),
            "y_pred_log_preclip": float(predictions.iloc[row_idx]["y_pred_log_preclip"]),
            "y_pred_log": float(predictions.iloc[row_idx]["y_pred_log"]),
            "y_true": float(predictions.iloc[row_idx]["y_true"]),
            "y_pred": float(predictions.iloc[row_idx]["y_pred"]),
            "y_pred_baseline": baseline_value,
            "abs_error_log": float(predictions.iloc[row_idx]["abs_error_log"]),
            "abs_error_raw": float(predictions.iloc[row_idx]["abs_error_raw"]),
            "rel_error_raw": float(predictions.iloc[row_idx]["rel_error_raw"]),
            "clip_applied": bool(predictions.iloc[row_idx]["clip_applied"]),
        }

        for col in _meta_cols(predictions):
            row[col] = predictions.iloc[row_idx][col]

        feature_order = np.argsort(np.abs(shap_values[row_idx]))[::-1][: max(1, int(top_k_features))]
        for pos, feature_idx in enumerate(feature_order, start=1):
            feature_name = str(feature_names[int(feature_idx)])
            row[f"top_{pos}_feature"] = feature_name
            row[f"top_{pos}_shap"] = float(shap_values[row_idx, feature_idx])
            row[f"top_{pos}_abs_shap"] = float(abs(shap_values[row_idx, feature_idx]))
            row[f"top_{pos}_feature_value"] = float(feature_df.iloc[row_idx, int(feature_idx)])

        rows.append(row)

    return pd.DataFrame(rows)


def _save_mean_abs_bar_plot(
    feature_importance: pd.DataFrame,
    out_path: Path,
    max_display: int,
) -> Dict[str, Any]:
    plot_df = (
        feature_importance.head(max(1, int(max_display)))
        .sort_values("mean_abs_shap", ascending=True, kind="mergesort")
        .reset_index(drop=True)
    )

    try:
        plt = _import_pyplot()
        fig, ax = plt.subplots(
            figsize=(10, max(4, 0.45 * len(plot_df) + 1.5)),
            constrained_layout=True,
        )
        ax.barh(plot_df["feature"], plot_df["mean_abs_shap"], color="#1f77b4")
        ax.set_xlabel("Mean |SHAP value| on model scale")
        ax.set_ylabel("Feature")
        ax.set_title("Valuation SHAP Global Importance")
        fig.savefig(out_path, dpi=200, bbox_inches="tight")
        plt.close(fig)
        return {"created": True, "path": str(out_path)}
    except Exception as exc:  # pragma: no cover - environment dependent
        return {"created": False, "reason": str(exc)}


def _save_beeswarm_plot(
    shap_values: np.ndarray,
    feature_df: pd.DataFrame,
    feature_names: Sequence[str],
    out_path: Path,
    max_display: int,
) -> Dict[str, Any]:
    if shap is None:
        return {"created": False, "reason": "shap_not_installed"}

    plt = None
    try:
        plt = _import_pyplot()
        plt.figure(figsize=(10, 7))
        shap.summary_plot(
            shap_values,
            features=feature_df,
            feature_names=list(feature_names),
            max_display=max(1, int(max_display)),
            show=False,
        )
        plt.tight_layout()
        plt.savefig(out_path, dpi=200, bbox_inches="tight")
        plt.close("all")
        return {"created": True, "path": str(out_path)}
    except Exception as exc:  # pragma: no cover - plotting backend dependent
        if plt is not None:
            plt.close("all")
        return {"created": False, "reason": str(exc)}


def _write_json(path: Path, payload: Dict[str, Any]) -> None:
    _ensure_dir(path.parent)
    path.write_text(
        json.dumps(payload, indent=2, default=_jsonable),
        encoding="utf-8",
    )


def _save_model_artifacts(
    model_name: str,
    model_dir: Path,
    all_predictions: pd.DataFrame,
    explained_predictions: pd.DataFrame,
    explained_feature_df: pd.DataFrame,
    feature_names: Sequence[str],
    shap_values: np.ndarray | None,
    base_values: np.ndarray | None,
    model_summary: Dict[str, Any],
    attribution_scale: str,
    shap_method: str,
    explanation_view: str,
    run_cfg: ValuationSHAPConfig,
) -> Dict[str, Any]:
    _ensure_dir(model_dir)

    predictions_path = model_dir / "validation_predictions.csv"
    all_predictions.to_csv(predictions_path, index=False)

    artifacts: Dict[str, Any] = {
        "predictions": str(predictions_path),
    }
    plot_status = {
        "importance_bar": {"created": False, "reason": "shap_not_computed"},
        "beeswarm": {"created": False, "reason": "shap_not_computed"},
    }

    shap_summary: Dict[str, Any] = {
        "shap_method": shap_method,
        "rows_explained": int(len(explained_predictions)),
        "attribution_scale": attribution_scale,
        "explanation_view": explanation_view,
        "feature_count": int(len(feature_names)),
    }

    if shap_values is None or base_values is None:
        shap_summary["status"] = "skipped"
        shap_summary["reason"] = "shap_values_unavailable"
    else:
        shap_feature_rows = pd.concat(
            [
                explained_predictions.reset_index(drop=True),
                explained_feature_df.reset_index(drop=True),
            ],
            axis=1,
        )
        shap_rows_path = model_dir / "validation_shap_rows.csv"
        shap_values_path = model_dir / "validation_shap_values.npz"
        feature_importance_path = model_dir / "validation_shap_feature_importance.csv"
        local_explanations_path = model_dir / "validation_shap_top_error_local_explanations.csv"
        bar_plot_path = model_dir / "validation_shap_importance_bar.png"
        beeswarm_plot_path = model_dir / "validation_shap_beeswarm.png"

        feature_importance = _build_feature_importance_table(
            feature_names,
            explained_feature_df,
            shap_values,
        )
        local_explanations = _build_local_explanations_table(
            explained_predictions.reset_index(drop=True),
            explained_feature_df.reset_index(drop=True),
            feature_names,
            shap_values,
            base_values,
            run_cfg.top_error_rows,
            run_cfg.top_k_local_features,
        )
        reconstructed = np.asarray(base_values, dtype=np.float64) + np.asarray(shap_values, dtype=np.float64).sum(axis=1)
        reconstruction_abs_diff = np.abs(
            reconstructed - explained_predictions["y_pred_log_preclip"].to_numpy(dtype=np.float64)
        )

        shap_feature_rows.to_csv(shap_rows_path, index=False)
        feature_importance.to_csv(feature_importance_path, index=False)
        local_explanations.to_csv(local_explanations_path, index=False)
        np.savez_compressed(
            shap_values_path,
            shap_values=np.asarray(shap_values, dtype=np.float32),
            base_values=np.asarray(base_values, dtype=np.float32),
            feature_names=np.asarray(feature_names, dtype=object),
            y_true_log=explained_predictions["y_true_log"].to_numpy(dtype=np.float32),
            y_pred_log_preclip=explained_predictions["y_pred_log_preclip"].to_numpy(dtype=np.float32),
            y_pred_log=explained_predictions["y_pred_log"].to_numpy(dtype=np.float32),
            y_true_raw=explained_predictions["y_true"].to_numpy(dtype=np.float32),
            y_pred_raw=explained_predictions["y_pred"].to_numpy(dtype=np.float32),
        )

        plot_status["importance_bar"] = _save_mean_abs_bar_plot(
            feature_importance,
            bar_plot_path,
            run_cfg.max_display,
        )
        if run_cfg.save_plots:
            plot_status["beeswarm"] = _save_beeswarm_plot(
                np.asarray(shap_values, dtype=np.float32),
                explained_feature_df,
                feature_names,
                beeswarm_plot_path,
                run_cfg.max_display,
            )
        else:
            plot_status["beeswarm"] = {"created": False, "reason": "disabled"}

        artifacts.update(
            {
                "shap_rows": str(shap_rows_path),
                "shap_values": str(shap_values_path),
                "feature_importance": str(feature_importance_path),
                "local_explanations": str(local_explanations_path),
                "importance_bar_plot": str(bar_plot_path)
                if plot_status["importance_bar"].get("created")
                else None,
                "beeswarm_plot": str(beeswarm_plot_path)
                if plot_status["beeswarm"].get("created")
                else None,
            }
        )
        shap_summary.update(
            {
                "status": "ok",
                "prediction_reconstruction_mean_abs_diff": float(np.mean(reconstruction_abs_diff)),
                "prediction_reconstruction_max_abs_diff": float(np.max(reconstruction_abs_diff)),
                "top_features_by_mean_abs_shap": feature_importance.head(10).to_dict(orient="records"),
            }
        )

    summary = {
        "model_name": model_name,
        "metrics": model_summary["metrics"],
        "model_details": model_summary["model_details"],
        "feature_representation": model_summary["feature_representation"],
        "feature_names": list(feature_names),
        "artifacts": artifacts,
        "shap": shap_summary,
        "plot_status": plot_status,
    }
    summary_path = model_dir / "model_shap_summary.json"
    _write_json(summary_path, summary)
    summary["artifacts"]["summary"] = str(summary_path)
    return summary


def _train_xgb_baseline(
    cfg: XGBValuationConfig,
) -> Dict[str, Any]:
    df = pd.read_csv(cfg.data_path)
    train_df, val_df = _xgb_time_aware_split(
        df=df,
        time_col=cfg.time_col,
        min_val_rows=cfg.min_val_rows,
        seed=cfg.random_seed,
        val_ratio_fallback=cfg.val_ratio_fallback,
    )

    X_train, y_train, feature_names = build_xgb_xy(train_df, cfg)
    X_val, y_val, _ = build_xgb_xy(val_df, cfg)
    feature_df = _feature_frame_from_df(val_df, feature_names)

    y_train_1d = y_train.reshape(-1).astype(np.float32)
    y_val_1d = y_val.reshape(-1).astype(np.float32)
    w_train = None
    if cfg.use_sample_weights:
        w_train = _make_xgb_asset_weights(
            y_train,
            power=cfg.weight_power,
            w_min=cfg.w_min,
            w_max=cfg.w_max,
        )

    dtrain = xgb.DMatrix(X_train, label=y_train_1d, weight=w_train, feature_names=feature_names)
    dval = xgb.DMatrix(X_val, label=y_val_1d, feature_names=feature_names)
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
        verbose_eval=False,
    )

    best_iteration = int(getattr(booster, "best_iteration", booster.num_boosted_rounds() - 1))
    iteration_range = (0, best_iteration + 1)
    yhat_train_log_preclip = booster.predict(dtrain, iteration_range=iteration_range).reshape(-1, 1)
    yhat_val_log_preclip = booster.predict(dval, iteration_range=iteration_range).reshape(-1, 1)

    clip_lo = float(np.min(y_train))
    clip_hi = float(np.max(y_train))
    if cfg.use_quantile_clip:
        clip_lo = float(np.quantile(y_train, cfg.clip_q_lo))
        clip_hi = float(np.quantile(y_train, cfg.clip_q_hi))
    yhat_train_log = np.clip(yhat_train_log_preclip, clip_lo, clip_hi)
    yhat_val_log = np.clip(yhat_val_log_preclip, clip_lo, clip_hi)

    if cfg.log_target:
        y_train_raw = np.expm1(y_train)
        y_val_raw = np.expm1(y_val)
        yhat_val_raw = np.expm1(yhat_val_log)
        yhat_val_raw_preclip = np.expm1(yhat_val_log_preclip)
    else:
        y_train_raw = y_train
        y_val_raw = y_val
        yhat_val_raw = yhat_val_log
        yhat_val_raw_preclip = yhat_val_log_preclip

    y_mean_log = float(np.mean(y_train))
    yhat_base_val_log = np.full_like(y_val, y_mean_log)
    yhat_base_val_raw = np.expm1(yhat_base_val_log) if cfg.log_target else yhat_base_val_log

    preds = val_df.loc[:, _meta_cols(val_df)].copy().reset_index(drop=True)
    preds["y_true_log"] = y_val.reshape(-1)
    preds["y_pred_log_preclip"] = yhat_val_log_preclip.reshape(-1)
    preds["y_pred_log"] = yhat_val_log.reshape(-1)
    preds["y_true"] = y_val_raw.reshape(-1)
    preds["y_pred"] = yhat_val_raw.reshape(-1)
    preds["y_pred_preclip"] = yhat_val_raw_preclip.reshape(-1)
    preds["y_pred_baseline"] = yhat_base_val_raw.reshape(-1)
    preds["abs_error_log"] = np.abs(preds["y_pred_log"] - preds["y_true_log"])
    preds["abs_error_raw"] = np.abs(preds["y_pred"] - preds["y_true"])
    preds["rel_error_raw"] = preds["abs_error_raw"] / np.maximum(preds["y_true"], 1e-8)
    preds["clip_applied"] = np.abs(preds["y_pred_log"] - preds["y_pred_log_preclip"]) > 1e-9

    return {
        "booster": booster,
        "feature_names": feature_names,
        "feature_df": feature_df,
        "predictions": preds,
        "iteration_range": iteration_range,
        "metrics": {
            "baseline_rmse_log": float(_rmse(y_val, yhat_base_val_log)),
            "baseline_rmse_raw": float(_rmse(y_val_raw, yhat_base_val_raw)),
            "baseline_mape_raw": float(_mape(y_val_raw, yhat_base_val_raw)),
            "train_rmse_log": float(_rmse(y_train, yhat_train_log)),
            "val_rmse_log": float(_rmse(y_val, yhat_val_log)),
            "train_r2_log": float(_r2(y_train, yhat_train_log)),
            "val_r2_log": float(_r2(y_val, yhat_val_log)),
            "val_rmse_raw": float(_rmse(y_val_raw, yhat_val_raw)),
            "val_r2_raw": float(_r2(y_val_raw, yhat_val_raw)),
            "val_mape_raw": float(_mape(y_val_raw, yhat_val_raw)),
        },
        "model_details": {
            "train_rows": int(len(train_df)),
            "validation_rows": int(len(val_df)),
            "years": sorted(df[cfg.time_col].dropna().unique().tolist()) if cfg.time_col in df.columns else None,
            "best_iteration": int(best_iteration),
            "clip": {
                "use_quantile_clip": bool(cfg.use_quantile_clip),
                "q_lo": float(cfg.clip_q_lo),
                "q_hi": float(cfg.clip_q_hi),
                "clip_lo": float(clip_lo),
                "clip_hi": float(clip_hi),
            },
            "config": {k: _jsonable(v) for k, v in asdict(cfg).items()},
        },
        "feature_representation": {
            "kind": "selected_numeric_features",
            "notes": "Matches src.models.valuation.xgb_valuation preprocessing.",
        },
    }


def _explain_xgb_model(
    run_cfg: ValuationSHAPConfig,
) -> Dict[str, Any]:
    cfg = XGBValuationConfig()
    cfg.data_path = run_cfg.data_path
    cfg.random_seed = run_cfg.random_seed
    cfg.out_dir = run_cfg.out_dir / "xgb"

    model_info = _train_xgb_baseline(cfg)
    booster: xgb.Booster = model_info["booster"]
    feature_names: List[str] = model_info["feature_names"]
    feature_df: pd.DataFrame = model_info["feature_df"]
    dval = xgb.DMatrix(
        feature_df.to_numpy(dtype=np.float32, copy=False),
        feature_names=feature_names,
    )
    contribs = booster.predict(
        dval,
        pred_contribs=True,
        iteration_range=model_info["iteration_range"],
    )
    shap_values = contribs[:, :-1].astype(np.float32)
    base_values = contribs[:, -1].astype(np.float32)

    return _save_model_artifacts(
        model_name="xgb",
        model_dir=run_cfg.out_dir / "xgb",
        all_predictions=model_info["predictions"],
        explained_predictions=model_info["predictions"],
        explained_feature_df=feature_df,
        feature_names=feature_names,
        shap_values=shap_values,
        base_values=base_values,
        model_summary=model_info,
        attribution_scale="log1p(total_assets)",
        shap_method="xgboost_pred_contribs",
        explanation_view="full_prediction",
        run_cfg=run_cfg,
    )


def _train_valuation2_mlp(
    cfg: Valuation2Config,
) -> Dict[str, Any]:
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
    yhat_train_model_preclip = yhat_train_s * y_scaler["sigma"] + y_scaler["mu"]
    yhat_val_model_preclip = yhat_val_s * y_scaler["sigma"] + y_scaler["mu"]

    if cfg.use_quantile_clip:
        clip_lo = float(np.quantile(y_train, cfg.clip_q_lo))
        clip_hi = float(np.quantile(y_train, cfg.clip_q_hi))
    else:
        clip_lo = float(np.min(y_train)) - float(cfg.clip_margin)
        clip_hi = float(np.max(y_train)) + float(cfg.clip_margin)

    yhat_train_model = np.clip(yhat_train_model_preclip, clip_lo, clip_hi)
    yhat_val_model = np.clip(yhat_val_model_preclip, clip_lo, clip_hi)

    if cfg.log_target:
        y_train_raw = np.expm1(y_train)
        y_val_raw = np.expm1(y_val)
        yhat_train_raw = np.expm1(yhat_train_model)
        yhat_val_raw = np.expm1(yhat_val_model)
        yhat_val_raw_preclip = np.expm1(yhat_val_model_preclip)
    else:
        y_train_raw = y_train
        y_val_raw = y_val
        yhat_train_raw = yhat_train_model
        yhat_val_raw = yhat_val_model
        yhat_val_raw_preclip = yhat_val_model_preclip

    y_mean_model = float(np.mean(y_train))
    yhat_base_val_model = np.full_like(y_val, y_mean_model)
    yhat_base_val_raw = np.expm1(yhat_base_val_model) if cfg.log_target else yhat_base_val_model

    preds = val_df.loc[:, _meta_cols(val_df)].copy().reset_index(drop=True)
    preds["y_true_log"] = y_val.reshape(-1)
    preds["y_pred_log_preclip"] = yhat_val_model_preclip.reshape(-1)
    preds["y_pred_log"] = yhat_val_model.reshape(-1)
    preds["y_true"] = y_val_raw.reshape(-1)
    preds["y_pred"] = yhat_val_raw.reshape(-1)
    preds["y_pred_preclip"] = yhat_val_raw_preclip.reshape(-1)
    preds["y_pred_baseline"] = yhat_base_val_raw.reshape(-1)
    preds["abs_error_log"] = np.abs(preds["y_pred_log"] - preds["y_true_log"])
    preds["abs_error_raw"] = np.abs(preds["y_pred"] - preds["y_true"])
    preds["rel_error_raw"] = preds["abs_error_raw"] / np.maximum(preds["y_true"], 1e-8)
    preds["clip_applied"] = np.abs(preds["y_pred_log"] - preds["y_pred_log_preclip"]) > 1e-9

    processed_train_df = pd.DataFrame(X_train, columns=feature_names)
    processed_val_df = pd.DataFrame(X_val, columns=feature_names)

    def predict_model_fn(processed_features: np.ndarray) -> np.ndarray:
        arr = np.asarray(processed_features, dtype=np.float32)
        if arr.ndim == 1:
            arr = arr.reshape(1, -1)
        feature_block_s = _transform_standard_scaler(arr, x_scaler)
        model.eval()
        yhat_s = model.forward(feature_block_s, training=False)
        yhat_model = yhat_s * y_scaler["sigma"] + y_scaler["mu"]
        return yhat_model.reshape(-1)

    return {
        "predict_model_fn": predict_model_fn,
        "train_feature_df": processed_train_df,
        "val_feature_df": processed_val_df,
        "feature_names": list(feature_names),
        "predictions": preds,
        "metrics": {
            "baseline_rmse_log": float(_rmse(y_val, yhat_base_val_model)),
            "baseline_rmse_raw": float(_rmse(y_val_raw, yhat_base_val_raw)),
            "baseline_mape_raw": float(_mape(y_val_raw, yhat_base_val_raw)),
            "train_rmse_log": float(_rmse(y_train, yhat_train_model)),
            "val_rmse_log": float(_rmse(y_val, yhat_val_model)),
            "train_r2_log": float(_r2(y_train, yhat_train_model)),
            "val_r2_log": float(_r2(y_val, yhat_val_model)),
            "val_rmse_raw": float(_rmse(y_val_raw, yhat_val_raw)),
            "val_r2_raw": float(_r2(y_val_raw, yhat_val_raw)),
            "val_mape_raw": float(_mape(y_val_raw, yhat_val_raw)),
        },
        "model_details": {
            "train_rows": int(len(train_df)),
            "validation_rows": int(len(val_df)),
            "years": sorted(df[cfg.time_col].dropna().unique().tolist()) if cfg.time_col in df.columns else None,
            "clip": {
                "use_quantile_clip": bool(cfg.use_quantile_clip),
                "q_lo": float(cfg.clip_q_lo),
                "q_hi": float(cfg.clip_q_hi),
                "clip_lo": float(clip_lo),
                "clip_hi": float(clip_hi),
                "clip_margin": float(cfg.clip_margin),
            },
            "log1p_feature_names": list(log1p_feature_names),
            "training": {
                "best_epoch": _to_jsonable(history.get("best_epoch")),
                "best_val_loss": _to_jsonable(history.get("best_val_loss")),
                "stopped_early": _to_jsonable(history.get("stopped_early")),
                "final_lr": float(history.get("lr", [float(cfg.lr)])[-1]) if history.get("lr") else float(cfg.lr),
                "lr_reductions": int(history.get("lr_reductions", 0)),
                "lr_increases": int(history.get("lr_increases", 0)),
                "lr_scheduler_switches": int(history.get("lr_scheduler_switches", 0)),
                "lr_scheduler_mode_last": history.get("lr_scheduler_mode", [None])[-1]
                if history.get("lr_scheduler_mode")
                else None,
            },
            "config": {k: _jsonable(v) for k, v in asdict(cfg).items()},
        },
        "feature_representation": {
            "kind": "processed_selected_features_direct_mlp_shap",
            "notes": "Selected features follow src.models.valuation.valuation2 preprocessing, including log1p transforms.",
        },
    }


def _explain_valuation2_model(
    run_cfg: ValuationSHAPConfig,
) -> Dict[str, Any]:
    cfg = Valuation2Config()
    cfg.data_path = run_cfg.data_path
    cfg.random_seed = run_cfg.random_seed
    cfg.out_dir = run_cfg.out_dir / "valuation2"
    cfg.print_every = 10

    model_info = _train_valuation2_mlp(cfg)
    all_predictions: pd.DataFrame = model_info["predictions"].reset_index(drop=True)
    val_feature_df: pd.DataFrame = model_info["val_feature_df"].reset_index(drop=True)
    train_feature_df: pd.DataFrame = model_info["train_feature_df"].reset_index(drop=True)

    explain_idx = _sample_indices(
        n_rows=len(val_feature_df),
        max_rows=run_cfg.mlp_explain_rows,
        seed=run_cfg.random_seed,
    )
    background_idx = _sample_indices(
        n_rows=len(train_feature_df),
        max_rows=run_cfg.mlp_background_size,
        seed=run_cfg.random_seed + 1,
    )

    explained_predictions = all_predictions.iloc[explain_idx].reset_index(drop=True).copy()
    explained_feature_df = val_feature_df.iloc[explain_idx].reset_index(drop=True)
    background = train_feature_df.iloc[background_idx].to_numpy(dtype=np.float32, copy=False)

    shap_values = None
    base_values = None
    if shap is not None:
        explainer = shap.KernelExplainer(model_info["predict_model_fn"], background)
        raw_shap = explainer.shap_values(
            explained_feature_df.to_numpy(dtype=np.float32, copy=False),
            nsamples=int(run_cfg.mlp_kernel_nsamples),
        )
        if isinstance(raw_shap, list):
            raw_shap = raw_shap[0]
        shap_values = np.asarray(raw_shap, dtype=np.float32)

        expected_value = explainer.expected_value
        expected_value_arr = np.asarray(expected_value, dtype=np.float64).reshape(-1)
        expected_scalar = float(expected_value_arr[0]) if expected_value_arr.size else float(expected_value)
        base_values = np.full(len(explained_feature_df), expected_scalar, dtype=np.float32)

    return _save_model_artifacts(
        model_name="valuation2",
        model_dir=run_cfg.out_dir / "valuation2",
        all_predictions=all_predictions,
        explained_predictions=explained_predictions,
        explained_feature_df=explained_feature_df,
        feature_names=model_info["feature_names"],
        shap_values=shap_values,
        base_values=base_values,
        model_summary=model_info,
        attribution_scale=DIRECT_MLP_ATTRIBUTION_SCALE,
        shap_method="shap_kernel_explainer_full_prediction",
        explanation_view="full_prediction",
        run_cfg=run_cfg,
    )


def _parse_models(values: Sequence[str] | None) -> tuple[str, ...]:
    if not values:
        return VALID_MODELS

    parsed = tuple(str(v).strip().lower() for v in values if str(v).strip())
    invalid = sorted(set(parsed) - set(VALID_MODELS))
    if invalid:
        raise SystemExit(f"Unsupported model(s): {invalid}. Valid choices: {list(VALID_MODELS)}")
    return parsed


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--main", type=str, default="data/processed/main_dataset.csv")
    ap.add_argument("--out-dir", type=str, default="experiments/valuation/SHAP")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--models", nargs="+", default=list(VALID_MODELS))
    ap.add_argument("--max-display", type=int, default=15)
    ap.add_argument("--top-error-rows", type=int, default=50)
    ap.add_argument("--top-k-local-features", type=int, default=5)
    ap.add_argument("--skip-plots", action="store_true")
    ap.add_argument("--mlp-background-size", type=int, default=64)
    ap.add_argument("--mlp-explain-rows", type=int, default=128)
    ap.add_argument("--mlp-kernel-nsamples", type=int, default=128)
    ap.add_argument("--residual-background-size", type=int, default=None, help=argparse.SUPPRESS)
    ap.add_argument("--residual-explain-rows", type=int, default=None, help=argparse.SUPPRESS)
    ap.add_argument("--residual-kernel-nsamples", type=int, default=None, help=argparse.SUPPRESS)
    args = ap.parse_args()

    mlp_background_size = (
        int(args.residual_background_size)
        if args.residual_background_size is not None
        else int(args.mlp_background_size)
    )
    mlp_explain_rows = (
        int(args.residual_explain_rows)
        if args.residual_explain_rows is not None
        else int(args.mlp_explain_rows)
    )
    mlp_kernel_nsamples = (
        int(args.residual_kernel_nsamples)
        if args.residual_kernel_nsamples is not None
        else int(args.mlp_kernel_nsamples)
    )

    run_cfg = ValuationSHAPConfig(
        data_path=Path(args.main),
        out_dir=Path(args.out_dir),
        random_seed=int(args.seed),
        models=_parse_models(args.models),
        max_display=int(args.max_display),
        top_error_rows=int(args.top_error_rows),
        top_k_local_features=int(args.top_k_local_features),
        save_plots=not bool(args.skip_plots),
        mlp_background_size=mlp_background_size,
        mlp_explain_rows=mlp_explain_rows,
        mlp_kernel_nsamples=mlp_kernel_nsamples,
    )

    _ensure_dir(run_cfg.out_dir)

    model_summaries: Dict[str, Any] = {}
    for model_name in run_cfg.models:
        if model_name == "xgb":
            model_summaries["xgb"] = _explain_xgb_model(run_cfg)
        elif model_name == "valuation2":
            model_summaries["valuation2"] = _explain_valuation2_model(run_cfg)

    summary = {
        "config": asdict(run_cfg),
        "inputs": {
            "main_dataset": str(run_cfg.data_path),
        },
        "primary_model": "valuation2",
        "comparison_model": "xgb" if "xgb" in model_summaries else None,
        "library_versions": {
            "xgboost": getattr(xgb, "__version__", None),
            "shap": getattr(shap, "__version__", None) if shap is not None else None,
        },
        "models": model_summaries,
    }
    summary_path = run_cfg.out_dir / "valuation_shap_summary.json"
    _write_json(summary_path, summary)

    print("Saved valuation SHAP artifacts:")
    print(f"- {summary_path}")
    for model_name in run_cfg.models:
        print(f"- {run_cfg.out_dir / model_name}")

    print("\nModel metrics:")
    for model_name in run_cfg.models:
        metrics = model_summaries[model_name]["metrics"]
        print(
            f"{model_name} val_rmse_log={metrics['val_rmse_log']:.6f} | "
            f"val_r2_raw={metrics['val_r2_raw']:.6f} | "
            f"val_mape_raw={metrics['val_mape_raw']:.6f}"
        )


if __name__ == "__main__":
    main()
