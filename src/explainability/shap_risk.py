from __future__ import annotations

import argparse
import json
import os
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, List, Sequence

import numpy as np
import pandas as pd
import xgboost as xgb

os.environ.setdefault("MPLCONFIGDIR", str((Path("/tmp") / "matplotlib").resolve()))
os.environ.setdefault("XDG_CACHE_HOME", str((Path("/tmp") / "xdg-cache").resolve()))

try:
    import shap
except ImportError:  # pragma: no cover - optional dependency
    shap = None

from src.models.nn.mlp import MLP
from src.models.risk.modeling import _clip_predictions, transform_standard_scaler
from src.models.valuation.valuation import _ensure_dir

VALID_MODELS = ("mlp", "xgb")


@dataclass
class RiskSHAPConfig:
    out_dir: Path = Path("experiments/risk/SHAP")
    mlp_run_dir: Path = Path("experiments/risk/runs/mlp_risk_artifacts")
    xgb_run_dir: Path = Path("experiments/risk/runs/xgb_risk_artifacts")
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


def _write_json(path: Path, payload: Dict[str, Any]) -> None:
    _ensure_dir(path.parent)
    path.write_text(json.dumps(payload, indent=2, default=_jsonable), encoding="utf-8")


def _read_json(path: Path) -> Dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _resolve_artifact(base_dir: Path, rel_path: str | None) -> Path | None:
    if not rel_path:
        return None
    path = Path(rel_path)
    if path.is_absolute():
        return path
    return (base_dir / path).resolve()


def _sample_indices(n_rows: int, max_rows: int, seed: int) -> np.ndarray:
    if max_rows <= 0 or n_rows <= max_rows:
        return np.arange(n_rows, dtype=np.int64)
    rng = np.random.default_rng(seed)
    return np.sort(rng.choice(n_rows, size=max_rows, replace=False).astype(np.int64))


def _feature_frame_from_df(df: pd.DataFrame, feature_names: Sequence[str]) -> pd.DataFrame:
    return (
        df.reindex(columns=list(feature_names))
        .apply(pd.to_numeric, errors="coerce")
        .replace([np.inf, -np.inf], np.nan)
        .fillna(0.0)
        .astype(np.float32)
        .reset_index(drop=True)
    )


def _meta_cols(df: pd.DataFrame) -> List[str]:
    preferred = ["ticker", "cik", "fiscal_year", "period_end", "timeframe", "split"]
    return [c for c in preferred if c in df.columns]


def _build_feature_importance_table(
    feature_names: Sequence[str],
    feature_df: pd.DataFrame,
    shap_values: np.ndarray,
) -> pd.DataFrame:
    values = np.asarray(shap_values, dtype=np.float64)
    rows: List[Dict[str, Any]] = []
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

    out = pd.DataFrame(rows).sort_values("mean_abs_shap", ascending=False, kind="mergesort")
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
        predictions["abs_error"]
        .to_numpy(dtype=np.float64)
        .argsort()[::-1][: max(1, int(top_rows))]
    )

    rows: List[Dict[str, Any]] = []
    for rank, row_idx in enumerate(ordered_idx, start=1):
        row = {
            "rank_by_abs_error": int(rank),
            "base_value": float(base_values[row_idx]),
            "y_true": float(predictions.iloc[row_idx]["y_true"]),
            "y_pred": float(predictions.iloc[row_idx]["y_pred"]),
            "y_pred_model_output": float(predictions.iloc[row_idx]["y_pred_model_output"]),
            "y_pred_baseline": float(predictions.iloc[row_idx]["y_pred_baseline"]),
            "abs_error": float(predictions.iloc[row_idx]["abs_error"]),
            "rel_error": float(predictions.iloc[row_idx]["rel_error"]),
            "clip_applied": bool(predictions.iloc[row_idx]["clip_applied"]),
        }
        if "y_true_raw_drawdown" in predictions.columns:
            row["y_true_raw_drawdown"] = float(predictions.iloc[row_idx]["y_true_raw_drawdown"])

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
        fig, ax = plt.subplots(figsize=(10, max(4, 0.45 * len(plot_df) + 1.5)), constrained_layout=True)
        ax.barh(plot_df["feature"], plot_df["mean_abs_shap"], color="#1f77b4")
        ax.set_xlabel("Mean |SHAP value| on predicted severity scale")
        ax.set_ylabel("Feature")
        ax.set_title("Risk SHAP Global Importance")
        fig.savefig(out_path, dpi=200, bbox_inches="tight")
        plt.close(fig)
        return {"created": True, "path": str(out_path)}
    except Exception as exc:  # pragma: no cover - plotting backend dependent
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
    shap_method: str,
    run_cfg: RiskSHAPConfig,
) -> Dict[str, Any]:
    _ensure_dir(model_dir)

    predictions_path = model_dir / "validation_predictions.csv"
    all_predictions.to_csv(predictions_path, index=False)

    artifacts: Dict[str, Any] = {"predictions": str(predictions_path)}
    plot_status = {
        "importance_bar": {"created": False, "reason": "shap_not_computed"},
        "beeswarm": {"created": False, "reason": "shap_not_computed"},
    }

    shap_summary: Dict[str, Any] = {
        "shap_method": shap_method,
        "rows_explained": int(len(explained_predictions)),
        "feature_count": int(len(feature_names)),
        "attribution_scale": "drawdown_severity_model_output",
        "explanation_view": "full_prediction",
    }

    if shap_values is None or base_values is None:
        shap_summary["status"] = "skipped"
        shap_summary["reason"] = "shap_values_unavailable"
    else:
        shap_feature_rows = pd.concat(
            [explained_predictions.reset_index(drop=True), explained_feature_df.reset_index(drop=True)],
            axis=1,
        )
        shap_rows_path = model_dir / "validation_shap_rows.csv"
        shap_values_path = model_dir / "validation_shap_values.npz"
        feature_importance_path = model_dir / "validation_shap_feature_importance.csv"
        local_explanations_path = model_dir / "validation_shap_top_error_local_explanations.csv"
        bar_plot_path = model_dir / "validation_shap_importance_bar.png"
        beeswarm_plot_path = model_dir / "validation_shap_beeswarm.png"

        feature_importance = _build_feature_importance_table(feature_names, explained_feature_df, shap_values)
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
            reconstructed - explained_predictions["y_pred_model_output"].to_numpy(dtype=np.float64)
        )

        shap_feature_rows.to_csv(shap_rows_path, index=False)
        feature_importance.to_csv(feature_importance_path, index=False)
        local_explanations.to_csv(local_explanations_path, index=False)
        np.savez_compressed(
            shap_values_path,
            shap_values=np.asarray(shap_values, dtype=np.float32),
            base_values=np.asarray(base_values, dtype=np.float32),
            feature_names=np.asarray(feature_names, dtype=object),
            y_true=explained_predictions["y_true"].to_numpy(dtype=np.float32),
            y_pred=explained_predictions["y_pred"].to_numpy(dtype=np.float32),
            y_pred_model_output=explained_predictions["y_pred_model_output"].to_numpy(dtype=np.float32),
            y_pred_baseline=explained_predictions["y_pred_baseline"].to_numpy(dtype=np.float32),
        )

        plot_status["importance_bar"] = _save_mean_abs_bar_plot(feature_importance, bar_plot_path, run_cfg.max_display)
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
                "importance_bar_plot": str(bar_plot_path) if plot_status["importance_bar"].get("created") else None,
                "beeswarm_plot": str(beeswarm_plot_path) if plot_status["beeswarm"].get("created") else None,
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


def _load_run_summary(run_dir: Path) -> Dict[str, Any]:
    summary_path = run_dir / "run_summary.json"
    if not summary_path.exists():
        raise FileNotFoundError(f"Run summary not found: {summary_path}")
    return _read_json(summary_path)


def _prepare_prediction_columns(preds: pd.DataFrame, model_output: np.ndarray) -> pd.DataFrame:
    out = preds.copy().reset_index(drop=True)
    out["y_pred_model_output"] = np.asarray(model_output, dtype=np.float64).reshape(-1)
    out["y_pred"] = _clip_predictions(out["y_pred_model_output"].to_numpy(dtype=np.float64))
    out["abs_error"] = np.abs(out["y_pred"] - pd.to_numeric(out["y_true"], errors="coerce").to_numpy(dtype=np.float64))
    out["rel_error"] = out["abs_error"] / np.maximum(pd.to_numeric(out["y_true"], errors="coerce"), 1e-8)
    out["clip_applied"] = np.abs(out["y_pred"] - out["y_pred_model_output"]) > 1e-9
    return out


def _load_mlp_model_from_artifacts(summary: Dict[str, Any], run_dir: Path) -> Dict[str, Any]:
    artifacts = summary.get("artifacts", {})
    scalers_path = _resolve_artifact(Path.cwd(), artifacts.get("scalers_path"))
    weights_path = _resolve_artifact(Path.cwd(), artifacts.get("weights_path"))
    predictions_path = _resolve_artifact(Path.cwd(), artifacts.get("predictions_path"))
    if scalers_path is None or not scalers_path.exists():
        raise FileNotFoundError(f"MLP scaler artifact not found for {run_dir}")
    if weights_path is None or not weights_path.exists():
        raise FileNotFoundError(f"MLP weights artifact not found for {run_dir}")
    if predictions_path is None or not predictions_path.exists():
        raise FileNotFoundError(f"MLP predictions artifact not found for {run_dir}")

    with np.load(scalers_path, allow_pickle=True) as loaded:
        feature_names = [str(x) for x in loaded["feature_names"].tolist()]
        x_scaler = {
            "mu": np.asarray(loaded["x_mu"], dtype=np.float64),
            "sigma": np.asarray(loaded["x_sigma"], dtype=np.float64),
        }
        y_scaler = {
            "mu": np.asarray(loaded["y_mu"], dtype=np.float64),
            "sigma": np.asarray(loaded["y_sigma"], dtype=np.float64),
        }

    config = summary["config"]
    model = MLP.from_dims(
        input_dim=len(feature_names),
        hidden_dims=tuple(int(v) for v in config["hidden_dims"]),
        output_dim=1,
        activation=str(config["activation"]).lower(),
        dropout=float(config["dropout"]),
        init=str(config["init"]),
        weight_scale=float(config["weight_scale"]),
        l2=float(config["l2_in_layers"]),
        l1=float(config["l1_in_layers"]),
    )
    with np.load(weights_path) as loaded:
        for ref in model.params_and_grads():
            key = f"layer_{ref.layer_idx}_{ref.name}"
            if key not in loaded:
                raise KeyError(f"Missing weight tensor {key!r} in {weights_path}")
            ref.value[...] = np.asarray(loaded[key], dtype=np.float64)

    predictions = pd.read_csv(predictions_path)

    def predict_raw(features: np.ndarray) -> np.ndarray:
        arr = np.asarray(features, dtype=np.float64)
        if arr.ndim == 1:
            arr = arr.reshape(1, -1)
        arr_s = transform_standard_scaler(arr, x_scaler).astype(np.float32, copy=False)
        model.eval()
        yhat_s = model.forward(arr_s, training=False)
        yhat = np.asarray(yhat_s, dtype=np.float64) * y_scaler["sigma"] + y_scaler["mu"]
        return yhat.reshape(-1)

    return {
        "summary": summary,
        "feature_names": feature_names,
        "predictions": predictions,
        "predict_raw": predict_raw,
    }


def _explain_mlp_model(run_cfg: RiskSHAPConfig) -> Dict[str, Any]:
    summary = _load_run_summary(run_cfg.mlp_run_dir)
    model_info = _load_mlp_model_from_artifacts(summary, run_cfg.mlp_run_dir)

    predictions = model_info["predictions"]
    feature_names = model_info["feature_names"]
    train_preds = predictions.loc[predictions["split"] == "train"].reset_index(drop=True)
    val_preds = predictions.loc[predictions["split"] == "val"].reset_index(drop=True)
    if val_preds.empty:
        raise ValueError(f"No validation rows found in {run_cfg.mlp_run_dir / 'predictions.csv'}")

    train_feature_df = _feature_frame_from_df(train_preds, feature_names)
    val_feature_df = _feature_frame_from_df(val_preds, feature_names)
    explain_idx = _sample_indices(len(val_feature_df), run_cfg.mlp_explain_rows, run_cfg.random_seed)
    background_idx = _sample_indices(len(train_feature_df), run_cfg.mlp_background_size, run_cfg.random_seed + 1)

    explained_feature_df = val_feature_df.iloc[explain_idx].reset_index(drop=True)
    raw_model_output = model_info["predict_raw"](val_feature_df.to_numpy(dtype=np.float32, copy=False))
    prepared_preds = _prepare_prediction_columns(val_preds, raw_model_output)
    explained_predictions = prepared_preds.iloc[explain_idx].reset_index(drop=True)

    shap_values = None
    base_values = None
    if shap is not None:
        background = train_feature_df.iloc[background_idx].to_numpy(dtype=np.float32, copy=False)
        explainer = shap.KernelExplainer(model_info["predict_raw"], background)
        raw_shap = explainer.shap_values(
            explained_feature_df.to_numpy(dtype=np.float32, copy=False),
            nsamples=int(run_cfg.mlp_kernel_nsamples),
        )
        if isinstance(raw_shap, list):
            raw_shap = raw_shap[0]
        shap_values = np.asarray(raw_shap, dtype=np.float32)

        expected_value = explainer.expected_value
        expected_arr = np.asarray(expected_value, dtype=np.float64).reshape(-1)
        expected_scalar = float(expected_arr[0]) if expected_arr.size else float(expected_value)
        base_values = np.full(len(explained_feature_df), expected_scalar, dtype=np.float32)

    model_summary = {
        "metrics": summary["metrics"],
        "model_details": {
            "run_dir": str(run_cfg.mlp_run_dir),
            "config": summary.get("config", {}),
            "training": summary.get("training", {}),
            "baseline": summary.get("baseline", {}),
            "validation_rows": int(len(val_preds)),
            "train_rows": int(len(train_preds)),
        },
        "feature_representation": {
            "kind": "raw_selected_features_with_internal_standard_scaling",
            "notes": "SHAP explains the direct risk MLP prediction on drawdown severity scale.",
        },
    }

    return _save_model_artifacts(
        model_name="mlp",
        model_dir=run_cfg.out_dir / "mlp",
        all_predictions=prepared_preds,
        explained_predictions=explained_predictions,
        explained_feature_df=explained_feature_df,
        feature_names=feature_names,
        shap_values=shap_values,
        base_values=base_values,
        model_summary=model_summary,
        shap_method="shap_kernel_explainer_full_prediction",
        run_cfg=run_cfg,
    )


def _explain_xgb_model(run_cfg: RiskSHAPConfig) -> Dict[str, Any]:
    summary = _load_run_summary(run_cfg.xgb_run_dir)
    artifacts = summary.get("artifacts", {})
    predictions_path = _resolve_artifact(Path.cwd(), artifacts.get("predictions_path"))
    model_path = _resolve_artifact(Path.cwd(), artifacts.get("model_path"))
    if predictions_path is None or not predictions_path.exists():
        raise FileNotFoundError(f"XGB predictions artifact not found for {run_cfg.xgb_run_dir}")
    if model_path is None or not model_path.exists():
        raise FileNotFoundError(f"XGB model artifact not found for {run_cfg.xgb_run_dir}")

    feature_names = [str(x) for x in summary["feature_names"]]
    predictions = pd.read_csv(predictions_path)
    val_preds = predictions.loc[predictions["split"] == "val"].reset_index(drop=True)
    if val_preds.empty:
        raise ValueError(f"No validation rows found in {predictions_path}")

    feature_df = _feature_frame_from_df(val_preds, feature_names)
    booster = xgb.Booster()
    booster.load_model(model_path)
    best_iteration = int(summary["metrics"].get("best_iteration", booster.num_boosted_rounds() - 1))
    dval = xgb.DMatrix(feature_df.to_numpy(dtype=np.float32, copy=False), feature_names=feature_names)
    raw_model_output = booster.predict(dval, iteration_range=(0, best_iteration + 1)).reshape(-1)
    contribs = booster.predict(
        dval,
        pred_contribs=True,
        iteration_range=(0, best_iteration + 1),
    )
    shap_values = contribs[:, :-1].astype(np.float32)
    base_values = contribs[:, -1].astype(np.float32)
    prepared_preds = _prepare_prediction_columns(val_preds, raw_model_output)

    model_summary = {
        "metrics": summary["metrics"],
        "model_details": {
            "run_dir": str(run_cfg.xgb_run_dir),
            "config": summary.get("config", {}),
            "baseline": summary.get("baseline", {}),
            "best_iteration": best_iteration,
            "validation_rows": int(len(val_preds)),
        },
        "feature_representation": {
            "kind": "selected_numeric_features",
            "notes": "Matches src.models.risk.xgb_risk preprocessing.",
        },
    }

    return _save_model_artifacts(
        model_name="xgb",
        model_dir=run_cfg.out_dir / "xgb",
        all_predictions=prepared_preds,
        explained_predictions=prepared_preds,
        explained_feature_df=feature_df,
        feature_names=feature_names,
        shap_values=shap_values,
        base_values=base_values,
        model_summary=model_summary,
        shap_method="xgboost_pred_contribs",
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
    ap.add_argument("--out-dir", type=str, default="experiments/risk/SHAP")
    ap.add_argument("--mlp-run-dir", type=str, default="experiments/risk/runs/mlp_risk_artifacts")
    ap.add_argument("--xgb-run-dir", type=str, default="experiments/risk/runs/xgb_risk_artifacts")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--models", nargs="+", default=list(VALID_MODELS))
    ap.add_argument("--max-display", type=int, default=15)
    ap.add_argument("--top-error-rows", type=int, default=50)
    ap.add_argument("--top-k-local-features", type=int, default=5)
    ap.add_argument("--skip-plots", action="store_true")
    ap.add_argument("--mlp-background-size", type=int, default=64)
    ap.add_argument("--mlp-explain-rows", type=int, default=128)
    ap.add_argument("--mlp-kernel-nsamples", type=int, default=128)
    args = ap.parse_args()

    run_cfg = RiskSHAPConfig(
        out_dir=Path(args.out_dir),
        mlp_run_dir=Path(args.mlp_run_dir),
        xgb_run_dir=Path(args.xgb_run_dir),
        random_seed=int(args.seed),
        models=_parse_models(args.models),
        max_display=int(args.max_display),
        top_error_rows=int(args.top_error_rows),
        top_k_local_features=int(args.top_k_local_features),
        save_plots=not bool(args.skip_plots),
        mlp_background_size=int(args.mlp_background_size),
        mlp_explain_rows=int(args.mlp_explain_rows),
        mlp_kernel_nsamples=int(args.mlp_kernel_nsamples),
    )

    _ensure_dir(run_cfg.out_dir)

    model_summaries: Dict[str, Any] = {}
    for model_name in run_cfg.models:
        if model_name == "mlp":
            model_summaries["mlp"] = _explain_mlp_model(run_cfg)
        elif model_name == "xgb":
            model_summaries["xgb"] = _explain_xgb_model(run_cfg)

    summary = {
        "config": asdict(run_cfg),
        "library_versions": {
            "xgboost": getattr(xgb, "__version__", None),
            "shap": getattr(shap, "__version__", None) if shap is not None else None,
        },
        "primary_model": "mlp" if "mlp" in model_summaries else None,
        "comparison_model": "xgb" if "xgb" in model_summaries else None,
        "models": model_summaries,
    }
    summary_path = run_cfg.out_dir / "risk_shap_summary.json"
    _write_json(summary_path, summary)

    print("Saved risk SHAP artifacts:")
    print(f"- {summary_path}")
    for model_name in run_cfg.models:
        print(f"- {run_cfg.out_dir / model_name}")

    print("\nModel metrics:")
    for model_name in run_cfg.models:
        metrics = model_summaries[model_name]["metrics"]
        print(
            f"{model_name} val_rmse={metrics['val_rmse']:.6f} | "
            f"val_mae={metrics['val_mae']:.6f} | "
            f"val_spearman={metrics['val_spearman']:.6f}"
        )


if __name__ == "__main__":
    main()
