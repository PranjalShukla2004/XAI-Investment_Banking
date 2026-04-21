from __future__ import annotations

import argparse
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, List

import numpy as np
import pandas as pd


@dataclass
class PlotSHAPConfig:
    shap_dir: Path = Path("experiments/valuation/SHAP")
    max_features: int = 15
    max_dependence_plots: int = 6
    max_local_plots: int = 5
    dpi: int = 200


def _ordered_model_names(payloads: Dict[str, Dict[str, Any]]) -> List[str]:
    preferred = ["valuation2", "xgb"]
    names = list(payloads.keys())
    ordered: List[str] = [name for name in preferred if name in names]
    ordered.extend(name for name in names if name not in ordered)
    return ordered


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


def _ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def _read_json(path: Path) -> Dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _save_figure(fig, out_path: Path, dpi: int) -> None:
    _ensure_dir(out_path.parent)
    fig.savefig(out_path, dpi=dpi, bbox_inches="tight")


def _sanitize_slug(text: str) -> str:
    safe = "".join(ch.lower() if ch.isalnum() else "_" for ch in str(text))
    while "__" in safe:
        safe = safe.replace("__", "_")
    return safe.strip("_") or "item"


def _resolve_artifact(base_dir: Path, rel_path: str | None) -> Path | None:
    if not rel_path:
        return None
    path = Path(rel_path)
    if path.is_absolute():
        return path
    return (base_dir / path).resolve()


def _shap_summary(model_summary: Dict[str, Any]) -> Dict[str, Any]:
    return model_summary.get("shap", {})


def _explanation_view(model_summary: Dict[str, Any]) -> str | None:
    return _shap_summary(model_summary).get("explanation_view")


def _attribution_scale(model_summary: Dict[str, Any]) -> str | None:
    return _shap_summary(model_summary).get("attribution_scale")


def _target_label(model_summary: Dict[str, Any]) -> str:
    model_details = model_summary.get("model_details", {})
    config = model_details.get("config", {})
    target_col = config.get("target_col")
    if target_col:
        return str(target_col)
    return "target"


def _explanation_label(model_summary: Dict[str, Any]) -> str:
    view = _explanation_view(model_summary)
    scale = _attribution_scale(model_summary)
    parts: List[str] = []
    if view == "full_prediction":
        parts.append("full prediction")
    elif view:
        parts.append(str(view).replace("_", " "))
    if scale:
        parts.append(str(scale))
    return " | ".join(parts)


def _has_required_shap_artifacts(model_summary: Dict[str, Any]) -> bool:
    shap_summary = _shap_summary(model_summary)
    artifacts = model_summary.get("artifacts", {})
    if shap_summary.get("status") != "ok":
        return False

    required = (
        artifacts.get("shap_values"),
        artifacts.get("feature_importance"),
        artifacts.get("local_explanations"),
        artifacts.get("shap_rows"),
    )
    for rel_path in required:
        path = _resolve_artifact(Path.cwd(), rel_path)
        if path is None or not path.exists():
            return False
    return True


def _materialize_valuation2_shap_if_missing(
    shap_dir: Path,
    summary: Dict[str, Any],
) -> Dict[str, Any]:
    valuation2_summary = summary.get("models", {}).get("valuation2")
    if not valuation2_summary:
        return {
            "attempted": False,
            "status": "not_applicable",
            "reason": "valuation2_not_present",
        }
    if _has_required_shap_artifacts(valuation2_summary):
        return {
            "attempted": False,
            "status": "already_available",
        }

    try:
        from src.explainability.shap_valuation import (
            ValuationSHAPConfig as SHAPRunConfig,
            _explain_valuation2_model,
            shap as shap_lib,
        )
    except Exception as exc:
        return {
            "attempted": True,
            "status": "failed",
            "reason": f"import_failed: {exc}",
        }

    if shap_lib is None:
        return {
            "attempted": True,
            "status": "failed",
            "reason": "shap_not_installed",
        }

    root_cfg = summary.get("config", {})
    run_cfg = SHAPRunConfig(
        data_path=Path(root_cfg.get("data_path", "data/processed/main_dataset.csv")),
        out_dir=shap_dir,
        random_seed=int(root_cfg.get("random_seed", 42)),
        models=("valuation2",),
        max_display=int(root_cfg.get("max_display", 15)),
        top_error_rows=int(root_cfg.get("top_error_rows", 50)),
        top_k_local_features=int(root_cfg.get("top_k_local_features", 5)),
        save_plots=bool(root_cfg.get("save_plots", True)),
        mlp_background_size=int(root_cfg.get("mlp_background_size", root_cfg.get("residual_background_size", 64))),
        mlp_explain_rows=int(root_cfg.get("mlp_explain_rows", root_cfg.get("residual_explain_rows", 128))),
        mlp_kernel_nsamples=int(root_cfg.get("mlp_kernel_nsamples", root_cfg.get("residual_kernel_nsamples", 128))),
    )

    try:
        updated_model_summary = _explain_valuation2_model(run_cfg)
    except Exception as exc:
        return {
            "attempted": True,
            "status": "failed",
            "reason": f"materialization_failed: {exc}",
        }

    summary.setdefault("models", {})["valuation2"] = updated_model_summary
    summary.setdefault("library_versions", {})["shap"] = getattr(shap_lib, "__version__", None)
    summary_path = shap_dir / "valuation_shap_summary.json"
    summary_path.write_text(
        json.dumps(summary, indent=2, default=_jsonable),
        encoding="utf-8",
    )
    return {
        "attempted": True,
        "status": "ok",
        "summary_path": str(summary_path),
    }


def _load_model_payloads(shap_dir: Path, summary: Dict[str, Any]) -> Dict[str, Dict[str, Any]]:
    payloads: Dict[str, Dict[str, Any]] = {}
    for model_name, model_summary in summary.get("models", {}).items():
        model_dir = shap_dir / model_name
        payload: Dict[str, Any] = {
            "summary": model_summary,
            "model_dir": model_dir,
        }

        artifacts = model_summary.get("artifacts", {})
        predictions_path = _resolve_artifact(Path.cwd(), artifacts.get("predictions"))
        if predictions_path and predictions_path.exists():
            payload["predictions"] = pd.read_csv(predictions_path)

        importance_path = _resolve_artifact(Path.cwd(), artifacts.get("feature_importance"))
        if importance_path and importance_path.exists():
            payload["feature_importance"] = pd.read_csv(importance_path)

        local_path = _resolve_artifact(Path.cwd(), artifacts.get("local_explanations"))
        if local_path and local_path.exists():
            payload["local_explanations"] = pd.read_csv(local_path)

        shap_rows_path = _resolve_artifact(Path.cwd(), artifacts.get("shap_rows"))
        if shap_rows_path and shap_rows_path.exists():
            payload["shap_rows"] = pd.read_csv(shap_rows_path)

        shap_values_path = _resolve_artifact(Path.cwd(), artifacts.get("shap_values"))
        if shap_values_path and shap_values_path.exists():
            with np.load(shap_values_path, allow_pickle=True) as loaded:
                payload["shap_npz"] = {
                    "feature_names": loaded["feature_names"].tolist(),
                    "shap_values": loaded["shap_values"],
                    "base_values": loaded["base_values"],
                    "y_true_log": loaded["y_true_log"],
                    "y_pred_log_preclip": loaded["y_pred_log_preclip"],
                    "y_pred_log": loaded["y_pred_log"],
                    "y_true_raw": loaded["y_true_raw"],
                    "y_pred_raw": loaded["y_pred_raw"],
                }

        payloads[model_name] = payload
    return payloads


def _plot_metric_comparison(
    payloads: Dict[str, Dict[str, Any]],
    out_path: Path,
    dpi: int,
) -> Path | None:
    if not payloads:
        return None

    plt = _import_pyplot()
    model_names = _ordered_model_names(payloads)
    metrics = ["val_rmse_log", "val_r2_raw", "val_mape_raw"]
    titles = {
        "val_rmse_log": "Validation RMSE (log)",
        "val_r2_raw": "Validation R² (raw)",
        "val_mape_raw": "Validation MAPE (raw)",
    }

    fig, axes = plt.subplots(1, 3, figsize=(14, 4.5), constrained_layout=True)
    x = np.arange(len(model_names))
    colors = ["#1f77b4", "#d2691e", "#2a9d8f", "#8b5cf6"]

    for ax, metric in zip(axes, metrics):
        values = [payloads[name]["summary"]["metrics"].get(metric, np.nan) for name in model_names]
        ax.bar(x, values, color=[colors[i % len(colors)] for i in range(len(model_names))])
        ax.set_xticks(x)
        ax.set_xticklabels(model_names, rotation=20)
        ax.set_title(titles[metric])
        ax.grid(axis="y", alpha=0.2)

    fig.suptitle("Valuation2 MLP vs XGBoost Benchmark", fontsize=14)
    _save_figure(fig, out_path, dpi)
    plt.close(fig)
    return out_path


def _plot_relative_error_cdf_comparison(
    payloads: Dict[str, Dict[str, Any]],
    out_path: Path,
    dpi: int,
) -> Path | None:
    available = {
        name: payload["predictions"]
        for name, payload in payloads.items()
        if "predictions" in payload and not payload["predictions"].empty
    }
    if not available:
        return None

    plt = _import_pyplot()
    fig, ax = plt.subplots(figsize=(8, 5), constrained_layout=True)
    colors = ["#1f77b4", "#d2691e", "#2a9d8f", "#8b5cf6"]

    for idx, model_name in enumerate(_ordered_model_names(available)):
        preds = available[model_name]
        rel = pd.to_numeric(preds["rel_error_raw"], errors="coerce").replace([np.inf, -np.inf], np.nan).dropna()
        if rel.empty:
            continue
        vals = np.sort(rel.to_numpy(dtype=np.float64))
        cdf = np.linspace(0.0, 1.0, len(vals), endpoint=True)
        ax.plot(vals, cdf, label=model_name, color=colors[idx % len(colors)], linewidth=2.0)

    ax.set_xlabel("Relative error")
    ax.set_ylabel("CDF")
    ax.set_title("Relative Error CDF by Model")
    ax.grid(alpha=0.2)
    ax.legend()
    _save_figure(fig, out_path, dpi)
    plt.close(fig)
    return out_path


def _plot_feature_importance_comparison(
    payloads: Dict[str, Dict[str, Any]],
    out_path: Path,
    max_features: int,
    dpi: int,
) -> Path | None:
    importance_frames = {
        name: payload["feature_importance"]
        for name, payload in payloads.items()
        if "feature_importance" in payload and not payload["feature_importance"].empty
    }
    if len(importance_frames) < 2:
        return None

    feature_pool: List[str] = []
    for model_name in _ordered_model_names(importance_frames):
        df = importance_frames[model_name]
        feature_pool.extend(df["feature"].head(max_features).astype(str).tolist())
    ordered_features = []
    for feature in feature_pool:
        if feature not in ordered_features:
            ordered_features.append(feature)
    ordered_features = ordered_features[:max_features]

    if not ordered_features:
        return None

    plt = _import_pyplot()
    fig, ax = plt.subplots(figsize=(12, 6), constrained_layout=True)
    x = np.arange(len(ordered_features))
    width = 0.8 / max(1, len(importance_frames))
    colors = ["#1f77b4", "#d2691e", "#2a9d8f", "#8b5cf6"]
    attribution_scales = {
        _attribution_scale(payloads[name]["summary"])
        for name in importance_frames
        if _attribution_scale(payloads[name]["summary"])
    }
    use_normalized_importance = len(attribution_scales) > 1

    for idx, model_name in enumerate(_ordered_model_names(importance_frames)):
        df = importance_frames[model_name]
        aligned_series = df.set_index("feature").reindex(ordered_features)["mean_abs_shap"].fillna(0.0)
        if use_normalized_importance:
            denom = float(df["mean_abs_shap"].sum())
            aligned = (
                aligned_series / denom if denom > 0.0 else aligned_series * 0.0
            ).to_numpy(dtype=np.float64)
        else:
            aligned = aligned_series.to_numpy(dtype=np.float64)
        offset = (idx - (len(importance_frames) - 1) / 2.0) * width
        ax.bar(x + offset, aligned, width=width, label=model_name, color=colors[idx % len(colors)])

    ax.set_xticks(x)
    ax.set_xticklabels(ordered_features, rotation=45, ha="right")
    if use_normalized_importance:
        ax.set_ylabel("Within-model SHAP importance share")
        ax.set_title("Top SHAP Feature Importance by Model (normalized for different attribution scales)")
    else:
        ax.set_ylabel("Mean |SHAP value|")
        ax.set_title("Top SHAP Feature Importance: Valuation2 MLP vs XGBoost")
    ax.grid(axis="y", alpha=0.2)
    ax.legend()
    _save_figure(fig, out_path, dpi)
    plt.close(fig)
    return out_path


def _plot_actual_vs_pred(
    preds: pd.DataFrame,
    model_name: str,
    model_summary: Dict[str, Any],
    out_path: Path,
    dpi: int,
) -> Path | None:
    if preds.empty:
        return None

    plt = _import_pyplot()
    fig, ax = plt.subplots(figsize=(6.5, 6.5), constrained_layout=True)
    y_true = pd.to_numeric(preds["y_true"], errors="coerce").to_numpy(dtype=np.float64)
    y_pred = pd.to_numeric(preds["y_pred"], errors="coerce").to_numpy(dtype=np.float64)
    ax.scatter(y_true, y_pred, alpha=0.35, s=16, color="#1f77b4", edgecolors="none")
    lo = float(min(np.nanmin(y_true), np.nanmin(y_pred)))
    hi = float(max(np.nanmax(y_true), np.nanmax(y_pred)))
    ax.plot([lo, hi], [lo, hi], linestyle="--", color="#444444", linewidth=1.25)
    target_label = _target_label(model_summary)
    ax.set_xlabel(f"True {target_label}")
    ax.set_ylabel(f"Predicted {target_label}")
    ax.set_title(f"{model_name}: Validation Actual vs Predicted")
    ax.grid(alpha=0.2)
    _save_figure(fig, out_path, dpi)
    plt.close(fig)
    return out_path


def _plot_relative_error_hist(
    preds: pd.DataFrame,
    model_name: str,
    out_path: Path,
    dpi: int,
) -> Path | None:
    if preds.empty:
        return None

    rel = pd.to_numeric(preds["rel_error_raw"], errors="coerce").replace([np.inf, -np.inf], np.nan).dropna()
    if rel.empty:
        return None

    clip_hi = float(np.quantile(rel, 0.99))
    vals = rel.clip(upper=clip_hi)

    plt = _import_pyplot()
    fig, ax = plt.subplots(figsize=(7.5, 4.5), constrained_layout=True)
    ax.hist(vals, bins=40, color="#d2691e", alpha=0.85)
    ax.set_xlabel("Relative error (capped at 99th percentile)")
    ax.set_ylabel("Count")
    ax.set_title(f"{model_name}: Validation Relative Error Distribution")
    ax.grid(axis="y", alpha=0.2)
    _save_figure(fig, out_path, dpi)
    plt.close(fig)
    return out_path


def _plot_error_vs_target(
    preds: pd.DataFrame,
    model_name: str,
    model_summary: Dict[str, Any],
    out_path: Path,
    dpi: int,
) -> Path | None:
    if preds.empty:
        return None

    plt = _import_pyplot()
    fig, ax = plt.subplots(figsize=(7, 4.8), constrained_layout=True)
    y_true = pd.to_numeric(preds["y_true"], errors="coerce").to_numpy(dtype=np.float64)
    rel = pd.to_numeric(preds["rel_error_raw"], errors="coerce").to_numpy(dtype=np.float64)
    ax.scatter(y_true, rel, alpha=0.3, s=14, color="#2a9d8f", edgecolors="none")
    ax.set_xlabel(f"True {_target_label(model_summary)}")
    ax.set_ylabel("Relative error")
    ax.set_title(f"{model_name}: Relative Error vs True Target")
    ax.grid(alpha=0.2)
    _save_figure(fig, out_path, dpi)
    plt.close(fig)
    return out_path


def _plot_feature_importance_bar(
    feature_importance: pd.DataFrame,
    model_name: str,
    model_summary: Dict[str, Any],
    out_path: Path,
    max_features: int,
    dpi: int,
) -> Path | None:
    if feature_importance.empty:
        return None

    plot_df = (
        feature_importance.head(max_features)
        .sort_values("mean_abs_shap", ascending=True, kind="mergesort")
        .reset_index(drop=True)
    )

    plt = _import_pyplot()
    fig, ax = plt.subplots(
        figsize=(9, max(4.5, 0.45 * len(plot_df) + 1.5)),
        constrained_layout=True,
    )
    ax.barh(plot_df["feature"], plot_df["mean_abs_shap"], color="#1f77b4")
    ax.set_xlabel("Mean |SHAP value|")
    ax.set_ylabel("Feature")
    label = _explanation_label(model_summary)
    title = f"{model_name}: SHAP Global Importance"
    if label:
        title = f"{title}\n{label}"
    ax.set_title(title)
    ax.grid(axis="x", alpha=0.2)
    _save_figure(fig, out_path, dpi)
    plt.close(fig)
    return out_path


def _plot_dependence_plots(
    shap_rows: pd.DataFrame,
    shap_npz: Dict[str, Any],
    feature_importance: pd.DataFrame,
    model_name: str,
    model_summary: Dict[str, Any],
    out_dir: Path,
    max_plots: int,
    dpi: int,
) -> List[str]:
    if shap_rows.empty:
        return []

    feature_names = [str(x) for x in shap_npz["feature_names"]]
    shap_values = np.asarray(shap_npz["shap_values"], dtype=np.float64)
    if shap_values.ndim != 2:
        return []

    if not feature_importance.empty:
        selected_features = feature_importance["feature"].astype(str).head(max_plots).tolist()
    else:
        selected_features = feature_names[:max_plots]

    feature_to_idx = {feature: idx for idx, feature in enumerate(feature_names)}
    created: List[str] = []
    plt = _import_pyplot()
    label = _explanation_label(model_summary)

    for rank, feature in enumerate(selected_features, start=1):
        if feature not in shap_rows.columns or feature not in feature_to_idx:
            continue

        idx = feature_to_idx[feature]
        x_vals = pd.to_numeric(shap_rows[feature], errors="coerce").to_numpy(dtype=np.float64)
        y_vals = shap_values[:, idx]
        if np.isnan(x_vals).all():
            continue

        fig, ax = plt.subplots(figsize=(6.5, 4.8), constrained_layout=True)
        ax.scatter(x_vals, y_vals, alpha=0.4, s=18, color="#d2691e", edgecolors="none")
        ax.axhline(0.0, color="#444444", linestyle="--", linewidth=1.0)
        ax.set_xlabel(feature)
        ax.set_ylabel("SHAP value")
        title = f"{model_name}: Dependence Plot {rank} ({feature})"
        if label:
            title = f"{title}\n{label}"
        ax.set_title(title)
        ax.grid(alpha=0.2)
        out_path = out_dir / f"{rank:02d}_{_sanitize_slug(feature)}.png"
        _save_figure(fig, out_path, dpi)
        plt.close(fig)
        created.append(str(out_path))

    return created


def _plot_local_explanations(
    local_df: pd.DataFrame,
    model_name: str,
    model_summary: Dict[str, Any],
    out_dir: Path,
    max_plots: int,
    dpi: int,
) -> List[str]:
    if local_df.empty:
        return []

    created: List[str] = []
    plt = _import_pyplot()
    top_df = local_df.head(max_plots).reset_index(drop=True)
    label = _explanation_label(model_summary)

    for idx, row in top_df.iterrows():
        contrib_rows: List[Dict[str, Any]] = []
        for pos in range(1, 6):
            feature = row.get(f"top_{pos}_feature")
            shap_val = row.get(f"top_{pos}_shap")
            if pd.isna(feature) or pd.isna(shap_val):
                continue
            contrib_rows.append(
                {
                    "feature": str(feature),
                    "shap": float(shap_val),
                }
            )

        if not contrib_rows:
            continue

        plot_df = pd.DataFrame(contrib_rows).sort_values("shap", ascending=True, kind="mergesort")
        colors = ["#d2691e" if v < 0 else "#1f77b4" for v in plot_df["shap"]]
        fig, ax = plt.subplots(
            figsize=(8.5, max(4.5, 0.55 * len(plot_df) + 2.0)),
            constrained_layout=True,
        )
        ax.barh(plot_df["feature"], plot_df["shap"], color=colors)
        ax.axvline(0.0, color="#444444", linestyle="--", linewidth=1.0)
        ticker = row.get("ticker", "unknown")
        fiscal_year = row.get("fiscal_year", "")
        ax.set_title(
            f"{model_name}: Local Explanation #{idx + 1} ({ticker}, {fiscal_year})\n"
            f"{label}\nbase={float(row['base_value_log']):.3f} | pred={float(row['y_pred_log_preclip']):.3f} | "
            f"true={float(row['y_true_log']):.3f}"
        )
        ax.set_xlabel("SHAP contribution")
        ax.set_ylabel("Feature")
        ax.grid(axis="x", alpha=0.2)
        out_path = out_dir / f"{idx + 1:02d}_{_sanitize_slug(str(ticker))}_{int(row.get('fiscal_year', 0) or 0)}.png"
        _save_figure(fig, out_path, dpi)
        plt.close(fig)
        created.append(str(out_path))

    return created


def _plot_model_bundle(
    model_name: str,
    payload: Dict[str, Any],
    cfg: PlotSHAPConfig,
) -> Dict[str, Any]:
    model_dir: Path = payload["model_dir"]
    plots_dir = model_dir / "plots"
    dependence_dir = plots_dir / "dependence"
    local_dir = plots_dir / "local_explanations"
    _ensure_dir(plots_dir)

    created: Dict[str, Any] = {
        "model_dir": str(model_dir),
        "plots_dir": str(plots_dir),
        "files": {},
    }

    preds = payload.get("predictions", pd.DataFrame())
    feature_importance = payload.get("feature_importance", pd.DataFrame())
    local_df = payload.get("local_explanations", pd.DataFrame())
    shap_rows = payload.get("shap_rows", pd.DataFrame())
    shap_npz = payload.get("shap_npz")
    model_summary = payload.get("summary", {})

    plot_path = _plot_actual_vs_pred(
        preds,
        model_name,
        model_summary,
        plots_dir / "validation_actual_vs_pred.png",
        cfg.dpi,
    )
    if plot_path:
        created["files"]["actual_vs_pred"] = str(plot_path)

    plot_path = _plot_relative_error_hist(
        preds,
        model_name,
        plots_dir / "validation_relative_error_hist.png",
        cfg.dpi,
    )
    if plot_path:
        created["files"]["relative_error_hist"] = str(plot_path)

    plot_path = _plot_error_vs_target(
        preds,
        model_name,
        model_summary,
        plots_dir / "validation_error_vs_target.png",
        cfg.dpi,
    )
    if plot_path:
        created["files"]["error_vs_target"] = str(plot_path)

    plot_path = _plot_feature_importance_bar(
        feature_importance,
        model_name,
        model_summary,
        plots_dir / "validation_feature_importance.png",
        cfg.max_features,
        cfg.dpi,
    )
    if plot_path:
        created["files"]["feature_importance"] = str(plot_path)

    if shap_npz is not None and not shap_rows.empty:
        dependence_paths = _plot_dependence_plots(
            shap_rows=shap_rows,
            shap_npz=shap_npz,
            feature_importance=feature_importance,
            model_name=model_name,
            model_summary=model_summary,
            out_dir=dependence_dir,
            max_plots=cfg.max_dependence_plots,
            dpi=cfg.dpi,
        )
        if dependence_paths:
            created["files"]["dependence_plots"] = dependence_paths

    if not local_df.empty:
        local_paths = _plot_local_explanations(
            local_df=local_df,
            model_name=model_name,
            model_summary=model_summary,
            out_dir=local_dir,
            max_plots=cfg.max_local_plots,
            dpi=cfg.dpi,
        )
        if local_paths:
            created["files"]["local_explanations"] = local_paths

    return created


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--shap-dir", type=str, default="experiments/valuation/SHAP")
    ap.add_argument("--max-features", type=int, default=15)
    ap.add_argument("--max-dependence-plots", type=int, default=6)
    ap.add_argument("--max-local-plots", type=int, default=5)
    ap.add_argument("--dpi", type=int, default=200)
    args = ap.parse_args()

    cfg = PlotSHAPConfig(
        shap_dir=Path(args.shap_dir),
        max_features=int(args.max_features),
        max_dependence_plots=int(args.max_dependence_plots),
        max_local_plots=int(args.max_local_plots),
        dpi=int(args.dpi),
    )

    summary_path = cfg.shap_dir / "valuation_shap_summary.json"
    if not summary_path.exists():
        raise SystemExit(f"SHAP summary not found: {summary_path}")

    summary = _read_json(summary_path)
    materialization = _materialize_valuation2_shap_if_missing(cfg.shap_dir, summary)
    if materialization.get("status") == "ok":
        summary = _read_json(summary_path)
    payloads = _load_model_payloads(cfg.shap_dir, summary)

    root_plots_dir = cfg.shap_dir / "plots"
    _ensure_dir(root_plots_dir)
    manifest: Dict[str, Any] = {
        "config": asdict(cfg),
        "materialization": materialization,
        "root_plots": {},
        "models": {},
    }

    plot_path = _plot_metric_comparison(
        payloads=payloads,
        out_path=root_plots_dir / "model_metric_comparison.png",
        dpi=cfg.dpi,
    )
    if plot_path:
        manifest["root_plots"]["metric_comparison"] = str(plot_path)

    plot_path = _plot_relative_error_cdf_comparison(
        payloads=payloads,
        out_path=root_plots_dir / "model_relative_error_cdf.png",
        dpi=cfg.dpi,
    )
    if plot_path:
        manifest["root_plots"]["relative_error_cdf"] = str(plot_path)

    plot_path = _plot_feature_importance_comparison(
        payloads=payloads,
        out_path=root_plots_dir / "model_feature_importance_comparison.png",
        max_features=cfg.max_features,
        dpi=cfg.dpi,
    )
    if plot_path:
        manifest["root_plots"]["feature_importance_comparison"] = str(plot_path)

    for model_name, payload in payloads.items():
        manifest["models"][model_name] = _plot_model_bundle(
            model_name=model_name,
            payload=payload,
            cfg=cfg,
        )

    manifest_path = cfg.shap_dir / "plot_manifest.json"
    manifest_path.write_text(
        json.dumps(manifest, indent=2, default=_jsonable),
        encoding="utf-8",
    )

    print("Saved SHAP plot artifacts:")
    print(f"- {manifest_path}")
    print(f"- {root_plots_dir}")
    for model_name in payloads:
        print(f"- {cfg.shap_dir / model_name / 'plots'}")


if __name__ == "__main__":
    main()
