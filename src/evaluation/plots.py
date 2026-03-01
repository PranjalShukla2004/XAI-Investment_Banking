from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


ARTIFACTS: Dict[str, Dict[str, Path]] = {
    "MLP (Direct)": {
        "summary": Path("data/processed/valuation_artifacts/run_summary.json"),
        "predictions": Path("data/processed/valuation_artifacts/predictions.csv"),
    },
    "XGBoost": {
        "summary": Path("data/processed/xgb_valuation_artifacts/run_summary.json"),
        "predictions": Path("data/processed/xgb_valuation_artifacts/predictions.csv"),
    },
    "MLP (Residual)": {
        "summary": Path("data/processed/residual_mlp_valuation_artifacts/run_summary.json"),
        "predictions": Path("data/processed/residual_mlp_valuation_artifacts/predictions.csv"),
    },
}

OUT_DIR = Path("experiments/plots")


def _load_json(path: Path) -> dict:
    if not path.exists():
        raise FileNotFoundError(f"Missing file: {path}")
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def _load_summaries() -> pd.DataFrame:
    rows: List[dict] = []
    for model_name, paths in ARTIFACTS.items():
        payload = _load_json(paths["summary"])
        metrics = payload.get("metrics", {})
        rows.append(
            {
                "model": model_name,
                "val_mape_raw": float(metrics.get("val_mape_raw", np.nan)),
                "val_rmse_log": float(metrics.get("val_rmse_log", np.nan)),
                "val_rmse_raw": float(metrics.get("val_rmse_raw", np.nan)),
                "val_r2_raw": float(metrics.get("val_r2_raw", np.nan)),
                "val_r2_log": float(metrics.get("val_r2_log", np.nan)),
            }
        )
    return pd.DataFrame(rows)


def _load_val_predictions(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Missing predictions file: {path}")
    df = pd.read_csv(path)
    if "split" not in df.columns or "y_true" not in df.columns or "y_pred" not in df.columns:
        raise ValueError(f"Predictions file missing required columns: {path}")
    out = df[df["split"] == "val"].copy()
    out["rel_err"] = np.abs(out["y_pred"] - out["y_true"]) / np.maximum(out["y_true"], 1e-8)
    return out


def _plot_core_metrics(df: pd.DataFrame, out_dir: Path) -> Path:
    metrics = [
        ("val_mape_raw", "Val MAPE (raw) ↓", False),
        ("val_rmse_log", "Val RMSE (log) ↓", False),
        ("val_rmse_raw", "Val RMSE (raw) ↓", True),
        ("val_r2_raw", "Val R2 (raw) ↑", False),
        ("val_r2_log", "Val R2 (log) ↑", False),
    ]

    fig, axes = plt.subplots(2, 3, figsize=(16, 9))
    axes = axes.ravel()
    colors = ["#457b9d", "#2a9d8f", "#e76f51"]

    for i, (col, title, use_log_y) in enumerate(metrics):
        ax = axes[i]
        vals = df[col].to_numpy(dtype=np.float64)
        bars = ax.bar(df["model"], vals, color=colors[: len(df)], edgecolor="black", linewidth=0.8)
        ax.set_title(title)
        ax.tick_params(axis="x", rotation=15)
        if use_log_y:
            ax.set_yscale("log")

        for b, v in zip(bars, vals):
            ax.annotate(
                f"{v:.4g}",
                xy=(b.get_x() + b.get_width() / 2, b.get_height()),
                xytext=(0, 4),
                textcoords="offset points",
                ha="center",
                va="bottom",
                fontsize=9,
            )

    # Hide the extra subplot (2x3 grid for 5 metrics).
    axes[-1].axis("off")
    fig.suptitle("Validation Performance Comparison", fontsize=15)
    fig.tight_layout()

    out_path = out_dir / "validation_performance_comparison.png"
    fig.savefig(out_path, dpi=220, bbox_inches="tight")
    plt.close(fig)
    return out_path


def _plot_error_quantiles(preds_by_model: Dict[str, pd.DataFrame], out_dir: Path) -> Path:
    quantiles = [0.5, 0.75, 0.9, 0.95]
    q_labels = [f"p{int(q*100)}" for q in quantiles]
    models = list(preds_by_model.keys())

    q_table: Dict[str, List[float]] = {}
    for model in models:
        rel_err = preds_by_model[model]["rel_err"].to_numpy(dtype=np.float64)
        q_table[model] = [float(np.quantile(rel_err, q)) for q in quantiles]

    x = np.arange(len(q_labels), dtype=np.float64)
    width = 0.25
    fig, ax = plt.subplots(figsize=(11, 6.5))
    colors = ["#457b9d", "#2a9d8f", "#e76f51"]

    for i, model in enumerate(models):
        offs = (i - (len(models) - 1) / 2.0) * width
        vals = q_table[model]
        bars = ax.bar(x + offs, vals, width=width, label=model, color=colors[i], edgecolor="black", linewidth=0.6)
        for b, v in zip(bars, vals):
            ax.annotate(
                f"{v:.3f}",
                xy=(b.get_x() + b.get_width() / 2, b.get_height()),
                xytext=(0, 3),
                textcoords="offset points",
                ha="center",
                va="bottom",
                fontsize=8,
            )

    ax.set_xticks(x)
    ax.set_xticklabels(q_labels)
    ax.set_ylabel("Relative Error")
    ax.set_title("Validation Relative Error Quantiles")
    ax.legend()
    ax.grid(axis="y", linestyle="--", alpha=0.35)
    fig.tight_layout()

    out_path = out_dir / "validation_relative_error_quantiles.png"
    fig.savefig(out_path, dpi=220, bbox_inches="tight")
    plt.close(fig)

    q_df = pd.DataFrame({"quantile": q_labels})
    for model in models:
        q_df[model] = q_table[model]
    q_df.to_csv(out_dir / "validation_relative_error_quantiles.csv", index=False)

    return out_path


def _plot_error_cdf(preds_by_model: Dict[str, pd.DataFrame], out_dir: Path) -> Path:
    fig, ax = plt.subplots(figsize=(11, 6.5))
    colors = ["#457b9d", "#2a9d8f", "#e76f51"]

    global_p995 = 0.0
    for df in preds_by_model.values():
        rel = df["rel_err"].to_numpy(dtype=np.float64)
        if rel.size:
            global_p995 = max(global_p995, float(np.quantile(rel, 0.995)))
    x_max = global_p995 if global_p995 > 0 else 1.0

    for i, (model, df) in enumerate(preds_by_model.items()):
        rel = np.sort(df["rel_err"].to_numpy(dtype=np.float64))
        if rel.size == 0:
            continue
        y = np.arange(1, rel.size + 1, dtype=np.float64) / rel.size
        mask = rel <= x_max
        ax.plot(rel[mask], y[mask], linewidth=2.0, color=colors[i], label=model)

    ax.set_xlabel("Relative Error")
    ax.set_ylabel("CDF")
    ax.set_title("Validation Relative Error CDF (truncated at p99.5)")
    ax.grid(True, linestyle="--", alpha=0.35)
    ax.legend()
    fig.tight_layout()

    out_path = out_dir / "validation_relative_error_cdf.png"
    fig.savefig(out_path, dpi=220, bbox_inches="tight")
    plt.close(fig)
    return out_path


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    summary_df = _load_summaries()
    summary_df.to_csv(OUT_DIR / "validation_metrics_table.csv", index=False)

    preds_by_model = {
        model: _load_val_predictions(paths["predictions"])
        for model, paths in ARTIFACTS.items()
    }

    p1 = _plot_core_metrics(summary_df, OUT_DIR)
    p2 = _plot_error_quantiles(preds_by_model, OUT_DIR)
    p3 = _plot_error_cdf(preds_by_model, OUT_DIR)

    print("Saved plots:")
    print(f"- {p1}")
    print(f"- {p2}")
    print(f"- {p3}")
    print(f"- {OUT_DIR / 'validation_metrics_table.csv'}")
    print(f"- {OUT_DIR / 'validation_relative_error_quantiles.csv'}")


if __name__ == "__main__":
    main()
