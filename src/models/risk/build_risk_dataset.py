from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from src.models.risk.dataset_utils import ensure_dir, normalize_tickers, write_json

PROJECT_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_SOURCE_DATASET = PROJECT_ROOT / "data/processed/main_dataset_with_market.csv"
DEFAULT_OUT_PATH = PROJECT_ROOT / "data/nprocessed/risk/risk_dataset.csv"
DEFAULT_SUMMARY_PATH = PROJECT_ROOT / "data/nprocessed/risk/risk_dataset_summary.json"

IDENTIFIER_COLUMNS = [
    "ticker",
    "cik",
    "fiscal_year",
    "period_end",
    "timeframe",
]
TARGET_COLUMNS = [
    "future_1y_max_drawdown",
    "drawdown_severity",
    "target_status",
    "target_usable",
    "target_issue_codes",
]
DROP_COLUMNS = {
    "news_description",
    "news_published_utc",
    "news_year",
    "anchor_trade_date",
    "future_anchor_trade_date",
    "future_horizon_end_date",
    "future_horizon_last_trade_date",
    "future_bar_count",
    "future_1y_total_return",
    "future_mdd_peak_date",
    "future_mdd_trough_date",
    "future_mdd_peak_close",
    "future_mdd_trough_close",
}


@dataclass
class BuildRiskDatasetConfig:
    source_dataset_path: Path = DEFAULT_SOURCE_DATASET
    out_path: Path = DEFAULT_OUT_PATH
    summary_path: Path = DEFAULT_SUMMARY_PATH
    timeframe: str = "annual"
    require_usable_target: bool = True
    target_col: str = "future_1y_max_drawdown"
    severity_col: str = "drawdown_severity"


def _log(message: str) -> None:
    print(message, flush=True)


def _resolve_column(columns: list[str], desired: str) -> str:
    col_map = {str(c).lower(): str(c) for c in columns}
    resolved = col_map.get(desired.lower())
    if not resolved:
        raise ValueError(
            f"Column '{desired}' was not found. Available columns include: {columns}"
        )
    return resolved


def _load_source_dataset(cfg: BuildRiskDatasetConfig) -> pd.DataFrame:
    if not cfg.source_dataset_path.exists():
        raise FileNotFoundError(f"Source dataset not found: {cfg.source_dataset_path}")

    _log(f"[risk-dataset] Loading source dataset from {cfg.source_dataset_path}")
    df = pd.read_csv(cfg.source_dataset_path)
    if df.empty:
        raise ValueError("Source dataset is empty.")

    header = list(df.columns)
    ticker_col = _resolve_column(header, "ticker")
    period_end_col = _resolve_column(header, "period_end")
    timeframe_col = _resolve_column(header, "timeframe")
    target_col = _resolve_column(header, cfg.target_col)
    target_usable_col = _resolve_column(header, "target_usable")
    target_status_col = _resolve_column(header, "target_status")

    df[ticker_col] = normalize_tickers(df[ticker_col])
    df[period_end_col] = pd.to_datetime(df[period_end_col], errors="coerce").dt.strftime("%Y-%m-%d")
    df[timeframe_col] = df[timeframe_col].astype(str).str.strip().str.lower()
    df[target_usable_col] = df[target_usable_col].fillna(False).astype(bool)
    df[target_status_col] = df[target_status_col].astype(str).str.strip()
    df[target_col] = pd.to_numeric(df[target_col], errors="coerce")

    df = df.rename(
        columns={
            ticker_col: "ticker",
            period_end_col: "period_end",
            timeframe_col: "timeframe",
            target_col: cfg.target_col,
            target_usable_col: "target_usable",
            target_status_col: "target_status",
        }
    )
    df = df.dropna(subset=["ticker", "period_end"]).copy()
    return df


def _filter_rows(df: pd.DataFrame, cfg: BuildRiskDatasetConfig) -> tuple[pd.DataFrame, dict[str, int]]:
    stats = {"rows_input": int(len(df))}

    out = df.copy()
    if cfg.timeframe:
        out = out.loc[out["timeframe"] == cfg.timeframe.lower()].copy()
    stats["rows_after_timeframe_filter"] = int(len(out))

    out = out.loc[out[cfg.target_col].notna()].copy()
    stats["rows_after_non_null_target"] = int(len(out))

    if cfg.require_usable_target:
        out = out.loc[out["target_usable"]].copy()
    stats["rows_after_usable_target_filter"] = int(len(out))

    out[cfg.severity_col] = (-pd.to_numeric(out[cfg.target_col], errors="coerce")).clip(lower=0.0)
    out = out.sort_values(["fiscal_year", "ticker", "period_end"], kind="stable").reset_index(drop=True)
    return out, stats


def _ordered_columns(df: pd.DataFrame, cfg: BuildRiskDatasetConfig) -> tuple[list[str], list[str]]:
    id_cols = [col for col in IDENTIFIER_COLUMNS if col in df.columns]
    target_cols = [col for col in TARGET_COLUMNS if col in df.columns]
    feature_cols = [
        col
        for col in df.columns
        if col not in set(id_cols) | set(target_cols) | DROP_COLUMNS
    ]
    return id_cols + target_cols + feature_cols, feature_cols


def _build_summary(
    cfg: BuildRiskDatasetConfig,
    source_df: pd.DataFrame,
    risk_df: pd.DataFrame,
    filter_stats: dict[str, int],
    feature_cols: list[str],
    kept_columns: list[str],
) -> dict[str, Any]:
    status_counts = (
        risk_df["target_status"].value_counts(dropna=False).to_dict()
        if "target_status" in risk_df.columns
        else {}
    )
    return {
        "config": asdict(cfg),
        "rows": {
            "source_dataset": int(len(source_df)),
            **filter_stats,
            "risk_dataset": int(len(risk_df)),
        },
        "unique_tickers": {
            "source_dataset": int(source_df["ticker"].nunique()) if not source_df.empty else 0,
            "risk_dataset": int(risk_df["ticker"].nunique()) if not risk_df.empty else 0,
        },
        "period_range": {
            "min_period_end": risk_df["period_end"].min() if not risk_df.empty else None,
            "max_period_end": risk_df["period_end"].max() if not risk_df.empty else None,
        },
        "fiscal_year_range": {
            "min_fiscal_year": int(pd.to_numeric(risk_df["fiscal_year"], errors="coerce").min())
            if not risk_df.empty and "fiscal_year" in risk_df.columns
            else None,
            "max_fiscal_year": int(pd.to_numeric(risk_df["fiscal_year"], errors="coerce").max())
            if not risk_df.empty and "fiscal_year" in risk_df.columns
            else None,
        },
        "target": {
            "target_col": cfg.target_col,
            "severity_col": cfg.severity_col,
            "severity_defined_as": "clip(-future_1y_max_drawdown, lower=0.0)",
            "status_counts": {str(k): int(v) for k, v in status_counts.items()},
            "severity_min": float(risk_df[cfg.severity_col].min()) if not risk_df.empty else None,
            "severity_max": float(risk_df[cfg.severity_col].max()) if not risk_df.empty else None,
            "severity_mean": float(risk_df[cfg.severity_col].mean()) if not risk_df.empty else None,
        },
        "columns": {
            "kept_columns": kept_columns,
            "feature_columns": feature_cols,
            "dropped_columns": [col for col in source_df.columns if col not in kept_columns],
        },
        "artifacts": {
            "source_dataset_path": str(cfg.source_dataset_path),
            "out_path": str(cfg.out_path),
            "summary_path": str(cfg.summary_path),
        },
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--source-dataset", type=str, default=str(DEFAULT_SOURCE_DATASET))
    ap.add_argument("--out-path", type=str, default=str(DEFAULT_OUT_PATH))
    ap.add_argument("--summary-path", type=str, default=str(DEFAULT_SUMMARY_PATH))
    ap.add_argument("--timeframe", type=str, default="annual")
    ap.add_argument(
        "--keep-nonusable-targets",
        action="store_true",
        help="Keep rows where target_usable is False instead of filtering to clean model-ready rows.",
    )
    args = ap.parse_args()

    cfg = BuildRiskDatasetConfig(
        source_dataset_path=Path(args.source_dataset),
        out_path=Path(args.out_path),
        summary_path=Path(args.summary_path),
        timeframe=str(args.timeframe).strip().lower(),
        require_usable_target=not bool(args.keep_nonusable_targets),
    )

    ensure_dir(cfg.out_path.parent)
    ensure_dir(cfg.summary_path.parent)

    source_df = _load_source_dataset(cfg)
    _log(
        f"[risk-dataset] Source rows={len(source_df)} | "
        f"tickers={source_df['ticker'].nunique() if not source_df.empty else 0}"
    )
    filtered_df, filter_stats = _filter_rows(source_df, cfg)
    _log(
        f"[risk-dataset] Filtered rows={len(filtered_df)} | "
        f"usable_only={cfg.require_usable_target} | timeframe={cfg.timeframe}"
    )

    kept_columns, feature_cols = _ordered_columns(filtered_df, cfg)
    risk_df = filtered_df.loc[:, kept_columns].copy()

    _log(f"[risk-dataset] Writing risk dataset to {cfg.out_path}")
    risk_df.to_csv(cfg.out_path, index=False)

    summary = _build_summary(cfg, source_df, risk_df, filter_stats, feature_cols, kept_columns)
    _log(f"[risk-dataset] Writing summary to {cfg.summary_path}")
    write_json(cfg.summary_path, summary)

    print("Risk dataset build complete:")
    print(f"rows={summary['rows']['risk_dataset']} | tickers={summary['unique_tickers']['risk_dataset']}")
    print(f"dataset={cfg.out_path}")
    print(f"summary={cfg.summary_path}")


if __name__ == "__main__":
    main()
