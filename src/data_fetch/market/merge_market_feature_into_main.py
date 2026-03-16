from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import pandas as pd

from src.models.risk.dataset_utils import ensure_dir, normalize_tickers, write_json

PROJECT_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_MAIN_DATASET = PROJECT_ROOT / "data/processed/main_dataset.csv"
DEFAULT_MARKET_FEATURE_MERGE = PROJECT_ROOT / "data/processed/market/market_feature_merge.csv"
DEFAULT_OUT_PATH = PROJECT_ROOT / "data/processed/main_dataset_with_market.csv"
DEFAULT_SUMMARY_PATH = PROJECT_ROOT / "data/processed/main_dataset_with_market_summary.json"

KEY_COLUMNS = ["ticker", "period_end"]
MARKET_NON_FEATURE_COLUMNS = {"fiscal_year", "timeframe"}


@dataclass
class MergeMarketFeatureIntoMainConfig:
    main_dataset_path: Path = DEFAULT_MAIN_DATASET
    market_feature_merge_path: Path = DEFAULT_MARKET_FEATURE_MERGE
    out_path: Path = DEFAULT_OUT_PATH
    summary_path: Path = DEFAULT_SUMMARY_PATH
    ticker_col: str = "ticker"
    period_end_col: str = "period_end"


def _log(message: str) -> None:
    print(message, flush=True)


def _resolve_column(columns: list[str], candidates: list[str], required: bool = True) -> str | None:
    col_map = {str(c).lower(): str(c) for c in columns}
    for candidate in candidates:
        resolved = col_map.get(candidate.lower())
        if resolved:
            return resolved
    if required:
        raise ValueError(
            f"None of the expected columns {candidates} were found. Available columns: {columns}"
        )
    return None


def _load_csv_with_normalized_keys(
    path: Path,
    ticker_col_name: str,
    period_end_col_name: str,
) -> tuple[pd.DataFrame, str, str]:
    if not path.exists():
        raise FileNotFoundError(f"Dataset not found: {path}")

    df = pd.read_csv(path)
    header = list(df.columns)
    ticker_col = _resolve_column(header, [ticker_col_name])
    period_end_col = _resolve_column(header, [period_end_col_name])

    df[ticker_col] = normalize_tickers(df[ticker_col])
    df[period_end_col] = pd.to_datetime(df[period_end_col], errors="coerce").dt.strftime("%Y-%m-%d")
    df = df.dropna(subset=[ticker_col, period_end_col]).copy()
    df = df.rename(columns={ticker_col: "ticker", period_end_col: "period_end"})
    return df, ticker_col, period_end_col


def _market_merge_columns(main_df: pd.DataFrame, market_df: pd.DataFrame) -> tuple[list[str], list[str]]:
    market_cols = [
        col
        for col in market_df.columns
        if col not in {*KEY_COLUMNS, *MARKET_NON_FEATURE_COLUMNS}
    ]
    new_cols = [col for col in market_cols if col not in main_df.columns]
    overlap_cols = [col for col in market_cols if col in main_df.columns]
    return new_cols, overlap_cols


def _merge_market_into_main(main_df: pd.DataFrame, market_df: pd.DataFrame) -> tuple[pd.DataFrame, list[str], list[str]]:
    new_cols, overlap_cols = _market_merge_columns(main_df, market_df)
    right_cols = [*KEY_COLUMNS, *new_cols, *overlap_cols]

    out = main_df.copy()
    out["__row_id"] = range(len(out))
    merged = out.merge(
        market_df.loc[:, right_cols],
        on=KEY_COLUMNS,
        how="left",
        validate="many_to_one",
        suffixes=("", "__market"),
    )

    for col in overlap_cols:
        new_col = f"{col}__market"
        if new_col not in merged.columns:
            continue
        merged[col] = merged[col].combine_first(merged[new_col])
        merged = merged.drop(columns=[new_col])

    merged = merged.sort_values("__row_id", kind="stable").drop(columns=["__row_id"]).reset_index(drop=True)
    return merged, new_cols, overlap_cols


def _build_summary(
    cfg: MergeMarketFeatureIntoMainConfig,
    main_df: pd.DataFrame,
    market_df: pd.DataFrame,
    merged_df: pd.DataFrame,
    new_cols: list[str],
    overlap_cols: list[str],
) -> dict[str, Any]:
    rows_with_market = 0
    if "anchor_trade_date" in merged_df.columns:
        rows_with_market = int(merged_df["anchor_trade_date"].notna().sum())

    return {
        "config": asdict(cfg),
        "rows": {
            "main_dataset": int(len(main_df)),
            "market_feature_merge": int(len(market_df)),
            "merged_output": int(len(merged_df)),
        },
        "unique_tickers": {
            "main_dataset": int(main_df["ticker"].nunique()) if not main_df.empty else 0,
            "market_feature_merge": int(market_df["ticker"].nunique()) if not market_df.empty else 0,
            "merged_output": int(merged_df["ticker"].nunique()) if not merged_df.empty else 0,
        },
        "market_columns_added": new_cols,
        "market_columns_filled_without_overwrite": overlap_cols,
        "rows_with_market_match": rows_with_market,
        "artifacts": {
            "main_dataset_path": str(cfg.main_dataset_path),
            "market_feature_merge_path": str(cfg.market_feature_merge_path),
            "out_path": str(cfg.out_path),
            "summary_path": str(cfg.summary_path),
        },
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--main-dataset", type=str, default=str(DEFAULT_MAIN_DATASET))
    ap.add_argument("--market-feature-merge", type=str, default=str(DEFAULT_MARKET_FEATURE_MERGE))
    ap.add_argument("--out-path", type=str, default=str(DEFAULT_OUT_PATH))
    ap.add_argument("--summary-path", type=str, default=str(DEFAULT_SUMMARY_PATH))
    ap.add_argument("--ticker-col", type=str, default="ticker")
    ap.add_argument("--period-end-col", type=str, default="period_end")
    args = ap.parse_args()

    cfg = MergeMarketFeatureIntoMainConfig(
        main_dataset_path=Path(args.main_dataset),
        market_feature_merge_path=Path(args.market_feature_merge),
        out_path=Path(args.out_path),
        summary_path=Path(args.summary_path),
        ticker_col=str(args.ticker_col),
        period_end_col=str(args.period_end_col),
    )

    ensure_dir(cfg.out_path.parent)
    ensure_dir(cfg.summary_path.parent)

    _log(f"[merge] Loading main dataset from {cfg.main_dataset_path}")
    main_df, _, _ = _load_csv_with_normalized_keys(
        cfg.main_dataset_path,
        cfg.ticker_col,
        cfg.period_end_col,
    )
    _log(f"[merge] Loading market feature merge dataset from {cfg.market_feature_merge_path}")
    market_df, _, _ = _load_csv_with_normalized_keys(
        cfg.market_feature_merge_path,
        cfg.ticker_col,
        cfg.period_end_col,
    )

    _log(
        f"[merge] Main rows={len(main_df)} | market rows={len(market_df)} | "
        f"main tickers={main_df['ticker'].nunique() if not main_df.empty else 0} | "
        f"market tickers={market_df['ticker'].nunique() if not market_df.empty else 0}"
    )
    _log("[merge] Merging market columns into main dataset without overwriting the source dataset")
    merged_df, new_cols, overlap_cols = _merge_market_into_main(main_df, market_df)

    _log(f"[merge] Writing merged dataset to {cfg.out_path}")
    merged_df.to_csv(cfg.out_path, index=False)

    summary = _build_summary(cfg, main_df, market_df, merged_df, new_cols, overlap_cols)
    _log(f"[merge] Writing summary to {cfg.summary_path}")
    write_json(cfg.summary_path, summary)

    print("Main dataset with market columns created:")
    print(f"rows={summary['rows']['merged_output']} | rows_with_market_match={summary['rows_with_market_match']}")
    print(f"dataset={cfg.out_path}")
    print(f"summary={cfg.summary_path}")


if __name__ == "__main__":
    main()
