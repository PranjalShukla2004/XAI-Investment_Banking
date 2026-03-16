from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import pandas as pd

from src.models.risk.dataset_utils import ensure_dir, normalize_tickers, write_json
from src.models.risk import build_future_drawdown_targets as _targets

PROJECT_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_MARKET_FEATURE_MERGE_PATH = PROJECT_ROOT / "data/processed/market/market_feature_merge.csv"
DEFAULT_MARKET_DATA_DIR = PROJECT_ROOT / "data/raw/market_data"
DEFAULT_SUMMARY_PATH = PROJECT_ROOT / "data/processed/market/market_feature_merge_drawdown_summary.json"

TARGET_COLUMN_MAP = {
    "anchor_trade_date": "future_anchor_trade_date",
    "horizon_end_date": "future_horizon_end_date",
    "horizon_last_trade_date": "future_horizon_last_trade_date",
    "future_bar_count": "future_bar_count",
    "future_1y_max_drawdown": "future_1y_max_drawdown",
    "future_1y_total_return": "future_1y_total_return",
    "mdd_peak_date": "future_mdd_peak_date",
    "mdd_trough_date": "future_mdd_trough_date",
    "mdd_peak_close": "future_mdd_peak_close",
    "mdd_trough_close": "future_mdd_trough_close",
    "target_status": "target_status",
    "target_usable": "target_usable",
    "target_issue_codes": "target_issue_codes",
}


def _log(message: str) -> None:
    print(message, flush=True)


@dataclass
class BuildFutureDrawdownConfig:
    merge_path: Path = DEFAULT_MARKET_FEATURE_MERGE_PATH
    market_data_dir: Path = DEFAULT_MARKET_DATA_DIR
    out_path: Path = DEFAULT_MARKET_FEATURE_MERGE_PATH
    summary_path: Path = DEFAULT_SUMMARY_PATH
    ticker_col: str = "ticker"
    period_end_col: str = "period_end"
    horizon_days: int = 365
    calendar_buffer_days: int = 7
    min_future_bars: int = 30
    max_tickers: int = 0


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


def _load_merge_dataset(cfg: BuildFutureDrawdownConfig) -> pd.DataFrame:
    if not cfg.merge_path.exists():
        raise FileNotFoundError(
            f"Market feature merge dataset not found: {cfg.merge_path}. "
            "Run src.data_fetch.market.build_market_feature_merge first."
        )

    _log(f"[drawdown] Loading market feature merge dataset from {cfg.merge_path}")
    df = pd.read_csv(cfg.merge_path)
    header = list(df.columns)
    ticker_col = _resolve_column(header, [cfg.ticker_col])
    period_end_col = _resolve_column(header, [cfg.period_end_col])

    df[ticker_col] = normalize_tickers(df[ticker_col])
    df[period_end_col] = pd.to_datetime(df[period_end_col], errors="coerce").dt.normalize()
    df = df.dropna(subset=[ticker_col, period_end_col]).copy()
    df = df.rename(columns={ticker_col: "ticker", period_end_col: "period_end"})
    df = df.sort_values(["ticker", "period_end"], kind="stable").reset_index(drop=True)

    if cfg.max_tickers and cfg.max_tickers > 0:
        keep = set(df["ticker"].drop_duplicates().head(int(cfg.max_tickers)))
        df = df.loc[df["ticker"].isin(keep)].copy().reset_index(drop=True)

    _log(
        f"[drawdown] Loaded {len(df)} rows across {df['ticker'].nunique() if not df.empty else 0} tickers"
    )
    return df


def _compute_target_rows(
    dataset: pd.DataFrame,
    cfg: BuildFutureDrawdownConfig,
) -> tuple[pd.DataFrame, str]:
    requested_tickers = dataset["ticker"].drop_duplicates().sort_values().tolist()
    market_layout = _targets._detect_market_layout(cfg.market_data_dir, requested_tickers)
    _log(
        f"[drawdown] Market layout detected: {market_layout} | tickers={len(requested_tickers)} | "
        f"market_data_dir={cfg.market_data_dir}"
    )
    flatfile_cache: dict[str, pd.DataFrame] = {}
    if market_layout == "daily_flatfiles":
        _log("[drawdown] Building daily flatfile cache")
        flatfile_cache = _targets._load_flatfile_market_cache(cfg.market_data_dir, set(requested_tickers))

    target_cfg = _targets.FutureDrawdownTargetConfig(
        market_data_dir=cfg.market_data_dir,
        out_path=cfg.out_path,
        summary_path=cfg.summary_path,
        horizon_days=int(cfg.horizon_days),
        calendar_buffer_days=int(cfg.calendar_buffer_days),
        min_future_bars=int(cfg.min_future_bars),
    )

    result_rows: list[dict[str, Any]] = []
    total_tickers = len(requested_tickers)
    for idx, ticker in enumerate(requested_tickers, start=1):
        if market_layout == "ticker_daily_json":
            bars = _targets._load_ticker_daily_json_bars(cfg.market_data_dir, ticker)
        elif market_layout == "ticker_files":
            bars = _targets._load_market_bars_legacy(cfg.market_data_dir / f"{ticker}.csv")
        elif market_layout == "daily_flatfiles":
            bars = flatfile_cache.get(ticker, pd.DataFrame(columns=["date", "close"]))
        else:
            bars = pd.DataFrame(columns=["date", "close"])

        group = dataset.loc[dataset["ticker"] == ticker].sort_values("period_end", kind="stable")
        for row in group.itertuples(index=False):
            target = _targets._compute_future_drawdown(pd.to_datetime(row.period_end), bars, target_cfg)
            renamed_target = {TARGET_COLUMN_MAP[key]: value for key, value in target.items()}
            result_rows.append(
                {
                    "ticker": str(row.ticker),
                    "period_end": pd.to_datetime(row.period_end).strftime("%Y-%m-%d"),
                    **renamed_target,
                }
            )

        if idx == 1 or idx % 100 == 0 or idx == total_tickers:
            _log(
                f"[drawdown] Computed targets for {idx}/{total_tickers} tickers | "
                f"latest={ticker} | accumulated_rows={len(result_rows)}"
            )

    targets_df = pd.DataFrame(result_rows)
    return targets_df, market_layout


def _merge_targets(dataset: pd.DataFrame, targets_df: pd.DataFrame) -> pd.DataFrame:
    out = dataset.copy()
    out["period_end"] = pd.to_datetime(out["period_end"], errors="coerce").dt.strftime("%Y-%m-%d")

    target_cols = list(TARGET_COLUMN_MAP.values())
    merged = out.merge(
        targets_df,
        on=["ticker", "period_end"],
        how="left",
        validate="one_to_one",
        suffixes=("", "__new"),
    )

    for col in target_cols:
        new_col = f"{col}__new"
        if new_col not in merged.columns:
            continue
        if col in out.columns:
            merged[col] = merged[col].combine_first(merged[new_col])
        else:
            merged[col] = merged[new_col]
        merged = merged.drop(columns=[new_col])

    return merged


def _build_summary(
    cfg: BuildFutureDrawdownConfig,
    merged: pd.DataFrame,
    market_layout: str,
) -> dict[str, Any]:
    status_counts = {}
    if "target_status" in merged.columns:
        status_counts = {
            str(k): int(v)
            for k, v in merged["target_status"].value_counts(dropna=False).to_dict().items()
        }

    return {
        "config": asdict(cfg),
        "rows": int(len(merged)),
        "unique_tickers": int(merged["ticker"].nunique()) if not merged.empty else 0,
        "market_layout": market_layout,
        "status_counts": status_counts,
        "usable_targets": int(merged["target_usable"].fillna(False).astype(bool).sum())
        if "target_usable" in merged.columns
        else 0,
        "artifacts": {
            "merge_path": str(cfg.merge_path),
            "out_path": str(cfg.out_path),
            "summary_path": str(cfg.summary_path),
            "market_data_dir": str(cfg.market_data_dir),
        },
        "target_columns": list(TARGET_COLUMN_MAP.values()),
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--merge-path", type=str, default=str(DEFAULT_MARKET_FEATURE_MERGE_PATH))
    ap.add_argument("--market-data-dir", type=str, default=str(DEFAULT_MARKET_DATA_DIR))
    ap.add_argument("--out-path", type=str, default=str(DEFAULT_MARKET_FEATURE_MERGE_PATH))
    ap.add_argument("--summary-path", type=str, default=str(DEFAULT_SUMMARY_PATH))
    ap.add_argument("--ticker-col", type=str, default="ticker")
    ap.add_argument("--period-end-col", type=str, default="period_end")
    ap.add_argument("--horizon-days", type=int, default=365)
    ap.add_argument("--calendar-buffer-days", type=int, default=7)
    ap.add_argument("--min-future-bars", type=int, default=30)
    ap.add_argument("--max-tickers", type=int, default=0)
    args = ap.parse_args()

    cfg = BuildFutureDrawdownConfig(
        merge_path=Path(args.merge_path),
        market_data_dir=Path(args.market_data_dir),
        out_path=Path(args.out_path),
        summary_path=Path(args.summary_path),
        ticker_col=str(args.ticker_col),
        period_end_col=str(args.period_end_col),
        horizon_days=int(args.horizon_days),
        calendar_buffer_days=int(args.calendar_buffer_days),
        min_future_bars=int(args.min_future_bars),
        max_tickers=int(args.max_tickers),
    )

    ensure_dir(cfg.out_path.parent)
    ensure_dir(cfg.summary_path.parent)

    dataset = _load_merge_dataset(cfg)
    _log("[drawdown] Computing forward 1-year drawdown targets")
    targets_df, market_layout = _compute_target_rows(dataset, cfg)
    _log("[drawdown] Merging target columns into market feature dataset")
    merged = _merge_targets(dataset, targets_df)
    _log(f"[drawdown] Writing merged dataset to {cfg.out_path}")
    merged.to_csv(cfg.out_path, index=False)

    summary = _build_summary(cfg, merged, market_layout)
    _log(f"[drawdown] Writing summary to {cfg.summary_path}")
    write_json(cfg.summary_path, summary)

    print("Future drawdown labels merged into market feature dataset:")
    print(f"rows={summary['rows']} | tickers={summary['unique_tickers']} | layout={market_layout}")
    print(f"usable_targets={summary['usable_targets']}")
    print(f"dataset={cfg.out_path}")
    print(f"summary={cfg.summary_path}")


if __name__ == "__main__":
    main()
