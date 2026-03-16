from __future__ import annotations

import argparse
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from src.models.risk.dataset_utils import (
    DatasetObservationConfig,
    ensure_dir,
    load_observation_universe,
    normalize_tickers,
    parse_dataset_paths,
    write_json,
)

DEFAULT_MARKET_DATA_DIR = Path("data/raw/market_data")
DEFAULT_OUT_PATH = Path("data/processed/risk/future_drawdown_targets.csv")
DEFAULT_SUMMARY_PATH = Path("data/processed/risk/future_drawdown_target_summary.json")


def _log(message: str) -> None:
    print(message, flush=True)


@dataclass
class FutureDrawdownTargetConfig:
    market_data_dir: Path = DEFAULT_MARKET_DATA_DIR
    out_path: Path = DEFAULT_OUT_PATH
    summary_path: Path = DEFAULT_SUMMARY_PATH
    horizon_days: int = 365
    calendar_buffer_days: int = 7
    min_future_bars: int = 30


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


def _ticker_path_component(ticker: str) -> str:
    return str(ticker).strip().upper().replace("/", "__SLASH__")


def _coerce_trade_dates(series: pd.Series) -> pd.Series:
    dt = pd.to_datetime(series, errors="coerce", utc=True)
    if dt.notna().any():
        return dt.dt.tz_localize(None).dt.normalize()

    numeric = pd.to_numeric(series, errors="coerce")
    if not numeric.notna().any():
        return pd.Series(pd.NaT, index=series.index, dtype="datetime64[ns]")

    max_abs = float(numeric.abs().max())
    if max_abs >= 1e17:
        unit = "ns"
    elif max_abs >= 1e14:
        unit = "us"
    elif max_abs >= 1e11:
        unit = "ms"
    else:
        unit = "s"
    return pd.to_datetime(numeric, unit=unit, errors="coerce", utc=True).dt.tz_localize(None).dt.normalize()


def _load_market_bars_legacy(csv_path: Path) -> pd.DataFrame:
    if not csv_path.exists():
        return pd.DataFrame(columns=["date", "close"])

    bars = pd.read_csv(csv_path, usecols=lambda c: str(c).lower() in {"date", "close"})
    if bars.empty or "date" not in bars.columns or "close" not in bars.columns:
        return pd.DataFrame(columns=["date", "close"])

    out = pd.DataFrame(
        {
            "date": pd.to_datetime(bars["date"], errors="coerce").dt.normalize(),
            "close": pd.to_numeric(bars["close"], errors="coerce"),
        }
    ).dropna(subset=["date", "close"])
    if out.empty:
        return pd.DataFrame(columns=["date", "close"])

    out = out.sort_values("date", kind="stable").drop_duplicates(subset=["date"], keep="last")
    return out.reset_index(drop=True)


def _read_json_payload(path: Path) -> dict[str, Any] | None:
    try:
        import json

        payload = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return None
    return payload if isinstance(payload, dict) else None


def _date_from_ticker_json_path(path: Path) -> pd.Timestamp | None:
    try:
        month_part = path.parent.name
        day_part = path.stem
        return pd.Timestamp(f"{month_part}-{day_part}").normalize()
    except Exception:
        return None


def _load_ticker_daily_json_bars(market_data_dir: Path, ticker: str) -> pd.DataFrame:
    ticker_dir = market_data_dir / _ticker_path_component(ticker)
    if not ticker_dir.exists():
        return pd.DataFrame(columns=["date", "close"])

    rows: list[dict[str, Any]] = []
    for path in sorted(ticker_dir.glob("*/*.json")):
        payload = _read_json_payload(path)
        if not payload:
            continue

        close_val = payload.get("close", payload.get("c"))
        close = pd.to_numeric(pd.Series([close_val]), errors="coerce").iloc[0]
        if pd.isna(close):
            continue

        date_val = payload.get("from", payload.get("date"))
        if date_val:
            date = pd.to_datetime(date_val, errors="coerce")
        else:
            date = _date_from_ticker_json_path(path)
        if pd.isna(date):
            timestamp_val = payload.get("t", payload.get("window_start"))
            if timestamp_val is not None:
                date = _coerce_trade_dates(pd.Series([timestamp_val])).iloc[0]
        if pd.isna(date):
            date = _date_from_ticker_json_path(path)
        if pd.isna(date):
            continue

        rows.append({"date": pd.Timestamp(date).normalize(), "close": float(close)})

    if not rows:
        return pd.DataFrame(columns=["date", "close"])

    bars = pd.DataFrame(rows)
    bars = bars.sort_values("date", kind="stable").drop_duplicates(subset=["date"], keep="last")
    return bars.reset_index(drop=True)


def _extract_date_from_filename(path: Path) -> pd.Timestamp | None:
    name = path.name
    if name.endswith(".csv.gz"):
        name = name[:-7]
    elif name.endswith(".csv"):
        name = name[:-4]
    try:
        return pd.Timestamp(name).normalize()
    except ValueError:
        return None


def _is_daily_market_file(path: Path) -> bool:
    return _extract_date_from_filename(path) is not None


def _load_flatfile_market_cache(market_data_dir: Path, requested_tickers: set[str]) -> dict[str, pd.DataFrame]:
    files = sorted(
        path
        for path in market_data_dir.rglob("*")
        if path.is_file() and _is_daily_market_file(path)
    )
    started_at = time.monotonic()
    _log(
        f"[market-cache] Scanning {len(files)} daily flatfiles for {len(requested_tickers)} requested tickers "
        f"from {market_data_dir}"
    )
    cache_rows: dict[str, list[pd.DataFrame]] = {ticker: [] for ticker in requested_tickers}

    for idx, path in enumerate(files, start=1):
        header = list(pd.read_csv(path, compression="infer", nrows=0).columns)
        ticker_col = _resolve_column(header, ["ticker", "symbol", "sym_root", "sym"])
        close_col = _resolve_column(header, ["close", "c"])
        date_col = _resolve_column(header, ["date", "window_start", "timestamp", "t"], required=False)

        usecols = [ticker_col, close_col]
        if date_col and date_col not in usecols:
            usecols.append(date_col)

        df = pd.read_csv(path, compression="infer", usecols=usecols)
        if df.empty:
            continue

        tickers = normalize_tickers(df[ticker_col])
        mask = tickers.isin(requested_tickers)
        if not bool(mask.any()):
            continue

        filtered = df.loc[mask].copy()
        filtered["ticker"] = tickers.loc[mask].to_numpy()
        filtered["close"] = pd.to_numeric(filtered[close_col], errors="coerce")
        if date_col:
            filtered["date"] = _coerce_trade_dates(filtered[date_col])
        else:
            trade_date = _extract_date_from_filename(path)
            filtered["date"] = trade_date

        filtered = filtered.loc[:, ["ticker", "date", "close"]].dropna(subset=["ticker", "date", "close"])
        if filtered.empty:
            continue

        for ticker, group in filtered.groupby("ticker", sort=False):
            cache_rows.setdefault(str(ticker), []).append(group.loc[:, ["date", "close"]].copy())

        if idx == 1 or idx % 10 == 0 or idx == len(files):
            elapsed = time.monotonic() - started_at
            _log(
                f"[market-cache] Processed {idx}/{len(files)} files | latest={path.name} | "
                f"matched_rows={len(filtered)} | elapsed={elapsed:.1f}s"
            )

    cache: dict[str, pd.DataFrame] = {}
    ticker_items = list(cache_rows.items())
    _log(f"[market-cache] Consolidating matched rows into per-ticker price histories for {len(ticker_items)} tickers")
    for idx, (ticker, frames) in enumerate(ticker_items, start=1):
        if not frames:
            continue
        bars = pd.concat(frames, ignore_index=True)
        bars = bars.sort_values("date", kind="stable").drop_duplicates(subset=["date"], keep="last")
        cache[ticker] = bars.reset_index(drop=True)
        if idx == 1 or idx % 250 == 0 or idx == len(ticker_items):
            elapsed = time.monotonic() - started_at
            _log(
                f"[market-cache] Consolidated {idx}/{len(ticker_items)} tickers | "
                f"latest={ticker} | bars={len(cache[ticker])} | elapsed={elapsed:.1f}s"
            )
    elapsed = time.monotonic() - started_at
    _log(f"[market-cache] Built ticker cache for {len(cache)} tickers | elapsed={elapsed:.1f}s")
    return cache


def _detect_market_layout(market_data_dir: Path, tickers: list[str]) -> str:
    if any(_is_daily_market_file(path) for path in market_data_dir.rglob("*") if path.is_file()):
        return "daily_flatfiles"
    for ticker in tickers[:25]:
        ticker_dir = market_data_dir / _ticker_path_component(ticker)
        if ticker_dir.exists() and any(ticker_dir.glob("*/*.json")):
            return "ticker_daily_json"
    for ticker in tickers[:25]:
        if (market_data_dir / f"{ticker}.csv").exists():
            return "ticker_files"
    return "missing"


def _compute_future_drawdown(
    period_end: pd.Timestamp,
    bars: pd.DataFrame,
    cfg: FutureDrawdownTargetConfig,
) -> dict[str, Any]:
    horizon_end = period_end + pd.Timedelta(days=int(cfg.horizon_days))
    if bars.empty:
        return {
            "anchor_trade_date": None,
            "horizon_end_date": horizon_end.strftime("%Y-%m-%d"),
            "horizon_last_trade_date": None,
            "future_bar_count": 0,
            "future_1y_max_drawdown": np.nan,
            "future_1y_total_return": np.nan,
            "mdd_peak_date": None,
            "mdd_trough_date": None,
            "mdd_peak_close": np.nan,
            "mdd_trough_close": np.nan,
            "target_status": "missing_market_data",
            "target_usable": False,
            "target_issue_codes": "missing_market_data",
        }

    eligible = bars.loc[bars["date"] >= period_end].copy()
    if eligible.empty:
        return {
            "anchor_trade_date": None,
            "horizon_end_date": horizon_end.strftime("%Y-%m-%d"),
            "horizon_last_trade_date": None,
            "future_bar_count": 0,
            "future_1y_max_drawdown": np.nan,
            "future_1y_total_return": np.nan,
            "mdd_peak_date": None,
            "mdd_trough_date": None,
            "mdd_peak_close": np.nan,
            "mdd_trough_close": np.nan,
            "target_status": "no_forward_bars",
            "target_usable": False,
            "target_issue_codes": "no_forward_bars",
        }

    window = eligible.loc[eligible["date"] <= horizon_end].copy()
    if window.empty:
        return {
            "anchor_trade_date": None,
            "horizon_end_date": horizon_end.strftime("%Y-%m-%d"),
            "horizon_last_trade_date": None,
            "future_bar_count": 0,
            "future_1y_max_drawdown": np.nan,
            "future_1y_total_return": np.nan,
            "mdd_peak_date": None,
            "mdd_trough_date": None,
            "mdd_peak_close": np.nan,
            "mdd_trough_close": np.nan,
            "target_status": "empty_horizon_window",
            "target_usable": False,
            "target_issue_codes": "empty_horizon_window",
        }

    prices = window["close"].to_numpy(dtype=np.float64)
    running_peak = np.maximum.accumulate(prices)
    drawdowns = prices / np.maximum(running_peak, 1e-12) - 1.0

    trough_idx = int(np.argmin(drawdowns))
    peak_idx = int(np.argmax(prices[: trough_idx + 1]))

    issue_codes: list[str] = []
    anchor_date = pd.to_datetime(window.iloc[0]["date"])
    last_trade_date = pd.to_datetime(window.iloc[-1]["date"])

    if anchor_date > period_end + pd.Timedelta(days=int(cfg.calendar_buffer_days)):
        issue_codes.append("anchor_after_period_end")
    if last_trade_date < horizon_end - pd.Timedelta(days=int(cfg.calendar_buffer_days)):
        issue_codes.append("insufficient_horizon_coverage")
    if len(window) < int(cfg.min_future_bars):
        issue_codes.append("too_few_future_bars")

    target_status = "ok" if not issue_codes else "partial_horizon"

    return {
        "anchor_trade_date": anchor_date.strftime("%Y-%m-%d"),
        "horizon_end_date": horizon_end.strftime("%Y-%m-%d"),
        "horizon_last_trade_date": last_trade_date.strftime("%Y-%m-%d"),
        "future_bar_count": int(len(window)),
        "future_1y_max_drawdown": float(np.min(drawdowns)),
        "future_1y_total_return": float((prices[-1] / max(prices[0], 1e-12)) - 1.0),
        "mdd_peak_date": pd.to_datetime(window.iloc[peak_idx]["date"]).strftime("%Y-%m-%d"),
        "mdd_trough_date": pd.to_datetime(window.iloc[trough_idx]["date"]).strftime("%Y-%m-%d"),
        "mdd_peak_close": float(prices[peak_idx]),
        "mdd_trough_close": float(prices[trough_idx]),
        "target_status": target_status,
        "target_usable": bool(target_status == "ok"),
        "target_issue_codes": ";".join(issue_codes),
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--datasets", nargs="*", default=None)
    ap.add_argument("--ticker-col", type=str, default="ticker")
    ap.add_argument("--period-end-col", type=str, default="period_end")
    ap.add_argument("--timeframe-col", type=str, default="timeframe")
    ap.add_argument("--timeframe", type=str, default="annual")
    ap.add_argument("--market-data-dir", type=str, default=str(DEFAULT_MARKET_DATA_DIR))
    ap.add_argument("--out-path", type=str, default=str(DEFAULT_OUT_PATH))
    ap.add_argument("--summary-path", type=str, default=str(DEFAULT_SUMMARY_PATH))
    ap.add_argument("--horizon-days", type=int, default=365)
    ap.add_argument("--calendar-buffer-days", type=int, default=7)
    ap.add_argument("--min-future-bars", type=int, default=30)
    ap.add_argument("--max-tickers", type=int, default=0)
    args = ap.parse_args()

    dataset_paths = parse_dataset_paths(args.datasets)
    obs_cfg = DatasetObservationConfig(
        ticker_col=args.ticker_col,
        period_end_col=args.period_end_col,
        timeframe_col=args.timeframe_col,
        timeframe=args.timeframe or None,
    )
    observations, dataset_summaries = load_observation_universe(dataset_paths, obs_cfg)
    if observations.empty:
        raise SystemExit("No observations were loaded from the provided datasets.")

    if args.max_tickers and args.max_tickers > 0:
        selected = set(observations["ticker"].drop_duplicates().head(int(args.max_tickers)))
        observations = observations.loc[observations["ticker"].isin(selected)].copy()

    cfg = FutureDrawdownTargetConfig(
        market_data_dir=Path(args.market_data_dir),
        out_path=Path(args.out_path),
        summary_path=Path(args.summary_path),
        horizon_days=int(args.horizon_days),
        calendar_buffer_days=int(args.calendar_buffer_days),
        min_future_bars=int(args.min_future_bars),
    )
    ensure_dir(cfg.out_path.parent)
    ensure_dir(cfg.summary_path.parent)

    requested_tickers = observations["ticker"].drop_duplicates().sort_values().tolist()
    market_layout = _detect_market_layout(cfg.market_data_dir, requested_tickers)
    flatfile_cache: dict[str, pd.DataFrame] = {}
    if market_layout == "daily_flatfiles":
        flatfile_cache = _load_flatfile_market_cache(cfg.market_data_dir, set(requested_tickers))

    result_rows: list[dict[str, Any]] = []
    for ticker in requested_tickers:
        if market_layout == "ticker_daily_json":
            bars = _load_ticker_daily_json_bars(cfg.market_data_dir, ticker)
        elif market_layout == "ticker_files":
            bars = _load_market_bars_legacy(cfg.market_data_dir / f"{ticker}.csv")
        elif market_layout == "daily_flatfiles":
            bars = flatfile_cache.get(ticker, pd.DataFrame(columns=["date", "close"]))
        else:
            bars = pd.DataFrame(columns=["date", "close"])

        group = observations.loc[observations["ticker"] == ticker].sort_values(
            ["period_end", "source_dataset"],
            kind="stable",
        )
        for row in group.itertuples(index=False):
            target = _compute_future_drawdown(pd.to_datetime(row.period_end), bars, cfg)
            result_rows.append(
                {
                    "source_dataset": row.source_dataset,
                    "ticker": row.ticker,
                    "period_end": pd.to_datetime(row.period_end).strftime("%Y-%m-%d"),
                    **target,
                }
            )

    targets_df = pd.DataFrame(result_rows).sort_values(
        ["target_status", "ticker", "period_end", "source_dataset"],
        kind="stable",
    )
    targets_df.to_csv(cfg.out_path, index=False)

    summary = {
        "config": asdict(cfg),
        "datasets": dataset_summaries,
        "rows": int(len(targets_df)),
        "unique_tickers": int(targets_df["ticker"].nunique()) if not targets_df.empty else 0,
        "market_layout": market_layout,
        "status_counts": {
            str(k): int(v) for k, v in targets_df["target_status"].value_counts(dropna=False).to_dict().items()
        },
        "usable_targets": int(targets_df["target_usable"].fillna(False).astype(bool).sum())
        if not targets_df.empty
        else 0,
        "artifacts": {
            "out_path": str(cfg.out_path),
            "summary_path": str(cfg.summary_path),
            "market_data_dir": str(cfg.market_data_dir),
        },
    }
    write_json(cfg.summary_path, summary)

    print("Future drawdown target build complete:")
    print(f"rows={summary['rows']} | tickers={summary['unique_tickers']} | layout={market_layout}")
    print(f"usable_targets={summary['usable_targets']}")
    print(f"targets={cfg.out_path}")
    print(f"summary={cfg.summary_path}")


if __name__ == "__main__":
    main()
