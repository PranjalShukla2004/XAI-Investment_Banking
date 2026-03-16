from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from src.models.risk.dataset_utils import ensure_dir, normalize_tickers, write_json

PROJECT_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_MAIN_DATASET = PROJECT_ROOT / "data/processed/main_dataset.csv"
DEFAULT_MARKET_DATA_DIR = PROJECT_ROOT / "data/raw/market_data"
DEFAULT_OUT_PATH = PROJECT_ROOT / "data/processed/market/market_feature_merge.csv"
DEFAULT_SUMMARY_PATH = PROJECT_ROOT / "data/processed/market/market_feature_merge_summary.json"

FEATURE_COLUMNS = [
    "volatility_21d",
    "volatility_63d",
    "drawdown_126d",
    "return_63d",
    "avg_dollar_volume_63d",
    "avg_transactions_21d",
    "high_low_range_21d",
]


@dataclass
class MarketFeatureMergeConfig:
    main_dataset_path: Path = DEFAULT_MAIN_DATASET
    market_data_dir: Path = DEFAULT_MARKET_DATA_DIR
    out_path: Path = DEFAULT_OUT_PATH
    summary_path: Path = DEFAULT_SUMMARY_PATH
    ticker_col: str = "ticker"
    period_end_col: str = "period_end"
    fiscal_year_col: str = "fiscal_year"
    timeframe_col: str = "timeframe"
    timeframe: str | None = "annual"
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


def _load_observations(cfg: MarketFeatureMergeConfig) -> pd.DataFrame:
    if not cfg.main_dataset_path.exists():
        raise FileNotFoundError(f"Main dataset not found: {cfg.main_dataset_path}")

    header = list(pd.read_csv(cfg.main_dataset_path, nrows=0).columns)
    ticker_col = _resolve_column(header, [cfg.ticker_col])
    period_end_col = _resolve_column(header, [cfg.period_end_col])

    usecols = [ticker_col, period_end_col]
    fiscal_year_col = _resolve_column(header, [cfg.fiscal_year_col], required=False)
    timeframe_col = _resolve_column(header, [cfg.timeframe_col], required=False)
    if fiscal_year_col and fiscal_year_col not in usecols:
        usecols.append(fiscal_year_col)
    if timeframe_col and timeframe_col not in usecols:
        usecols.append(timeframe_col)

    raw = pd.read_csv(cfg.main_dataset_path, usecols=usecols)
    obs = pd.DataFrame(
        {
            "ticker": normalize_tickers(raw[ticker_col]),
            "period_end": pd.to_datetime(raw[period_end_col], errors="coerce").dt.normalize().astype("datetime64[ns]"),
        }
    )
    if fiscal_year_col and fiscal_year_col in raw.columns:
        obs["fiscal_year"] = raw[fiscal_year_col]
    if timeframe_col and timeframe_col in raw.columns:
        obs["timeframe"] = raw[timeframe_col].astype(str).str.strip()

    obs = obs.replace({"ticker": {"": np.nan, "NAN": np.nan, "NONE": np.nan}})
    if cfg.timeframe and "timeframe" in obs.columns:
        obs = obs.loc[obs["timeframe"].str.lower() == str(cfg.timeframe).lower()].copy()

    obs = obs.dropna(subset=["ticker", "period_end"]).copy()
    sort_cols = ["ticker", "period_end"]
    if "fiscal_year" in obs.columns:
        sort_cols.append("fiscal_year")
    obs = obs.sort_values(sort_cols, kind="stable")
    obs = obs.drop_duplicates(subset=["ticker", "period_end"], keep="last").reset_index(drop=True)

    if cfg.max_tickers and cfg.max_tickers > 0:
        keep = set(obs["ticker"].drop_duplicates().head(int(cfg.max_tickers)))
        obs = obs.loc[obs["ticker"].isin(keep)].copy().reset_index(drop=True)

    return obs


def _load_market_panel(market_data_dir: Path, requested_tickers: set[str]) -> pd.DataFrame:
    files = sorted(
        path
        for path in market_data_dir.rglob("*")
        if path.is_file() and _is_daily_market_file(path)
    )
    frames: list[pd.DataFrame] = []

    for path in files:
        header = list(pd.read_csv(path, compression="infer", nrows=0).columns)
        ticker_col = _resolve_column(header, ["ticker", "symbol", "sym_root", "sym"])
        close_col = _resolve_column(header, ["close", "c"])
        high_col = _resolve_column(header, ["high", "h"])
        low_col = _resolve_column(header, ["low", "l"])
        volume_col = _resolve_column(header, ["volume", "v"], required=False)
        transactions_col = _resolve_column(header, ["transactions", "n"], required=False)
        date_col = _resolve_column(header, ["date", "window_start", "timestamp", "t"], required=False)

        usecols = [ticker_col, close_col, high_col, low_col]
        for col in [volume_col, transactions_col, date_col]:
            if col and col not in usecols:
                usecols.append(col)

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
        filtered["high"] = pd.to_numeric(filtered[high_col], errors="coerce")
        filtered["low"] = pd.to_numeric(filtered[low_col], errors="coerce")
        filtered["volume"] = (
            pd.to_numeric(filtered[volume_col], errors="coerce") if volume_col else np.nan
        )
        filtered["transactions"] = (
            pd.to_numeric(filtered[transactions_col], errors="coerce") if transactions_col else np.nan
        )
        if date_col:
            filtered["date"] = _coerce_trade_dates(filtered[date_col]).astype("datetime64[ns]")
        else:
            trade_date = _extract_date_from_filename(path)
            filtered["date"] = pd.Series(trade_date, index=filtered.index, dtype="datetime64[ns]")

        filtered = filtered.loc[
            :,
            ["ticker", "date", "close", "high", "low", "volume", "transactions"],
        ].dropna(subset=["ticker", "date", "close"])
        if filtered.empty:
            continue

        frames.append(filtered)

    if not frames:
        return pd.DataFrame(columns=["ticker", "date", "close", "high", "low", "volume", "transactions"])

    panel = pd.concat(frames, ignore_index=True)
    panel = panel.sort_values(["ticker", "date"], kind="stable")
    panel = panel.drop_duplicates(subset=["ticker", "date"], keep="last")
    return panel.reset_index(drop=True)


def _compute_market_features_for_ticker(group: pd.DataFrame) -> pd.DataFrame:
    g = group.sort_values("date", kind="stable").copy()
    close = pd.to_numeric(g["close"], errors="coerce")
    high = pd.to_numeric(g["high"], errors="coerce")
    low = pd.to_numeric(g["low"], errors="coerce")
    volume = pd.to_numeric(g["volume"], errors="coerce")
    transactions = pd.to_numeric(g["transactions"], errors="coerce")

    log_close = np.log(close.where(close > 0.0))
    log_return = log_close.diff()
    close_safe = close.replace(0.0, np.nan)

    g["volatility_21d"] = log_return.rolling(21, min_periods=21).std() * np.sqrt(252.0)
    g["volatility_63d"] = log_return.rolling(63, min_periods=63).std() * np.sqrt(252.0)
    g["drawdown_126d"] = (close / close.rolling(126, min_periods=126).max()) - 1.0
    g["return_63d"] = close.pct_change(periods=63)
    g["avg_dollar_volume_63d"] = (close * volume).rolling(63, min_periods=63).mean()
    g["avg_transactions_21d"] = transactions.rolling(21, min_periods=21).mean()
    g["high_low_range_21d"] = ((high - low) / close_safe).rolling(21, min_periods=21).mean()

    return g.loc[:, ["ticker", "date", *FEATURE_COLUMNS]].reset_index(drop=True)


def _compute_market_feature_panel(market_panel: pd.DataFrame) -> pd.DataFrame:
    if market_panel.empty:
        return pd.DataFrame(columns=["ticker", "date", *FEATURE_COLUMNS])

    feature_frames = [
        _compute_market_features_for_ticker(group)
        for _, group in market_panel.groupby("ticker", sort=True)
    ]
    feature_panel = pd.concat(feature_frames, ignore_index=True)
    feature_panel = feature_panel.sort_values(["ticker", "date"], kind="stable")
    return feature_panel.reset_index(drop=True)


def _merge_features(observations: pd.DataFrame, feature_panel: pd.DataFrame) -> pd.DataFrame:
    if observations.empty:
        cols = list(observations.columns) + ["anchor_trade_date", *FEATURE_COLUMNS]
        return pd.DataFrame(columns=cols)

    obs = observations.sort_values(["ticker", "period_end"], kind="stable").reset_index(drop=True)
    feats = feature_panel.sort_values(["ticker", "date"], kind="stable").reset_index(drop=True)
    if feats.empty:
        merged = obs.copy()
        merged["anchor_trade_date"] = pd.NaT
        for col in FEATURE_COLUMNS:
            merged[col] = np.nan
        return merged

    feature_groups = {
        str(ticker): group.loc[:, ["date", *FEATURE_COLUMNS]].sort_values("date", kind="stable").reset_index(drop=True)
        for ticker, group in feats.groupby("ticker", sort=False)
    }
    merged_groups: list[pd.DataFrame] = []

    for ticker, obs_group in obs.groupby("ticker", sort=False):
        obs_group_sorted = obs_group.sort_values("period_end", kind="stable").reset_index(drop=True)
        feat_group = feature_groups.get(str(ticker))
        if feat_group is None or feat_group.empty:
            out = obs_group_sorted.copy()
            out["anchor_trade_date"] = pd.NaT
            for col in FEATURE_COLUMNS:
                out[col] = np.nan
            merged_groups.append(out)
            continue

        joined = pd.merge_asof(
            obs_group_sorted,
            feat_group,
            left_on="period_end",
            right_on="date",
            direction="backward",
            allow_exact_matches=True,
        ).rename(columns={"date": "anchor_trade_date"})
        merged_groups.append(joined)

    merged = pd.concat(merged_groups, ignore_index=True)
    merged = merged.sort_values(["ticker", "period_end"], kind="stable").reset_index(drop=True)
    return merged


def _build_summary(
    cfg: MarketFeatureMergeConfig,
    observations: pd.DataFrame,
    market_panel: pd.DataFrame,
    merged: pd.DataFrame,
) -> dict[str, Any]:
    market_min = None
    market_max = None
    if not market_panel.empty:
        market_min = market_panel["date"].min()
        market_max = market_panel["date"].max()

    return {
        "config": asdict(cfg),
        "rows": int(len(merged)),
        "unique_tickers": int(merged["ticker"].nunique()) if not merged.empty else 0,
        "observation_range": {
            "min_period_end": observations["period_end"].min().strftime("%Y-%m-%d") if not observations.empty else None,
            "max_period_end": observations["period_end"].max().strftime("%Y-%m-%d") if not observations.empty else None,
        },
        "market_range": {
            "min_trade_date": market_min.strftime("%Y-%m-%d") if market_min is not None else None,
            "max_trade_date": market_max.strftime("%Y-%m-%d") if market_max is not None else None,
        },
        "coverage": {
            "anchor_trade_date_non_null": int(merged["anchor_trade_date"].notna().sum()) if "anchor_trade_date" in merged.columns else 0,
            **{f"{col}_non_null": int(merged[col].notna().sum()) for col in FEATURE_COLUMNS if col in merged.columns},
        },
        "feature_definitions": {
            "volatility_21d": "Annualized standard deviation of daily log returns over the trailing 21 trading days.",
            "volatility_63d": "Annualized standard deviation of daily log returns over the trailing 63 trading days.",
            "drawdown_126d": "Current drawdown versus the trailing 126-trading-day rolling peak, computed as close / rolling_max(close, 126) - 1.",
            "return_63d": "Trailing 63-trading-day close-to-close return, computed as close_t / close_t-63 - 1.",
            "avg_dollar_volume_63d": "Average trailing 63-trading-day dollar volume, computed as mean(close * volume).",
            "avg_transactions_21d": "Average trailing 21-trading-day transaction count.",
            "high_low_range_21d": "Average trailing 21-trading-day intraday range, computed as mean((high - low) / close).",
        },
        "artifacts": {
            "out_path": str(cfg.out_path),
            "summary_path": str(cfg.summary_path),
            "main_dataset_path": str(cfg.main_dataset_path),
            "market_data_dir": str(cfg.market_data_dir),
        },
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--main-dataset", type=str, default=str(DEFAULT_MAIN_DATASET))
    ap.add_argument("--market-data-dir", type=str, default=str(DEFAULT_MARKET_DATA_DIR))
    ap.add_argument("--out-path", type=str, default=str(DEFAULT_OUT_PATH))
    ap.add_argument("--summary-path", type=str, default=str(DEFAULT_SUMMARY_PATH))
    ap.add_argument("--ticker-col", type=str, default="ticker")
    ap.add_argument("--period-end-col", type=str, default="period_end")
    ap.add_argument("--fiscal-year-col", type=str, default="fiscal_year")
    ap.add_argument("--timeframe-col", type=str, default="timeframe")
    ap.add_argument("--timeframe", type=str, default="annual")
    ap.add_argument("--max-tickers", type=int, default=0)
    args = ap.parse_args()

    cfg = MarketFeatureMergeConfig(
        main_dataset_path=Path(args.main_dataset),
        market_data_dir=Path(args.market_data_dir),
        out_path=Path(args.out_path),
        summary_path=Path(args.summary_path),
        ticker_col=str(args.ticker_col),
        period_end_col=str(args.period_end_col),
        fiscal_year_col=str(args.fiscal_year_col),
        timeframe_col=str(args.timeframe_col),
        timeframe=str(args.timeframe).strip() or None,
        max_tickers=int(args.max_tickers),
    )

    ensure_dir(cfg.out_path.parent)
    ensure_dir(cfg.summary_path.parent)

    observations = _load_observations(cfg)
    requested_tickers = set(observations["ticker"].dropna().astype(str))
    market_panel = _load_market_panel(cfg.market_data_dir, requested_tickers)
    feature_panel = _compute_market_feature_panel(market_panel)
    merged = _merge_features(observations, feature_panel)

    out_cols = [
        col
        for col in ["ticker", "period_end", "fiscal_year", "timeframe", "anchor_trade_date", *FEATURE_COLUMNS]
        if col in merged.columns
    ]
    merged.loc[:, out_cols].to_csv(cfg.out_path, index=False)

    summary = _build_summary(cfg, observations, market_panel, merged)
    write_json(cfg.summary_path, summary)

    print("Saved market feature merge artifacts:")
    print(f"- {cfg.out_path}")
    print(f"- {cfg.summary_path}")
    print(
        "Coverage:"
        + "".join(
            f" {col}={summary['coverage'].get(f'{col}_non_null', 0)}"
            for col in FEATURE_COLUMNS
        )
    )


if __name__ == "__main__":
    main()
