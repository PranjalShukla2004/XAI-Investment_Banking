from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Sequence

import numpy as np
import pandas as pd

DEFAULT_RISK_DATASETS: tuple[Path, ...] = (
    Path("data/processed/main_dataset.csv"),
    Path("data/processed/test_dataset_out_of_time.csv"),
    Path("data/processed/test_dataset_unseen_tickers.csv"),
)


@dataclass(frozen=True)
class DatasetObservationConfig:
    ticker_col: str = "ticker"
    period_end_col: str = "period_end"
    timeframe_col: str = "timeframe"
    timeframe: str | None = "annual"


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def jsonable(value):
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, (np.integer, np.floating)):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, pd.Timestamp):
        return value.strftime("%Y-%m-%d")
    return value


def write_json(path: Path, payload: object) -> None:
    ensure_dir(path.parent)
    path.write_text(
        json.dumps(payload, default=jsonable, indent=2, sort_keys=True),
        encoding="utf-8",
    )


def normalize_tickers(series: pd.Series) -> pd.Series:
    return series.astype(str).str.upper().str.strip()


def parse_dataset_paths(values: Sequence[str] | None) -> list[Path]:
    if values:
        return [Path(v) for v in values]
    return [Path(p) for p in DEFAULT_RISK_DATASETS]


def _resolve_column(columns: Iterable[str], desired: str) -> str:
    col_map = {str(c).lower(): str(c) for c in columns}
    resolved = col_map.get(desired.lower())
    if not resolved:
        raise ValueError(
            f"Column '{desired}' was not found. Available columns: {list(columns)}"
        )
    return resolved


def read_dataset_observations(
    dataset_path: Path,
    cfg: DatasetObservationConfig,
) -> tuple[pd.DataFrame, dict[str, object]]:
    if not dataset_path.exists():
        raise FileNotFoundError(f"Dataset not found: {dataset_path}")

    header = list(pd.read_csv(dataset_path, nrows=0).columns)
    ticker_col = _resolve_column(header, cfg.ticker_col)
    period_end_col = _resolve_column(header, cfg.period_end_col)

    usecols = [ticker_col, period_end_col]
    timeframe_col = None
    if cfg.timeframe_col:
        try:
            timeframe_col = _resolve_column(header, cfg.timeframe_col)
            if timeframe_col not in usecols:
                usecols.append(timeframe_col)
        except ValueError:
            timeframe_col = None

    raw = pd.read_csv(dataset_path, usecols=usecols)
    rows_read = int(len(raw))

    out = pd.DataFrame(
        {
            "source_dataset": str(dataset_path),
            "ticker": normalize_tickers(raw[ticker_col]),
            "period_end": pd.to_datetime(raw[period_end_col], errors="coerce").dt.normalize(),
        }
    )

    timeframe_filter_applied = False
    if cfg.timeframe and timeframe_col and timeframe_col in raw.columns:
        timeframe_filter_applied = True
        tf = raw[timeframe_col].astype(str).str.strip().str.lower()
        out = out.loc[tf == cfg.timeframe.lower()].copy()

    out = out.replace({"ticker": {"": np.nan, "NAN": np.nan, "NONE": np.nan}})
    before_valid = len(out)
    out = out.dropna(subset=["ticker", "period_end"]).copy()
    rows_dropped_invalid = int(before_valid - len(out))

    out = out.sort_values(["ticker", "period_end"], kind="stable")
    rows_before_dedup = int(len(out))
    out = out.drop_duplicates(subset=["source_dataset", "ticker", "period_end"], keep="last")
    rows_deduped = int(rows_before_dedup - len(out))

    summary = {
        "dataset_path": str(dataset_path),
        "rows_read": rows_read,
        "rows_after_timeframe_filter": int(before_valid),
        "rows_dropped_invalid": rows_dropped_invalid,
        "rows_deduped": rows_deduped,
        "rows_after_cleaning": int(len(out)),
        "timeframe_filter_applied": timeframe_filter_applied,
        "timeframe": cfg.timeframe,
        "unique_tickers": int(out["ticker"].nunique()) if not out.empty else 0,
        "min_period_end": out["period_end"].min().strftime("%Y-%m-%d") if not out.empty else None,
        "max_period_end": out["period_end"].max().strftime("%Y-%m-%d") if not out.empty else None,
    }
    return out.reset_index(drop=True), summary


def load_observation_universe(
    dataset_paths: Sequence[Path],
    cfg: DatasetObservationConfig,
) -> tuple[pd.DataFrame, list[dict[str, object]]]:
    frames: list[pd.DataFrame] = []
    summaries: list[dict[str, object]] = []
    for path in dataset_paths:
        frame, summary = read_dataset_observations(path, cfg)
        frames.append(frame)
        summaries.append(summary)

    if not frames:
        raise ValueError("No dataset paths were provided.")

    observations = pd.concat(frames, ignore_index=True)
    observations = observations.sort_values(
        ["ticker", "period_end", "source_dataset"],
        kind="stable",
    ).reset_index(drop=True)
    return observations, summaries


def build_ticker_window_table(observations: pd.DataFrame) -> pd.DataFrame:
    if observations.empty:
        return pd.DataFrame(
            columns=[
                "ticker",
                "observation_count",
                "source_dataset_count",
                "first_observation_date",
                "last_observation_date",
            ]
        )

    grouped = (
        observations.groupby("ticker", sort=True)
        .agg(
            observation_count=("period_end", "size"),
            source_dataset_count=("source_dataset", "nunique"),
            first_observation_date=("period_end", "min"),
            last_observation_date=("period_end", "max"),
        )
        .reset_index()
    )
    return grouped


def build_global_fetch_window(
    observations: pd.DataFrame,
    horizon_days: int,
    calendar_buffer_days: int,
) -> tuple[pd.Timestamp, pd.Timestamp]:
    if observations.empty:
        raise ValueError("Observation universe is empty.")

    start_date = observations["period_end"].min()
    end_date = observations["period_end"].max() + pd.Timedelta(
        days=int(horizon_days) + int(calendar_buffer_days)
    )
    return start_date.normalize(), end_date.normalize()

