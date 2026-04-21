from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from src.data_fetch.market.build_market_feature_merge import _load_market_panel
from src.models.risk.dataset_utils import ensure_dir, normalize_tickers, write_json

PROJECT_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_DATASET_PATH = PROJECT_ROOT / "data/processed/main_dataset_expanded_fundamentals.csv"
DEFAULT_MARKET_DATA_DIR = PROJECT_ROOT / "data/raw/market_data"
DEFAULT_OUT_PATH = PROJECT_ROOT / "data/processed/valuation/main_dataset_valuation_ready.csv"
DEFAULT_TARGETS_PATH = PROJECT_ROOT / "data/processed/valuation/main_dataset_valuation_targets.csv"
DEFAULT_SUMMARY_PATH = PROJECT_ROOT / "data/processed/valuation/main_dataset_valuation_target_summary.json"

KEY_COLUMNS = ["ticker", "fiscal_year", "period_end"]
TARGET_COLUMNS = [
    "anchor_trade_date",
    "anchor_close",
    "anchor_lag_days",
    "shares_target",
    "shares_target_source",
    "market_cap_basic_target",
    "market_cap_diluted_target",
    "market_cap_target",
    "log_market_cap_target",
    "enterprise_value_target",
    "enterprise_value_target_clipped",
    "log_enterprise_value_target",
    "enterprise_value_negative_flag",
    "price_to_sales_target",
    "price_to_book_target",
    "ev_to_ebitda_target",
    "valuation_target_status",
    "valuation_target_issue_codes",
    "valuation_target_usable",
    "enterprise_value_target_usable",
]


@dataclass
class ValuationTargetDatasetConfig:
    dataset_path: Path = DEFAULT_DATASET_PATH
    market_data_dir: Path = DEFAULT_MARKET_DATA_DIR
    out_path: Path = DEFAULT_OUT_PATH
    targets_path: Path = DEFAULT_TARGETS_PATH
    summary_path: Path = DEFAULT_SUMMARY_PATH
    ticker_col: str = "ticker"
    fiscal_year_col: str = "fiscal_year"
    period_end_col: str = "period_end"
    timeframe_col: str = "timeframe"
    timeframe: str | None = "annual"
    diluted_shares_col: str = "diluted_shares_outstanding"
    basic_shares_col: str = "basic_shares_outstanding"
    debt_col: str = "total_debt"
    cash_col: str = "cash_and_equivalents"
    revenue_col: str = "revenue"
    equity_col: str = "total_equity"
    ebitda_col: str = "ebitda"
    max_anchor_lag_days: int = 10
    max_tickers: int = 0


def _log(message: str) -> None:
    print(message, flush=True)


def _resolve_column(columns: list[str], desired: str, required: bool = True) -> str | None:
    col_map = {str(c).lower(): str(c) for c in columns}
    resolved = col_map.get(desired.lower())
    if resolved:
        return resolved
    if required:
        raise ValueError(f"Column '{desired}' was not found. Available columns: {columns}")
    return None


def _to_numeric(series: pd.Series) -> pd.Series:
    return pd.to_numeric(series, errors="coerce")


def _safe_div(num: pd.Series, den: pd.Series) -> pd.Series:
    num_s = _to_numeric(num)
    den_s = _to_numeric(den)
    out = num_s / den_s.replace(0.0, np.nan)
    out = out.where(np.isfinite(out))
    return out


def _load_dataset(cfg: ValuationTargetDatasetConfig) -> tuple[pd.DataFrame, dict[str, str]]:
    if not cfg.dataset_path.exists():
        raise FileNotFoundError(f"Dataset not found: {cfg.dataset_path}")

    df = pd.read_csv(cfg.dataset_path)
    header = list(df.columns)
    ticker_col = _resolve_column(header, cfg.ticker_col)
    fiscal_year_col = _resolve_column(header, cfg.fiscal_year_col)
    period_end_col = _resolve_column(header, cfg.period_end_col)
    timeframe_col = _resolve_column(header, cfg.timeframe_col, required=False)

    out = df.copy()
    out[ticker_col] = normalize_tickers(out[ticker_col])
    out[fiscal_year_col] = pd.to_numeric(out[fiscal_year_col], errors="coerce").astype("Int64")
    out[period_end_col] = pd.to_datetime(out[period_end_col], errors="coerce").dt.normalize().astype("datetime64[ns]")
    if timeframe_col and cfg.timeframe:
        tf = out[timeframe_col].astype(str).str.strip().str.lower()
        out = out.loc[tf == str(cfg.timeframe).lower()].copy()

    out = out.dropna(subset=[ticker_col, fiscal_year_col, period_end_col]).copy()
    out = out.rename(
        columns={
            ticker_col: "ticker",
            fiscal_year_col: "fiscal_year",
            period_end_col: "period_end",
            **({timeframe_col: "timeframe"} if timeframe_col else {}),
        }
    )
    out["fiscal_year"] = out["fiscal_year"].astype(int)
    out = out.sort_values(["ticker", "period_end", "fiscal_year"], kind="stable").reset_index(drop=True)

    if cfg.max_tickers and cfg.max_tickers > 0:
        keep = set(out["ticker"].drop_duplicates().head(int(cfg.max_tickers)))
        out = out.loc[out["ticker"].isin(keep)].copy().reset_index(drop=True)

    return out, {
        "ticker": ticker_col,
        "fiscal_year": fiscal_year_col,
        "period_end": period_end_col,
        "timeframe": timeframe_col or "",
    }


def _align_anchor_prices(dataset: pd.DataFrame, market_panel: pd.DataFrame) -> pd.DataFrame:
    if dataset.empty:
        return dataset.copy()

    base = dataset.sort_values(["ticker", "period_end"], kind="stable").copy()
    base["__row_id"] = np.arange(len(base), dtype=np.int64)

    if market_panel.empty:
        base["anchor_trade_date"] = pd.NaT
        base["anchor_close"] = np.nan
        return base.sort_values("__row_id", kind="stable").drop(columns=["__row_id"]).reset_index(drop=True)

    panel = (
        market_panel.loc[:, ["ticker", "date", "close"]]
        .rename(columns={"date": "anchor_trade_date", "close": "anchor_close"})
        .sort_values(["ticker", "anchor_trade_date"], kind="stable")
        .reset_index(drop=True)
    )
    panel["anchor_trade_date"] = pd.to_datetime(panel["anchor_trade_date"], errors="coerce").astype("datetime64[ns]")
    groups = {
        str(ticker): group.loc[:, ["anchor_trade_date", "anchor_close"]].reset_index(drop=True)
        for ticker, group in panel.groupby("ticker", sort=False)
    }

    aligned_groups: list[pd.DataFrame] = []
    for ticker, obs_group in base.groupby("ticker", sort=False):
        obs_sorted = obs_group.sort_values("period_end", kind="stable").reset_index(drop=True)
        obs_sorted["period_end"] = pd.to_datetime(obs_sorted["period_end"], errors="coerce").astype("datetime64[ns]")
        price_group = groups.get(str(ticker))
        if price_group is None or price_group.empty:
            out = obs_sorted.copy()
            out["anchor_trade_date"] = pd.NaT
            out["anchor_close"] = np.nan
            aligned_groups.append(out)
            continue

        joined = pd.merge_asof(
            obs_sorted,
            price_group,
            left_on="period_end",
            right_on="anchor_trade_date",
            direction="backward",
            allow_exact_matches=True,
        )
        aligned_groups.append(joined)

    aligned = pd.concat(aligned_groups, ignore_index=True)
    aligned = aligned.sort_values("__row_id", kind="stable").drop(columns=["__row_id"]).reset_index(drop=True)
    return aligned


def _compute_targets(df: pd.DataFrame, cfg: ValuationTargetDatasetConfig) -> pd.DataFrame:
    out = df.copy()

    diluted_col = _resolve_column(list(out.columns), cfg.diluted_shares_col, required=False)
    basic_col = _resolve_column(list(out.columns), cfg.basic_shares_col, required=False)
    debt_col = _resolve_column(list(out.columns), cfg.debt_col, required=False)
    cash_col = _resolve_column(list(out.columns), cfg.cash_col, required=False)
    revenue_col = _resolve_column(list(out.columns), cfg.revenue_col, required=False)
    equity_col = _resolve_column(list(out.columns), cfg.equity_col, required=False)
    ebitda_col = _resolve_column(list(out.columns), cfg.ebitda_col, required=False)

    diluted_shares = _to_numeric(out[diluted_col]) if diluted_col else pd.Series(np.nan, index=out.index)
    basic_shares = _to_numeric(out[basic_col]) if basic_col else pd.Series(np.nan, index=out.index)
    anchor_close = _to_numeric(out["anchor_close"])
    total_debt = _to_numeric(out[debt_col]) if debt_col else pd.Series(np.nan, index=out.index)
    cash = _to_numeric(out[cash_col]) if cash_col else pd.Series(np.nan, index=out.index)
    revenue = _to_numeric(out[revenue_col]) if revenue_col else pd.Series(np.nan, index=out.index)
    total_equity = _to_numeric(out[equity_col]) if equity_col else pd.Series(np.nan, index=out.index)
    ebitda = _to_numeric(out[ebitda_col]) if ebitda_col else pd.Series(np.nan, index=out.index)

    diluted_valid = diluted_shares > 0.0
    basic_valid = basic_shares > 0.0
    close_valid = anchor_close > 0.0

    shares_target = diluted_shares.where(diluted_valid, basic_shares.where(basic_valid))
    shares_source = pd.Series("", index=out.index, dtype=object)
    shares_source.loc[diluted_valid] = "diluted_shares_outstanding"
    shares_source.loc[(~diluted_valid) & basic_valid] = "basic_shares_outstanding"

    market_cap_basic = (anchor_close * basic_shares).where(close_valid & basic_valid)
    market_cap_diluted = (anchor_close * diluted_shares).where(close_valid & diluted_valid)
    market_cap_target = (anchor_close * shares_target).where(close_valid & shares_target.notna())

    enterprise_value = market_cap_target + total_debt - cash
    enterprise_value = enterprise_value.where(market_cap_target.notna())
    enterprise_value_clipped = enterprise_value.clip(lower=0.0)
    enterprise_value_negative = enterprise_value < 0.0

    out["anchor_lag_days"] = (
        (pd.to_datetime(out["period_end"]) - pd.to_datetime(out["anchor_trade_date"])).dt.days.astype("float")
    )

    market_cap_log = np.log1p(market_cap_target.clip(lower=0.0)).where(market_cap_target > 0.0)
    enterprise_value_log = np.log1p(enterprise_value_clipped).where(enterprise_value_clipped.notna())

    price_to_sales = _safe_div(market_cap_target, revenue.where(revenue > 0.0))
    price_to_book = _safe_div(market_cap_target, total_equity.where(total_equity > 0.0))
    ev_to_ebitda = _safe_div(enterprise_value_clipped, ebitda.where(ebitda > 0.0))

    missing_price = out["anchor_trade_date"].isna() | (~close_valid)
    stale_anchor = out["anchor_lag_days"].gt(float(cfg.max_anchor_lag_days))
    missing_shares = shares_target.isna()
    invalid_market_cap = market_cap_target.isna() | (market_cap_target <= 0.0)

    issue_rows: list[str] = []
    status_rows: list[str] = []
    usable_rows: list[bool] = []
    for miss_price, stale, miss_sh, bad_cap in zip(
        missing_price.to_numpy(dtype=bool),
        stale_anchor.fillna(False).to_numpy(dtype=bool),
        missing_shares.to_numpy(dtype=bool),
        invalid_market_cap.to_numpy(dtype=bool),
    ):
        codes: list[str] = []
        if miss_price:
            codes.append("missing_market_price")
        if stale:
            codes.append("stale_anchor_price")
        if miss_sh:
            codes.append("missing_share_count")
        if bad_cap and not miss_price and not miss_sh:
            codes.append("invalid_market_cap")

        issue_rows.append("|".join(codes))
        status_rows.append("ok" if not codes else codes[0])
        usable_rows.append(len(codes) == 0)

    out["shares_target"] = shares_target
    out["shares_target_source"] = shares_source.replace({"": pd.NA})
    out["market_cap_basic_target"] = market_cap_basic
    out["market_cap_diluted_target"] = market_cap_diluted
    out["market_cap_target"] = market_cap_target
    out["log_market_cap_target"] = market_cap_log
    out["enterprise_value_target"] = enterprise_value
    out["enterprise_value_target_clipped"] = enterprise_value_clipped
    out["log_enterprise_value_target"] = enterprise_value_log
    out["enterprise_value_negative_flag"] = enterprise_value_negative.fillna(False).astype(int)
    out["price_to_sales_target"] = price_to_sales
    out["price_to_book_target"] = price_to_book
    out["ev_to_ebitda_target"] = ev_to_ebitda
    out["valuation_target_status"] = status_rows
    out["valuation_target_issue_codes"] = issue_rows
    out["valuation_target_usable"] = usable_rows
    out["enterprise_value_target_usable"] = [
        bool(usable and pd.notna(ev_log)) for usable, ev_log in zip(usable_rows, enterprise_value_log)
    ]
    return out


def _build_summary(
    cfg: ValuationTargetDatasetConfig,
    source_columns: dict[str, str],
    dataset: pd.DataFrame,
    market_panel: pd.DataFrame,
    valuation_df: pd.DataFrame,
) -> dict[str, Any]:
    market_min = None
    market_max = None
    if not market_panel.empty:
        market_min = market_panel["date"].min()
        market_max = market_panel["date"].max()

    shares_source_counts = (
        valuation_df["shares_target_source"].fillna("missing").value_counts(dropna=False).to_dict()
        if "shares_target_source" in valuation_df.columns
        else {}
    )

    coverage = {
        "anchor_trade_date_non_null": int(valuation_df["anchor_trade_date"].notna().sum()),
        "anchor_close_non_null": int(pd.to_numeric(valuation_df["anchor_close"], errors="coerce").notna().sum()),
        "market_cap_target_non_null": int(pd.to_numeric(valuation_df["market_cap_target"], errors="coerce").notna().sum()),
        "valuation_target_usable": int(pd.Series(valuation_df["valuation_target_usable"]).fillna(False).sum()),
        "enterprise_value_target_non_null": int(pd.to_numeric(valuation_df["enterprise_value_target"], errors="coerce").notna().sum()),
        "enterprise_value_target_usable": int(pd.Series(valuation_df["enterprise_value_target_usable"]).fillna(False).sum()),
        "negative_enterprise_value_rows": int(pd.Series(valuation_df["enterprise_value_negative_flag"]).fillna(0).sum()),
    }

    return {
        "config": asdict(cfg),
        "source_columns": source_columns,
        "rows": {
            "dataset_input": int(len(dataset)),
            "valuation_output": int(len(valuation_df)),
        },
        "unique_tickers": int(valuation_df["ticker"].nunique()) if not valuation_df.empty else 0,
        "market_range": {
            "min_trade_date": market_min.strftime("%Y-%m-%d") if market_min is not None else None,
            "max_trade_date": market_max.strftime("%Y-%m-%d") if market_max is not None else None,
        },
        "coverage": coverage,
        "shares_target_source_counts": shares_source_counts,
        "recommended_primary_target": {
            "column": "market_cap_target",
            "log_column": "log_market_cap_target",
            "reason": "Cleanest equity valuation target with no debt/cash arithmetic built into the label.",
        },
        "recommended_secondary_target": {
            "column": "enterprise_value_target_clipped",
            "log_column": "log_enterprise_value_target",
            "reason": "Firm valuation target; keep in mind it partly embeds debt and cash.",
        },
        "target_definitions": {
            "anchor_trade_date": "Last trading date on or before the statement period_end used for target pricing.",
            "anchor_close": "Closing price on anchor_trade_date.",
            "market_cap_target": "Preferred equity valuation target = anchor_close * diluted shares if available, otherwise basic shares.",
            "enterprise_value_target": "market_cap_target + total_debt - cash_and_equivalents.",
            "enterprise_value_target_clipped": "Enterprise value clipped at zero for log-space modeling.",
            "price_to_sales_target": "market_cap_target / revenue.",
            "price_to_book_target": "market_cap_target / total_equity when equity is positive.",
            "ev_to_ebitda_target": "enterprise_value_target_clipped / EBITDA when EBITDA is positive.",
        },
        "artifacts": {
            "out_path": str(cfg.out_path),
            "targets_path": str(cfg.targets_path),
            "summary_path": str(cfg.summary_path),
            "dataset_path": str(cfg.dataset_path),
            "market_data_dir": str(cfg.market_data_dir),
        },
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", type=str, default=str(DEFAULT_DATASET_PATH))
    ap.add_argument("--market-data-dir", type=str, default=str(DEFAULT_MARKET_DATA_DIR))
    ap.add_argument("--out-path", type=str, default=str(DEFAULT_OUT_PATH))
    ap.add_argument("--targets-path", type=str, default=str(DEFAULT_TARGETS_PATH))
    ap.add_argument("--summary-path", type=str, default=str(DEFAULT_SUMMARY_PATH))
    ap.add_argument("--ticker-col", type=str, default="ticker")
    ap.add_argument("--fiscal-year-col", type=str, default="fiscal_year")
    ap.add_argument("--period-end-col", type=str, default="period_end")
    ap.add_argument("--timeframe-col", type=str, default="timeframe")
    ap.add_argument("--timeframe", type=str, default="annual")
    ap.add_argument("--diluted-shares-col", type=str, default="diluted_shares_outstanding")
    ap.add_argument("--basic-shares-col", type=str, default="basic_shares_outstanding")
    ap.add_argument("--debt-col", type=str, default="total_debt")
    ap.add_argument("--cash-col", type=str, default="cash_and_equivalents")
    ap.add_argument("--revenue-col", type=str, default="revenue")
    ap.add_argument("--equity-col", type=str, default="total_equity")
    ap.add_argument("--ebitda-col", type=str, default="ebitda")
    ap.add_argument("--max-anchor-lag-days", type=int, default=10)
    ap.add_argument("--max-tickers", type=int, default=0)
    args = ap.parse_args()

    cfg = ValuationTargetDatasetConfig(
        dataset_path=Path(args.dataset),
        market_data_dir=Path(args.market_data_dir),
        out_path=Path(args.out_path),
        targets_path=Path(args.targets_path),
        summary_path=Path(args.summary_path),
        ticker_col=str(args.ticker_col),
        fiscal_year_col=str(args.fiscal_year_col),
        period_end_col=str(args.period_end_col),
        timeframe_col=str(args.timeframe_col),
        timeframe=str(args.timeframe).strip() or None,
        diluted_shares_col=str(args.diluted_shares_col),
        basic_shares_col=str(args.basic_shares_col),
        debt_col=str(args.debt_col),
        cash_col=str(args.cash_col),
        revenue_col=str(args.revenue_col),
        equity_col=str(args.equity_col),
        ebitda_col=str(args.ebitda_col),
        max_anchor_lag_days=int(args.max_anchor_lag_days),
        max_tickers=int(args.max_tickers),
    )

    ensure_dir(cfg.out_path.parent)
    ensure_dir(cfg.targets_path.parent)
    ensure_dir(cfg.summary_path.parent)

    _log(f"[valuation-targets] Loading dataset from {cfg.dataset_path}")
    dataset, source_columns = _load_dataset(cfg)
    if dataset.empty:
        raise SystemExit("Input dataset is empty after normalization/filtering.")

    requested_tickers = set(dataset["ticker"].dropna().astype(str))
    _log(
        f"[valuation-targets] Rows={len(dataset)} | tickers={dataset['ticker'].nunique()} | "
        f"year_range={dataset['fiscal_year'].min()}..{dataset['fiscal_year'].max()}"
    )
    _log(f"[valuation-targets] Loading raw market panel for {len(requested_tickers)} tickers from {cfg.market_data_dir}")
    market_panel = _load_market_panel(cfg.market_data_dir, requested_tickers)
    _log(f"[valuation-targets] Market rows loaded={len(market_panel)}")

    _log("[valuation-targets] Aligning anchor prices to statement dates")
    aligned = _align_anchor_prices(dataset, market_panel)

    _log("[valuation-targets] Computing market-cap and enterprise-value targets")
    valuation_df = _compute_targets(aligned, cfg)

    target_table = valuation_df.loc[:, [*KEY_COLUMNS, *TARGET_COLUMNS]].copy()

    _log(f"[valuation-targets] Writing target table to {cfg.targets_path}")
    target_table.to_csv(cfg.targets_path, index=False)

    _log(f"[valuation-targets] Writing valuation-ready dataset to {cfg.out_path}")
    valuation_df.to_csv(cfg.out_path, index=False)

    summary = _build_summary(cfg, source_columns, dataset, market_panel, valuation_df)
    _log(f"[valuation-targets] Writing summary to {cfg.summary_path}")
    write_json(cfg.summary_path, summary)

    print("Valuation target artifacts created:")
    print(f"rows={summary['rows']['valuation_output']} | usable_market_cap={summary['coverage']['valuation_target_usable']}")
    print(f"dataset={cfg.out_path}")
    print(f"targets={cfg.targets_path}")
    print(f"summary={cfg.summary_path}")


if __name__ == "__main__":
    main()
