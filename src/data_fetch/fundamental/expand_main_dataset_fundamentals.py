from __future__ import annotations

import argparse
import json
import math
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from src.models.risk.dataset_utils import ensure_dir, normalize_tickers, write_json

PROJECT_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_MAIN_DATASET = PROJECT_ROOT / "data/processed/main_dataset.csv"
DEFAULT_RAW_ROOT = PROJECT_ROOT / "data/raw"
DEFAULT_FEATURES_ONLY_PATH = PROJECT_ROOT / "data/processed/fundamental/main_dataset_fundamental_expansion_features.csv"
DEFAULT_OUT_PATH = PROJECT_ROOT / "data/processed/main_dataset_expanded_fundamentals.csv"
DEFAULT_SUMMARY_PATH = PROJECT_ROOT / "data/processed/main_dataset_expanded_fundamentals_summary.json"

KEY_COLUMNS = ["ticker", "fiscal_year"]
MAIN_NON_FEATURE_COLUMNS = {
    "ticker",
    "fiscal_year",
    "period_end",
    "cik",
    "timeframe",
    "has_income_statement",
    "has_cash_flow",
    "news_count",
    "news_description",
    "news_published_utc",
    "news_year",
    "news_sentiment_score",
}

FEATURE_SPECS: dict[str, dict[str, str]] = {
    "ebitda": {"source": "income_statement", "family": "raw_income", "description": "Reported EBITDA."},
    "gross_profit_raw": {"source": "income_statement", "family": "raw_income", "description": "Reported gross profit."},
    "cost_of_revenue": {"source": "income_statement", "family": "raw_income", "description": "Reported cost of revenue."},
    "income_before_tax": {"source": "income_statement", "family": "raw_income", "description": "Income before taxes."},
    "income_taxes": {"source": "income_statement", "family": "raw_income", "description": "Income tax expense."},
    "selling_general_administrative": {
        "source": "income_statement",
        "family": "raw_income",
        "description": "Selling, general and administrative expense.",
    },
    "research_and_development": {
        "source": "income_statement",
        "family": "raw_income",
        "description": "Research and development expense.",
    },
    "interest_income": {"source": "income_statement", "family": "raw_income", "description": "Interest income."},
    "depreciation_and_amortization": {
        "source": "income_or_cashflow",
        "family": "raw_income",
        "description": "Depreciation and amortization, using income statement first then cash flow fallback.",
    },
    "basic_eps": {"source": "income_statement", "family": "per_share", "description": "Basic EPS."},
    "diluted_eps": {"source": "income_statement", "family": "per_share", "description": "Diluted EPS."},
    "basic_shares_outstanding": {
        "source": "income_statement",
        "family": "per_share",
        "description": "Basic weighted-average shares outstanding.",
    },
    "diluted_shares_outstanding": {
        "source": "income_statement",
        "family": "per_share",
        "description": "Diluted weighted-average shares outstanding.",
    },
    "dividends_paid": {"source": "cash_flow", "family": "raw_cashflow", "description": "Dividends from cash flow statement."},
    "investing_cash_flow": {
        "source": "cash_flow",
        "family": "raw_cashflow",
        "description": "Net cash from investing activities.",
    },
    "financing_cash_flow": {
        "source": "cash_flow",
        "family": "raw_cashflow",
        "description": "Net cash from financing activities.",
    },
    "long_term_debt_issuances_repayments": {
        "source": "cash_flow",
        "family": "raw_cashflow",
        "description": "Net long-term debt issuances or repayments.",
    },
    "short_term_debt_issuances_repayments": {
        "source": "cash_flow",
        "family": "raw_cashflow",
        "description": "Net short-term debt issuances or repayments.",
    },
    "cash_change": {
        "source": "cash_flow",
        "family": "raw_cashflow",
        "description": "Change in cash and equivalents.",
    },
    "working_capital_change_cf": {
        "source": "cash_flow",
        "family": "raw_cashflow",
        "description": "Change in other operating assets and liabilities net.",
    },
    "sale_of_ppe": {
        "source": "cash_flow",
        "family": "raw_cashflow",
        "description": "Cash proceeds from sale of property, plant and equipment.",
    },
    "accounts_payable": {"source": "balance_sheet", "family": "raw_balance", "description": "Accounts payable."},
    "inventory": {"source": "balance_sheet", "family": "raw_balance", "description": "Inventories."},
    "ppe_net": {"source": "balance_sheet", "family": "raw_balance", "description": "Net property, plant and equipment."},
    "goodwill": {"source": "balance_sheet", "family": "raw_balance", "description": "Goodwill."},
    "intangible_assets": {"source": "balance_sheet", "family": "raw_balance", "description": "Net intangible assets."},
    "retained_earnings": {"source": "balance_sheet", "family": "raw_balance", "description": "Retained earnings or deficit."},
    "short_term_investments": {
        "source": "balance_sheet",
        "family": "raw_balance",
        "description": "Short-term investments.",
    },
    "accrued_current_liabilities": {
        "source": "balance_sheet",
        "family": "raw_balance",
        "description": "Accrued and other current liabilities.",
    },
    "other_current_assets": {
        "source": "balance_sheet",
        "family": "raw_balance",
        "description": "Other current assets.",
    },
    "deferred_revenue_current": {
        "source": "balance_sheet",
        "family": "raw_balance",
        "description": "Current deferred revenue.",
    },
    "additional_paid_in_capital": {
        "source": "balance_sheet",
        "family": "raw_balance",
        "description": "Additional paid-in capital.",
    },
    "accumulated_other_comprehensive_income": {
        "source": "balance_sheet",
        "family": "raw_balance",
        "description": "Accumulated other comprehensive income.",
    },
    "debt_current_component": {
        "source": "balance_sheet",
        "family": "raw_balance",
        "description": "Current debt component.",
    },
    "debt_long_term_component": {
        "source": "balance_sheet",
        "family": "raw_balance",
        "description": "Long-term debt component.",
    },
    "ebitda_margin": {"source": "derived", "family": "margin", "description": "EBITDA divided by revenue."},
    "pretax_margin": {"source": "derived", "family": "margin", "description": "Income before tax divided by revenue."},
    "sga_to_revenue": {"source": "derived", "family": "margin", "description": "SG&A divided by revenue."},
    "rd_to_revenue": {"source": "derived", "family": "margin", "description": "R&D divided by revenue."},
    "tax_rate": {"source": "derived", "family": "margin", "description": "Income taxes divided by pretax income."},
    "depreciation_to_revenue": {
        "source": "derived",
        "family": "margin",
        "description": "Depreciation and amortization divided by revenue.",
    },
    "dividends_to_cfo": {"source": "derived", "family": "cash_quality", "description": "Dividends divided by CFO."},
    "cfo_to_ebitda": {"source": "derived", "family": "cash_quality", "description": "CFO divided by EBITDA."},
    "capex_to_depreciation": {
        "source": "derived",
        "family": "cash_quality",
        "description": "Capex divided by depreciation and amortization.",
    },
    "accruals_to_revenue": {
        "source": "derived",
        "family": "cash_quality",
        "description": "Net income minus CFO, scaled by revenue.",
    },
    "accounts_payable_to_revenue": {
        "source": "derived",
        "family": "working_capital",
        "description": "Accounts payable divided by revenue.",
    },
    "inventory_to_revenue": {
        "source": "derived",
        "family": "working_capital",
        "description": "Inventory divided by revenue.",
    },
    "deferred_revenue_to_revenue": {
        "source": "derived",
        "family": "working_capital",
        "description": "Current deferred revenue divided by revenue.",
    },
    "ppe_turnover": {"source": "derived", "family": "efficiency", "description": "Revenue divided by net PP&E."},
    "current_debt_share": {
        "source": "derived",
        "family": "capital_structure",
        "description": "Current debt divided by current plus long-term debt.",
    },
    "revenue_per_share": {
        "source": "derived",
        "family": "per_share",
        "description": "Revenue divided by diluted shares outstanding.",
    },
    "gross_profit_per_share": {
        "source": "derived",
        "family": "per_share",
        "description": "Gross profit divided by diluted shares outstanding.",
    },
    "ebitda_per_share": {
        "source": "derived",
        "family": "per_share",
        "description": "EBITDA divided by diluted shares outstanding.",
    },
    "cfo_per_share": {
        "source": "derived",
        "family": "per_share",
        "description": "CFO divided by diluted shares outstanding.",
    },
    "book_value_per_share": {
        "source": "derived",
        "family": "per_share",
        "description": "Total equity divided by diluted shares outstanding.",
    },
    "revenue_yoy": {
        "source": "derived_growth",
        "family": "growth",
        "description": "Year-over-year signed growth in revenue.",
    },
    "gross_profit_yoy": {
        "source": "derived_growth",
        "family": "growth",
        "description": "Year-over-year signed growth in gross profit.",
    },
    "ebitda_yoy": {
        "source": "derived_growth",
        "family": "growth",
        "description": "Year-over-year signed growth in EBITDA.",
    },
    "operating_income_yoy": {
        "source": "derived_growth",
        "family": "growth",
        "description": "Year-over-year signed growth in operating income.",
    },
    "net_income_yoy": {
        "source": "derived_growth",
        "family": "growth",
        "description": "Year-over-year signed growth in net income.",
    },
    "cfo_yoy": {
        "source": "derived_growth",
        "family": "growth",
        "description": "Year-over-year signed growth in CFO.",
    },
    "capex_yoy": {
        "source": "derived_growth",
        "family": "growth",
        "description": "Year-over-year signed growth in capex.",
    },
    "dividends_yoy": {
        "source": "derived_growth",
        "family": "growth",
        "description": "Year-over-year signed growth in dividends.",
    },
    "total_debt_yoy": {
        "source": "derived_growth",
        "family": "growth",
        "description": "Year-over-year signed growth in total debt.",
    },
    "total_equity_yoy": {
        "source": "derived_growth",
        "family": "growth",
        "description": "Year-over-year signed growth in total equity.",
    },
    "has_research_and_development": {
        "source": "derived_flag",
        "family": "missingness",
        "description": "Indicator that R&D was reported.",
    },
    "has_inventory": {"source": "derived_flag", "family": "missingness", "description": "Indicator that inventory was reported."},
    "has_goodwill": {"source": "derived_flag", "family": "missingness", "description": "Indicator that goodwill was reported."},
    "has_intangible_assets": {
        "source": "derived_flag",
        "family": "missingness",
        "description": "Indicator that intangible assets were reported.",
    },
    "has_short_term_investments": {
        "source": "derived_flag",
        "family": "missingness",
        "description": "Indicator that short-term investments were reported.",
    },
    "has_deferred_revenue_current": {
        "source": "derived_flag",
        "family": "missingness",
        "description": "Indicator that current deferred revenue was reported.",
    },
}


@dataclass
class ExpandFundamentalDatasetConfig:
    main_dataset_path: Path = DEFAULT_MAIN_DATASET
    raw_root: Path = DEFAULT_RAW_ROOT
    features_only_path: Path = DEFAULT_FEATURES_ONLY_PATH
    out_path: Path = DEFAULT_OUT_PATH
    summary_path: Path = DEFAULT_SUMMARY_PATH
    ticker_col: str = "ticker"
    fiscal_year_col: str = "fiscal_year"
    period_end_col: str = "period_end"
    timeframe_col: str = "timeframe"
    timeframe_value: str | None = "annual"
    max_tickers: int | None = None


def _log(message: str) -> None:
    print(message, flush=True)


def _resolve_column(columns: list[str], candidates: list[str], required: bool = True) -> str | None:
    col_map = {str(c).lower(): str(c) for c in columns}
    for candidate in candidates:
        resolved = col_map.get(candidate.lower())
        if resolved:
            return resolved
    if required:
        raise ValueError(f"None of the expected columns {candidates} were found. Available columns: {columns}")
    return None


def _to_float(value: Any) -> float:
    try:
        if value is None:
            return float("nan")
        out = float(value)
        if not np.isfinite(out):
            return float("nan")
        return out
    except Exception:
        return float("nan")


def _safe_div(num: Any, den: Any) -> float:
    num_f = _to_float(num)
    den_f = _to_float(den)
    if not np.isfinite(num_f) or not np.isfinite(den_f) or math.isclose(den_f, 0.0, abs_tol=1e-12):
        return float("nan")
    return num_f / den_f


def _signed_growth(current: Any, previous: Any) -> float:
    cur = _to_float(current)
    prev = _to_float(previous)
    if not np.isfinite(cur) or not np.isfinite(prev) or math.isclose(prev, 0.0, abs_tol=1e-12):
        return float("nan")
    return (cur - prev) / max(abs(prev), 1e-12)


def _pick_first(row: dict[str, Any], keys: list[str]) -> Any:
    for key in keys:
        value = row.get(key)
        if value is not None:
            return value
    return None


def _load_massive_results(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    try:
        obj = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return []
    if isinstance(obj, dict) and isinstance(obj.get("results"), list):
        return [row for row in obj["results"] if isinstance(row, dict)]
    return []


def _rows_by_fiscal_year(rows: list[dict[str, Any]]) -> dict[int, dict[str, Any]]:
    out: dict[int, dict[str, Any]] = {}
    for row in rows:
        fy = row.get("fiscal_year")
        try:
            fy_int = int(fy)
        except Exception:
            continue
        out[fy_int] = row
    return out


def _normalize_main_dataset(cfg: ExpandFundamentalDatasetConfig) -> tuple[pd.DataFrame, dict[str, str]]:
    if not cfg.main_dataset_path.exists():
        raise FileNotFoundError(f"Main dataset not found: {cfg.main_dataset_path}")

    df = pd.read_csv(cfg.main_dataset_path)
    header = list(df.columns)
    ticker_col = _resolve_column(header, [cfg.ticker_col])
    fiscal_year_col = _resolve_column(header, [cfg.fiscal_year_col])
    period_end_col = _resolve_column(header, [cfg.period_end_col], required=False)
    timeframe_col = _resolve_column(header, [cfg.timeframe_col], required=False)

    out = df.copy()
    out[ticker_col] = normalize_tickers(out[ticker_col])
    out[fiscal_year_col] = pd.to_numeric(out[fiscal_year_col], errors="coerce").astype("Int64")
    if period_end_col:
        out[period_end_col] = pd.to_datetime(out[period_end_col], errors="coerce").dt.strftime("%Y-%m-%d")
    if timeframe_col and cfg.timeframe_value:
        tf = out[timeframe_col].astype(str).str.strip().str.lower()
        out = out.loc[tf == str(cfg.timeframe_value).lower()].copy()

    out = out.dropna(subset=[ticker_col, fiscal_year_col]).copy()
    out = out.rename(
        columns={
            ticker_col: "ticker",
            fiscal_year_col: "fiscal_year",
            **({period_end_col: "period_end"} if period_end_col else {}),
            **({timeframe_col: "timeframe"} if timeframe_col else {}),
        }
    )
    out["fiscal_year"] = out["fiscal_year"].astype(int)
    out = out.sort_values(["ticker", "fiscal_year"], kind="stable").reset_index(drop=True)
    return out, {
        "ticker": ticker_col,
        "fiscal_year": fiscal_year_col,
        "period_end": period_end_col or "",
        "timeframe": timeframe_col or "",
    }


def _capex_from_cash_flow(cf_row: dict[str, Any]) -> float:
    purchase = _to_float(cf_row.get("purchase_of_property_plant_and_equipment"))
    if not np.isfinite(purchase):
        return float("nan")
    return -purchase if purchase < 0.0 else purchase


def _compute_feature_row(
    ticker: str,
    fiscal_year: int,
    period_end: str | None,
    bs_row: dict[str, Any],
    inc_row: dict[str, Any],
    cf_row: dict[str, Any],
) -> dict[str, Any]:
    revenue = _to_float(inc_row.get("revenue"))
    gross_profit = _to_float(inc_row.get("gross_profit"))
    ebitda = _to_float(inc_row.get("ebitda"))
    cost_of_revenue = _to_float(inc_row.get("cost_of_revenue"))
    income_before_tax = _to_float(inc_row.get("income_before_income_taxes"))
    income_taxes = _to_float(inc_row.get("income_taxes"))
    sga = _to_float(inc_row.get("selling_general_administrative"))
    rd = _to_float(inc_row.get("research_development"))
    interest_income = _to_float(inc_row.get("interest_income"))
    operating_income = _to_float(inc_row.get("operating_income"))
    net_income = _to_float(_pick_first(inc_row, ["consolidated_net_income_loss", "net_income"]))
    da = _to_float(inc_row.get("depreciation_depletion_amortization"))
    if not np.isfinite(da):
        da = _to_float(cf_row.get("depreciation_depletion_and_amortization"))

    basic_eps = _to_float(inc_row.get("basic_earnings_per_share"))
    diluted_eps = _to_float(inc_row.get("diluted_earnings_per_share"))
    basic_shares = _to_float(inc_row.get("basic_shares_outstanding"))
    diluted_shares = _to_float(inc_row.get("diluted_shares_outstanding"))

    dividends_paid = _to_float(cf_row.get("dividends"))
    investing_cash_flow = _to_float(cf_row.get("net_cash_from_investing_activities"))
    financing_cash_flow = _to_float(cf_row.get("net_cash_from_financing_activities"))
    long_term_debt_cf = _to_float(cf_row.get("long_term_debt_issuances_repayments"))
    short_term_debt_cf = _to_float(cf_row.get("short_term_debt_issuances_repayments"))
    cash_change = _to_float(cf_row.get("change_in_cash_and_equivalents"))
    working_capital_change_cf = _to_float(cf_row.get("change_in_other_operating_assets_and_liabilities_net"))
    sale_of_ppe = _to_float(cf_row.get("sale_of_property_plant_and_equipment"))

    accounts_payable = _to_float(bs_row.get("accounts_payable"))
    inventory = _to_float(bs_row.get("inventories"))
    ppe_net = _to_float(bs_row.get("property_plant_equipment_net"))
    goodwill = _to_float(bs_row.get("goodwill"))
    intangible_assets = _to_float(bs_row.get("intangible_assets_net"))
    retained_earnings = _to_float(bs_row.get("retained_earnings_deficit"))
    short_term_investments = _to_float(bs_row.get("short_term_investments"))
    accrued_current_liabilities = _to_float(bs_row.get("accrued_and_other_current_liabilities"))
    other_current_assets = _to_float(bs_row.get("other_current_assets"))
    deferred_revenue_current = _to_float(bs_row.get("deferred_revenue_current"))
    additional_paid_in_capital = _to_float(bs_row.get("additional_paid_in_capital"))
    accumulated_other_comprehensive_income = _to_float(bs_row.get("accumulated_other_comprehensive_income"))
    debt_current_component = _to_float(bs_row.get("debt_current"))
    debt_long_term_component = _to_float(bs_row.get("long_term_debt_and_capital_lease_obligations"))

    total_equity = _to_float(_pick_first(bs_row, ["total_equity", "total_equity_attributable_to_parent"]))
    total_debt = float("nan")
    if np.isfinite(debt_current_component) or np.isfinite(debt_long_term_component):
        total_debt = np.nansum([debt_current_component, debt_long_term_component])

    cfo = _to_float(
        _pick_first(
            cf_row,
            [
                "net_cash_from_operating_activities",
                "cash_from_operating_activities_continuing_operations",
            ],
        )
    )
    capex = _capex_from_cash_flow(cf_row)

    row = {
        "ticker": ticker,
        "fiscal_year": int(fiscal_year),
        "raw_period_end": period_end,
        "statement_period_end": bs_row.get("period_end") or inc_row.get("period_end") or cf_row.get("period_end"),
        "ebitda": ebitda,
        "gross_profit_raw": gross_profit,
        "cost_of_revenue": cost_of_revenue,
        "income_before_tax": income_before_tax,
        "income_taxes": income_taxes,
        "selling_general_administrative": sga,
        "research_and_development": rd,
        "interest_income": interest_income,
        "depreciation_and_amortization": da,
        "basic_eps": basic_eps,
        "diluted_eps": diluted_eps,
        "basic_shares_outstanding": basic_shares,
        "diluted_shares_outstanding": diluted_shares,
        "dividends_paid": dividends_paid,
        "investing_cash_flow": investing_cash_flow,
        "financing_cash_flow": financing_cash_flow,
        "long_term_debt_issuances_repayments": long_term_debt_cf,
        "short_term_debt_issuances_repayments": short_term_debt_cf,
        "cash_change": cash_change,
        "working_capital_change_cf": working_capital_change_cf,
        "sale_of_ppe": sale_of_ppe,
        "accounts_payable": accounts_payable,
        "inventory": inventory,
        "ppe_net": ppe_net,
        "goodwill": goodwill,
        "intangible_assets": intangible_assets,
        "retained_earnings": retained_earnings,
        "short_term_investments": short_term_investments,
        "accrued_current_liabilities": accrued_current_liabilities,
        "other_current_assets": other_current_assets,
        "deferred_revenue_current": deferred_revenue_current,
        "additional_paid_in_capital": additional_paid_in_capital,
        "accumulated_other_comprehensive_income": accumulated_other_comprehensive_income,
        "debt_current_component": debt_current_component,
        "debt_long_term_component": debt_long_term_component,
        "ebitda_margin": _safe_div(ebitda, revenue),
        "pretax_margin": _safe_div(income_before_tax, revenue),
        "sga_to_revenue": _safe_div(sga, revenue),
        "rd_to_revenue": _safe_div(rd, revenue),
        "tax_rate": _safe_div(income_taxes, income_before_tax),
        "depreciation_to_revenue": _safe_div(da, revenue),
        "dividends_to_cfo": _safe_div(abs(dividends_paid), abs(cfo)),
        "cfo_to_ebitda": _safe_div(cfo, ebitda),
        "capex_to_depreciation": _safe_div(capex, da),
        "accruals_to_revenue": _safe_div(net_income - cfo, revenue),
        "accounts_payable_to_revenue": _safe_div(accounts_payable, revenue),
        "inventory_to_revenue": _safe_div(inventory, revenue),
        "deferred_revenue_to_revenue": _safe_div(deferred_revenue_current, revenue),
        "ppe_turnover": _safe_div(revenue, ppe_net),
        "current_debt_share": _safe_div(debt_current_component, total_debt),
        "revenue_per_share": _safe_div(revenue, diluted_shares),
        "gross_profit_per_share": _safe_div(gross_profit, diluted_shares),
        "ebitda_per_share": _safe_div(ebitda, diluted_shares),
        "cfo_per_share": _safe_div(cfo, diluted_shares),
        "book_value_per_share": _safe_div(total_equity, diluted_shares),
        "has_research_and_development": int(np.isfinite(rd)),
        "has_inventory": int(np.isfinite(inventory)),
        "has_goodwill": int(np.isfinite(goodwill)),
        "has_intangible_assets": int(np.isfinite(intangible_assets)),
        "has_short_term_investments": int(np.isfinite(short_term_investments)),
        "has_deferred_revenue_current": int(np.isfinite(deferred_revenue_current)),
        # Internal helper columns kept for YoY engineering then dropped before merge.
        "__operating_income": operating_income,
        "__net_income": net_income,
        "__cfo": cfo,
        "__capex": capex,
        "__dividends_paid": dividends_paid,
        "__total_debt": total_debt,
        "__total_equity": total_equity,
        "__revenue": revenue,
        "__gross_profit_raw": gross_profit,
        "__ebitda": ebitda,
    }
    return row


def _add_growth_features(frame: pd.DataFrame) -> pd.DataFrame:
    if frame.empty:
        return frame

    out = frame.sort_values(["ticker", "fiscal_year"], kind="stable").copy()
    growth_map = {
        "revenue_yoy": "__revenue",
        "gross_profit_yoy": "__gross_profit_raw",
        "ebitda_yoy": "__ebitda",
        "operating_income_yoy": "__operating_income",
        "net_income_yoy": "__net_income",
        "cfo_yoy": "__cfo",
        "capex_yoy": "__capex",
        "dividends_yoy": "__dividends_paid",
        "total_debt_yoy": "__total_debt",
        "total_equity_yoy": "__total_equity",
    }

    for out_col, src_col in growth_map.items():
        prev = out.groupby("ticker", sort=False)[src_col].shift(1)
        cur = pd.to_numeric(out[src_col], errors="coerce")
        denom = prev.abs().clip(lower=1e-12)
        growth = (cur - prev) / denom
        growth = growth.where(prev.notna())
        out[out_col] = growth.astype(float)

    helper_cols = [c for c in out.columns if c.startswith("__")]
    return out.drop(columns=helper_cols)


def _extract_feature_frame(
    main_df: pd.DataFrame,
    cfg: ExpandFundamentalDatasetConfig,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    raw_root = cfg.raw_root
    bs_dir = raw_root / "balance-sheets"
    is_dir = raw_root / "income-statements"
    cf_dir = raw_root / "cash-flow-statements"

    tickers = main_df["ticker"].dropna().astype(str).unique().tolist()
    tickers = sorted(tickers)
    if cfg.max_tickers is not None:
        tickers = tickers[: int(cfg.max_tickers)]
        main_df = main_df.loc[main_df["ticker"].isin(tickers)].copy()

    rows_out: list[dict[str, Any]] = []
    matched_rows = 0
    missing_statement_counts = {"income": 0, "cash": 0, "balance": 0}

    period_end_lookup = (
        main_df.loc[:, ["ticker", "fiscal_year", "period_end"]]
        if "period_end" in main_df.columns
        else main_df.assign(period_end=None).loc[:, ["ticker", "fiscal_year", "period_end"]]
    )
    year_rows = {
        ticker: group.set_index("fiscal_year")["period_end"].to_dict()
        for ticker, group in period_end_lookup.groupby("ticker", sort=False)
    }

    for idx, ticker in enumerate(tickers, start=1):
        if idx % 500 == 0 or idx == len(tickers):
            _log(f"[expand] processed tickers {idx}/{len(tickers)}")

        bs_rows = _rows_by_fiscal_year(_load_massive_results(bs_dir / f"{ticker}.json"))
        inc_rows = _rows_by_fiscal_year(_load_massive_results(is_dir / f"{ticker}.json"))
        cf_rows = _rows_by_fiscal_year(_load_massive_results(cf_dir / f"{ticker}.json"))

        if not bs_rows:
            missing_statement_counts["balance"] += 1
        if not inc_rows:
            missing_statement_counts["income"] += 1
        if not cf_rows:
            missing_statement_counts["cash"] += 1

        fiscal_years = sorted(year_rows.get(ticker, {}).keys())
        for fiscal_year in fiscal_years:
            bs_row = bs_rows.get(int(fiscal_year), {})
            inc_row = inc_rows.get(int(fiscal_year), {})
            cf_row = cf_rows.get(int(fiscal_year), {})
            if not bs_row and not inc_row and not cf_row:
                continue
            matched_rows += 1
            rows_out.append(
                _compute_feature_row(
                    ticker=ticker,
                    fiscal_year=int(fiscal_year),
                    period_end=year_rows.get(ticker, {}).get(int(fiscal_year)),
                    bs_row=bs_row,
                    inc_row=inc_row,
                    cf_row=cf_row,
                )
            )

    feature_df = pd.DataFrame(rows_out)
    if not feature_df.empty:
        feature_df = _add_growth_features(feature_df)
        feature_df = feature_df.sort_values(KEY_COLUMNS, kind="stable").drop_duplicates(KEY_COLUMNS, keep="last")
        feature_df = feature_df.reset_index(drop=True)

    summary = {
        "requested_tickers": int(len(tickers)),
        "matched_feature_rows": int(matched_rows),
        "missing_statement_files": missing_statement_counts,
    }
    return feature_df, summary


def _merge_feature_frame(main_df: pd.DataFrame, feature_df: pd.DataFrame) -> tuple[pd.DataFrame, list[str], list[str]]:
    if feature_df.empty:
        return main_df.copy(), [], []

    feature_cols = [col for col in feature_df.columns if col not in {*KEY_COLUMNS, "raw_period_end", "statement_period_end"}]
    new_cols = [col for col in feature_cols if col not in main_df.columns]
    overlap_cols = [col for col in feature_cols if col in main_df.columns]

    out = main_df.copy()
    out["__row_id"] = range(len(out))
    merged = out.merge(
        feature_df.loc[:, [*KEY_COLUMNS, "raw_period_end", "statement_period_end", *new_cols]],
        on=KEY_COLUMNS,
        how="left",
        validate="many_to_one",
        suffixes=("", "__fundamental"),
    )
    merged = merged.sort_values("__row_id", kind="stable").drop(columns=["__row_id"]).reset_index(drop=True)
    return merged, new_cols, overlap_cols


def _coverage_rows(merged_df: pd.DataFrame, added_cols: list[str]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    total_rows = max(len(merged_df), 1)
    for col in added_cols:
        non_null = int(merged_df[col].notna().sum()) if col in merged_df.columns else 0
        spec = FEATURE_SPECS.get(col, {})
        rows.append(
            {
                "feature": col,
                "family": spec.get("family", "unknown"),
                "source": spec.get("source", "unknown"),
                "description": spec.get("description", ""),
                "non_null_rows": non_null,
                "coverage": float(non_null / total_rows),
            }
        )
    rows.sort(key=lambda item: (-float(item["coverage"]), str(item["feature"])))
    return rows


def _build_summary(
    cfg: ExpandFundamentalDatasetConfig,
    main_df: pd.DataFrame,
    feature_df: pd.DataFrame,
    merged_df: pd.DataFrame,
    source_columns: dict[str, str],
    extract_summary: dict[str, Any],
    added_cols: list[str],
    overlap_cols: list[str],
) -> dict[str, Any]:
    rows_with_feature_match = 0
    if feature_df is not None and not feature_df.empty:
        matched_keys = set(zip(feature_df["ticker"], feature_df["fiscal_year"]))
        rows_with_feature_match = int(
            sum((ticker, fiscal_year) in matched_keys for ticker, fiscal_year in zip(main_df["ticker"], main_df["fiscal_year"]))
        )

    coverage = _coverage_rows(merged_df, added_cols)
    candidate_recommendations = [
        row["feature"]
        for row in coverage
        if float(row["coverage"]) >= 0.5 and row["feature"] in {
            "ebitda",
            "gross_profit_raw",
            "cost_of_revenue",
            "selling_general_administrative",
            "income_before_tax",
            "depreciation_and_amortization",
            "basic_eps",
            "diluted_eps",
            "basic_shares_outstanding",
            "diluted_shares_outstanding",
            "dividends_paid",
            "investing_cash_flow",
            "financing_cash_flow",
            "accounts_payable",
            "ppe_net",
            "retained_earnings",
            "ebitda_margin",
            "pretax_margin",
            "sga_to_revenue",
            "cfo_to_ebitda",
            "revenue_per_share",
            "ebitda_per_share",
            "book_value_per_share",
            "revenue_yoy",
            "ebitda_yoy",
            "cfo_yoy",
        }
    ]

    return {
        "config": asdict(cfg),
        "source_columns": source_columns,
        "rows": {
            "main_dataset": int(len(main_df)),
            "feature_frame": int(len(feature_df)),
            "merged_output": int(len(merged_df)),
        },
        "unique_tickers": {
            "main_dataset": int(main_df["ticker"].nunique()) if not main_df.empty else 0,
            "feature_frame": int(feature_df["ticker"].nunique()) if not feature_df.empty else 0,
            "merged_output": int(merged_df["ticker"].nunique()) if not merged_df.empty else 0,
        },
        "rows_with_feature_match": rows_with_feature_match,
        "fundamental_columns_added": added_cols,
        "fundamental_columns_skipped_existing_overlap": overlap_cols,
        "feature_extraction": extract_summary,
        "coverage_by_added_feature": coverage,
        "recommended_high_value_features": candidate_recommendations,
        "artifacts": {
            "main_dataset_path": str(cfg.main_dataset_path),
            "raw_root": str(cfg.raw_root),
            "features_only_path": str(cfg.features_only_path),
            "out_path": str(cfg.out_path),
            "summary_path": str(cfg.summary_path),
        },
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--main-dataset", type=str, default=str(DEFAULT_MAIN_DATASET))
    ap.add_argument("--raw-root", type=str, default=str(DEFAULT_RAW_ROOT))
    ap.add_argument("--features-only-path", type=str, default=str(DEFAULT_FEATURES_ONLY_PATH))
    ap.add_argument("--out-path", type=str, default=str(DEFAULT_OUT_PATH))
    ap.add_argument("--summary-path", type=str, default=str(DEFAULT_SUMMARY_PATH))
    ap.add_argument("--ticker-col", type=str, default="ticker")
    ap.add_argument("--fiscal-year-col", type=str, default="fiscal_year")
    ap.add_argument("--period-end-col", type=str, default="period_end")
    ap.add_argument("--timeframe-col", type=str, default="timeframe")
    ap.add_argument("--timeframe-value", type=str, default="annual")
    ap.add_argument("--max-tickers", type=int, default=None)
    args = ap.parse_args()

    cfg = ExpandFundamentalDatasetConfig(
        main_dataset_path=Path(args.main_dataset),
        raw_root=Path(args.raw_root),
        features_only_path=Path(args.features_only_path),
        out_path=Path(args.out_path),
        summary_path=Path(args.summary_path),
        ticker_col=str(args.ticker_col),
        fiscal_year_col=str(args.fiscal_year_col),
        period_end_col=str(args.period_end_col),
        timeframe_col=str(args.timeframe_col),
        timeframe_value=None if args.timeframe_value.lower() == "none" else str(args.timeframe_value),
        max_tickers=args.max_tickers,
    )

    ensure_dir(cfg.features_only_path.parent)
    ensure_dir(cfg.out_path.parent)
    ensure_dir(cfg.summary_path.parent)

    _log(f"[expand] Loading main dataset from {cfg.main_dataset_path}")
    main_df, source_columns = _normalize_main_dataset(cfg)
    if main_df.empty:
        raise SystemExit("Main dataset is empty after normalization/filtering.")

    duplicate_count = int(main_df.duplicated(KEY_COLUMNS).sum())
    if duplicate_count > 0:
        raise SystemExit(
            f"Main dataset has {duplicate_count} duplicate rows on keys {KEY_COLUMNS}; "
            "the expansion merge expects a unique annual observation per ticker and fiscal year."
        )

    _log(
        f"[expand] Main rows={len(main_df)} | tickers={main_df['ticker'].nunique()} | "
        f"year_range={main_df['fiscal_year'].min()}..{main_df['fiscal_year'].max()}"
    )
    _log("[expand] Extracting additional fundamental features from raw statements")
    feature_df, extract_summary = _extract_feature_frame(main_df, cfg)

    _log(f"[expand] Writing extracted feature table to {cfg.features_only_path}")
    feature_df.to_csv(cfg.features_only_path, index=False)

    _log("[expand] Merging new fundamental columns into the main dataset without overwriting existing columns")
    merged_df, added_cols, overlap_cols = _merge_feature_frame(main_df, feature_df)

    _log(f"[expand] Writing expanded dataset to {cfg.out_path}")
    merged_df.to_csv(cfg.out_path, index=False)

    summary = _build_summary(
        cfg=cfg,
        main_df=main_df,
        feature_df=feature_df,
        merged_df=merged_df,
        source_columns=source_columns,
        extract_summary=extract_summary,
        added_cols=added_cols,
        overlap_cols=overlap_cols,
    )
    _log(f"[expand] Writing summary to {cfg.summary_path}")
    write_json(cfg.summary_path, summary)

    print("Expanded main dataset with additional fundamental features created:")
    print(f"rows={summary['rows']['merged_output']} | rows_with_feature_match={summary['rows_with_feature_match']}")
    print(f"added_columns={len(added_cols)} | overlap_skipped={len(overlap_cols)}")
    print(f"dataset={cfg.out_path}")
    print(f"features={cfg.features_only_path}")
    print(f"summary={cfg.summary_path}")


if __name__ == "__main__":
    main()
