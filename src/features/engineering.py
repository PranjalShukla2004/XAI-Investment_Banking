from __future__ import annotations

from dataclasses import dataclass
from typing import Dict

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class RatioConfig:
    eps: float = 1e-8


def _safe_div(numer: pd.Series, denom: pd.Series, eps: float) -> pd.Series:
    return numer / (denom.replace(0, np.nan) + eps)


def compute_core_ratios(
    income_stmt: pd.DataFrame,
    balance_sheet: pd.DataFrame,
    cash_flow: pd.DataFrame,
    cfg: RatioConfig | None = None,
) -> pd.DataFrame:
    """
    Compute common financial ratios from statements.

    Expected columns (case-sensitive):
      income_stmt: ["fiscalDateEnding", "totalRevenue", "grossProfit",
                    "ebit", "ebitda", "netIncome"]
      balance_sheet: ["fiscalDateEnding", "totalAssets", "totalLiabilities",
                      "totalShareholderEquity", "cashAndCashEquivalentsAtCarryingValue",
                      "shortTermDebt", "longTermDebt"]
      cash_flow: ["fiscalDateEnding", "operatingCashflow", "capitalExpenditures"]
    """
    cfg = cfg or RatioConfig()

    inc = income_stmt.copy()
    bal = balance_sheet.copy()
    cfs = cash_flow.copy()

    for df in (inc, bal, cfs):
        if "fiscalDateEnding" in df.columns:
            df["fiscalDateEnding"] = pd.to_datetime(df["fiscalDateEnding"])

    merged = inc.merge(bal, on="fiscalDateEnding", how="inner").merge(
        cfs, on="fiscalDateEnding", how="inner"
    )

    merged["debt_total"] = merged.get("shortTermDebt", 0) + merged.get("longTermDebt", 0)
    merged["net_debt"] = merged["debt_total"] - merged.get(
        "cashAndCashEquivalentsAtCarryingValue", 0
    )

    merged["ebitda_margin"] = _safe_div(merged["ebitda"], merged["totalRevenue"], cfg.eps)
    merged["ebit_margin"] = _safe_div(merged["ebit"], merged["totalRevenue"], cfg.eps)
    merged["net_margin"] = _safe_div(merged["netIncome"], merged["totalRevenue"], cfg.eps)
    merged["gross_margin"] = _safe_div(merged["grossProfit"], merged["totalRevenue"], cfg.eps)

    merged["debt_to_equity"] = _safe_div(
        merged["debt_total"], merged["totalShareholderEquity"], cfg.eps
    )
    merged["debt_to_assets"] = _safe_div(merged["debt_total"], merged["totalAssets"], cfg.eps)
    merged["net_debt_to_ebitda"] = _safe_div(merged["net_debt"], merged["ebitda"], cfg.eps)

    merged["fcf"] = merged["operatingCashflow"] - merged["capitalExpenditures"]
    merged["fcf_margin"] = _safe_div(merged["fcf"], merged["totalRevenue"], cfg.eps)

    return merged.sort_values("fiscalDateEnding").reset_index(drop=True)


def standardize_features(df: pd.DataFrame, feature_cols: list[str]) -> pd.DataFrame:
    """
    Z-score standardization for numeric columns. Returns a new DataFrame.
    """
    out = df.copy()
    for col in feature_cols:
        if col not in out.columns:
            continue
        mean = out[col].mean()
        std = out[col].std(ddof=0)
        if std == 0 or np.isnan(std):
            out[col] = 0.0
        else:
            out[col] = (out[col] - mean) / std
    return out


def select_feature_columns() -> Dict[str, list[str]]:
    """
    Centralized feature list to keep training/inference consistent.
    """
    return {
        "base": [
            "totalRevenue",
            "grossProfit",
            "ebit",
            "ebitda",
            "netIncome",
            "totalAssets",
            "totalLiabilities",
            "totalShareholderEquity",
            "cashAndCashEquivalentsAtCarryingValue",
            "debt_total",
            "net_debt",
            "operatingCashflow",
            "capitalExpenditures",
            "fcf",
        ],
        "ratios": [
            "ebitda_margin",
            "ebit_margin",
            "net_margin",
            "gross_margin",
            "debt_to_equity",
            "debt_to_assets",
            "net_debt_to_ebitda",
            "fcf_margin",
        ],
    }
