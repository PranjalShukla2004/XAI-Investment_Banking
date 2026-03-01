#!/usr/bin/env python3
"""
Build final ML dataset from Massive raw JSON dumps:
- data/raw/balance-sheets/{TICKER}.json
- data/raw/income-statements/{TICKER}.json
- data/raw/cash-flow-statements/{TICKER}.json

Outputs:
- data/processed/final_features_dataset.csv

Computed features (annual):
Liquidity:
  current_ratio = total_current_assets / total_current_liabilities
  quick_ratio   = (cash_and_equivalents + receivables) / total_current_liabilities
  working_capital_to_assets = (total_current_assets - total_current_liabilities) / total_assets

Leverage & solvency:
  debt_to_equity = total_debt / total_equity
  liabilities_to_assets = total_liabilities / total_assets
  interest_coverage = EBIT / interest_expense
  net_debt = total_debt - cash_and_equivalents

Profitability:
  gross_margin = gross_profit / revenue
  operating_margin = operating_income / revenue
  net_margin = net_income / revenue
  roa = net_income / total_assets
  roe = net_income / total_equity

Cash generation:
  fcf_margin = fcf / revenue
  cfo_to_net_income = cfo / net_income
  capex_to_revenue = capex / revenue

Notes:
- EBIT is approximated as operating_income (common in many datasets).
- total_debt = debt_current + long_term_debt_and_capital_lease_obligations (if present).
- CFO uses net_cash_from_operating_activities, else cash_from_operating_activities_continuing_operations.
- Capex uses purchase_of_property_plant_and_equipment (usually negative cash flow). We convert to positive capex = -value if negative.
- Div-by-zero and missing fields => NaN.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any, Dict, List, Optional
import pandas as pd


# -----------------------------
# IO helpers
# -----------------------------
def load_massive_file(path: Path) -> List[Dict[str, Any]]:
    """
    Your saved files look like:
      {"status":"OK","ticker":"...","timeframe":"annual",...,"results":[{...},{...}]}
    We return the list under "results" (or [] if missing).
    """
    obj = json.loads(path.read_text(encoding="utf-8"))
    if isinstance(obj, dict) and "results" in obj and isinstance(obj["results"], list):
        return obj["results"]
    # fallback: sometimes you might save raw API response with "results"
    if isinstance(obj, dict) and "results" in obj:
        return list(obj["results"] or [])
    return []


def safe_div(num: Any, den: Any) -> float:
    try:
        if num is None or den is None:
            return float("nan")
        num_f = float(num)
        den_f = float(den)
        if den_f == 0.0 or math.isclose(den_f, 0.0):
            return float("nan")
        return num_f / den_f
    except Exception:
        return float("nan")


def to_float(x: Any) -> float:
    try:
        return float(x) if x is not None else float("nan")
    except Exception:
        return float("nan")


def pick_first(row: Dict[str, Any], keys: List[str]) -> Any:
    for k in keys:
        v = row.get(k, None)
        if v is not None:
            return v
    return None


# -----------------------------
# Feature computation per year
# -----------------------------
def compute_features(
    bs: Dict[str, Any],
    inc: Dict[str, Any],
    cf: Dict[str, Any],
) -> Dict[str, Any]:
    # --- Balance sheet fields ---
    current_assets = bs.get("total_current_assets")
    current_liab = bs.get("total_current_liabilities")
    total_assets = bs.get("total_assets")
    total_liab = bs.get("total_liabilities")
    equity = pick_first(bs, ["total_equity", "total_equity_attributable_to_parent"])
    cash = bs.get("cash_and_equivalents")
    receivables = bs.get("receivables")

    debt_current = bs.get("debt_current")
    debt_long = bs.get("long_term_debt_and_capital_lease_obligations")
    total_debt = (
        (to_float(debt_current) if debt_current is not None else 0.0)
        + (to_float(debt_long) if debt_long is not None else 0.0)
    )
    # if both were None, make it NaN rather than 0
    if debt_current is None and debt_long is None:
        total_debt = float("nan")

    # --- Income statement fields ---
    revenue = inc.get("revenue")
    gross_profit = inc.get("gross_profit")
    operating_income = inc.get("operating_income")  # used as EBIT proxy
    # your sample uses consolidated_net_income_loss for net income :contentReference[oaicite:3]{index=3}
    net_income = pick_first(inc, ["consolidated_net_income_loss", "net_income"])

    interest_expense = inc.get("interest_expense")

    # --- Cash flow fields ---
    cfo = pick_first(
        cf,
        [
            "net_cash_from_operating_activities",
            "cash_from_operating_activities_continuing_operations",
        ],
    )

    ppe_purchase = cf.get("purchase_of_property_plant_and_equipment")

    # Capex convention: purchase_of_PPE is usually negative (cash outflow).
    # We want capex as a positive magnitude.
    capex = float("nan")
    if ppe_purchase is not None:
        p = to_float(ppe_purchase)
        if not math.isnan(p):
            capex = -p if p < 0 else p

    # Free cash flow: FCF = CFO - Capex
    fcf = float("nan")
    if cfo is not None and not math.isnan(to_float(cfo)) and not math.isnan(capex):
        fcf = to_float(cfo) - capex

    # Interest coverage: EBIT / interest_expense
    # Some data sources encode interest_expense with weird signs.
    # We use abs(interest_expense) as denominator if non-zero.
    interest_cov = float("nan")
    if operating_income is not None and interest_expense is not None:
        denom = abs(to_float(interest_expense))
        if denom and not math.isclose(denom, 0.0):
            interest_cov = to_float(operating_income) / denom

    # --- Ratios ---
    out = {
        # Liquidity
        "current_ratio": safe_div(current_assets, current_liab),
        "quick_ratio": safe_div(
            (to_float(cash) if cash is not None else float("nan"))
            + (to_float(receivables) if receivables is not None else float("nan")),
            current_liab,
        ),
        "working_capital_to_assets": safe_div(
            (to_float(current_assets) - to_float(current_liab))
            if current_assets is not None and current_liab is not None
            else float("nan"),
            total_assets,
        ),
        # Leverage & solvency
        "debt_to_equity": safe_div(total_debt, equity),
        "liabilities_to_assets": safe_div(total_liab, total_assets),
        "interest_coverage": interest_cov,
        "net_debt": (
            to_float(total_debt) - to_float(cash)
            if not math.isnan(to_float(total_debt)) and cash is not None
            else float("nan")
        ),
        # Profitability
        "gross_margin": safe_div(gross_profit, revenue),
        "operating_margin": safe_div(operating_income, revenue),
        "net_margin": safe_div(net_income, revenue),
        "roa": safe_div(net_income, total_assets),
        "roe": safe_div(net_income, equity),
        # Cash generation
        "fcf_margin": safe_div(fcf, revenue),
        "cfo_to_net_income": safe_div(cfo, net_income),
        "capex_to_revenue": safe_div(capex, revenue),
        # Helpful raw fields for debugging / modeling
        "revenue": to_float(revenue),
        "net_income": to_float(net_income),
        "operating_income": to_float(operating_income),
        "total_assets": to_float(total_assets),
        "total_liabilities": to_float(total_liab),
        "total_equity": to_float(equity),
        "cash_and_equivalents": to_float(cash),
        "total_debt": to_float(total_debt),
        "cfo": to_float(cfo),
        "capex": to_float(capex),
        "fcf": to_float(fcf),
    }
    return out


# -----------------------------
# Main merge logic
# -----------------------------
def index_by_year(rows: List[Dict[str, Any]]) -> Dict[int, Dict[str, Any]]:
    out: Dict[int, Dict[str, Any]] = {}
    for r in rows:
        fy = r.get("fiscal_year")
        if fy is None:
            continue
        try:
            out[int(fy)] = r
        except Exception:
            continue
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--raw", type=str, default="data/raw", help="Raw root (default: data/raw)")
    ap.add_argument("--out", type=str, default="data/processed/final_features_dataset.csv")
    ap.add_argument("--min_years", type=int, default=1, help="Require at least this many annual rows per ticker")
    args = ap.parse_args()

    raw_root = Path(args.raw)
    bs_dir = raw_root / "balance-sheets"
    is_dir = raw_root / "income-statements"
    cf_dir = raw_root / "cash-flow-statements"

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    # Discover tickers by balance sheet files (you can change to intersection if you want strict)
    tickers = sorted(p.stem.upper() for p in bs_dir.glob("*.json"))

    rows_out: List[Dict[str, Any]] = []
    skipped = 0

    for t in tickers:
        bs_path = bs_dir / f"{t}.json"
        is_path = is_dir / f"{t}.json"
        cf_path = cf_dir / f"{t}.json"

        if not bs_path.exists():
            continue

        bs_rows = load_massive_file(bs_path)
        inc_rows = load_massive_file(is_path) if is_path.exists() else []
        cf_rows = load_massive_file(cf_path) if cf_path.exists() else []

        bs_by_year = index_by_year(bs_rows)
        inc_by_year = index_by_year(inc_rows)
        cf_by_year = index_by_year(cf_rows)

        years = sorted(bs_by_year.keys())
        if len(years) < args.min_years:
            skipped += 1
            continue

        for y in years:
            bs = bs_by_year.get(y, {})
            inc = inc_by_year.get(y, {})
            cf = cf_by_year.get(y, {})

            # Basic metadata
            period_end = bs.get("period_end") or inc.get("period_end") or cf.get("period_end")
            cik = bs.get("cik") or inc.get("cik") or cf.get("cik")

            feats = compute_features(bs, inc, cf)
            feats.update(
                {
                    "ticker": t,
                    "fiscal_year": y,
                    "period_end": period_end,
                    "cik": cik,
                    "timeframe": "annual",
                    "has_income_statement": int(bool(inc)),
                    "has_cash_flow": int(bool(cf)),
                }
            )
            rows_out.append(feats)

    df = pd.DataFrame(rows_out)

    # Optional: sort and de-duplicate
    if not df.empty:
        df = df.sort_values(["ticker", "fiscal_year"]).drop_duplicates(subset=["ticker", "fiscal_year"], keep="last")

    df.to_csv(out_path, index=False)
    print(f"Saved: {out_path} | rows={len(df):,} | tickers={df['ticker'].nunique() if not df.empty else 0:,}")
    if skipped:
        print(f"Skipped tickers with < {args.min_years} years: {skipped}")


if __name__ == "__main__":
    main()
