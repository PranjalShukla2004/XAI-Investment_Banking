#!/usr/bin/env python3
"""
build_combined_dataset.py

Combine annual financial statements from three ZIPs (cash flow, balance sheet, income statement)
and compute derived metrics used in acquisition/risk analysis.

Outputs:
  - combined_financials_long.csv  : One row per ticker per fiscal year (normalized metrics).
  - combined_latest_with_market.csv: One row per ticker (latest annual + market data if available).

Usage:
  python build_combined_dataset.py \
      --cashflow_zip /path/to/us-cashflow-annual.zip \
      --balance_zip  /path/to/us-balance-annual.zip  \
      --income_zip   /path/to/us-income-annual.zip   \
      --tickers AAPL MSFT GOOGL   \
      --years 5 \
      --outdir ./out

Notes:
- Market data (volatility, 1y return, P/E) requires internet. If yfinance cannot fetch,
  those fields will be left empty with warnings.
- Column names differ across providers; this script performs case-insensitive matching
  and uses reasonable fallbacks where possible.
"""

import argparse, os, zipfile, io, json, math, sys, warnings
from typing import Dict, List, Optional, Tuple
import pandas as pd

# -------------------------- Helpers: flexible column access --------------------------

def pick_col(d: Dict[str, any], candidates: List[str]):
    """Pick the first present (case-insensitive) column name from candidates; returns (key, value) or (None, None)."""
    lower = {k.lower(): k for k in d.keys()}
    for c in candidates:
        k = lower.get(c.lower())
        if k in d:
            return k, d[k]
    return None, None

def get_num(row: Dict[str, any], candidates: List[str]) -> Optional[float]:
    """Fetch numeric value from row for any candidate key. Returns float or None."""
    _, v = pick_col(row, candidates)
    if v is None or (isinstance(v, float) and (math.isnan(v) or v is None)):
        return None
    # Coerce strings like "123456" or "123,456"
    try:
        if isinstance(v, str):
            v = v.replace(",", "")
        return float(v)
    except Exception:
        try:
            return pd.to_numeric(v)
        except Exception:
            return None

def coalesce_str(row: Dict[str, any], candidates: List[str]) -> Optional[str]:
    _, v = pick_col(row, candidates)
    return None if v is None else str(v)

# -------------------------- Loaders for ZIP members --------------------------

def read_zip_table(zpath: str) -> pd.DataFrame:
    """Read a ZIP that contains per-ticker CSV/JSON annual statements, return a long dataframe with symbol+fiscalDateEnding."""
    rows: List[pd.DataFrame] = []
    with zipfile.ZipFile(zpath, 'r') as z:
        for name in z.namelist():
            if name.lower().endswith(".csv"):
                try:
                    df = pd.read_csv(z.open(name))
                    # Try to add symbol from filename if not present
                    if "symbol" not in df.columns and "Symbol" not in df.columns:
                        # crude parse: /AAPL.csv or /AAPL.json -> "AAPL"
                        base = os.path.basename(name)
                        sym = os.path.splitext(base)[0]
                        df["symbol"] = sym
                    rows.append(df)
                except Exception as e:
                    warnings.warn(f"Failed to read CSV {name} from {os.path.basename(zpath)}: {e}")
            elif name.lower().endswith(".json"):
                try:
                    raw = json.load(z.open(name))
                    if isinstance(raw, dict) and "annualReports" in raw:
                        df = pd.json_normalize(raw["annualReports"])
                    elif isinstance(raw, list):
                        df = pd.json_normalize(raw)
                    elif isinstance(raw, dict):
                        df = pd.json_normalize(raw)
                    else:
                        continue
                    if "symbol" not in df.columns and "Symbol" not in df.columns:
                        base = os.path.basename(name)
                        sym = os.path.splitext(base)[0]
                        df["symbol"] = sym
                    rows.append(df)
                except Exception as e:
                    warnings.warn(f"Failed to read JSON {name} from {os.path.basename(zpath)}: {e}")
    if not rows:
        raise RuntimeError(f"No readable CSV/JSON members found in {zpath}")
    out = pd.concat(rows, ignore_index=True)
    # Normalize column names: strip spaces
    out.columns = [c.strip() for c in out.columns]
    # Ensure fiscal date column exists
    # Common names: fiscalDateEnding, fiscalDate, date, periodEnding
    date_col = None
    for c in ["fiscalDateEnding", "fiscalDate", "date", "periodEnding", "calendarYear"]:
        if c in out.columns:
            date_col = c
            break
    if date_col is None:
        # fabricate a date order if needed
        out["fiscalDateEnding"] = None
    return out

# -------------------------- Statement merge & metrics --------------------------

def normalize_statement(df: pd.DataFrame, tag: str) -> pd.DataFrame:
    """Keep essential keys and add a 'stmt' tag to identify type."""
    df = df.copy()
    df["__stmt__"] = tag
    return df

def compute_metrics_per_row(row: pd.Series, cash: Dict[str, any], bal: Dict[str, any], inc: Dict[str, any]) -> Dict[str, any]:
    """
    Compute derived metrics using available fields with fallbacks:
      - working_capital, current_ratio
      - total_liabilities
      - retained_earnings
      - ebit, ebitda, ebitda_margin
      - sales (revenue)
    """
    out = {}

    # Working capital & liquidity
    ca = get_num(bal, ["totalCurrentAssets", "currentAssets", "CurrentAssets"])
    cl = get_num(bal, ["totalCurrentLiabilities", "currentLiabilities", "CurrentLiabilities"])
    if ca is not None and cl is not None:
        out["working_capital"] = ca - cl
        out["current_ratio"] = (ca / cl) if cl not in (0, None) else None

    # Solvency / leverage
    out["total_liabilities"] = get_num(bal, ["totalLiabilities", "TotalLiabilities", "liabilities"])
    out["retained_earnings"] = get_num(bal, ["retainedEarnings", "RetainedEarnings", "retainedEarningsAccumulated"])

    # Profitability
    ebit = get_num(inc, ["ebit", "EBIT", "operatingIncome", "OperatingIncome"])
    out["ebit"] = ebit

    # EBITDA from income if present, else operatingIncome + D&A from cash flow
    ebitda_inc = get_num(inc, ["ebitda", "EBITDA"])
    da_cf = get_num(cash, ["depreciationAndAmortization", "DepreciationAmortization", "depreciationAndAmortizationDepreciation"])
    if ebitda_inc is not None:
        out["ebitda"] = ebitda_inc
    else:
        oi = get_num(inc, ["operatingIncome", "OperatingIncome", "ebit"])
        out["ebitda"] = (oi + da_cf) if (oi is not None and da_cf is not None) else None

    # Sales / Revenue
    sales = get_num(inc, ["totalRevenue", "revenue", "Sales", "salesRevenueNet", "Revenue"])
    out["sales"] = sales

    # EBITDA margin
    if out.get("ebitda") is not None and sales not in (None, 0):
        out["ebitda_margin"] = out["ebitda"] / sales

    return out

def build_long_table(cash_df: pd.DataFrame, bal_df: pd.DataFrame, inc_df: pd.DataFrame, years: Optional[int]) -> pd.DataFrame:
    """Merge statements into a long table keyed by symbol + fiscal date, compute metrics."""
    # Normalize
    cash_df = normalize_statement(cash_df, "cashflow")
    bal_df  = normalize_statement(bal_df , "balance")
    inc_df  = normalize_statement(inc_df , "income")

    # Identify date column
    def find_date_col(df):
        for c in ["fiscalDateEnding", "fiscalDate", "date", "periodEnding", "calendarYear"]:
            if c in df.columns:
                return c
        return None

    # Prepare for merge
    for df in (cash_df, bal_df, inc_df):
        if "symbol" not in df.columns and "Symbol" in df.columns:
            df.rename(columns={"Symbol": "symbol"}, inplace=True)
        if "symbol" not in df.columns:
            raise RuntimeError("Missing 'symbol' column in a statement file.")
        # Select useful columns (avoid exploding memory)
        # Keep everything, but ensure symbol/date present
        pass

    date_cols = { "cash": find_date_col(cash_df), "bal": find_date_col(bal_df), "inc": find_date_col(inc_df) }
    # Rename date columns to a common name for merge
    if date_cols["cash"]: cash_df.rename(columns={date_cols["cash"]:"fiscalDateEnding"}, inplace=True)
    if date_cols["bal"] : bal_df .rename(columns={date_cols["bal"] :"fiscalDateEnding"}, inplace=True)
    if date_cols["inc"] : inc_df .rename(columns={date_cols["inc"] :"fiscalDateEnding"}, inplace=True)

    # Inner/outer merge strategy: use outer to keep data even if one statement missing
    merged = pd.merge(bal_df, inc_df, on=["symbol","fiscalDateEnding"], how="outer", suffixes=("_bal", "_inc"))
    merged = pd.merge(merged, cash_df, on=["symbol","fiscalDateEnding"], how="outer", suffixes=("", "_cash"))

    # Optionally limit years per symbol (most recent N)
    if years is not None:
        def _take_most_recent(group):
            # Try to parse date
            if "fiscalDateEnding" in group.columns:
                # Coerce to datetime where possible
                g = group.copy()
                g["_dt"] = pd.to_datetime(g["fiscalDateEnding"], errors="coerce")
                g = g.sort_values(["_dt", "fiscalDateEnding"], ascending=[False, False])
                return g.head(years).drop(columns=["_dt"])
            return group
        merged = merged.groupby("symbol", group_keys=False).apply(_take_most_recent)

    # Compute derived metrics per row
    metric_rows = []
    cols = merged.columns
    for _, row in merged.iterrows():
        cash = row.to_dict()
        bal  = row.to_dict()
        inc  = row.to_dict()
        metrics = compute_metrics_per_row(row, cash, bal, inc)
        base = {
            "symbol": row.get("symbol"),
            "fiscalDateEnding": row.get("fiscalDateEnding"),
        }
        base.update(metrics)
        metric_rows.append(base)
    metrics_df = pd.DataFrame(metric_rows)

    # Ensure expected columns exist
    expected_cols = ["sales","ebit","ebitda","ebitda_margin","working_capital","current_ratio","total_liabilities","retained_earnings"]
    for ec in expected_cols:
        if ec not in metrics_df.columns:
            metrics_df[ec] = None

    # Attach company name if present
    cname_col = None
    for c in ["name", "companyName", "CompanyName", "longName"]:
        if c in merged.columns:
            cname_col = c
            break
    if cname_col:
        # best non-null value per symbol
        name_map = merged.groupby("symbol")[cname_col].agg(lambda s: s.dropna().iloc[0] if len(s.dropna()) else None)
        metrics_df = metrics_df.merge(name_map.rename("company_name"), on="symbol", how="left")
    else:
        metrics_df["company_name"] = None

    # Keep order
    metrics_df = metrics_df[["symbol","company_name","fiscalDateEnding","sales","ebit","ebitda","ebitda_margin","working_capital","current_ratio","total_liabilities","retained_earnings"]]
    return metrics_df

# -------------------------- Market data (optional via yfinance) --------------------------
# -------------------------- Companies metadata (optional) --------------------------


def read_company_zip(zpath: str) -> pd.DataFrame:
    """Read companies ZIP (CSV/JSON). Returns one row per symbol with company_* columns + company_name_from_companies."""
    import io, json, zipfile
    def smart_read_member(z, name):
        raw = z.read(name)
        # Try common delimiters
        for sep in [",",";","\t","|"]:
            try:
                df = pd.read_csv(io.BytesIO(raw), sep=sep)
                # Heuristic: if it has > 1 column it's likely correct
                if df.shape[1] >= 2:
                    return df
            except Exception:
                pass
        # JSON fallback
        try:
            obj = json.loads(raw.decode("utf-8", errors="replace"))
            if isinstance(obj, list):
                return pd.json_normalize(obj)
            if isinstance(obj, dict):
                key = "data" if "data" in obj else None
                return pd.json_normalize(obj[key]) if key else pd.json_normalize(obj)
        except Exception:
            pass
        raise RuntimeError(f"Unable to parse {name} in {os.path.basename(zpath)}")

    frames = []
    with zipfile.ZipFile(zpath, "r") as z:
        for name in z.namelist():
            if not (name.lower().endswith(".csv") or name.lower().endswith(".json")):
                continue
            try:
                df = smart_read_member(z, name)
                # Inject symbol from filename if missing
                if "symbol" not in df.columns:
                    base = os.path.basename(name)
                    sym = os.path.splitext(base)[0]
                    # If the member is a single CSV for many companies (e.g., us-companies.csv), keep as-is
                    if df.shape[0] == 1:
                        df["symbol"] = sym
                frames.append(df)
            except Exception as e:
                warnings.warn(f"Failed to parse {name} in companies ZIP: {e}")
    if not frames:
        raise RuntimeError(f"No readable CSV/JSON members found in {zpath}")
    base = pd.concat(frames, ignore_index=True)

    # Normalize likely ticker column to 'symbol'
    ticker_candidates = ["symbol","Symbol","Ticker","ticker"]
    found = None
    for c in ticker_candidates:
        if c in base.columns:
            found = c
            break
    if found and found != "symbol":
        base = base.rename(columns={found: "symbol"})
    if "symbol" not in base.columns:
        raise RuntimeError("Companies ZIP missing 'symbol'/'Ticker' column and filename inference failed.")

    # Clean symbol
    base["symbol"] = base["symbol"].astype(str).str.strip().str.upper()

    # One row per symbol; first non-null per column
    def first_non_null(series: pd.Series):
        s = series.dropna()
        return s.iloc[0] if len(s) else None
    agg = {c: first_non_null for c in base.columns if c != "symbol"}
    comp = base.groupby("symbol", as_index=False).agg(agg)

    # Extract company name
    name_candidates = ["name","companyName","Company Name","CompanyName","longName","shortName"]
    name_col = next((c for c in name_candidates if c in comp.columns), None)
    comp["company_name_from_companies"] = comp[name_col] if name_col is not None else None

    # Prefix all other metadata columns with 'company_' (except symbol and company_name_from_companies)
    keep = [c for c in comp.columns if c not in ["symbol","company_name_from_companies"]]
    rename_map = {c: f"company_{c}" for c in keep}
    comp = comp.rename(columns=rename_map)
    # Friendly alias for company name
    if name_col is not None and f"company_{name_col}" in comp.columns:
        comp["company_name_companies"] = comp[f"company_{name_col}"]
    else:
        comp["company_name_companies"] = comp["company_name_from_companies"]

    comp = comp.drop_duplicates("symbol")
    return comp



def fetch_market_metrics(tickers: List[str]) -> pd.DataFrame:
    """
    Fetch volatility (annualized), 1-year return, and P/E using yfinance.
    Returns a DataFrame with columns: symbol, vol_1y, return_1y, pe.
    If yfinance is unavailable or offline, returns empty columns.
    """
    try:
        import yfinance as yf
    except Exception as e:
        warnings.warn(f"yfinance not available: {e}. Market metrics will be empty.")
        return pd.DataFrame({"symbol": tickers, "vol_1y": [None]*len(tickers), "return_1y": [None]*len(tickers), "pe": [None]*len(tickers), "company_name_mkt":[None]*len(tickers)})

    rows = []
    for t in tickers:
        try:
            tk = yf.Ticker(t)
            hist = tk.history(period="1y", auto_adjust=True)
            if hist is None or hist.empty:
                vol = ret = None
            else:
                rets = hist["Close"].pct_change().dropna()
                vol = float(rets.std() * (252**0.5)) if len(rets) > 2 else None
                ret = float(hist["Close"].iloc[-1] / hist["Close"].iloc[0] - 1.0) if len(hist) > 1 else None

            # Try P/E via fast_info or info
            pe = None
            try:
                fi = getattr(tk, "fast_info", None)
                info = tk.get_info() if hasattr(tk, "get_info") else getattr(tk, "info", {})
                if info and isinstance(info, dict):
                    pe = info.get("trailingPE") or info.get("forwardPE")
                if pe is None and fi and hasattr(fi, "__dict__"):
                    pe = getattr(fi, "trailing_pe", None) or getattr(fi, "pe_ratio", None)
            except Exception:
                pass

            cname = None
            try:
                info = tk.get_info() if hasattr(tk, "get_info") else getattr(tk, "info", {})
                if isinstance(info, dict):
                    cname = info.get("longName") or info.get("shortName")
            except Exception:
                pass

            rows.append({"symbol": t, "vol_1y": vol, "return_1y": ret, "pe": pe, "company_name_mkt": cname})
        except Exception as e:
            warnings.warn(f"Failed to fetch market data for {t}: {e}")
            rows.append({"symbol": t, "vol_1y": None, "return_1y": None, "pe": None, "company_name_mkt": None})
    return pd.DataFrame(rows)

# -------------------------- Main --------------------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cashflow_zip", required=True, help="./datasets/us-cashflow-annual.zip")
    ap.add_argument("--balance_zip",  required=True, help="./datasets/us-balance-annual.zip")
    ap.add_argument("--income_zip",   required=True, help="./datasets/us-income-annual.zip")
    ap.add_argument("--companies_zip", required=False, help="./datasets/us-companies.zip")
    ap.add_argument("--tickers", nargs="*", help="Optional list of tickers to restrict to")
    ap.add_argument("--years", type=int, default=None, help="Keep the most recent N years per ticker")
    ap.add_argument("--outdir", default=".", help="Output directory")
    args = ap.parse_args()

    os.makedirs(args.outdir, exist_ok=True)

    cash_df = read_zip_table(args.cashflow_zip)
    bal_df  = read_zip_table(args.balance_zip)
    inc_df  = read_zip_table(args.income_zip)

    # Optionally filter by tickers
    if args.tickers:
        tickset = set([t.upper() for t in args.tickers])
        cash_df = cash_df[cash_df["symbol"].str.upper().isin(tickset)]
        bal_df  = bal_df [bal_df ["symbol"].str.upper().isin(tickset)]
        inc_df  = inc_df [inc_df ["symbol"].str.upper().isin(tickset)]

    long_df = build_long_table(cash_df, bal_df, inc_df, years=args.years)

    # Merge companies metadata if provided
    companies_df = None
    if args.companies_zip:
        try:
            companies_df = read_company_zip(args.companies_zip)
            # Merge onto long
            long_df = long_df.merge(companies_df, on="symbol", how="left")
            # Prefer company name from companies ZIP, then from statements, then market
            if "company_name_from_companies" in long_df.columns:
                long_df["company_name"] = long_df["company_name_from_companies"].fillna(long_df["company_name"])
        except Exception as e:
            warnings.warn(f"Failed to merge companies ZIP: {e}")

    long_path = os.path.join(args.outdir, "combined_financials_long.csv")
    long_df.to_csv(long_path, index=False)

    # Latest-year snapshot per ticker + market metrics
    latest = long_df.copy()
    latest["_dt"] = pd.to_datetime(latest["fiscalDateEnding"], errors="coerce")
    latest = latest.sort_values(["symbol","_dt"], ascending=[True, False]).drop_duplicates("symbol").drop(columns=["_dt"])

    # Attach market metrics
    tickers = latest["symbol"].dropna().astype(str).str.upper().unique().tolist()
    mkt = fetch_market_metrics(tickers)
    final = latest.merge(mkt, on="symbol", how="left")

    # Merge companies metadata into final snapshot too (only if not already present)
    if args.companies_zip and companies_df is not None:
        already_has_company_cols = any(c.startswith("company_") for c in final.columns) or ("company_name_from_companies" in final.columns)
        if not already_has_company_cols:
            final = final.merge(companies_df, on="symbol", how="left")
        else:
            # If merged earlier, avoid re-merging. Also normalize potential _x/_y leftovers.
            dup_suffixes = ["_x","_y"]
            to_fix = {}
            for col in list(final.columns):
                for suf in dup_suffixes:
                    if col.endswith(suf):
                        base = col[:-2]
                        # Prefer the non-suffixed or _x first
                        if base in final.columns:
                            # drop this suffixed col
                            to_fix[col] = None
                        elif base+"_x" in final.columns:
                            final.rename(columns={base+"_x": base}, inplace=True)
                            to_fix[col] = None
                        else:
                            # rename this suffixed version to base
                            final.rename(columns={col: base}, inplace=True)
            if to_fix:
                final.drop(columns=[c for c in to_fix if c in final.columns], inplace=True, errors="ignore")

    # Prefer company name: companies ZIP > statements > market
    if "company_name_from_companies" in final.columns:
        final["company_name"] = final["company_name"].fillna(final["company_name_from_companies"])
    final["company_name"] = final["company_name"].fillna(final.get("company_name_mkt"))
    if "company_name_mkt" in final.columns:
        final = final.drop(columns=["company_name_mkt"])


    final_path = os.path.join(args.outdir, "combined_latest_with_market.csv")
    final.to_csv(final_path, index=False)

    print(f"Saved:\n - {long_path}\n - {final_path}")
    print("Columns in latest_with_market:")
    print(", ".join(final.columns))

if __name__ == "__main__":
    pd.options.display.width = 200
    pd.options.display.max_columns = 200
    main()
