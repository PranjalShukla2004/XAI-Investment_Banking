#!/usr/bin/env python3
"""
Fetch last 5 years of ANNUAL financial statements from Massive API for all tickers in tickers_3000.

Per ticker, fetch in this order:
1) balance sheets
2) income statements
3) cash flow statements

Saves raw JSON to:
data/raw/balance-sheets/{TICKER}.json
data/raw/income-statements/{TICKER}.json
data/raw/cash-flow-statements/{TICKER}.json

Usage:
  export MASSIVE_API_KEY="YOUR_KEY"
  python src/scripts/fetch_massive_annual_5y.py --tickers tickers_3000 --out data/raw --years 5

Notes:
- Uses timeframe="annual"
- Pulls enough rows then trims to the most recent N fiscal_years
- Processes tickers sequentially (one ticker fully, then next)
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

from massive import RESTClient
from dotenv import load_dotenv


load_dotenv()

api_key = (os.getenv("MASSIVE_API_KEY") or "").strip()
if not api_key:
    raise SystemExit("Missing MASSIVE_API_KEY in environment/.env")

# -----------------------------
# Ticker loading
# -----------------------------
def load_tickers(path: Path) -> List[str]:
    if not path.exists():
        raise FileNotFoundError(f"Tickers file not found: {path}")

    ext = path.suffix.lower()

    tickers: List[str] = []
    if ext in (".txt", ""):
        # one ticker per line, allow commas/spaces too
        text = path.read_text(encoding="utf-8", errors="ignore")
        raw = []
        for line in text.splitlines():
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            raw.extend([t.strip() for t in line.replace(",", " ").split() if t.strip()])
        tickers = raw

    elif ext == ".csv":
        with path.open("r", encoding="utf-8", newline="") as f:
            reader = csv.DictReader(f)
            # try common column names; else fallback to first column
            cols = [c.lower() for c in (reader.fieldnames or [])]
            preferred = None
            for c in ("ticker", "tickers", "symbol"):
                if c in cols:
                    preferred = reader.fieldnames[cols.index(c)]
                    break
            for row in reader:
                if not row:
                    continue
                val = row.get(preferred) if preferred else next(iter(row.values()), "")
                if val:
                    tickers.append(val.strip())

    elif ext == ".json":
        obj = json.loads(path.read_text(encoding="utf-8"))
        if isinstance(obj, list):
            tickers = [str(x).strip() for x in obj if str(x).strip()]
        elif isinstance(obj, dict):
            # allow {"tickers":[...]} or {"symbols":[...]}
            for key in ("tickers", "symbols", "ticker_list"):
                if key in obj and isinstance(obj[key], list):
                    tickers = [str(x).strip() for x in obj[key] if str(x).strip()]
                    break
            if not tickers:
                raise ValueError(f"JSON format not recognized in {path}. Expected list or dict with 'tickers'.")
        else:
            raise ValueError(f"JSON format not recognized in {path}. Expected list/dict.")

    else:
        raise ValueError(f"Unsupported tickers file type: {ext} (use .txt/.csv/.json)")

    # normalize, dedupe, keep order
    out: List[str] = []
    seen = set()
    for t in tickers:
        t = t.upper().strip()
        if not t or t in seen:
            continue
        seen.add(t)
        out.append(t)
    return out


# -----------------------------
# Fetch helpers
# -----------------------------
def _coerce_results(resp: Any) -> Dict[str, Any]:
    """
    RESTClient may return dict-like objects. Normalize to plain dict for JSON saving.
    """
    if isinstance(resp, dict):
        return resp
    # try common conversions
    if hasattr(resp, "to_dict"):
        return resp.to_dict()
    try:
        return json.loads(json.dumps(resp, default=lambda o: getattr(o, "__dict__", str(o))))
    except Exception:
        return {"raw": str(resp)}


def _select_last_n_years(results: Sequence[Dict[str, Any]], n_years: int) -> List[Dict[str, Any]]:
    """
    Keep rows for the most recent n unique fiscal_year values.
    Assumes results are sorted descending by period_end already.
    """
    kept: List[Dict[str, Any]] = []
    years_seen: List[int] = []

    for row in results:
        fy = row.get("fiscal_year")
        if fy is None:
            # if missing, keep anyway but it won't help with "last 5 years" logic
            kept.append(row)
            continue
        try:
            fy_i = int(fy)
        except Exception:
            kept.append(row)
            continue

        if fy_i not in years_seen:
            years_seen.append(fy_i)
        if len(years_seen) > n_years:
            break
        kept.append(row)

    return kept


def fetch_endpoint_with_retry(
    fetch_fn,
    *,
    ticker: str,
    years: int,
    limit: int,
    sleep_s: float,
    max_retries: int,
) -> Dict[str, Any]:
    """
    Calls a Massive RESTClient list_* generator, collects results, trims to last N years.
    """
    attempt = 0
    last_err: Optional[Exception] = None

    while attempt <= max_retries:
        try:
            # Massive SDK uses generator-style iteration in your examples
            rows: List[Dict[str, Any]] = []
            for item in fetch_fn(
                tickers=ticker,
                timeframe="annual",
                limit=limit,
                sort="period_end.desc",  # get most recent first
            ):
                # item may be dict-like
                if isinstance(item, dict):
                    rows.append(item)
                elif hasattr(item, "to_dict"):
                    rows.append(item.to_dict())
                else:
                    rows.append(_coerce_results(item))

            trimmed = _select_last_n_years(rows, years)

            return {
                "status": "OK",
                "ticker": ticker,
                "timeframe": "annual",
                "requested_years": years,
                "returned_rows": len(rows),
                "kept_rows": len(trimmed),
                "results": trimmed,
            }

        except Exception as e:
            last_err = e
            attempt += 1
            if attempt > max_retries:
                break
            # exponential-ish backoff
            time.sleep(sleep_s * (2 ** (attempt - 1)))

    return {
        "status": "ERROR",
        "ticker": ticker,
        "timeframe": "annual",
        "requested_years": years,
        "error": repr(last_err) if last_err else "unknown error",
        "results": [],
    }


# -----------------------------
# Main script
# -----------------------------
def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--tickers", type=str, required=True, help="Path to tickers_3000 (.txt/.csv/.json)")
    ap.add_argument("--out", type=str, default="data/raw", help="Output root folder (default: data/raw)")
    ap.add_argument("--years", type=int, default=5, help="How many years to keep (default: 5)")
    ap.add_argument("--limit", type=int, default=200, help="API limit per request (default: 200)")
    ap.add_argument("--sleep", type=float, default=0.15, help="Sleep between requests (default: 0.15s)")
    ap.add_argument("--retries", type=int, default=3, help="Retries per endpoint on failure (default: 3)")
    args = ap.parse_args()

    api_key = (os.getenv("MASSIVE_API_KEY") or "").strip()
    if not api_key:
        raise SystemExit("Missing MASSIVE_API_KEY env var. Example: export MASSIVE_API_KEY='...'\n")

    tickers_path = Path(args.tickers)
    out_root = Path(args.out)

    tickers = load_tickers(tickers_path)
    if not tickers:
        raise SystemExit(f"No tickers found in {tickers_path}")

    # output dirs
    bs_dir = out_root / "balance-sheets"
    is_dir = out_root / "income-statements"
    cf_dir = out_root / "cash-flow-statements"
    for d in (bs_dir, is_dir, cf_dir):
        d.mkdir(parents=True, exist_ok=True)

    client = RESTClient(api_key)

    total = len(tickers)
    for idx, ticker in enumerate(tickers, start=1):
        print(f"[{idx}/{total}] {ticker}: fetching annual last {args.years} years...")

        # 1) Balance Sheets
        bs_payload = fetch_endpoint_with_retry(
            client.list_financials_balance_sheets,
            ticker=ticker,
            years=args.years,
            limit=args.limit,
            sleep_s=args.sleep,
            max_retries=args.retries,
        )
        (bs_dir / f"{ticker}.json").write_text(json.dumps(bs_payload, indent=2), encoding="utf-8")
        time.sleep(args.sleep)

        # 2) Income Statements
        is_payload = fetch_endpoint_with_retry(
            client.list_financials_income_statements,
            ticker=ticker,
            years=args.years,
            limit=args.limit,
            sleep_s=args.sleep,
            max_retries=args.retries,
        )
        (is_dir / f"{ticker}.json").write_text(json.dumps(is_payload, indent=2), encoding="utf-8")
        time.sleep(args.sleep)

        # 3) Cash Flow Statements
        cf_payload = fetch_endpoint_with_retry(
            client.list_financials_cash_flow_statements,
            ticker=ticker,
            years=args.years,
            limit=args.limit,
            sleep_s=args.sleep,
            max_retries=args.retries,
        )
        (cf_dir / f"{ticker}.json").write_text(json.dumps(cf_payload, indent=2), encoding="utf-8")
        time.sleep(args.sleep)

        # simple per-ticker summary
        ok_count = sum(1 for p in (bs_payload, is_payload, cf_payload) if p.get("status") == "OK")
        print(f"    done: {ticker} ({ok_count}/3 endpoints OK)")

    print("\nAll done.")
    print(f"Saved to: {out_root.resolve()}")


if __name__ == "__main__":
    main()
