#!/usr/bin/env python3
"""
Fetch ALL available tickers via Massive RESTClient.list_tickers() and save to:
  data/raw/tickers/final_tickers.txt

Usage:
  python src/data_fetch/fetch_final_tickers.py

Env:
  MASSIVE_API_KEY=...
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Optional

from dotenv import load_dotenv
from massive import RESTClient


OUT_PATH = Path("data/raw/tickers/final_tickers.txt")


def extract_ticker(item: Any) -> Optional[str]:
    """
    RESTClient.list_tickers may yield:
      - dicts: {"ticker": "..."} or {"symbol": "..."}
      - objects with .ticker / .symbol
      - strings
    This function normalizes to uppercase ticker string.
    """
    if item is None:
        return None

    # If it's already a string
    if isinstance(item, str):
        t = item.strip()
        return t.upper() if t else None

    # If it's a dict-like record
    if isinstance(item, dict):
        t = item.get("ticker") or item.get("symbol")
        if not t:
            return None
        t = str(t).strip()
        return t.upper() if t else None

    # If it's an object with attributes
    for attr in ("ticker", "symbol"):
        if hasattr(item, attr):
            t = getattr(item, attr)
            if t is None:
                continue
            t = str(t).strip()
            return t.upper() if t else None

    # Fallback: last resort string conversion
    t = str(item).strip()
    return t.upper() if t else None


def main() -> None:
    load_dotenv()

    api_key = (os.getenv("MASSIVE_API_KEY") or "").strip()
    if not api_key:
        raise SystemExit("Missing MASSIVE_API_KEY. Put it in your .env or export it.")

    client = RESTClient(api_key)

    # Params mirror your screenshot (strings are fine)
    params = dict(
        market="stocks",
        active="true",
        order="asc",
        limit="1000",   # if API supports bigger pages, try "1000"
        sort="ticker",
    )

    tickers = []
    seen = set()

    # RESTClient.list_tickers is expected to handle pagination internally
    for rec in client.list_tickers(**params):
        t = extract_ticker(rec)
        if not t or t in seen:
            continue
        seen.add(t)
        tickers.append(t)

    tickers.sort()

    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    OUT_PATH.write_text("\n".join(tickers) + "\n", encoding="utf-8")

    print(f"Saved {len(tickers)} tickers to: {OUT_PATH}")
    if tickers:
        print("Sample:", ", ".join(tickers[:20]))


if __name__ == "__main__":
    main()
