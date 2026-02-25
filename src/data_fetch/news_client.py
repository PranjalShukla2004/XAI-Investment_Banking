#!/usr/bin/env python3
"""
Fetch Massive API:
  News for a ticker

Saves to:
  data/raw/news/<TICKER>/news.jsonl
  data/raw/news/<TICKER>/meta.json

Usage examples:
  export MASSIVE_API_KEY="..."
  python -m src.data_fetch.news_client --ticker UBS
  python -m src.data_fetch.news_client --ticker UBS --max-news-items 7 --news-since 2024-01-01
  python -m src.data_fetch.news_client --tickers-file data/raw/tickers/final_tickers.txt
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, Optional, Set
import requests

try:
    from dotenv import load_dotenv
except ImportError:  # optional dependency
    def load_dotenv() -> bool:
        return False


DEFAULT_OUT_DIR = Path("data/raw/news")
DEFAULT_DATASET_PATH = Path("data/processed/main_dataset.csv")
DEFAULT_TICKER_COLUMN = "ticker"
DEFAULT_SINCE_DATE = "2024-01-01"
DEFAULT_MAX_NEWS_ITEMS = 7
DEFAULT_BASE_URL = "https://api.massive.com/v2/reference"

# These paths match the sample next_url you showed.
NEWS_PATH = "/news"
load_dotenv()

@dataclass
class FetchConfig:
    api_key: str
    base_url: str = DEFAULT_BASE_URL
    out_dir: Path = DEFAULT_OUT_DIR
    limit: int = 100
    sleep_s: float = 0.15
    timeout_s: int = 60
    max_pages: int = 2000  # safety guard
    max_news_items: int = DEFAULT_MAX_NEWS_ITEMS


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _ensure_dir(p: Path) -> None:
    p.mkdir(parents=True, exist_ok=True)


def _read_tickers_file(path: Path) -> list[str]:
    txt = path.read_text(encoding="utf-8", errors="ignore")
    tickers: list[str] = []
    for line in txt.splitlines():
        line = line.strip()
        if not line:
            continue
        # allow comma-separated or whitespace
        parts = [x.strip() for x in line.replace(",", " ").split()]
        tickers.extend([p for p in parts if p])
    # de-dupe, preserve order
    seen = set()
    out = []
    for t in tickers:
        t = t.upper()
        if t not in seen:
            seen.add(t)
            out.append(t)
    return out


def _read_dataset_tickers(dataset_path: Path, ticker_column: str = DEFAULT_TICKER_COLUMN) -> Set[str]:
    if not dataset_path.exists():
        raise FileNotFoundError(f"Dataset file not found: {dataset_path}")

    with dataset_path.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        if not reader.fieldnames:
            raise ValueError(f"No header found in dataset: {dataset_path}")

        # case-insensitive lookup so both 'ticker' and 'Ticker' work
        col_by_lower = {c.lower(): c for c in reader.fieldnames if c}
        resolved_col = col_by_lower.get(ticker_column.lower())
        if not resolved_col:
            cols = ", ".join(reader.fieldnames)
            raise ValueError(f"Ticker column '{ticker_column}' not found. Available columns: {cols}")

        tickers: Set[str] = set()
        for row in reader:
            ticker = (row.get(resolved_col) or "").strip().upper()
            if ticker:
                tickers.add(ticker)

    if not tickers:
        raise ValueError(f"No tickers found in column '{ticker_column}' from {dataset_path}")
    return tickers


def _filter_tickers_by_dataset(input_tickers: list[str], dataset_tickers: Set[str]) -> tuple[list[str], list[str]]:
    kept: list[str] = []
    skipped: list[str] = []
    for ticker in input_tickers:
        if ticker in dataset_tickers:
            kept.append(ticker)
        else:
            skipped.append(ticker)
    return kept, skipped


def _validate_iso_date(date_str: Optional[str], arg_name: str) -> Optional[str]:
    if date_str is None:
        return None
    try:
        datetime.strptime(date_str, "%Y-%m-%d")
    except ValueError as e:
        raise ValueError(f"Invalid {arg_name}='{date_str}'. Expected YYYY-MM-DD.") from e
    return date_str


def _safe_append_api_key(url: str, api_key: str) -> str:
    # next_url often does NOT include apiKey; append if missing
    if "apiKey=" in url:
        return url
    joiner = "&" if ("?" in url) else "?"
    return f"{url}{joiner}apiKey={api_key}"


def _get_json(url: str, cfg: FetchConfig) -> Dict[str, Any]:
    url = _safe_append_api_key(url, cfg.api_key)
    r = requests.get(url, timeout=cfg.timeout_s)
    r.raise_for_status()
    return r.json()


def _write_jsonl(path: Path, rows: Iterable[Dict[str, Any]]) -> int:
    n = 0
    with path.open("a", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
            n += 1
    return n


def _load_existing_keys_jsonl(path: Path, key_fn) -> Set[str]:
    if not path.exists():
        return set()
    keys: Set[str] = set()
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
                keys.add(key_fn(obj))
            except Exception:
                continue
    return keys


def _news_key(item: Dict[str, Any]) -> str:
    # Your sample has "id"
    if "id" in item and item["id"]:
        return str(item["id"])
    # fallback hash
    h = hashlib.sha256(
        (str(item.get("article_url", "")) + "|" + str(item.get("published_utc", "")) + "|" + str(item.get("title", ""))).encode(
            "utf-8", errors="ignore"
        )
    ).hexdigest()
    return h


def _paginate(start_url: str, cfg: FetchConfig) -> Iterable[Dict[str, Any]]:
    url = start_url
    pages = 0
    while url:
        pages += 1
        if pages > cfg.max_pages:
            raise RuntimeError(f"Hit max_pages={cfg.max_pages}. Stopping to avoid infinite pagination.")
        data = _get_json(url, cfg)
        yield data
        url = data.get("next_url")
        if cfg.sleep_s:
            time.sleep(cfg.sleep_s)


def _build_news_url(cfg: FetchConfig, ticker: str, since: Optional[str], order: str, sort: str, limit: int) -> str:
    # Massive supports query params similar to your examples
    # We include published_utc.gte if provided
    params = {
        "ticker": ticker,
        "limit": str(limit),
        "sort": sort,
        "order": order,
    }
    if since:
        # Based on your sample cursor structure using published_utc.gte
        params["published_utc.gte"] = since
    qs = "&".join([f"{k}={requests.utils.quote(v)}" for k, v in params.items()])
    return f"{cfg.base_url}{NEWS_PATH}?{qs}&apiKey={cfg.api_key}"


def fetch_one_ticker(cfg: FetchConfig, ticker: str, news_since: Optional[str]) -> None:
    ticker = ticker.upper()
    out_ticker_dir = cfg.out_dir / ticker
    _ensure_dir(out_ticker_dir)

    news_path = out_ticker_dir / "news.jsonl"
    meta_path = out_ticker_dir / "meta.json"

    existing_news = _load_existing_keys_jsonl(news_path, _news_key)

    # ---- NEWS ----
    news_url = _build_news_url(cfg, ticker, news_since, order="desc", sort="published_utc", limit=cfg.limit)
    new_news_written = 0
    news_total_seen = 0

    for page in _paginate(news_url, cfg):
        results = page.get("results") or []
        to_write = []
        for item in results:
            news_total_seen += 1
            k = _news_key(item)
            if k in existing_news:
                continue
            existing_news.add(k)
            to_write.append(item)
            if cfg.max_news_items > 0 and (new_news_written + len(to_write)) >= cfg.max_news_items:
                break
        if to_write:
            new_news_written += _write_jsonl(news_path, to_write)
        if cfg.max_news_items > 0 and new_news_written >= cfg.max_news_items:
            break

    meta = {
        "ticker": ticker,
        "fetched_at_utc": _now_iso(),
        "news": {
            "since": news_since,
            "seen_total_this_run": news_total_seen,
            "new_items_written_this_run": new_news_written,
            "max_items_this_run": cfg.max_news_items,
            "jsonl_path": str(news_path),
        },
    }
    meta_path.write_text(json.dumps(meta, indent=2), encoding="utf-8")

    print(f"[{ticker}] news: +{new_news_written} (seen {news_total_seen})")


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--ticker", type=str, default=None, help="Single ticker (e.g., AAPL)")
    p.add_argument("--tickers-file", type=str, default=None, help="Path to a txt file of tickers")
    p.add_argument(
        "--dataset-path",
        type=str,
        default=str(DEFAULT_DATASET_PATH),
        help=f"Dataset CSV used to whitelist tickers (default: {DEFAULT_DATASET_PATH})",
    )
    p.add_argument(
        "--dataset-ticker-column",
        type=str,
        default=DEFAULT_TICKER_COLUMN,
        help=f"Ticker column name in dataset CSV (default: {DEFAULT_TICKER_COLUMN})",
    )
    p.add_argument("--out-dir", type=str, default=str(DEFAULT_OUT_DIR))
    p.add_argument("--base-url", type=str, default=DEFAULT_BASE_URL)
    p.add_argument("--limit", type=int, default=100)
    p.add_argument("--sleep", type=float, default=0.15)
    p.add_argument(
        "--max-news-items",
        type=int,
        default=DEFAULT_MAX_NEWS_ITEMS,
        help=f"Max new news rows to write per ticker per run (default: {DEFAULT_MAX_NEWS_ITEMS})",
    )

    p.add_argument(
        "--news-since",
        type=str,
        default=DEFAULT_SINCE_DATE,
        help=f"YYYY-MM-DD (filters published_utc.gte). Default: {DEFAULT_SINCE_DATE}",
    )

    args = p.parse_args()

    if args.max_news_items < 1:
        raise SystemExit("--max-news-items must be >= 1")
    try:
        args.news_since = _validate_iso_date(args.news_since, "news-since")
    except ValueError as e:
        raise SystemExit(str(e)) from e

    api_key = (os.getenv("MASSIVE_API_KEY") or "").strip()
    if not api_key:
        raise SystemExit("Missing MASSIVE_API_KEY in environment.")

    dataset_path = Path(args.dataset_path)
    try:
        dataset_tickers = _read_dataset_tickers(dataset_path, args.dataset_ticker_column)
    except Exception as e:
        raise SystemExit(f"Failed to load dataset ticker whitelist from {dataset_path}: {e}") from e

    requested_tickers: list[str] = []
    if args.ticker:
        requested_tickers = [args.ticker.upper()]
    elif args.tickers_file:
        requested_tickers = _read_tickers_file(Path(args.tickers_file))
    else:
        raise SystemExit("Provide either --ticker or --tickers-file")

    tickers, skipped = _filter_tickers_by_dataset(requested_tickers, dataset_tickers)
    if skipped:
        skipped_preview = ", ".join(skipped[:10])
        suffix = " ..." if len(skipped) > 10 else ""
        print(
            f"Skipping {len(skipped)} ticker(s) not present in dataset "
            f"{dataset_path}: {skipped_preview}{suffix}"
        )
    if not tickers:
        raise SystemExit(f"No requested tickers are present in dataset ticker set from {dataset_path}.")
    print(f"Using {len(tickers)} ticker(s) present in {dataset_path}.")

    cfg = FetchConfig(
        api_key=api_key,
        base_url=args.base_url.rstrip("/"),
        out_dir=Path(args.out_dir),
        limit=args.limit,
        sleep_s=args.sleep,
        max_news_items=args.max_news_items,
    )
    _ensure_dir(cfg.out_dir)

    for t in tickers:
        try:
            fetch_one_ticker(cfg, t, args.news_since)
        except requests.HTTPError as e:
            print(f"[{t}] HTTPError: {e}")
        except Exception as e:
            print(f"[{t}] Error: {e}")


if __name__ == "__main__":
    main()
