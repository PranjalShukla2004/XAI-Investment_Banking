#!/usr/bin/env python3
"""
Attach fetched news to the main dataset by ticker and year.

Input layout:
  data/raw/news/<TICKER>/news.jsonl

Default behavior:
  - Reads data/processed/main_dataset.csv
  - For each dataset row, attaches news where:
      ticker matches
      published_utc.year == fiscal_year (fallback: period_end.year)
  - Keeps at most 4 news items per row (configurable)
  - Writes back to the same CSV (in-place)

Added/updated columns:
  news_count
  news_description
  news_published_utc
  news_year
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from datetime import date, datetime
from pathlib import Path
from typing import Any, Dict, Iterable, List, Set, Tuple


DEFAULT_DATASET_PATH = Path("data/processed/main_dataset.csv")
DEFAULT_NEWS_ROOT = Path("data/raw/news")
DEFAULT_TICKER_COLUMN = "ticker"
DEFAULT_PERIOD_END_COLUMN = "period_end"  # fallback year source only
DEFAULT_FISCAL_YEAR_COLUMN = "fiscal_year"
DEFAULT_MAX_NEWS_PER_ROW = 4

NEWS_COUNT_COL = "news_count"
NEWS_DESCRIPTION_COL = "news_description"
NEWS_PUBLISHED_UTC_COL = "news_published_utc"
NEWS_YEAR_COL = "news_year"

OBSOLETE_NEWS_COLS = {
    "ticker_news",
    "latest_news_published_utc",
    "latest_news_title",
    "news_items_json",
    "news_sentiment",
    "news_reasoning",
}


def _resolve_column(fieldnames: List[str], requested: str) -> str:
    mapping = {c.lower(): c for c in fieldnames if c}
    resolved = mapping.get(requested.lower())
    if not resolved:
        cols = ", ".join(fieldnames)
        raise ValueError(f"Column '{requested}' not found. Available columns: {cols}")
    return resolved


def _read_dataset(dataset_path: Path, ticker_column: str) -> Tuple[List[str], List[Dict[str, str]], str]:
    if not dataset_path.exists():
        raise FileNotFoundError(f"Dataset not found: {dataset_path}")

    with dataset_path.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        if not reader.fieldnames:
            raise ValueError(f"No header found in dataset: {dataset_path}")
        fieldnames = list(reader.fieldnames)
        resolved_ticker_col = _resolve_column(fieldnames, ticker_column)
        rows = list(reader)
    return fieldnames, rows, resolved_ticker_col


def _parse_date(value: Any) -> date | None:
    text = str(value or "").strip()
    if not text:
        return None
    if len(text) >= 10 and text[4] == "-" and text[7] == "-":
        try:
            return date.fromisoformat(text[:10])
        except ValueError:
            pass
    try:
        return datetime.fromisoformat(text.replace("Z", "+00:00")).date()
    except ValueError:
        return None


def _parse_year(value: Any) -> int | None:
    text = str(value or "").strip()
    if not text:
        return None
    try:
        return int(float(text))
    except ValueError:
        return None


def _normalize_tickers(values: Any) -> List[str]:
    if values is None:
        return []
    if not isinstance(values, list):
        values = [values]
    out: List[str] = []
    seen: Set[str] = set()
    for value in values:
        ticker = str(value).strip().upper()
        if not ticker or ticker in seen:
            continue
        seen.add(ticker)
        out.append(ticker)
    return out


def _news_key(item: Dict[str, Any]) -> str:
    if item.get("id"):
        return str(item["id"])
    payload = (
        str(item.get("article_url", ""))
        + "|"
        + str(item.get("published_utc", ""))
        + "|"
        + str(item.get("title", ""))
    )
    return hashlib.sha256(payload.encode("utf-8", errors="ignore")).hexdigest()


def _iter_news_files(news_root: Path) -> Iterable[Tuple[str, Path]]:
    if not news_root.exists():
        return
    for ticker_dir in sorted(p for p in news_root.iterdir() if p.is_dir()):
        yield ticker_dir.name.strip().upper(), ticker_dir / "news.jsonl"


def _write_dataset(path: Path, fieldnames: List[str], rows: List[Dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_suffix(path.suffix + ".tmp")
    with tmp_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    tmp_path.replace(path)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", type=str, default=str(DEFAULT_DATASET_PATH))
    ap.add_argument("--ticker-column", type=str, default=DEFAULT_TICKER_COLUMN)
    ap.add_argument("--period-end-column", type=str, default=DEFAULT_PERIOD_END_COLUMN)
    ap.add_argument("--fiscal-year-column", type=str, default=DEFAULT_FISCAL_YEAR_COLUMN)
    ap.add_argument("--news-root", type=str, default=str(DEFAULT_NEWS_ROOT))
    ap.add_argument(
        "--max-news-per-row",
        "--max-news-per-ticker",
        dest="max_news_per_row",
        type=int,
        default=DEFAULT_MAX_NEWS_PER_ROW,
        help=f"Max attached news items per ticker-year per dataset row (default: {DEFAULT_MAX_NEWS_PER_ROW}).",
    )
    ap.add_argument(
        "--out",
        type=str,
        default=None,
        help="Output CSV path. If omitted, updates --dataset in place.",
    )
    ap.add_argument(
        "--keep-mismatched",
        action="store_true",
        help="Keep rows where directory ticker is not present in article tickers list.",
    )
    args = ap.parse_args()

    if args.max_news_per_row < 1:
        raise SystemExit("--max-news-per-row must be >= 1")

    dataset_path = Path(args.dataset)
    news_root = Path(args.news_root)
    out_path = Path(args.out) if args.out else dataset_path

    fieldnames, dataset_rows, resolved_ticker_col = _read_dataset(dataset_path, args.ticker_column)
    resolved_period_end_col = _resolve_column(fieldnames, args.period_end_column)
    resolved_fiscal_year_col = _resolve_column(fieldnames, args.fiscal_year_column)

    dataset_tickers = {
        str(row.get(resolved_ticker_col, "")).strip().upper()
        for row in dataset_rows
        if str(row.get(resolved_ticker_col, "")).strip()
    }
    if not dataset_tickers:
        raise SystemExit(f"No tickers found in dataset column '{resolved_ticker_col}'.")

    # keyed by (ticker, year)
    grouped: Dict[Tuple[str, int], List[Dict[str, Any]]] = {}
    seen_ids: Dict[Tuple[str, int], Set[str]] = {}

    total_dirs = 0
    missing_news_files = 0
    dirs_not_in_dataset = 0
    json_errors = 0
    mismatched_skipped = 0
    missing_news_date_skipped = 0
    missing_description_skipped = 0
    duplicate_skipped = 0

    for dir_ticker, news_path in _iter_news_files(news_root):
        total_dirs += 1
        if dir_ticker not in dataset_tickers:
            dirs_not_in_dataset += 1
            continue
        if not news_path.exists():
            missing_news_files += 1
            continue

        with news_path.open("r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    item = json.loads(line)
                except Exception:
                    json_errors += 1
                    continue

                article_tickers = _normalize_tickers(item.get("tickers"))
                ticker_match = (not article_tickers) or (dir_ticker in article_tickers)
                if (not ticker_match) and (not args.keep_mismatched):
                    mismatched_skipped += 1
                    continue

                published_utc = str(item.get("published_utc", "") or "").strip()
                published_date = _parse_date(published_utc)
                if published_date is None:
                    missing_news_date_skipped += 1
                    continue

                description = str(item.get("description", "") or "").strip()
                if not description:
                    missing_description_skipped += 1
                    continue

                key = (dir_ticker, published_date.year)
                nid = _news_key(item)
                ids = seen_ids.setdefault(key, set())
                if nid in ids:
                    duplicate_skipped += 1
                    continue
                ids.add(nid)

                grouped.setdefault(key, []).append(
                    {
                        "published_utc": published_utc,
                        "_published_date": published_date,
                        "description": description,
                    }
                )

    for key, items in grouped.items():
        items.sort(key=lambda r: (r["_published_date"], r["published_utc"]), reverse=True)

    rows_missing_target_year = 0
    for row in dataset_rows:
        ticker = str(row.get(resolved_ticker_col, "")).strip().upper()
        fiscal_year = _parse_year(row.get(resolved_fiscal_year_col))
        period_end = _parse_date(row.get(resolved_period_end_col))
        target_year = fiscal_year if fiscal_year is not None else (period_end.year if period_end is not None else None)

        if target_year is None:
            rows_missing_target_year += 1
            items: List[Dict[str, Any]] = []
        else:
            source_items = grouped.get((ticker, target_year), [])
            items = source_items[: args.max_news_per_row]

        row[NEWS_COUNT_COL] = str(len(items))
        row[NEWS_DESCRIPTION_COL] = json.dumps([x["description"] for x in items], ensure_ascii=False)
        row[NEWS_PUBLISHED_UTC_COL] = json.dumps([x["published_utc"] for x in items], ensure_ascii=False)
        row[NEWS_YEAR_COL] = json.dumps([x["_published_date"].year for x in items], ensure_ascii=False)

    out_fieldnames = [c for c in fieldnames if c not in OBSOLETE_NEWS_COLS]
    for row in dataset_rows:
        for col in OBSOLETE_NEWS_COLS:
            row.pop(col, None)

    for col in (NEWS_COUNT_COL, NEWS_DESCRIPTION_COL, NEWS_PUBLISHED_UTC_COL, NEWS_YEAR_COL):
        if col not in out_fieldnames:
            out_fieldnames.append(col)

    _write_dataset(out_path, out_fieldnames, dataset_rows)

    print(f"Saved dataset with attached news: {out_path}")
    print(f"Rows updated: {len(dataset_rows):,}")
    print(
        "Summary: "
        f"ticker_dirs={total_dirs:,}, "
        f"dirs_not_in_dataset={dirs_not_in_dataset:,}, "
        f"missing_news_files={missing_news_files:,}, "
        f"mismatched_skipped={mismatched_skipped:,}, "
        f"missing_news_date_skipped={missing_news_date_skipped:,}, "
        f"missing_description_skipped={missing_description_skipped:,}, "
        f"rows_missing_target_year={rows_missing_target_year:,}, "
        f"duplicate_skipped={duplicate_skipped:,}, "
        f"json_errors={json_errors:,}"
    )


if __name__ == "__main__":
    main()
