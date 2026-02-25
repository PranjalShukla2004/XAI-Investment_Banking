#!/usr/bin/env python3
"""
Attach fetched news to the main dataset by ticker.

Input layout:
  data/raw/news/<TICKER>/news.jsonl

Default behavior:
  - Reads data/processed/main_dataset.csv
  - Matches news to ticker
  - Adds/updates columns in the dataset:
      news_count
      news_sentiment
      news_reasoning
      news_description
      news_published_utc
    (each news_* column stores a JSON array with up to max-news-per-ticker items)
  - Writes back to the same CSV (in-place)

Usage:
  python -m src.data_fetch.attach_news_to_tickers
  python -m src.data_fetch.attach_news_to_tickers --max-news-per-ticker 4
  python -m src.data_fetch.attach_news_to_tickers --out data/processed/main_dataset_with_news.csv
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path
from typing import Any, Dict, Iterable, List, Set, Tuple


DEFAULT_DATASET_PATH = Path("data/processed/main_dataset.csv")
DEFAULT_NEWS_ROOT = Path("data/raw/news")
DEFAULT_TICKER_COLUMN = "ticker"
DEFAULT_MAX_NEWS_PER_TICKER = 4

NEWS_COUNT_COL = "news_count"
NEWS_SENTIMENT_COL = "news_sentiment"
NEWS_REASONING_COL = "news_reasoning"
NEWS_DESCRIPTION_COL = "news_description"
NEWS_PUBLISHED_UTC_COL = "news_published_utc"
OBSOLETE_NEWS_COLS = {
    "ticker_news",
    "latest_news_published_utc",
    "latest_news_title",
    "news_items_json",
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


def _news_id(item: Dict[str, Any]) -> str:
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


def _publisher_name(item: Dict[str, Any]) -> str:
    publisher = item.get("publisher")
    if isinstance(publisher, dict):
        name = publisher.get("name")
        return str(name).strip() if name is not None else ""
    return ""


def _find_ticker_insight(item: Dict[str, Any], ticker: str) -> Dict[str, Any] | None:
    insights = item.get("insights")
    if not isinstance(insights, list):
        return None
    target = ticker.upper().strip()
    for insight in insights:
        if not isinstance(insight, dict):
            continue
        insight_ticker = str(insight.get("ticker", "")).strip().upper()
        if insight_ticker == target:
            return insight
    return None


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
    ap.add_argument("--news-root", type=str, default=str(DEFAULT_NEWS_ROOT))
    ap.add_argument(
        "--max-news-per-ticker",
        type=int,
        default=DEFAULT_MAX_NEWS_PER_TICKER,
        help=f"Max attached news items per ticker (default: {DEFAULT_MAX_NEWS_PER_TICKER}).",
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

    if args.max_news_per_ticker < 1:
        raise SystemExit("--max-news-per-ticker must be >= 1")

    dataset_path = Path(args.dataset)
    news_root = Path(args.news_root)
    out_path = Path(args.out) if args.out else dataset_path

    fieldnames, dataset_rows, resolved_ticker_col = _read_dataset(dataset_path, args.ticker_column)
    dataset_tickers = {
        str(row.get(resolved_ticker_col, "")).strip().upper()
        for row in dataset_rows
        if str(row.get(resolved_ticker_col, "")).strip()
    }
    if not dataset_tickers:
        raise SystemExit(f"No tickers found in dataset column '{resolved_ticker_col}'.")

    per_ticker_rows: Dict[str, List[Dict[str, str]]] = {}
    per_ticker_seen_ids: Dict[str, Set[str]] = {}
    per_ticker_seen_reasonings: Dict[str, Set[str]] = {}

    total_dirs = 0
    missing_news_files = 0
    dirs_not_in_dataset = 0
    json_errors = 0
    mismatched_skipped = 0
    missing_insight_skipped = 0
    duplicate_skipped = 0

    for dir_ticker, news_path in _iter_news_files(news_root):
        total_dirs += 1
        if dir_ticker not in dataset_tickers:
            dirs_not_in_dataset += 1
            continue
        if not news_path.exists():
            missing_news_files += 1
            continue

        rows_for_ticker = per_ticker_rows.setdefault(dir_ticker, [])
        seen_ids_for_ticker = per_ticker_seen_ids.setdefault(dir_ticker, set())
        seen_reasonings_for_ticker = per_ticker_seen_reasonings.setdefault(dir_ticker, set())

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

                insight = _find_ticker_insight(item, dir_ticker)
                if insight is None:
                    missing_insight_skipped += 1
                    continue

                nid = _news_id(item)
                if nid in seen_ids_for_ticker:
                    duplicate_skipped += 1
                    continue
                reasoning = str(insight.get("sentiment_reasoning", "") or "").strip()
                if not reasoning:
                    continue
                norm_reasoning = " ".join(reasoning.lower().split())
                if norm_reasoning in seen_reasonings_for_ticker:
                    duplicate_skipped += 1
                    continue

                rows_for_ticker.append(
                    {
                        "news_id": nid,
                        "published_utc": str(item.get("published_utc", "") or ""),
                        "description": str(item.get("description", "") or "").strip(),
                        "sentiment": str(insight.get("sentiment", "") or "").strip(),
                        "reasoning": reasoning,
                        "article_url": str(item.get("article_url", "") or ""),
                        "publisher_name": _publisher_name(item),
                    }
                )
                seen_ids_for_ticker.add(nid)
                seen_reasonings_for_ticker.add(norm_reasoning)

    for ticker, items in per_ticker_rows.items():
        items.sort(key=lambda r: r["published_utc"], reverse=True)
        if len(items) > args.max_news_per_ticker:
            per_ticker_rows[ticker] = items[: args.max_news_per_ticker]

    for row in dataset_rows:
        ticker = str(row.get(resolved_ticker_col, "")).strip().upper()
        items = per_ticker_rows.get(ticker, [])

        row[NEWS_COUNT_COL] = str(len(items))
        row[NEWS_SENTIMENT_COL] = json.dumps([item["sentiment"] for item in items], ensure_ascii=False)
        row[NEWS_REASONING_COL] = json.dumps([item["reasoning"] for item in items], ensure_ascii=False)
        row[NEWS_DESCRIPTION_COL] = json.dumps([item["description"] for item in items], ensure_ascii=False)
        row[NEWS_PUBLISHED_UTC_COL] = json.dumps([item["published_utc"] for item in items], ensure_ascii=False)

    out_fieldnames = [c for c in fieldnames if c not in OBSOLETE_NEWS_COLS]
    for row in dataset_rows:
        for col in OBSOLETE_NEWS_COLS:
            row.pop(col, None)
    for col in (
        NEWS_COUNT_COL,
        NEWS_SENTIMENT_COL,
        NEWS_REASONING_COL,
        NEWS_DESCRIPTION_COL,
        NEWS_PUBLISHED_UTC_COL,
    ):
        if col not in out_fieldnames:
            out_fieldnames.append(col)

    _write_dataset(out_path, out_fieldnames, dataset_rows)

    print(f"Saved dataset with attached news: {out_path}")
    print(f"Rows updated: {len(dataset_rows):,}")
    print(f"Tickers with news: {len(per_ticker_rows):,}")
    print(
        "Summary: "
        f"ticker_dirs={total_dirs:,}, "
        f"dirs_not_in_dataset={dirs_not_in_dataset:,}, "
        f"missing_news_files={missing_news_files:,}, "
        f"mismatched_skipped={mismatched_skipped:,}, "
        f"missing_insight_skipped={missing_insight_skipped:,}, "
        f"duplicate_skipped={duplicate_skipped:,}, "
        f"json_errors={json_errors:,}"
    )


if __name__ == "__main__":
    main()
