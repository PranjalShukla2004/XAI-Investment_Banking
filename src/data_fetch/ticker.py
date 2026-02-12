# src/data_fetch/get_3000_tickers_reference.py
from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Set

from dotenv import load_dotenv
from tqdm import tqdm

from massive_client import MassiveClient, MassiveConfig


@dataclass
class TickerConfig:
    n_companies: int = 3000

    # Your screenshot endpoint
    path: str = "v3/reference/tickers"
    params: Dict[str, Any] = None  # filled in __post_init__

    # Safety
    max_pages: int = 2000  # enough to reach 3000 even with filtering

    # Output
    out_dir: str = "data/raw/tickers"
    out_file: str = "tickers_3000.txt"

    # Filters (matches your call + typical cleanup)
    market: str = "stocks"
    active: bool = True
    sort: str = "ticker"
    order: str = "asc"
    limit: int = 100

    # Keep only common stocks by default (your sample shows "type": "CS")
    type_allowlist: Optional[Set[str]] = None  # e.g. {"CS"}; None = keep all

    def __post_init__(self):
        if self.params is None:
            self.params = {
                "market": self.market,
                "active": str(self.active).lower(),
                "order": self.order,
                "limit": self.limit,
                "sort": self.sort,
            }


def _get_ticker(rec: Dict[str, Any]) -> Optional[str]:
    t = rec.get("ticker") or rec.get("symbol")
    if not t:
        return None
    return str(t).strip().upper()


def get_3000_tickers(cfg: TickerConfig) -> str:
    load_dotenv()
    client = MassiveClient(MassiveConfig())

    tickers: List[str] = []
    seen: Set[str] = set()

    it = client.iter_paginated(
        cfg.path,
        params=cfg.params,
        results_key="results",
        use_cache=False,       # IMPORTANT: ticker discovery should not write tons of cache files
        max_pages=cfg.max_pages,
    )

    for rec in tqdm(it, desc=f"Collecting {cfg.n_companies} tickers (reference endpoint)"):
        # optional type filter
        if cfg.type_allowlist is not None:
            typ = rec.get("type")
            if typ not in cfg.type_allowlist:
                continue

        t = _get_ticker(rec)
        if not t or t in seen:
            continue

        seen.add(t)
        tickers.append(t)

        if len(tickers) >= cfg.n_companies:
            break

    os.makedirs(cfg.out_dir, exist_ok=True)
    out_path = os.path.join(cfg.out_dir, cfg.out_file)
    with open(out_path, "w", encoding="utf-8") as f:
        f.write("\n".join(tickers))

    print(f"Saved {len(tickers)} tickers to {out_path}")
    return out_path


if __name__ == "__main__":
    # Example: only common stocks ("CS")
    cfg = TickerConfig(type_allowlist={"CS"})
    get_3000_tickers(cfg)
