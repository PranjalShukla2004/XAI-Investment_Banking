# src/data_fetch/massive_client.py
from __future__ import annotations

import hashlib
import json
import os
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterator, Optional
from urllib.parse import urlparse, parse_qs

import requests


@dataclass
class MassiveConfig:
    base_url: str = "https://api.massive.com"
    api_key_env: str = "MASSIVE_API_KEY"  
    cache_dir: str = "data/raw"
    timeout_sec: int = 30

    max_retries: int = 6
    backoff_base_sec: float = 1.0
    polite_sleep_sec: float = 0.0


class MassiveClient:
    """
    Massive REST client:
      - apiKey query parameter injection (as shown in your screenshot)
      - disk caching
      - retries/backoff for 429/5xx
      - simple pagination support via next_url (if the API returns it)
    """
    def __init__(self, cfg: MassiveConfig = MassiveConfig()):
        self.cfg = cfg
        self.base_url = cfg.base_url.rstrip("/")
        self.api_key = os.getenv(cfg.api_key_env)
        if not self.api_key:
            raise RuntimeError(f"Missing {cfg.api_key_env} in environment/.env")

        self.session = requests.Session()
        self.cache_root = Path(cfg.cache_dir)
        self.cache_root.mkdir(parents=True, exist_ok=True)

    def _cache_path(self, path: str, params: Dict[str, Any]) -> Path:
        safe_path = path.strip("/").replace("/", "__")
        items = sorted((k, str(v)) for k, v in params.items())
        digest = hashlib.md5(json.dumps(items).encode("utf-8")).hexdigest()  # noqa: S324
        folder = self.cache_root / safe_path
        folder.mkdir(parents=True, exist_ok=True)
        return folder / f"{digest}.json"

    def get_json(self, path: str, params: Optional[Dict[str, Any]] = None, use_cache: bool = True) -> Dict[str, Any]:
        params = dict(params or {})
        params["apiKey"] = self.api_key  # Massive uses apiKey in query string (per your screenshot)

        cache_file = self._cache_path(path, params)
        if use_cache and cache_file.exists():
            return json.loads(cache_file.read_text(encoding="utf-8"))

        url = f"{self.base_url}/{path.lstrip('/')}"
        data = self._request_with_retries(url, params)

        if use_cache:
            cache_file.write_text(json.dumps(data), encoding="utf-8")

        if self.cfg.polite_sleep_sec > 0:
            time.sleep(self.cfg.polite_sleep_sec)

        return data

    def _request_with_retries(self, url: str, params: Dict[str, Any]) -> Dict[str, Any]:
        last_err: Optional[Exception] = None
        for attempt in range(self.cfg.max_retries):
            try:
                resp = self.session.get(url, params=params, timeout=self.cfg.timeout_sec)

                if resp.status_code in (429, 500, 502, 503, 504):
                    sleep_s = self.cfg.backoff_base_sec * (2 ** attempt)
                    ra = resp.headers.get("Retry-After")
                    if ra:
                        try:
                            sleep_s = max(sleep_s, float(ra))
                        except ValueError:
                            pass
                    time.sleep(sleep_s)
                    continue

                resp.raise_for_status()
                return resp.json()

            except Exception as e:
                last_err = e
                time.sleep(self.cfg.backoff_base_sec * (2 ** attempt))

        raise RuntimeError(f"Request failed after retries: {url}") from last_err

    def iter_paginated(
    self,
    path: str,
    params: Optional[Dict[str, Any]] = None,
    results_key: str = "results",
    use_cache: bool = True,
    max_pages: Optional[int] = None,
) -> Iterator[Dict[str, Any]]:
        """
        Iterate pages using next_url.

        Safety:
        - Detect repeated next_url (prevents infinite loops)
        - Break if a page returns zero items (prevents cursor stalls)
        - Optional max_pages cap
        """
        params = dict(params or {})
        page = 0

        data = self.get_json(path, params=params, use_cache=use_cache)
        items = (data.get(results_key) or [])
        for item in items:
            yield item

        next_url = data.get("next_url")
        seen_next_urls = set()

        while next_url:
            # loop detection
            if next_url in seen_next_urls:
                # next_url repeating => API/cursor loop
                break
            seen_next_urls.add(next_url)

            page += 1
            if max_pages is not None and page >= max_pages:
                break

            parsed = urlparse(next_url)
            next_path = parsed.path.lstrip("/")
            q = {k: v[-1] for k, v in parse_qs(parsed.query).items()}

            data = self.get_json(next_path, params=q, use_cache=use_cache)

            items = (data.get(results_key) or [])
            if not items:
                # cursor advanced but no results => stop
                break

            for item in items:
                yield item

            next_url = data.get("next_url")

