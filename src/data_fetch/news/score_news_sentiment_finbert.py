#!/usr/bin/env python3
"""
Score dataset news text with FinBERT and store per-news sentiment scores.

Default behavior:
  - Reads data/processed/main_dataset.csv
  - Uses news_description as input text column
  - Writes/updates news_sentiment_score in the output CSV
  - Stores one score per split news item as a JSON list string
  - Updates dataset in place unless --out is provided

Sentiment score definition:
  news_sentiment_score = P(positive) - P(negative)
  Range: [-1.0, 1.0]
"""

from __future__ import annotations

import argparse
import ast
import csv
import json
from pathlib import Path
from typing import Any, Iterable, List

import numpy as np
import pandas as pd
import torch
from tqdm import tqdm

DEFAULT_DATASET_PATH = Path("data/processed/main_dataset.csv")
DEFAULT_TEXT_COLUMN = "news_description"
DEFAULT_OUTPUT_COLUMN = "news_sentiment_score"
DEFAULT_MODEL_NAME = "ProsusAI/finbert"
DEFAULT_BATCH_SIZE = 32
DEFAULT_MAX_LENGTH = 256


def _resolve_column(columns: Iterable[str], requested: str) -> str:
    mapping = {str(col).lower(): str(col) for col in columns}
    resolved = mapping.get(requested.lower())
    if not resolved:
        available = ", ".join(str(c) for c in columns)
        raise ValueError(f"Column '{requested}' not found. Available columns: {available}")
    return resolved


def _split_quoted_csv(text: str) -> List[str]:
    try:
        fields = next(csv.reader([text], skipinitialspace=True))
    except Exception:
        fields = text.split(",")
    parts = [p.strip().strip('"').strip("'") for p in fields]
    return [p for p in parts if p]


def _parse_news_texts(value: object) -> List[str]:
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return []

    text = str(value).strip()
    if not text:
        return []

    parsed: object = text
    if text.startswith("[") and text.endswith("]"):
        try:
            parsed = json.loads(text)
        except json.JSONDecodeError:
            try:
                parsed = ast.literal_eval(text)
            except (ValueError, SyntaxError):
                parsed = text

    if isinstance(parsed, list):
        out: List[str] = []
        for item in parsed:
            s = str(item).strip()
            if s:
                out.append(s)
        return out

    s = str(parsed).strip()
    if not s:
        return []

    # Fallback when the column is plain quoted CSV text.
    return _split_quoted_csv(s)


def _get_label_indices(model: Any) -> tuple[int, int]:
    id2label = getattr(model.config, "id2label", {})
    if not id2label:
        raise ValueError("Model config does not expose id2label.")

    pos_idx = None
    neg_idx = None
    for idx, label in id2label.items():
        normalized = str(label).strip().lower()
        if normalized == "positive":
            pos_idx = int(idx)
        elif normalized == "negative":
            neg_idx = int(idx)

    if pos_idx is None or neg_idx is None:
        raise ValueError(f"Could not find positive/negative labels in id2label={id2label}")
    return pos_idx, neg_idx


def _compute_scores(
    texts: List[str],
    model_name: str,
    batch_size: int,
    max_length: int,
    local_files_only: bool,
    cache_dir: str | None,
) -> np.ndarray:
    # In some local environments, sklearn is installed but linked against a broken SciPy.
    # Disabling sklearn availability in transformers avoids unrelated import crashes.
    import transformers.utils.import_utils as hf_import_utils

    hf_import_utils._sklearn_available = False
    from transformers import AutoModelForSequenceClassification, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(
        model_name,
        local_files_only=local_files_only,
        cache_dir=cache_dir,
    )
    model = AutoModelForSequenceClassification.from_pretrained(
        model_name,
        local_files_only=local_files_only,
        cache_dir=cache_dir,
    )

    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")
    print(f"[finbert] device={device} | texts={len(texts):,} | batch={batch_size} | max_len={max_length}")

    model = model.to(device)
    model.eval()

    pos_idx, neg_idx = _get_label_indices(model)
    scores = np.zeros(len(texts), dtype=np.float32)

    with torch.inference_mode():
        for start in tqdm(
            range(0, len(texts), batch_size),
            desc="[finbert] scoring",
            unit="batch",
        ):
            batch = texts[start : start + batch_size]
            enc = tokenizer(
                batch,
                padding=True,
                truncation=True,
                max_length=max_length,
                return_tensors="pt",
            )
            enc = {k: v.to(device) for k, v in enc.items()}
            logits = model(**enc).logits
            probs = torch.softmax(logits, dim=-1).detach().cpu().numpy()
            batch_scores = probs[:, pos_idx] - probs[:, neg_idx]
            scores[start : start + len(batch)] = batch_scores

    return scores


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", type=str, default=str(DEFAULT_DATASET_PATH))
    ap.add_argument("--text-column", type=str, default=DEFAULT_TEXT_COLUMN)
    ap.add_argument("--output-column", type=str, default=DEFAULT_OUTPUT_COLUMN)
    ap.add_argument("--model", type=str, default=DEFAULT_MODEL_NAME)
    ap.add_argument("--batch-size", type=int, default=DEFAULT_BATCH_SIZE)
    ap.add_argument("--max-length", type=int, default=DEFAULT_MAX_LENGTH)
    ap.add_argument(
        "--local-files-only",
        action="store_true",
        help="Load model/tokenizer from local cache only (no network calls).",
    )
    ap.add_argument(
        "--cache-dir",
        type=str,
        default=None,
        help="Optional Hugging Face cache directory for model files.",
    )
    ap.add_argument(
        "--out",
        type=str,
        default=None,
        help="Output CSV path. If omitted, updates --dataset in place.",
    )
    args = ap.parse_args()

    if args.batch_size < 1:
        raise SystemExit("--batch-size must be >= 1")
    if args.max_length < 8:
        raise SystemExit("--max-length must be >= 8")

    dataset_path = Path(args.dataset)
    out_path = Path(args.out) if args.out else dataset_path
    if not dataset_path.exists():
        raise FileNotFoundError(f"Dataset not found: {dataset_path}")

    df = pd.read_csv(dataset_path)
    text_column = _resolve_column(df.columns, args.text_column)

    row_texts: List[List[str]] = [_parse_news_texts(v) for v in df[text_column].tolist()]
    flattened_texts: List[str] = [item for items in row_texts for item in items]

    if flattened_texts:
        article_scores = _compute_scores(
            texts=flattened_texts,
            model_name=args.model,
            batch_size=args.batch_size,
            max_length=args.max_length,
            local_files_only=args.local_files_only,
            cache_dir=args.cache_dir,
        )
    else:
        article_scores = np.array([], dtype=np.float32)

    row_scores: List[str] = []
    cursor = 0
    for items in row_texts:
        if not items:
            row_scores.append("[]")
            continue
        next_cursor = cursor + len(items)
        score_list = [float(x) for x in article_scores[cursor:next_cursor]]
        row_scores.append(json.dumps(score_list, ensure_ascii=False))
        cursor = next_cursor

    df[args.output_column] = row_scores
    out_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_path, index=False)

    print(f"Saved dataset with FinBERT sentiment scores: {out_path}")
    print(f"Rows scored: {len(df):,}")
    print(f"Input text column: {text_column}")
    print(f"Output score column: {args.output_column}")
    print(f"Model: {args.model}")
    print(f"Local files only: {args.local_files_only}")
    print(f"Total news items scored: {len(flattened_texts):,}")


if __name__ == "__main__":
    main()
