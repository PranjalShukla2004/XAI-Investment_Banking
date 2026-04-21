from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from src.data_fetch.market.build_future_drawdown import (
    BuildFutureDrawdownConfig,
    _build_summary as _build_drawdown_summary,
    _compute_target_rows,
    _load_merge_dataset,
    _merge_targets,
)
from src.data_fetch.market.build_market_feature_merge import (
    MarketFeatureMergeConfig,
    _build_summary as _build_market_summary,
    _compute_market_feature_panel,
    _load_market_panel,
    _load_observations,
    _merge_features,
)
from src.models.risk.build_risk_dataset import (
    BuildRiskDatasetConfig,
    _build_summary as _build_risk_summary,
    _filter_rows,
    _load_source_dataset,
    _ordered_columns,
)
from src.models.risk.dataset_utils import ensure_dir, write_json

PROJECT_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_MARKET_DATA_DIR = PROJECT_ROOT / "data/raw/market_data"
DEFAULT_OUT_DIR = PROJECT_ROOT / "data/processed/risk"
DEFAULT_DATASET_SPECS = (
    "main=data/processed/main_dataset.csv",
    "test_out_of_time=data/processed/test_dataset_out_of_time.csv",
    "test_unseen_tickers=data/processed/test_dataset_unseen_tickers.csv",
)


@dataclass
class PrepareRiskDatasetsConfig:
    market_data_dir: Path = DEFAULT_MARKET_DATA_DIR
    out_dir: Path = DEFAULT_OUT_DIR
    timeframe: str = "annual"
    require_usable_target: bool = True
    horizon_days: int = 365
    calendar_buffer_days: int = 7
    min_future_bars: int = 30
    max_tickers: int = 0


def _parse_dataset_specs(values: list[str] | None) -> list[tuple[str, Path]]:
    specs = list(values or DEFAULT_DATASET_SPECS)
    parsed: list[tuple[str, Path]] = []
    for spec in specs:
        if "=" not in spec:
            raise ValueError(f"Invalid dataset spec {spec!r}. Expected NAME=PATH.")
        name, raw_path = spec.split("=", 1)
        parsed.append((name.strip(), Path(raw_path.strip())))
    return parsed


def _market_artifact_paths(out_dir: Path, name: str) -> tuple[Path, Path]:
    return out_dir / f"{name}_with_market.csv", out_dir / f"{name}_with_market_summary.json"


def _drawdown_artifact_paths(out_dir: Path, name: str) -> tuple[Path, Path]:
    return out_dir / f"{name}_with_market_targets.csv", out_dir / f"{name}_with_market_targets_summary.json"


def _risk_artifact_paths(out_dir: Path, name: str) -> tuple[Path, Path]:
    return out_dir / f"{name}_risk_dataset.csv", out_dir / f"{name}_risk_dataset_summary.json"


def _prepare_market_dataset(
    name: str,
    dataset_path: Path,
    cfg: PrepareRiskDatasetsConfig,
) -> dict[str, object]:
    market_out, market_summary_path = _market_artifact_paths(cfg.out_dir, name)
    market_cfg = MarketFeatureMergeConfig(
        main_dataset_path=dataset_path,
        market_data_dir=cfg.market_data_dir,
        out_path=market_out,
        summary_path=market_summary_path,
        timeframe=cfg.timeframe or None,
        max_tickers=int(cfg.max_tickers),
    )

    observations = _load_observations(market_cfg)
    requested_tickers = set(observations["ticker"].dropna().astype(str))
    market_panel = _load_market_panel(cfg.market_data_dir, requested_tickers)
    feature_panel = _compute_market_feature_panel(market_panel)
    merged = _merge_features(observations, feature_panel)

    out_cols = [
        col
        for col in ["ticker", "period_end", "fiscal_year", "timeframe", "anchor_trade_date", *feature_panel.columns[2:].tolist()]
        if col in merged.columns
    ]
    merged.loc[:, out_cols].to_csv(market_out, index=False)

    market_summary = _build_market_summary(market_cfg, observations, market_panel, merged)
    write_json(market_summary_path, market_summary)
    return {
        "market_dataset_path": str(market_out),
        "market_summary_path": str(market_summary_path),
        "market_rows": int(len(merged)),
        "market_tickers": int(merged["ticker"].nunique()) if not merged.empty else 0,
    }


def _prepare_drawdown_dataset(name: str, cfg: PrepareRiskDatasetsConfig) -> dict[str, object]:
    market_out, _ = _market_artifact_paths(cfg.out_dir, name)
    drawdown_out, drawdown_summary_path = _drawdown_artifact_paths(cfg.out_dir, name)
    drawdown_cfg = BuildFutureDrawdownConfig(
        merge_path=market_out,
        market_data_dir=cfg.market_data_dir,
        out_path=drawdown_out,
        summary_path=drawdown_summary_path,
        horizon_days=int(cfg.horizon_days),
        calendar_buffer_days=int(cfg.calendar_buffer_days),
        min_future_bars=int(cfg.min_future_bars),
        max_tickers=int(cfg.max_tickers),
    )

    merge_dataset = _load_merge_dataset(drawdown_cfg)
    targets_df, market_layout = _compute_target_rows(merge_dataset, drawdown_cfg)
    merged = _merge_targets(merge_dataset, targets_df)
    merged.to_csv(drawdown_out, index=False)

    drawdown_summary = _build_drawdown_summary(drawdown_cfg, merged, market_layout)
    write_json(drawdown_summary_path, drawdown_summary)
    return {
        "market_targets_path": str(drawdown_out),
        "market_targets_summary_path": str(drawdown_summary_path),
        "market_layout": market_layout,
        "usable_targets": int(drawdown_summary.get("usable_targets", 0)),
    }


def _prepare_model_dataset(name: str, cfg: PrepareRiskDatasetsConfig) -> dict[str, object]:
    drawdown_out, _ = _drawdown_artifact_paths(cfg.out_dir, name)
    risk_out, risk_summary_path = _risk_artifact_paths(cfg.out_dir, name)
    risk_cfg = BuildRiskDatasetConfig(
        source_dataset_path=drawdown_out,
        out_path=risk_out,
        summary_path=risk_summary_path,
        timeframe=cfg.timeframe,
        require_usable_target=cfg.require_usable_target,
    )

    source_df = _load_source_dataset(risk_cfg)
    filtered_df, filter_stats = _filter_rows(source_df, risk_cfg)
    kept_columns, feature_cols = _ordered_columns(filtered_df, risk_cfg)
    risk_df = filtered_df.loc[:, kept_columns].copy()
    risk_df.to_csv(risk_out, index=False)

    risk_summary = _build_risk_summary(risk_cfg, source_df, risk_df, filter_stats, feature_cols, kept_columns)
    write_json(risk_summary_path, risk_summary)
    return {
        "risk_dataset_path": str(risk_out),
        "risk_summary_path": str(risk_summary_path),
        "risk_rows": int(len(risk_df)),
        "risk_tickers": int(risk_df["ticker"].nunique()) if not risk_df.empty else 0,
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--dataset-spec",
        action="append",
        default=[],
        help="Repeat NAME=PATH to prepare multiple split datasets. Defaults to main, out-of-time, and unseen tickers.",
    )
    ap.add_argument("--market-data-dir", type=str, default=str(DEFAULT_MARKET_DATA_DIR))
    ap.add_argument("--out-dir", type=str, default=str(DEFAULT_OUT_DIR))
    ap.add_argument("--timeframe", type=str, default="annual")
    ap.add_argument("--keep-nonusable-targets", action="store_true")
    ap.add_argument("--horizon-days", type=int, default=365)
    ap.add_argument("--calendar-buffer-days", type=int, default=7)
    ap.add_argument("--min-future-bars", type=int, default=30)
    ap.add_argument("--max-tickers", type=int, default=0)
    args = ap.parse_args()

    cfg = PrepareRiskDatasetsConfig(
        market_data_dir=Path(args.market_data_dir),
        out_dir=Path(args.out_dir),
        timeframe=str(args.timeframe).strip().lower(),
        require_usable_target=not bool(args.keep_nonusable_targets),
        horizon_days=int(args.horizon_days),
        calendar_buffer_days=int(args.calendar_buffer_days),
        min_future_bars=int(args.min_future_bars),
        max_tickers=int(args.max_tickers),
    )
    ensure_dir(cfg.out_dir)

    dataset_specs = _parse_dataset_specs(args.dataset_spec or None)
    summary: dict[str, Any] = {
        "config": asdict(cfg),
        "datasets": {},
    }

    for name, dataset_path in dataset_specs:
        dataset_summary: dict[str, object] = {"source_dataset_path": str(dataset_path)}
        dataset_summary.update(_prepare_market_dataset(name, dataset_path, cfg))
        dataset_summary.update(_prepare_drawdown_dataset(name, cfg))
        dataset_summary.update(_prepare_model_dataset(name, cfg))
        summary["datasets"][name] = dataset_summary

    summary_path = cfg.out_dir / "prepare_risk_datasets_summary.json"
    write_json(summary_path, summary)

    print("Prepared risk datasets:")
    for name in summary["datasets"]:
        info = summary["datasets"][name]
        print(
            f"- {name}: risk_rows={info['risk_rows']} | risk_tickers={info['risk_tickers']} | "
            f"dataset={info['risk_dataset_path']}"
        )
    print(f"- summary: {summary_path}")


if __name__ == "__main__":
    main()
