from __future__ import annotations

import argparse
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.model_selection import GroupKFold, KFold

from src.models.valuation.exp_xgb_valuation import (
    _deduplicate_entity_period_rows,
    _ensure_dir,
    _jsonable,
    _make_asset_weights_from_log_target,
    _rmse,
    _select_feature_columns,
)

RAW_LEVEL_FEATURES = {
    "revenue",
    "net_income",
    "operating_income",
    "total_assets",
    "total_liabilities",
    "total_equity",
    "cash_and_equivalents",
    "total_debt",
    "cfo",
    "capex",
    "fcf",
    "net_debt",
}


@dataclass
class GroupCVXGBValuationConfig:
    # Data and output
    data_path: Path = Path("data/processed/main_dataset.csv")
    out_dir: Path = Path("experiments/valuation/runs/exp_xgb_groupcv_valuation_artifacts")

    # Target
    target_col: str = "total_assets"
    log_target: bool = True

    # Split / dedup
    time_col: str = "fiscal_year"
    random_seed: int = 42
    enable_deduplicate_entity_period: bool = True
    dedup_ticker_col: str = "ticker"
    dedup_period_col: str = "period_end"

    # Feature set:
    # - all_numeric: same as default numeric feature selection
    # - ratio_plus_news: removes raw level features to reduce ticker memorization
    feature_set: str = "ratio_plus_news"

    # Clipping in model space (log space if log_target=True)
    use_quantile_clip: bool = True
    clip_q_hi: float = 0.995
    min_target_raw: float = 0.0

    # Weighting
    use_ticker_frequency_weights: bool = True
    use_asset_weights: bool = False
    weight_power: float = 0.25
    weight_clip_min: float = 0.25
    weight_clip_max: float = 4.0

    # CV and search
    cv_splits: int = 5
    num_boost_round: int = 4000
    early_stopping_rounds: int = 80
    max_candidates: int = 8

    # XGBoost base
    tree_method: str = "hist"
    eta: float = 0.03
    nthread: int = -1


def normalize_tickers(series: pd.Series) -> pd.Series:
    return series.astype(str).str.upper().str.strip()


def target_to_model_scale(df: pd.DataFrame, cfg: GroupCVXGBValuationConfig) -> np.ndarray:
    y = (
        pd.to_numeric(df[cfg.target_col], errors="coerce")
        .replace([np.inf, -np.inf], np.nan)
        .fillna(0.0)
        .to_numpy(dtype=np.float32)
    )
    if cfg.log_target:
        y = np.log1p(np.clip(y, a_min=0.0, a_max=None))
    return y.reshape(-1, 1)


def features_from_cols(df: pd.DataFrame, feature_cols: List[str]) -> np.ndarray:
    return (
        df.reindex(columns=feature_cols)
        .replace([np.inf, -np.inf], np.nan)
        .fillna(0.0)
        .to_numpy(dtype=np.float32)
    )


def select_feature_columns_for_mode(df: pd.DataFrame, cfg: GroupCVXGBValuationConfig) -> List[str]:
    feature_cols = _select_feature_columns(df, cfg.target_col, cfg.time_col)
    if cfg.feature_set == "all_numeric":
        return feature_cols
    if cfg.feature_set != "ratio_plus_news":
        raise ValueError(f"Unsupported feature_set='{cfg.feature_set}'.")

    filtered = [c for c in feature_cols if c not in RAW_LEVEL_FEATURES]
    if not filtered:
        raise ValueError("No features remain after ratio_plus_news filtering.")
    return filtered


def build_train_weights(
    df: pd.DataFrame,
    y_model: np.ndarray,
    cfg: GroupCVXGBValuationConfig,
) -> np.ndarray:
    n = int(len(df))
    if n == 0:
        return np.zeros(0, dtype=np.float32)

    w = np.ones(n, dtype=np.float64)

    if cfg.use_ticker_frequency_weights and "ticker" in df.columns:
        tick = normalize_tickers(df["ticker"])
        counts = tick.value_counts().to_dict()
        inv_freq = tick.map(lambda t: 1.0 / max(float(counts.get(t, 1)), 1.0)).to_numpy(dtype=np.float64)
        w *= inv_freq

    if cfg.use_asset_weights:
        asset_w = _make_asset_weights_from_log_target(
            y_model,
            power=cfg.weight_power,
            w_min=cfg.weight_clip_min,
            w_max=cfg.weight_clip_max,
        ).astype(np.float64)
        w *= asset_w

    w /= np.mean(w) + 1e-12
    w = np.clip(w, cfg.weight_clip_min, cfg.weight_clip_max)
    return w.astype(np.float32)


def _candidate_params(cfg: GroupCVXGBValuationConfig) -> List[Dict[str, float]]:
    base = {
        "objective": "reg:squarederror",
        "eval_metric": "rmse",
        "tree_method": cfg.tree_method,
        "eta": cfg.eta,
        "seed": cfg.random_seed,
        "nthread": cfg.nthread,
    }
    candidates: List[Dict[str, float]] = [
        # Strongly regularized
        {"max_depth": 3, "min_child_weight": 8.0, "subsample": 0.75, "colsample_bytree": 0.75, "lambda": 10.0, "alpha": 3.0, "gamma": 0.0},
        {"max_depth": 4, "min_child_weight": 8.0, "subsample": 0.75, "colsample_bytree": 0.75, "lambda": 10.0, "alpha": 2.0, "gamma": 0.0},
        # Balanced
        {"max_depth": 4, "min_child_weight": 5.0, "subsample": 0.80, "colsample_bytree": 0.80, "lambda": 6.0, "alpha": 1.0, "gamma": 0.0},
        {"max_depth": 5, "min_child_weight": 5.0, "subsample": 0.80, "colsample_bytree": 0.80, "lambda": 4.0, "alpha": 1.0, "gamma": 0.0},
        # Slightly less regularized
        {"max_depth": 5, "min_child_weight": 3.0, "subsample": 0.90, "colsample_bytree": 0.90, "lambda": 2.0, "alpha": 0.5, "gamma": 0.0},
        {"max_depth": 6, "min_child_weight": 3.0, "subsample": 0.90, "colsample_bytree": 0.90, "lambda": 1.0, "alpha": 0.0, "gamma": 0.0},
        # Additional controlled variants
        {"max_depth": 4, "min_child_weight": 10.0, "subsample": 0.70, "colsample_bytree": 0.85, "lambda": 12.0, "alpha": 4.0, "gamma": 0.2},
        {"max_depth": 5, "min_child_weight": 6.0, "subsample": 0.70, "colsample_bytree": 0.85, "lambda": 8.0, "alpha": 2.0, "gamma": 0.1},
        {"max_depth": 3, "min_child_weight": 12.0, "subsample": 0.80, "colsample_bytree": 0.70, "lambda": 14.0, "alpha": 5.0, "gamma": 0.0},
        {"max_depth": 6, "min_child_weight": 2.0, "subsample": 0.85, "colsample_bytree": 0.85, "lambda": 1.0, "alpha": 0.0, "gamma": 0.0},
    ]
    out: List[Dict[str, float]] = []
    for raw in candidates[: max(1, int(cfg.max_candidates))]:
        p = dict(base)
        p.update(raw)
        out.append(p)
    return out


def make_cv_splits(
    train_df: pd.DataFrame,
    cfg: GroupCVXGBValuationConfig,
) -> Tuple[List[Tuple[np.ndarray, np.ndarray]], str]:
    n_rows = int(len(train_df))
    idx = np.arange(n_rows, dtype=np.int64)
    if n_rows < 2:
        return [(idx, idx)], "single_row"

    if "ticker" in train_df.columns:
        groups = normalize_tickers(train_df["ticker"]).to_numpy()
        n_groups = int(pd.Series(groups).nunique())
        n_splits = min(int(cfg.cv_splits), n_groups)
        if n_splits >= 2:
            splitter = GroupKFold(n_splits=n_splits)
            splits = [(tr.astype(np.int64), va.astype(np.int64)) for tr, va in splitter.split(idx, groups=groups)]
            return splits, "group_ticker"

    n_splits = min(int(cfg.cv_splits), n_rows)
    if n_splits < 2:
        return [(idx, idx)], "single_row"
    splitter = KFold(n_splits=n_splits, shuffle=True, random_state=cfg.random_seed)
    splits = [(tr.astype(np.int64), va.astype(np.int64)) for tr, va in splitter.split(idx)]
    return splits, "row"


def run_groupcv_search(
    X: np.ndarray,
    y: np.ndarray,
    w: np.ndarray | None,
    splits: Sequence[Tuple[np.ndarray, np.ndarray]],
    cfg: GroupCVXGBValuationConfig,
) -> pd.DataFrame:
    candidates = _candidate_params(cfg)
    rows: List[Dict[str, object]] = []

    y_1d = y.reshape(-1).astype(np.float32)
    for cand_id, params in enumerate(candidates):
        fold_rmse: List[float] = []
        fold_best_iter: List[int] = []

        for fold_idx, (tr_idx, va_idx) in enumerate(splits):
            w_tr = None if w is None else w[tr_idx]
            dtr = xgb.DMatrix(X[tr_idx], label=y_1d[tr_idx], weight=w_tr)
            dva = xgb.DMatrix(X[va_idx], label=y_1d[va_idx])
            booster = xgb.train(
                params=params,
                dtrain=dtr,
                num_boost_round=cfg.num_boost_round,
                evals=[(dva, "val")],
                early_stopping_rounds=cfg.early_stopping_rounds,
                verbose_eval=False,
            )
            best_iter = int(getattr(booster, "best_iteration", booster.num_boosted_rounds() - 1))
            pred = booster.predict(dva, iteration_range=(0, best_iter + 1)).reshape(-1, 1)
            rmse = float(_rmse(y[va_idx], pred))
            fold_rmse.append(rmse)
            fold_best_iter.append(best_iter)

            rows.append(
                {
                    "candidate_id": int(cand_id),
                    "fold": int(fold_idx),
                    "fold_rmse_log": rmse,
                    "fold_best_iteration": int(best_iter),
                    "params": json.dumps(params, sort_keys=True),
                }
            )

        rows.append(
            {
                "candidate_id": int(cand_id),
                "fold": "mean",
                "fold_rmse_log": float(np.mean(fold_rmse)),
                "fold_best_iteration": int(np.median(fold_best_iter)),
                "rmse_log_std": float(np.std(fold_rmse)),
                "params": json.dumps(params, sort_keys=True),
            }
        )

    cv_df = pd.DataFrame(rows)
    mean_rows = cv_df[cv_df["fold"] == "mean"].copy()
    mean_rows = mean_rows.sort_values(["fold_rmse_log", "rmse_log_std"], ascending=[True, True], kind="mergesort")
    ordered_ids = mean_rows["candidate_id"].astype(int).tolist()

    cv_df["rank"] = np.nan
    for rank, cid in enumerate(ordered_ids, start=1):
        cv_df.loc[cv_df["candidate_id"] == cid, "rank"] = float(rank)
    return cv_df


def train_groupcv_xgb(
    train_df: pd.DataFrame,
    cfg: GroupCVXGBValuationConfig,
) -> Dict[str, object]:
    feature_cols = select_feature_columns_for_mode(train_df, cfg)
    X = features_from_cols(train_df, feature_cols)
    y = target_to_model_scale(train_df, cfg)
    w = build_train_weights(train_df, y, cfg)

    splits, split_mode = make_cv_splits(train_df, cfg)
    cv_df = run_groupcv_search(X, y, w, splits, cfg)

    mean_rows = cv_df[cv_df["fold"] == "mean"].copy()
    mean_rows = mean_rows.sort_values(["fold_rmse_log", "rmse_log_std"], ascending=[True, True], kind="mergesort")
    if mean_rows.empty:
        raise RuntimeError("No CV mean rows were generated.")

    best = mean_rows.iloc[0]
    best_candidate_id = int(best["candidate_id"])
    best_params = json.loads(str(best["params"]))
    best_iteration = int(best["fold_best_iteration"])
    best_num_boost_round = max(1, best_iteration + 1)

    dtrain = xgb.DMatrix(X, label=y.reshape(-1).astype(np.float32), weight=w)
    booster = xgb.train(
        params=best_params,
        dtrain=dtrain,
        num_boost_round=best_num_boost_round,
        evals=[(dtrain, "train")],
        verbose_eval=False,
    )

    yhat_train_preclip = booster.predict(dtrain, iteration_range=(0, best_num_boost_round)).reshape(-1, 1)
    if cfg.log_target:
        clip_lo = float(np.log1p(max(float(cfg.min_target_raw), 0.0)))
    else:
        clip_lo = float(max(float(cfg.min_target_raw), 0.0))
    clip_hi = float(np.max(y))
    if cfg.use_quantile_clip:
        clip_hi = float(np.quantile(y, cfg.clip_q_hi))
    yhat_train = np.clip(yhat_train_preclip, clip_lo, clip_hi)

    if cfg.log_target:
        y_train_raw = np.expm1(y)
        yhat_train_raw = np.expm1(yhat_train)
    else:
        y_train_raw = y
        yhat_train_raw = yhat_train

    metrics = {
        "rows_train": int(len(train_df)),
        "feature_count": int(len(feature_cols)),
        "cv_split_mode": split_mode,
        "cv_splits": int(len(splits)),
        "best_candidate_id": best_candidate_id,
        "best_iteration": int(best_iteration),
        "best_num_boost_round": int(best_num_boost_round),
        "cv_rmse_log_mean": float(best["fold_rmse_log"]),
        "cv_rmse_log_std": float(best.get("rmse_log_std", np.nan)),
        "train_rmse_log_preclip": float(_rmse(y, yhat_train_preclip)),
        "train_rmse_log": float(_rmse(y, yhat_train)),
        "train_rmse_raw": float(_rmse(y_train_raw, yhat_train_raw)),
    }

    preds = train_df.copy()
    preds["y_true"] = y_train_raw.reshape(-1)
    preds["y_pred"] = yhat_train_raw.reshape(-1)
    preds["y_pred_preclip"] = np.expm1(yhat_train_preclip).reshape(-1) if cfg.log_target else yhat_train_preclip.reshape(-1)

    return {
        "booster": booster,
        "feature_names": feature_cols,
        "weights_used": bool(cfg.use_ticker_frequency_weights or cfg.use_asset_weights),
        "best_params": best_params,
        "clip_lo": float(clip_lo),
        "clip_hi": float(clip_hi),
        "cv_results": cv_df,
        "metrics": metrics,
        "train_predictions": preds,
    }


def _save_training_artifacts(
    out_dir: Path,
    cfg: GroupCVXGBValuationConfig,
    train_info: Dict[str, object],
    dedup_stats: Dict[str, object],
) -> Dict[str, str]:
    _ensure_dir(out_dir)

    model_path = out_dir / "xgb_groupcv_model.json"
    cv_path = out_dir / "xgb_groupcv_cv_results.csv"
    preds_path = out_dir / "xgb_groupcv_train_predictions.csv"
    summary_path = out_dir / "xgb_groupcv_run_summary.json"

    booster: xgb.Booster = train_info["booster"]  # type: ignore[assignment]
    booster.save_model(str(model_path))
    cv_df: pd.DataFrame = train_info["cv_results"]  # type: ignore[assignment]
    cv_df.to_csv(cv_path, index=False)
    train_preds: pd.DataFrame = train_info["train_predictions"]  # type: ignore[assignment]
    train_preds.to_csv(preds_path, index=False)

    summary = {
        "config": {k: _jsonable(v) for k, v in asdict(cfg).items()},
        "feature_names": train_info["feature_names"],
        "best_params": train_info["best_params"],
        "clip": {
            "use_quantile_clip": bool(cfg.use_quantile_clip),
            "clip_q_hi": float(cfg.clip_q_hi),
            "min_target_raw": float(cfg.min_target_raw),
            "clip_lo": float(train_info["clip_lo"]),
            "clip_hi": float(train_info["clip_hi"]),
        },
        "dedup": dedup_stats,
        "metrics": train_info["metrics"],
        "artifacts": {
            "model_path": str(model_path),
            "cv_results_path": str(cv_path),
            "train_predictions_path": str(preds_path),
        },
    }
    with summary_path.open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    return {
        "model_path": str(model_path),
        "cv_results_path": str(cv_path),
        "train_predictions_path": str(preds_path),
        "summary_path": str(summary_path),
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--main", type=str, default="data/processed/main_dataset.csv")
    ap.add_argument("--out-dir", type=str, default="experiments/valuation/runs/exp_xgb_groupcv_valuation_artifacts")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--feature-set", type=str, choices=["all_numeric", "ratio_plus_news"], default="ratio_plus_news")
    ap.add_argument("--cv-splits", type=int, default=5)
    ap.add_argument("--max-candidates", type=int, default=8)
    ap.add_argument("--num-boost-round", type=int, default=4000)
    ap.add_argument("--early-stopping-rounds", type=int, default=80)
    ap.add_argument("--disable-ticker-frequency-weights", action="store_true")
    ap.add_argument("--use-asset-weights", action="store_true")
    args = ap.parse_args()

    cfg = GroupCVXGBValuationConfig()
    cfg.data_path = Path(args.main)
    cfg.out_dir = Path(args.out_dir)
    cfg.random_seed = int(args.seed)
    cfg.feature_set = str(args.feature_set)
    cfg.cv_splits = int(args.cv_splits)
    cfg.max_candidates = int(args.max_candidates)
    cfg.num_boost_round = int(args.num_boost_round)
    cfg.early_stopping_rounds = int(args.early_stopping_rounds)
    cfg.use_ticker_frequency_weights = not bool(args.disable_ticker_frequency_weights)
    cfg.use_asset_weights = bool(args.use_asset_weights)

    if not cfg.data_path.exists():
        raise SystemExit(f"Dataset not found: {cfg.data_path.resolve()}")

    raw_df = pd.read_csv(cfg.data_path)
    if cfg.enable_deduplicate_entity_period:
        df, dedup_stats = _deduplicate_entity_period_rows(
            raw_df,
            ticker_col=cfg.dedup_ticker_col,
            period_col=cfg.dedup_period_col,
            time_col=cfg.time_col,
        )
    else:
        df = raw_df.copy()
        dedup_stats = {
            "enabled": False,
            "applied": False,
            "rows_before": int(len(raw_df)),
            "rows_after": int(len(raw_df)),
            "rows_dropped": 0,
        }

    train_info = train_groupcv_xgb(df, cfg)
    saved = _save_training_artifacts(cfg.out_dir, cfg, train_info, dedup_stats)

    metrics = train_info["metrics"]
    print("Saved:")
    print(f"- {saved['summary_path']}")
    print(f"- {saved['model_path']}")
    print(f"- {saved['cv_results_path']}")
    print(f"- {saved['train_predictions_path']}")
    print("\nGroupCV XGB training summary:")
    print(f"rows={metrics['rows_train']} | features={metrics['feature_count']} | cv_mode={metrics['cv_split_mode']}")
    print(f"cv_rmse_log_mean={metrics['cv_rmse_log_mean']:.6f} | cv_rmse_log_std={metrics['cv_rmse_log_std']:.6f}")
    print(
        f"best_candidate_id={metrics['best_candidate_id']} | "
        f"best_iteration={metrics['best_iteration']} | "
        f"best_num_boost_round={metrics['best_num_boost_round']}"
    )
    print(
        f"train_rmse_log={metrics['train_rmse_log']:.6f} | "
        f"train_rmse_log_preclip={metrics['train_rmse_log_preclip']:.6f}"
    )


if __name__ == "__main__":
    main()
