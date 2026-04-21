from __future__ import annotations

import argparse
from pathlib import Path

from src.models.risk.mlp_risk import RiskMLPConfig, train_and_eval_risk_mlp
from src.models.risk.modeling import LEGACY_MAIN_RISK_DATASET, ensure_dir, load_risk_dataset, write_json

DEFAULT_MAIN = Path("data/processed/risk/main_risk_dataset.csv")
DEFAULT_TEST_OUT_OF_TIME = Path("data/processed/risk/test_out_of_time_risk_dataset.csv")
DEFAULT_TEST_UNSEEN = Path("data/processed/risk/test_unseen_tickers_risk_dataset.csv")
DEFAULT_OUT_DIR = Path("experiments/risk/runs/mlp_risk_evaluation_artifacts")


def _default_main_dataset() -> Path:
    if DEFAULT_MAIN.exists():
        return DEFAULT_MAIN
    return LEGACY_MAIN_RISK_DATASET


def _normalize_tickers(series):
    return series.astype(str).str.upper().str.strip()


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--main", type=str, default=str(_default_main_dataset()))
    ap.add_argument("--test-out-of-time", type=str, default=str(DEFAULT_TEST_OUT_OF_TIME))
    ap.add_argument("--test-unseen-tickers", type=str, default=str(DEFAULT_TEST_UNSEEN))
    ap.add_argument("--out-dir", type=str, default=str(DEFAULT_OUT_DIR))
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--epochs", type=int, default=400)
    ap.add_argument("--batch-size", type=int, default=128)
    ap.add_argument("--disable-sample-weights", action="store_true")
    ap.add_argument(
        "--hidden-dims",
        type=int,
        nargs="+",
        default=None,
        help="Override hidden layer sizes, e.g. --hidden-dims 128 64",
    )
    args = ap.parse_args()

    cfg = RiskMLPConfig()
    cfg.random_seed = int(args.seed)
    cfg.epochs = int(args.epochs)
    cfg.batch_size = int(args.batch_size)
    cfg.use_sample_weights = not bool(args.disable_sample_weights)
    if args.hidden_dims is not None:
        if len(args.hidden_dims) == 0:
            raise SystemExit("--hidden-dims was provided but no values were supplied.")
        cfg.hidden_dims = tuple(int(v) for v in args.hidden_dims)

    main_df = load_risk_dataset(Path(args.main), cfg)
    oot_df = load_risk_dataset(Path(args.test_out_of_time), cfg)
    unseen_df = load_risk_dataset(Path(args.test_unseen_tickers), cfg)

    out_dir = Path(args.out_dir)
    ensure_dir(out_dir)

    oot_years = sorted(oot_df[cfg.time_col].dropna().astype(int).unique().tolist()) if cfg.time_col in oot_df.columns else []
    train_oot = (
        main_df.loc[~main_df[cfg.time_col].isin(oot_years)].copy()
        if cfg.time_col in main_df.columns and oot_years
        else main_df.copy()
    )
    if train_oot.empty:
        raise SystemExit(
            "Out-of-time training slice is empty after excluding the test years from the main risk dataset. "
            "Prepare a main risk dataset with earlier years or provide a different test split."
        )
    res_oot = train_and_eval_risk_mlp(train_oot, oot_df, cfg)
    oot_preds_path = out_dir / "risk_mlp_test_out_of_time_predictions.csv"
    res_oot["predictions"].to_csv(oot_preds_path, index=False)
    res_oot["feature_profile"].to_csv(out_dir / "risk_mlp_test_out_of_time_feature_profile.csv", index=False)

    unseen_tickers = set(_normalize_tickers(unseen_df[cfg.ticker_col]).dropna()) if cfg.ticker_col in unseen_df.columns else set()
    train_unseen = (
        main_df.loc[~_normalize_tickers(main_df[cfg.ticker_col]).isin(unseen_tickers)].copy()
        if cfg.ticker_col in main_df.columns and unseen_tickers
        else main_df.copy()
    )
    if train_unseen.empty:
        raise SystemExit(
            "Unseen-ticker training slice is empty after excluding the held-out tickers from the main risk dataset. "
            "Prepare a broader main risk dataset or provide a different unseen-ticker split."
        )
    res_unseen = train_and_eval_risk_mlp(train_unseen, unseen_df, cfg)
    unseen_preds_path = out_dir / "risk_mlp_test_unseen_tickers_predictions.csv"
    res_unseen["predictions"].to_csv(unseen_preds_path, index=False)
    res_unseen["feature_profile"].to_csv(out_dir / "risk_mlp_test_unseen_tickers_feature_profile.csv", index=False)

    summary = {
        "config": {
            "main_dataset": args.main,
            "test_out_of_time": args.test_out_of_time,
            "test_unseen_tickers": args.test_unseen_tickers,
            "seed": cfg.random_seed,
            "epochs": cfg.epochs,
            "batch_size": cfg.batch_size,
            "use_sample_weights": cfg.use_sample_weights,
            "hidden_dims": list(cfg.hidden_dims),
        },
        "out_of_time": {
            "years": oot_years,
            "train_rows": int(len(train_oot)),
            "test_rows": int(len(oot_df)),
            "feature_names": res_oot["feature_cols"],
            "baseline": res_oot["baseline"],
            "training": res_oot["training"],
            "test_metrics": res_oot["metrics"],
            "predictions_path": str(oot_preds_path),
        },
        "unseen_tickers": {
            "heldout_tickers": int(len(unseen_tickers)),
            "train_rows": int(len(train_unseen)),
            "test_rows": int(len(unseen_df)),
            "feature_names": res_unseen["feature_cols"],
            "baseline": res_unseen["baseline"],
            "training": res_unseen["training"],
            "test_metrics": res_unseen["metrics"],
            "predictions_path": str(unseen_preds_path),
        },
    }
    summary_path = out_dir / "risk_mlp_test_set_evaluation_summary.json"
    write_json(summary_path, summary)

    print("Saved risk MLP test-set evaluation artifacts:")
    print(f"- {summary_path}")
    print(f"- {oot_preds_path}")
    print(f"- {unseen_preds_path}")
    print(
        f"out_of_time rmse={summary['out_of_time']['test_metrics']['test_rmse']:.6f} | "
        f"mae={summary['out_of_time']['test_metrics']['test_mae']:.6f} | "
        f"spearman={summary['out_of_time']['test_metrics']['test_spearman']:.6f}"
    )
    print(
        f"unseen_ticker rmse={summary['unseen_tickers']['test_metrics']['test_rmse']:.6f} | "
        f"mae={summary['unseen_tickers']['test_metrics']['test_mae']:.6f} | "
        f"spearman={summary['unseen_tickers']['test_metrics']['test_spearman']:.6f}"
    )


if __name__ == "__main__":
    main()
