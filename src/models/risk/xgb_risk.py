from __future__ import annotations

import argparse
from pathlib import Path

from src.models.risk.modeling import (
    RiskXGBConfig,
    ensure_dir,
    load_risk_dataset,
    summary_payload,
    train_risk_xgb,
    write_json,
)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", type=str, default=None)
    ap.add_argument("--out-dir", type=str, default="experiments/risk/runs/xgb_risk_artifacts")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--disable-sample-weights", action="store_true")
    ap.add_argument("--num-boost-round", type=int, default=4000)
    ap.add_argument("--early-stopping-rounds", type=int, default=100)
    args = ap.parse_args()

    cfg = RiskXGBConfig()
    cfg.out_dir = Path(args.out_dir)
    cfg.random_seed = int(args.seed)
    cfg.use_sample_weights = not bool(args.disable_sample_weights)
    cfg.num_boost_round = int(args.num_boost_round)
    cfg.early_stopping_rounds = int(args.early_stopping_rounds)
    if args.data:
        cfg.data_path = Path(args.data)

    ensure_dir(cfg.out_dir)

    df = load_risk_dataset(cfg.data_path, cfg)
    result = train_risk_xgb(df, cfg)

    preds_path = cfg.out_dir / "predictions.csv"
    result["predictions"].to_csv(preds_path, index=False)

    feature_profile_path = cfg.out_dir / "feature_profile.csv"
    result["feature_profile"].to_csv(feature_profile_path, index=False)

    model_path = cfg.out_dir / "xgb_model.json"
    result["booster"].save_model(model_path)

    summary = summary_payload(cfg, result["metrics"], result["feature_cols"], result["baseline"])
    summary["artifacts"] = {
        "predictions_path": str(preds_path),
        "feature_profile_path": str(feature_profile_path),
        "model_path": str(model_path),
    }
    summary_path = cfg.out_dir / "run_summary.json"
    write_json(summary_path, summary)

    print("Saved risk model artifacts:")
    print(f"- {summary_path}")
    print(f"- {preds_path}")
    print(f"- {feature_profile_path}")
    print(f"- {model_path}")
    print(
        f"val_rmse={summary['metrics']['val_rmse']:.6f} | "
        f"val_mae={summary['metrics']['val_mae']:.6f} | "
        f"val_spearman={summary['metrics']['val_spearman']:.6f}"
    )


if __name__ == "__main__":
    main()
