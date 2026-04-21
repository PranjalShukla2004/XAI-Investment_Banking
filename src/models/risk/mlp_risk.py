from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from src.models.nn.losses import HuberLoss
from src.models.nn.mlp import MLP
from src.models.nn.optimizer import Adam
from src.models.nn.train import TrainConfig, fit
from src.models.risk.modeling import (
    DEFAULT_OUT_DIR,
    _add_prefixed_metrics,
    _baseline_predictions,
    _build_matrix,
    _clip_predictions,
    _default_risk_dataset_path,
    _inner_train_val_indices,
    _make_sample_weights,
    _prediction_frame,
    _target_array,
    ensure_dir,
    fit_standard_scaler,
    load_risk_dataset,
    risk_metrics,
    select_feature_columns,
    transform_standard_scaler,
    write_json,
)

DEFAULT_MLP_OUT_DIR = DEFAULT_OUT_DIR.parent / "mlp_risk_artifacts"


@dataclass
class RiskMLPConfig:
    data_path: Path = _default_risk_dataset_path()
    out_dir: Path = DEFAULT_MLP_OUT_DIR
    target_col: str = "drawdown_severity"
    raw_target_col: str = "future_1y_max_drawdown"
    time_col: str = "fiscal_year"
    ticker_col: str = "ticker"
    period_end_col: str = "period_end"
    baseline_feature_col: str = "drawdown_126d"
    val_ratio_fallback: float = 0.15
    min_val_rows: int = 250
    random_seed: int = 42
    use_sample_weights: bool = True
    sample_weight_power: float = 1.0
    sample_weight_min: float = 1.0
    sample_weight_max: float = 2.0
    top_fraction: float = 0.1
    event_thresholds: tuple[float, ...] = (0.3, 0.5)

    epochs: int = 400
    batch_size: int = 128
    lr: float = 5e-4
    weight_decay: float = 1e-5
    print_every: int = 10
    early_stopping: bool = True
    patience: int = 40
    min_delta: float = 1e-4
    restore_best: bool = True
    lr_scheduler: str = "reduce_on_plateau"
    lr_factor: float = 0.5
    lr_patience: int = 15
    lr_min: float = 1e-6
    lr_threshold: float = 1e-5
    lr_cooldown: int = 5
    huber_delta: float = 0.05

    hidden_dims: tuple[int, ...] = (128, 64)
    dropout: float = 0.15
    activation: str = "relu"
    init: str = "he"
    weight_scale: float = 0.01
    l2_in_layers: float = 5e-6
    l1_in_layers: float = 0.0


def _inverse_target_scaler(y_scaled: np.ndarray, scaler: dict[str, np.ndarray]) -> np.ndarray:
    return np.asarray(y_scaled, dtype=np.float64) * scaler["sigma"] + scaler["mu"]


def _fit_model(train_df, cfg: RiskMLPConfig) -> dict[str, object]:
    feature_cols, feature_profile = select_feature_columns(train_df, cfg)
    X_train = _build_matrix(train_df, feature_cols)
    y_train = _target_array(train_df, cfg.target_col).reshape(-1, 1)

    x_scaler = fit_standard_scaler(X_train)
    X_train_s = transform_standard_scaler(X_train, x_scaler).astype(np.float32)

    y_scaler = fit_standard_scaler(y_train)
    y_train_s = transform_standard_scaler(y_train, y_scaler).astype(np.float32)

    fit_idx, val_idx, inner_split_mode = _inner_train_val_indices(train_df, cfg)
    X_fit = X_train_s[fit_idx]
    y_fit = y_train_s[fit_idx]
    X_val = X_train_s[val_idx]
    y_val = y_train_s[val_idx]

    model = MLP.from_dims(
        input_dim=int(X_train_s.shape[1]),
        hidden_dims=cfg.hidden_dims,
        output_dim=1,
        activation=cfg.activation.lower(),
        dropout=float(cfg.dropout),
        init=cfg.init,
        weight_scale=float(cfg.weight_scale),
        l2=float(cfg.l2_in_layers),
        l1=float(cfg.l1_in_layers),
    )

    loss_fn = HuberLoss(delta=float(cfg.huber_delta))
    opt = Adam(lr=float(cfg.lr), weight_decay=float(cfg.weight_decay))
    train_cfg = TrainConfig(
        epochs=int(cfg.epochs),
        batch_size=int(cfg.batch_size),
        seed=int(cfg.random_seed),
        print_every=int(cfg.print_every),
        early_stopping=bool(cfg.early_stopping),
        patience=int(cfg.patience),
        min_delta=float(cfg.min_delta),
        restore_best=bool(cfg.restore_best),
        lr_scheduler=str(cfg.lr_scheduler),
        lr_factor=float(cfg.lr_factor),
        lr_patience=int(cfg.lr_patience),
        lr_min=float(cfg.lr_min),
        lr_threshold=float(cfg.lr_threshold),
        lr_cooldown=int(cfg.lr_cooldown),
    )

    w_fit = _make_sample_weights(y_train[fit_idx].reshape(-1), cfg)
    history = fit(
        model=model,
        criterion=loss_fn,
        optimizer=opt,
        X_train=X_fit,
        y_train=y_fit,
        X_val=X_val,
        y_val=y_val,
        cfg=train_cfg,
        metric_fn=None,
        w_train=w_fit,
        w_val=None,
    )

    return {
        "model": model,
        "feature_cols": feature_cols,
        "feature_profile": feature_profile,
        "x_scaler": x_scaler,
        "y_scaler": y_scaler,
        "history": history,
        "fit_idx": fit_idx,
        "val_idx": val_idx,
        "inner_split_mode": inner_split_mode,
        "train_target_mean": float(np.mean(y_train)),
    }


def _predict(model: MLP, df, feature_cols: list[str], x_scaler, y_scaler) -> np.ndarray:
    X = _build_matrix(df, feature_cols)
    X_s = transform_standard_scaler(X, x_scaler).astype(np.float32)
    model.eval()
    yhat_s = model.forward(X_s, training=False)
    yhat = _inverse_target_scaler(yhat_s, y_scaler)
    return _clip_predictions(yhat)


def train_risk_mlp(df, cfg: RiskMLPConfig) -> dict[str, object]:
    fitted = _fit_model(df, cfg)
    train_target_mean = float(fitted["train_target_mean"])
    train_pred = _predict(
        fitted["model"],
        df,
        fitted["feature_cols"],
        fitted["x_scaler"],
        fitted["y_scaler"],
    )
    baseline_pred, baseline_info = _baseline_predictions(df, train_target_mean, cfg)

    fit_idx = np.asarray(fitted["fit_idx"], dtype=np.int64)
    val_idx = np.asarray(fitted["val_idx"], dtype=np.int64)
    y_true = _target_array(df, cfg.target_col)

    train_metrics = risk_metrics(y_true[fit_idx], train_pred[fit_idx], cfg)
    val_metrics = risk_metrics(y_true[val_idx], train_pred[val_idx], cfg)
    baseline_train_metrics = risk_metrics(y_true[fit_idx], baseline_pred[fit_idx], cfg)
    baseline_val_metrics = risk_metrics(y_true[val_idx], baseline_pred[val_idx], cfg)

    preds_fit = _prediction_frame(df.iloc[fit_idx].copy(), train_pred[fit_idx], baseline_pred[fit_idx], cfg, split="train")
    preds_val = _prediction_frame(df.iloc[val_idx].copy(), train_pred[val_idx], baseline_pred[val_idx], cfg, split="val")
    predictions = pd.concat([preds_fit, preds_val], ignore_index=True)

    metrics: dict[str, object] = {
        "rows_total": int(len(df)),
        "rows_train": int(fit_idx.size),
        "rows_val": int(val_idx.size),
        "unique_tickers_total": int(df[cfg.ticker_col].nunique()) if cfg.ticker_col in df.columns else None,
        "feature_count": int(len(fitted["feature_cols"])),
        "inner_split_mode": str(fitted["inner_split_mode"]),
    }
    _add_prefixed_metrics("train", train_metrics, metrics)
    _add_prefixed_metrics("val", val_metrics, metrics)
    _add_prefixed_metrics("baseline_train", baseline_train_metrics, metrics)
    _add_prefixed_metrics("baseline_val", baseline_val_metrics, metrics)

    training = {
        "best_epoch": fitted["history"].get("best_epoch"),
        "best_val_loss": fitted["history"].get("best_val_loss"),
        "stopped_early": fitted["history"].get("stopped_early"),
        "final_lr": fitted["history"]["lr"][-1] if fitted["history"].get("lr") else None,
        "lr_reductions": fitted["history"].get("lr_reductions"),
    }

    return {
        "metrics": metrics,
        "training": training,
        "history": fitted["history"],
        "predictions": predictions,
        "feature_cols": fitted["feature_cols"],
        "feature_profile": fitted["feature_profile"],
        "baseline": baseline_info,
        "model": fitted["model"],
        "x_scaler": fitted["x_scaler"],
        "y_scaler": fitted["y_scaler"],
    }


def train_and_eval_risk_mlp(train_df, test_df, cfg: RiskMLPConfig) -> dict[str, object]:
    fitted = _fit_model(train_df, cfg)
    train_target_mean = float(fitted["train_target_mean"])
    test_pred = _predict(
        fitted["model"],
        test_df,
        fitted["feature_cols"],
        fitted["x_scaler"],
        fitted["y_scaler"],
    )
    baseline_pred, baseline_info = _baseline_predictions(test_df, train_target_mean, cfg)
    y_true = _target_array(test_df, cfg.target_col)
    test_metrics = risk_metrics(y_true, test_pred, cfg)
    baseline_metrics = risk_metrics(y_true, baseline_pred, cfg)

    metrics: dict[str, object] = {
        "rows_test": int(len(test_df)),
        "unique_tickers_test": int(test_df[cfg.ticker_col].nunique()) if cfg.ticker_col in test_df.columns else None,
        "feature_count": int(len(fitted["feature_cols"])),
        "inner_split_mode": str(fitted["inner_split_mode"]),
    }
    _add_prefixed_metrics("test", test_metrics, metrics)
    _add_prefixed_metrics("baseline_test", baseline_metrics, metrics)

    training = {
        "best_epoch": fitted["history"].get("best_epoch"),
        "best_val_loss": fitted["history"].get("best_val_loss"),
        "stopped_early": fitted["history"].get("stopped_early"),
        "final_lr": fitted["history"]["lr"][-1] if fitted["history"].get("lr") else None,
        "lr_reductions": fitted["history"].get("lr_reductions"),
    }

    predictions = _prediction_frame(test_df, test_pred, baseline_pred, cfg, split="test")
    return {
        "metrics": metrics,
        "training": training,
        "history": fitted["history"],
        "predictions": predictions,
        "feature_cols": fitted["feature_cols"],
        "feature_profile": fitted["feature_profile"],
        "baseline": baseline_info,
        "model": fitted["model"],
        "x_scaler": fitted["x_scaler"],
        "y_scaler": fitted["y_scaler"],
    }


def _save_scalers(path: Path, feature_cols: list[str], x_scaler, y_scaler) -> None:
    ensure_dir(path.parent)
    np.savez_compressed(
        path,
        feature_names=np.array(feature_cols, dtype=object),
        x_mu=np.asarray(x_scaler["mu"]),
        x_sigma=np.asarray(x_scaler["sigma"]),
        y_mu=np.asarray(y_scaler["mu"]),
        y_sigma=np.asarray(y_scaler["sigma"]),
    )


def _save_model_weights(path: Path, model: MLP) -> None:
    ensure_dir(path.parent)
    payload: dict[str, np.ndarray] = {}
    for ref in model.params_and_grads():
        payload[f"layer_{ref.layer_idx}_{ref.name}"] = np.asarray(ref.value)
    np.savez_compressed(path, **payload)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", type=str, default=None)
    ap.add_argument("--out-dir", type=str, default=str(DEFAULT_MLP_OUT_DIR))
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
    cfg.out_dir = Path(args.out_dir)
    cfg.random_seed = int(args.seed)
    cfg.epochs = int(args.epochs)
    cfg.batch_size = int(args.batch_size)
    cfg.use_sample_weights = not bool(args.disable_sample_weights)
    if args.data:
        cfg.data_path = Path(args.data)
    if args.hidden_dims is not None:
        if len(args.hidden_dims) == 0:
            raise SystemExit("--hidden-dims was provided but no values were supplied.")
        cfg.hidden_dims = tuple(int(v) for v in args.hidden_dims)

    ensure_dir(cfg.out_dir)
    df = load_risk_dataset(cfg.data_path, cfg)
    result = train_risk_mlp(df, cfg)

    preds_path = cfg.out_dir / "predictions.csv"
    result["predictions"].to_csv(preds_path, index=False)

    feature_profile_path = cfg.out_dir / "feature_profile.csv"
    result["feature_profile"].to_csv(feature_profile_path, index=False)

    history_path = cfg.out_dir / "history.json"
    write_json(history_path, result["history"])

    scalers_path = cfg.out_dir / "scalers_and_features.npz"
    _save_scalers(scalers_path, result["feature_cols"], result["x_scaler"], result["y_scaler"])

    weights_path = cfg.out_dir / "model_weights.npz"
    _save_model_weights(weights_path, result["model"])

    summary = {
        "config": asdict(cfg),
        "feature_names": list(result["feature_cols"]),
        "baseline": result["baseline"],
        "training": result["training"],
        "metrics": result["metrics"],
        "artifacts": {
            "predictions_path": str(preds_path),
            "feature_profile_path": str(feature_profile_path),
            "history_path": str(history_path),
            "scalers_path": str(scalers_path),
            "weights_path": str(weights_path),
        },
    }
    summary_path = cfg.out_dir / "run_summary.json"
    write_json(summary_path, summary)

    print("Saved risk MLP artifacts:")
    print(f"- {summary_path}")
    print(f"- {preds_path}")
    print(f"- {feature_profile_path}")
    print(f"- {history_path}")
    print(f"- {scalers_path}")
    print(f"- {weights_path}")
    print(
        f"val_rmse={summary['metrics']['val_rmse']:.6f} | "
        f"val_mae={summary['metrics']['val_mae']:.6f} | "
        f"val_spearman={summary['metrics']['val_spearman']:.6f}"
    )


if __name__ == "__main__":
    main()
