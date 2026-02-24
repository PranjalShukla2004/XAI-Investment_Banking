from __future__ import annotations

import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Sequence

import numpy as np
import pandas as pd

from .nn.losses import HuberLoss
from .nn.mlp import MLP
from .nn.optimizer import Adam
from .nn.train import TrainConfig, fit
from .valuation import (
    ValuationConfig,
    _fit_standard_scaler,
    _make_asset_weights_from_log_target,
    _mape,
    _r2,
    _rmse,
    _select_feature_columns,
    _select_log1p_features,
    _time_aware_split,
    _to_jsonable,
    _transform_standard_scaler,
    build_xy,
)


@dataclass
class CustomValuationConfig:
    data_path: Path = Path("data/processed/main_dataset.csv")
    out_dir: Path = Path("data/processed/custom_valuation_artifacts")
    target_col: str = "total_assets"
    log_target: bool = True
    time_col: str = "fiscal_year"
    min_val_rows: int = 20
    random_seed: int = 42
    val_ratio_fallback: float = 0.2

    # Only sweep huber delta; everything else follows valuation defaults.
    delta_values: Sequence[float] = (0.05, 0.1, 0.2, 0.3, 0.5, 1.0)

    # Keep ReduceLROnPlateau behavior enabled in custom valuation.
    lr_scheduler: str = "reduce_on_plateau"
    lr_factor: float = 0.5
    lr_patience: int = 20
    lr_min: float = 1e-6
    lr_threshold: float = 5e-5
    lr_cooldown: int = 5


def _ensure_dir(p: Path) -> None:
    p.mkdir(parents=True, exist_ok=True)


def _meta_cols(d: pd.DataFrame) -> Dict[str, np.ndarray]:
    out: Dict[str, np.ndarray] = {}
    for c in ["ticker", "fiscal_year", "period_end", "timeframe"]:
        if c in d.columns:
            out[c] = d[c].to_numpy()
    return out


def _is_better(candidate: dict, best: dict | None) -> bool:
    if best is None:
        return True
    c_rmse = float(candidate["metrics"]["val_rmse_log"])
    b_rmse = float(best["metrics"]["val_rmse_log"])
    if c_rmse < b_rmse - 1e-12:
        return True
    if abs(c_rmse - b_rmse) <= 1e-12:
        return float(candidate["metrics"]["val_mape_raw"]) < float(best["metrics"]["val_mape_raw"])
    return False


def main() -> None:
    run_cfg = CustomValuationConfig()
    base_cfg = ValuationConfig()

    env_path = os.getenv("VAL_DATA_PATH")
    if env_path:
        run_cfg.data_path = Path(env_path)

    if not run_cfg.data_path.exists():
        raise SystemExit(f"Dataset not found at: {run_cfg.data_path.resolve()}")

    _ensure_dir(run_cfg.out_dir)

    # Follow valuation config for model/training defaults.
    base_cfg.data_path = run_cfg.data_path
    base_cfg.target_col = run_cfg.target_col
    base_cfg.log_target = run_cfg.log_target
    base_cfg.time_col = run_cfg.time_col
    base_cfg.min_val_rows = run_cfg.min_val_rows
    base_cfg.random_seed = run_cfg.random_seed
    base_cfg.val_ratio_fallback = run_cfg.val_ratio_fallback

    df = pd.read_csv(run_cfg.data_path)

    train_df, val_df = _time_aware_split(
        df=df,
        time_col=base_cfg.time_col,
        min_val_rows=base_cfg.min_val_rows,
        seed=base_cfg.random_seed,
        val_ratio_fallback=base_cfg.val_ratio_fallback,
    )

    feature_names = _select_feature_columns(train_df, base_cfg.target_col)
    log1p_feature_names = _select_log1p_features(
        train_df=train_df,
        feature_cols=feature_names,
        enabled=base_cfg.use_log1p_feature_transform,
    )

    X_train, y_train, feature_names = build_xy(
        train_df,
        base_cfg,
        feature_cols=feature_names,
        log1p_features=log1p_feature_names,
    )
    X_val, y_val, _ = build_xy(
        val_df,
        base_cfg,
        feature_cols=feature_names,
        log1p_features=log1p_feature_names,
    )

    x_scaler = _fit_standard_scaler(X_train)
    X_train_s = _transform_standard_scaler(X_train, x_scaler)
    X_val_s = _transform_standard_scaler(X_val, x_scaler)

    y_scaler = _fit_standard_scaler(y_train)
    y_train_s = _transform_standard_scaler(y_train, y_scaler)
    y_val_s = _transform_standard_scaler(y_val, y_scaler)

    if base_cfg.use_quantile_clip:
        clip_lo = float(np.quantile(y_train, base_cfg.clip_q_lo))
        clip_hi = float(np.quantile(y_train, base_cfg.clip_q_hi))
    else:
        train_log_min = float(np.min(y_train))
        train_log_max = float(np.max(y_train))
        clip_lo = train_log_min - float(base_cfg.clip_margin)
        clip_hi = train_log_max + float(base_cfg.clip_margin)

    w_train = _make_asset_weights_from_log_target(y_train, power=0.25, w_min=0.5, w_max=2.0)
    train_meta = _meta_cols(train_df)
    val_meta = _meta_cols(val_df)

    sweep_rows: List[dict] = []
    best_payload: dict | None = None

    for idx, delta in enumerate(run_cfg.delta_values, start=1):
        model = MLP.from_dims(
            input_dim=int(X_train_s.shape[1]),
            hidden_dims=base_cfg.hidden_dims,
            output_dim=1,
            activation=base_cfg.activation.lower(),
            dropout=float(base_cfg.dropout),
            init=base_cfg.init,
            weight_scale=float(base_cfg.weight_scale),
            l2=float(base_cfg.l2_in_layers),
        )

        loss_fn = HuberLoss(delta=float(delta))
        opt = Adam(lr=float(base_cfg.lr), weight_decay=float(base_cfg.weight_decay))
        train_cfg = TrainConfig(
            epochs=base_cfg.epochs,
            batch_size=base_cfg.batch_size,
            seed=base_cfg.random_seed,
            print_every=0,
            early_stopping=base_cfg.early_stopping,
            patience=base_cfg.patience,
            min_delta=base_cfg.min_delta,
            restore_best=base_cfg.restore_best,
            lr_scheduler=run_cfg.lr_scheduler,
            lr_factor=run_cfg.lr_factor,
            lr_patience=run_cfg.lr_patience,
            lr_min=run_cfg.lr_min,
            lr_threshold=run_cfg.lr_threshold,
            lr_cooldown=run_cfg.lr_cooldown,
        )

        history = fit(
            model=model,
            criterion=loss_fn,
            optimizer=opt,
            X_train=X_train_s,
            y_train=y_train_s,
            X_val=X_val_s,
            y_val=y_val_s,
            cfg=train_cfg,
            metric_fn=None,
            w_train=w_train,
            w_val=None,
        )

        model.eval()
        yhat_train_s = model.forward(X_train_s, training=False)
        yhat_val_s = model.forward(X_val_s, training=False)

        yhat_train_log = yhat_train_s * y_scaler["sigma"] + y_scaler["mu"]
        yhat_val_log = yhat_val_s * y_scaler["sigma"] + y_scaler["mu"]
        yhat_train_log_clip = np.clip(yhat_train_log, clip_lo, clip_hi)
        yhat_val_log_clip = np.clip(yhat_val_log, clip_lo, clip_hi)

        if base_cfg.log_target:
            yhat_train = np.expm1(yhat_train_log_clip)
            yhat_val = np.expm1(yhat_val_log_clip)
            y_train_orig = np.expm1(y_train)
            y_val_orig = np.expm1(y_val)
        else:
            yhat_train = yhat_train_log
            yhat_val = yhat_val_log
            y_train_orig = y_train
            y_val_orig = y_val

        metrics = {
            "baseline_rmse_log": float(_rmse(y_val, np.full_like(y_val, float(np.mean(y_train))))),
            "baseline_factor": float(np.exp(_rmse(y_val, np.full_like(y_val, float(np.mean(y_train)))))),
            "train_rmse_log": float(_rmse(y_train, yhat_train_log_clip)),
            "train_factor": float(np.exp(_rmse(y_train, yhat_train_log_clip))),
            "val_rmse_log": float(_rmse(y_val, yhat_val_log_clip)),
            "val_factor": float(np.exp(_rmse(y_val, yhat_val_log_clip))),
            "train_r2_log": float(_r2(y_train, yhat_train_log_clip)),
            "val_r2_log": float(_r2(y_val, yhat_val_log_clip)),
            "train_r2_raw": float(_r2(y_train_orig, yhat_train)),
            "val_r2_raw": float(_r2(y_val_orig, yhat_val)),
            "val_mape_raw": float(_mape(y_val_orig, yhat_val)),
            "train_rmse_raw": float(_rmse(y_train_orig, yhat_train)),
            "val_rmse_raw": float(_rmse(y_val_orig, yhat_val)),
        }

        trial = {
            "delta": float(delta),
            "training": {
                "epochs_ran": int(len(history.get("train_loss", []))),
                "best_epoch": _to_jsonable(history.get("best_epoch")),
                "best_val_loss": _to_jsonable(history.get("best_val_loss")),
                "stopped_early": _to_jsonable(history.get("stopped_early")),
                "final_lr": float(history.get("lr", [float(base_cfg.lr)])[-1]) if history.get("lr") else float(base_cfg.lr),
                "lr_reductions": int(history.get("lr_reductions", 0)),
            },
            "metrics": metrics,
            "history": history,
            "yhat_train": yhat_train,
            "yhat_val": yhat_val,
            "y_train_orig": y_train_orig,
            "y_val_orig": y_val_orig,
        }
        sweep_rows.append(
            {
                "delta": float(delta),
                "val_rmse_log": metrics["val_rmse_log"],
                "val_mape_raw": metrics["val_mape_raw"],
                "val_r2_raw": metrics["val_r2_raw"],
                "epochs_ran": trial["training"]["epochs_ran"],
                "final_lr": trial["training"]["final_lr"],
                "lr_reductions": trial["training"]["lr_reductions"],
            }
        )

        if _is_better(trial, best_payload):
            best_payload = trial

        print(
            f"[{idx}/{len(run_cfg.delta_values)}] "
            f"delta={delta:.4g} | val_rmse_log={metrics['val_rmse_log']:.6f} "
            f"| val_mape_raw={metrics['val_mape_raw']:.6f}"
        )

    if best_payload is None:
        raise SystemExit("No delta trials ran.")

    best_delta = float(best_payload["delta"])

    # Save sweep table.
    pd.DataFrame(sweep_rows).sort_values(["val_rmse_log", "val_mape_raw"]).to_csv(
        run_cfg.out_dir / "delta_sweep.csv", index=False
    )

    # Save best history.
    with open(run_cfg.out_dir / "history.json", "w") as f:
        json.dump(best_payload["history"], f, indent=2)

    # Save best scalers + features.
    np.savez(
        run_cfg.out_dir / "scalers_and_features.npz",
        x_mu=x_scaler["mu"],
        x_sigma=x_scaler["sigma"],
        y_mu=y_scaler["mu"],
        y_sigma=y_scaler["sigma"],
        feature_names=np.array(feature_names, dtype=object),
        feature_log1p_names=np.array(log1p_feature_names, dtype=object),
        target_col=np.array([base_cfg.target_col], dtype=object),
        log_target=np.array([base_cfg.log_target], dtype=object),
        use_quantile_clip=np.array([base_cfg.use_quantile_clip], dtype=object),
        clip_q_lo=np.array([base_cfg.clip_q_lo], dtype=np.float32),
        clip_q_hi=np.array([base_cfg.clip_q_hi], dtype=np.float32),
        clip_lo=np.array([clip_lo], dtype=np.float32),
        clip_hi=np.array([clip_hi], dtype=np.float32),
        clip_margin=np.array([base_cfg.clip_margin], dtype=np.float32),
    )

    # Save best predictions.
    rows: List[dict] = []
    for i in range(len(train_df)):
        r = {
            "split": "train",
            "y_true": float(best_payload["y_train_orig"][i, 0]),
            "y_pred": float(best_payload["yhat_train"][i, 0]),
        }
        for k, arr in train_meta.items():
            r[k] = arr[i]
        rows.append(r)
    for i in range(len(val_df)):
        r = {
            "split": "val",
            "y_true": float(best_payload["y_val_orig"][i, 0]),
            "y_pred": float(best_payload["yhat_val"][i, 0]),
        }
        for k, arr in val_meta.items():
            r[k] = arr[i]
        rows.append(r)
    pd.DataFrame(rows).to_csv(run_cfg.out_dir / "predictions.csv", index=False)

    run_summary = {
        "target_col": base_cfg.target_col,
        "log_target": base_cfg.log_target,
        "train_rows": int(len(train_df)),
        "val_rows": int(len(val_df)),
        "years": sorted(df[base_cfg.time_col].dropna().unique().tolist()) if base_cfg.time_col in df.columns else None,
        "best_delta": best_delta,
        "delta_values": [float(x) for x in run_cfg.delta_values],
        "scheduler": {
            "type": run_cfg.lr_scheduler,
            "factor": float(run_cfg.lr_factor),
            "patience": int(run_cfg.lr_patience),
            "min_lr": float(run_cfg.lr_min),
            "threshold": float(run_cfg.lr_threshold),
            "cooldown": int(run_cfg.lr_cooldown),
        },
        "clip": {
            "use_quantile_clip": bool(base_cfg.use_quantile_clip),
            "q_lo": float(base_cfg.clip_q_lo),
            "q_hi": float(base_cfg.clip_q_hi),
            "lo": float(clip_lo),
            "hi": float(clip_hi),
            "margin": float(base_cfg.clip_margin),
        },
        "config": {
            "hidden_dims": list(base_cfg.hidden_dims),
            "dropout": float(base_cfg.dropout),
            "lr": float(base_cfg.lr),
            "weight_decay": float(base_cfg.weight_decay),
            "loss_fn": "HuberLoss",
            "huber_delta": best_delta,
            "batch_size": int(base_cfg.batch_size),
            "epochs_max": int(base_cfg.epochs),
            "early_stopping": bool(base_cfg.early_stopping),
            "patience": int(base_cfg.patience),
            "min_delta": float(base_cfg.min_delta),
            "restore_best": bool(base_cfg.restore_best),
            "log1p_feature_transform": bool(base_cfg.use_log1p_feature_transform),
            "log1p_feature_count": int(len(log1p_feature_names)),
            "log1p_features": list(log1p_feature_names),
        },
        "training": best_payload["training"],
        "metrics": best_payload["metrics"],
        "delta_sweep": sweep_rows,
    }

    with open(run_cfg.out_dir / "run_summary.json", "w") as f:
        json.dump(run_summary, f, indent=2)

    print("=== Custom Valuation Delta Sweep Done ===")
    print(f"Best delta: {best_delta}")
    print(
        f"Best val_rmse_log={best_payload['metrics']['val_rmse_log']:.6f}, "
        f"val_mape_raw={best_payload['metrics']['val_mape_raw']:.6f}, "
        f"val_r2_raw={best_payload['metrics']['val_r2_raw']:.6f}"
    )
    print(f"Saved: {run_cfg.out_dir / 'run_summary.json'}")


if __name__ == "__main__":
    main()
