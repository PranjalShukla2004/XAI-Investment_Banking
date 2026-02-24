# src/models/nn/train.py
from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Dict, Optional
import numpy as np

Array = np.ndarray


def iterate_minibatches(
    X: Array,
    y: Array,
    batch_size: int,
    shuffle: bool = True,
    seed: int = 42,
    w: Optional[Array] = None,
):
    """Yield (X_batch, y_batch[, w_batch])."""
    n = X.shape[0]
    idx = np.arange(n)

    if shuffle:
        rng = np.random.default_rng(seed)
        rng.shuffle(idx)

    for start in range(0, n, batch_size):
        end = min(start + batch_size, n)
        b = idx[start:end]
        if w is None:
            yield X[b], y[b]
        else:
            yield X[b], y[b], w[b]


@dataclass
class TrainConfig:
    epochs: int = 30
    batch_size: int = 128
    seed: int = 42
    print_every: int = 1
    early_stopping: bool = False
    patience: int = 20
    min_delta: float = 0.0
    restore_best: bool = True
    # LR scheduler: set lr_scheduler="reduce_on_plateau" to enable
    lr_scheduler: str = "none"  # "none" | "reduce_on_plateau"
    lr_factor: float = 0.5
    lr_patience: int = 15
    lr_min: float = 1e-6
    lr_threshold: float = 1e-4
    lr_cooldown: int = 0


def _safe_reg_loss(model) -> float:
    if hasattr(model, "reg_loss") and callable(getattr(model, "reg_loss")):
        try:
            return float(model.reg_loss())
        except Exception:
            return 0.0
    return 0.0


def _criterion_forward(criterion, pred: Array, y: Array, w: Optional[Array]) -> float:
    if w is None:
        return float(criterion.forward(pred, y))
    try:
        return float(criterion.forward(pred, y, sample_weight=w))
    except TypeError:
        return float(criterion.forward(pred, y))


def _snapshot_params(model) -> Dict[tuple, Array]:
    snap: Dict[tuple, Array] = {}
    for ref in model.params_and_grads():
        snap[(ref.layer_idx, ref.name)] = ref.value.copy()
    return snap


def _restore_params(model, snap: Dict[tuple, Array]) -> None:
    for ref in model.params_and_grads():
        key = (ref.layer_idx, ref.name)
        if key in snap:
            ref.value[...] = snap[key]


def fit(
    model,
    criterion,
    optimizer,
    X_train: Array,
    y_train: Array,
    X_val: Optional[Array] = None,
    y_val: Optional[Array] = None,
    cfg: TrainConfig = TrainConfig(),
    metric_fn: Optional[Callable[[Array, Array], float]] = None,
    w_train: Optional[Array] = None,
    w_val: Optional[Array] = None,
) -> Dict[str, list]:
    """
    Train a model using mini-batch GD.

    Supports optional sample weights if criterion.forward supports sample_weight.
    """
    history = {
        "train_loss": [],
        "val_loss": [],
        "train_metric": [],
        "val_metric": [],
        "lr": [],
        "lr_reductions": 0,
        "best_epoch": None,
        "best_val_loss": None,
        "stopped_early": False,
    }

    if w_train is not None:
        w_train = np.asarray(w_train, dtype=float).reshape(-1)
        if w_train.shape[0] != X_train.shape[0]:
            raise ValueError(f"w_train must have shape (N_train,), got {w_train.shape} for N_train={X_train.shape[0]}")
    if w_val is not None and X_val is not None:
        w_val = np.asarray(w_val, dtype=float).reshape(-1)
        if w_val.shape[0] != X_val.shape[0]:
            raise ValueError(f"w_val must have shape (N_val,), got {w_val.shape} for N_val={X_val.shape[0]}")

    use_early_stopping = bool(cfg.early_stopping and X_val is not None and y_val is not None)
    use_lr_scheduler = bool(
        cfg.lr_scheduler == "reduce_on_plateau"
        and X_val is not None
        and y_val is not None
        and hasattr(optimizer, "lr")
    )
    best_val_loss = np.inf
    best_epoch = None
    epochs_without_improve = 0
    best_params = None

    # Scheduler state
    sched_best_val = np.inf
    sched_bad_epochs = 0
    sched_cooldown_left = 0
    lr_reductions = 0

    for epoch in range(1, cfg.epochs + 1):
        # ---- train ----
        if hasattr(model, "train") and callable(getattr(model, "train")):
            model.train()

        batch_losses = []
        batch_metrics = []

        mb_iter = iterate_minibatches(
            X_train,
            y_train,
            batch_size=cfg.batch_size,
            shuffle=True,
            seed=cfg.seed + epoch,
            w=w_train,
        )

        for batch in mb_iter:
            if w_train is None:
                xb, yb = batch
                wb = None
            else:
                xb, yb, wb = batch

            pred = model.forward(xb, training=True)
            data_loss = _criterion_forward(criterion, pred, yb, wb)
            total_loss = float(data_loss) + _safe_reg_loss(model)

            dpred = criterion.backward()
            model.backward(dpred)

            optimizer.step(model.params_and_grads())

            batch_losses.append(total_loss)
            if metric_fn is not None:
                batch_metrics.append(float(metric_fn(pred, yb)))

        train_loss = float(np.mean(batch_losses)) if batch_losses else 0.0
        history["train_loss"].append(train_loss)
        history["train_metric"].append(float(np.mean(batch_metrics)) if batch_metrics else None)

        # ---- validation ----
        if X_val is not None and y_val is not None:
            if hasattr(model, "eval") and callable(getattr(model, "eval")):
                model.eval()

            pred_val = model.forward(X_val, training=False)
            val_data_loss = _criterion_forward(criterion, pred_val, y_val, w_val)
            val_loss = float(val_data_loss) + _safe_reg_loss(model)
            history["val_loss"].append(val_loss)

            history["val_metric"].append(float(metric_fn(pred_val, y_val)) if metric_fn is not None else None)

            if use_early_stopping:
                if val_loss < (best_val_loss - float(cfg.min_delta)):
                    best_val_loss = val_loss
                    best_epoch = epoch
                    epochs_without_improve = 0
                    if cfg.restore_best:
                        best_params = _snapshot_params(model)
                else:
                    epochs_without_improve += 1

            if use_lr_scheduler:
                if val_loss < (sched_best_val - float(cfg.lr_threshold)):
                    sched_best_val = val_loss
                    sched_bad_epochs = 0
                else:
                    if sched_cooldown_left > 0:
                        sched_cooldown_left -= 1
                    else:
                        sched_bad_epochs += 1
                        if sched_bad_epochs >= int(cfg.lr_patience):
                            old_lr = float(getattr(optimizer, "lr"))
                            new_lr = max(float(cfg.lr_min), old_lr * float(cfg.lr_factor))
                            if new_lr < old_lr:
                                optimizer.lr = new_lr
                                lr_reductions += 1
                                sched_cooldown_left = int(cfg.lr_cooldown)
                            sched_bad_epochs = 0
        else:
            history["val_loss"].append(None)
            history["val_metric"].append(None)

        history["lr"].append(float(getattr(optimizer, "lr", np.nan)))

        if cfg.print_every and epoch % cfg.print_every == 0:
            msg = f"Epoch {epoch:03d} | train_loss={train_loss:.6f}"
            if history["val_loss"][-1] is not None:
                msg += f" | val_loss={history['val_loss'][-1]:.6f}"
            if history["train_metric"][-1] is not None:
                msg += f" | train_metric={history['train_metric'][-1]:.4f}"
            if history["val_metric"][-1] is not None:
                msg += f" | val_metric={history['val_metric'][-1]:.4f}"
            if history["lr"][-1] is not None:
                msg += f" | lr={history['lr'][-1]:.6g}"
            print(msg)

        if use_early_stopping and epochs_without_improve >= int(cfg.patience):
            history["stopped_early"] = True
            if cfg.restore_best and best_params is not None:
                _restore_params(model, best_params)
            break

    if use_early_stopping and np.isfinite(best_val_loss):
        history["best_val_loss"] = float(best_val_loss)
        history["best_epoch"] = int(best_epoch) if best_epoch is not None else None
    history["lr_reductions"] = int(lr_reductions)

    return history
