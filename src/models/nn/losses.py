# src/models/nn/losses.py
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Literal, Union
import numpy as np

Array = np.ndarray
Reduction = Literal["mean", "sum", "none"]


def _ensure_2d(x: Array) -> Array:
    if x.ndim == 1:
        return x.reshape(-1, 1)
    return x


def _one_hot(y: Array, num_classes: int) -> Array:
    if y.ndim != 1:
        raise ValueError(f"Expected class indices shape (N,), got {y.shape}")
    if y.size == 0:
        return np.zeros((0, num_classes), dtype=float)
    oh = np.zeros((y.shape[0], num_classes), dtype=float)
    oh[np.arange(y.shape[0]), y.astype(int)] = 1.0
    return oh


def softmax(logits: Array, axis: int = -1) -> Array:
    # Stable softmax
    z = logits - np.max(logits, axis=axis, keepdims=True)
    expz = np.exp(z)
    return expz / np.sum(expz, axis=axis, keepdims=True)


def log_softmax(logits: Array, axis: int = -1) -> Array:
    # Stable log-softmax
    z = logits - np.max(logits, axis=axis, keepdims=True)
    logsumexp = np.log(np.sum(np.exp(z), axis=axis, keepdims=True))
    return z - logsumexp


def sigmoid(x: Array) -> Array:
    # Stable sigmoid
    x = np.clip(x, -50.0, 50.0)
    return 1.0 / (1.0 + np.exp(-x))


@dataclass
class Loss:
    reduction: Reduction = "mean"

    def forward(self, pred: Array, target: Array) -> Union[float, Array]:
        raise NotImplementedError

    def backward(self) -> Array:
        """Return gradient w.r.t. the prediction input passed to forward()."""
        raise NotImplementedError


class MSELoss(Loss):
    """
    Mean Squared Error (optionally sample-weighted):

      pred:   (N, D) or (N,)
      target: same shape

    If sample_weight is provided (shape (N,) or (N,1)), the loss is:

      loss = sum_i w_i * 0.5 * ||pred_i - target_i||^2  / sum_i w_i    (for reduction="mean")

    and dL/dpred_i = w_i * (pred_i - target_i) / sum_i w_i            (for reduction="mean")
    """
    def __init__(self, reduction: Reduction = "mean"):
        super().__init__(reduction=reduction)
        self._pred: Optional[Array] = None
        self._target: Optional[Array] = None
        self._w: Optional[Array] = None  # (N,)

    def forward(
        self,
        pred: Array,
        target: Array,
        sample_weight: Optional[Array] = None,
    ) -> Union[float, Array]:
        pred = np.asarray(pred, dtype=float)
        target = np.asarray(target, dtype=float)

        if pred.shape != target.shape:
            raise ValueError(f"MSELoss shape mismatch: pred {pred.shape}, target {target.shape}")

        self._pred = pred
        self._target = target

        w = None
        if sample_weight is not None:
            w = np.asarray(sample_weight, dtype=float).reshape(-1)
            if w.shape[0] != pred.shape[0]:
                raise ValueError(
                    f"sample_weight must have shape (N,), got {sample_weight.shape} for N={pred.shape[0]}"
                )
            w = np.clip(w, a_min=0.0, a_max=None)  # allow zero, forbid negative
        self._w = w

        loss_per = 0.5 * (pred - target) ** 2  # 0.5 for nicer gradient

        # reduce over feature dims -> per-sample (N,)
        if loss_per.ndim >= 2:
            loss_per = np.sum(loss_per, axis=tuple(range(1, loss_per.ndim)))

        if w is not None:
            loss_per = loss_per * w

        if self.reduction == "none":
            return loss_per
        if self.reduction == "sum":
            return float(np.sum(loss_per))

        # mean
        if w is None:
            return float(np.mean(loss_per))
        denom = float(np.sum(w))
        if denom <= 0.0:
            return 0.0
        return float(np.sum(loss_per) / denom)

    def backward(self) -> Array:
        if self._pred is None or self._target is None:
            raise RuntimeError("Call forward() before backward().")

        pred, target = self._pred, self._target
        grad = (pred - target)

        w = self._w
        if w is not None:
            # expand (N,) -> (N,1,1,...) to match pred shape
            expand_shape = (w.shape[0],) + (1,) * (grad.ndim - 1)
            grad = grad * w.reshape(expand_shape)

        if self.reduction == "none":
            return grad
        if self.reduction == "sum":
            return grad

        # mean
        if w is None:
            return grad / max(pred.shape[0], 1)
        denom = float(np.sum(w))
        if denom <= 0.0:
            return np.zeros_like(grad)
        return grad / denom


class HuberLoss(Loss):
    """
    Huber loss (optionally sample-weighted), robust to outliers.

      pred:   (N, D) or (N,)
      target: same shape

    For error e = pred - target and threshold delta:
      0.5 * e^2                    if |e| <= delta
      delta * (|e| - 0.5 * delta)  otherwise
    """
    def __init__(self, delta: float = 1.0, reduction: Reduction = "mean"):
        super().__init__(reduction=reduction)
        if delta <= 0.0:
            raise ValueError("HuberLoss delta must be > 0.")
        self.delta = float(delta)
        self._pred: Optional[Array] = None
        self._target: Optional[Array] = None
        self._w: Optional[Array] = None

    def forward(
        self,
        pred: Array,
        target: Array,
        sample_weight: Optional[Array] = None,
    ) -> Union[float, Array]:
        pred = np.asarray(pred, dtype=float)
        target = np.asarray(target, dtype=float)

        if pred.shape != target.shape:
            raise ValueError(f"HuberLoss shape mismatch: pred {pred.shape}, target {target.shape}")

        self._pred = pred
        self._target = target

        w = None
        if sample_weight is not None:
            w = np.asarray(sample_weight, dtype=float).reshape(-1)
            if w.shape[0] != pred.shape[0]:
                raise ValueError(
                    f"sample_weight must have shape (N,), got {sample_weight.shape} for N={pred.shape[0]}"
                )
            w = np.clip(w, a_min=0.0, a_max=None)
        self._w = w

        err = pred - target
        abs_err = np.abs(err)
        quad = np.minimum(abs_err, self.delta)
        lin = abs_err - quad
        loss_per = 0.5 * quad * quad + self.delta * lin

        if loss_per.ndim >= 2:
            loss_per = np.sum(loss_per, axis=tuple(range(1, loss_per.ndim)))

        if w is not None:
            loss_per = loss_per * w

        if self.reduction == "none":
            return loss_per
        if self.reduction == "sum":
            return float(np.sum(loss_per))

        if w is None:
            return float(np.mean(loss_per))
        denom = float(np.sum(w))
        if denom <= 0.0:
            return 0.0
        return float(np.sum(loss_per) / denom)

    def backward(self) -> Array:
        if self._pred is None or self._target is None:
            raise RuntimeError("Call forward() before backward().")

        err = self._pred - self._target
        abs_err = np.abs(err)
        grad = np.where(abs_err <= self.delta, err, self.delta * np.sign(err))

        w = self._w
        if w is not None:
            expand_shape = (w.shape[0],) + (1,) * (grad.ndim - 1)
            grad = grad * w.reshape(expand_shape)

        if self.reduction == "none":
            return grad
        if self.reduction == "sum":
            return grad

        if w is None:
            return grad / max(self._pred.shape[0], 1)
        denom = float(np.sum(w))
        if denom <= 0.0:
            return np.zeros_like(grad)
        return grad / denom



class CrossEntropyLoss(Loss):
    """
    Multi-class cross entropy on logits (recommended):
      logits: (N, C)
      target: (N,) integer class indices OR (N, C) one-hot/prob distribution

    Supports label smoothing for index targets and one-hot targets.

    Backward gives dlogits with same shape as logits.
    """
    def __init__(
        self,
        reduction: Reduction = "mean",
        label_smoothing: float = 0.0,
        eps: float = 1e-12,
    ):
        super().__init__(reduction=reduction)
        if not (0.0 <= label_smoothing < 1.0):
            raise ValueError("label_smoothing must be in [0, 1).")
        self.label_smoothing = float(label_smoothing)
        self.eps = float(eps)

        self._logits: Optional[Array] = None
        self._target_dist: Optional[Array] = None
        self._probs: Optional[Array] = None

    def forward(self, logits: Array, target: Array) -> Union[float, Array]:
        logits = np.asarray(logits, dtype=float)
        if logits.ndim != 2:
            raise ValueError(f"CrossEntropyLoss expects logits shape (N, C), got {logits.shape}")
        N, C = logits.shape

        # Convert targets to distribution
        target = np.asarray(target)
        if target.ndim == 1:
            y = _one_hot(target, C)
        elif target.ndim == 2:
            if target.shape != (N, C):
                raise ValueError(f"Target distribution must be (N, C)={N,C}, got {target.shape}")
            y = target.astype(float)
        else:
            raise ValueError(f"Unsupported target shape for CE: {target.shape}")

        # Label smoothing (always safe on distributions too)
        if self.label_smoothing > 0.0:
            y = (1.0 - self.label_smoothing) * y + (self.label_smoothing / C)

        # Compute stable log softmax
        lsm = log_softmax(logits, axis=1)
        loss_per = -np.sum(y * lsm, axis=1)  # (N,)

        self._logits = logits
        self._target_dist = y
        self._probs = np.exp(lsm)  # softmax(logits)

        if self.reduction == "none":
            return loss_per
        if self.reduction == "sum":
            return float(np.sum(loss_per))
        return float(np.mean(loss_per))

    def backward(self) -> Array:
        if self._logits is None or self._target_dist is None or self._probs is None:
            raise RuntimeError("Call forward() before backward().")

        N = self._logits.shape[0]
        dlogits = self._probs - self._target_dist  # (N, C)

        if self.reduction == "none":
            return dlogits
        if self.reduction == "sum":
            return dlogits
        return dlogits / max(N, 1)


class BCEWithLogitsLoss(Loss):
    """
    Binary cross entropy on logits (stable):
      logits: (N,) or (N, 1) or (N, D)
      target: same shape, values in {0,1} (or [0,1])

    Backward gives dlogits with same shape as logits.
    """
    def __init__(self, reduction: Reduction = "mean"):
        super().__init__(reduction=reduction)
        self._logits: Optional[Array] = None
        self._target: Optional[Array] = None

    def forward(self, logits: Array, target: Array) -> Union[float, Array]:
        logits = np.asarray(logits, dtype=float)
        target = np.asarray(target, dtype=float)

        if logits.shape != target.shape:
            raise ValueError(f"BCEWithLogitsLoss shape mismatch: logits {logits.shape}, target {target.shape}")

        self._logits = logits
        self._target = target

        # Stable BCE with logits:
        # loss = max(x,0) - x*y + log(1 + exp(-|x|))
        x = logits
        y = target
        loss = np.maximum(x, 0.0) - x * y + np.log1p(np.exp(-np.abs(x)))

        # reduce over feature dims, keep batch
        if loss.ndim >= 2:
            loss_per = np.sum(loss, axis=tuple(range(1, loss.ndim)))
        else:
            loss_per = loss  # (N,)

        if self.reduction == "none":
            return loss_per
        if self.reduction == "sum":
            return float(np.sum(loss_per))
        return float(np.mean(loss_per))

    def backward(self) -> Array:
        if self._logits is None or self._target is None:
            raise RuntimeError("Call forward() before backward().")

        x = self._logits
        y = self._target
        p = sigmoid(x)
        dlogits = p - y  # derivative of BCE-with-logits wrt logits

        if self.reduction == "none":
            return dlogits
        if self.reduction == "sum":
            return dlogits
        return dlogits / max(x.shape[0], 1)


# Optional convenience metrics (useful during training)
def accuracy_from_logits(logits: Array, y: Array) -> float:
    """
    logits: (N, C)
    y: (N,) int labels OR (N, C) one-hot/prob dist
    """
    logits = np.asarray(logits, dtype=float)
    if logits.ndim != 2:
        raise ValueError(f"accuracy_from_logits expects (N, C), got {logits.shape}")
    pred = np.argmax(logits, axis=1)

    y = np.asarray(y)
    if y.ndim == 2:
        y_true = np.argmax(y, axis=1)
    elif y.ndim == 1:
        y_true = y.astype(int)
    else:
        raise ValueError(f"Unsupported y shape: {y.shape}")

    if y_true.shape[0] == 0:
        return 0.0
    return float(np.mean(pred == y_true))
