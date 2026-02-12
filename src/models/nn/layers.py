from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional, Tuple
import numpy as np


Array = np.ndarray


class Layer:
    """
    Minimal base class for layers.

    Contract:
      - forward(x, training=True) -> y
      - backward(dy) -> dx
      - params() -> dict of trainable parameters
      - grads()  -> dict of gradients (same keys as params)
    """
    def __init__(self) -> None:
        self._training = True

    def train(self) -> None:
        self._training = True

    def eval(self) -> None:
        self._training = False


##### Example layers left empty because each of them is implemented later differently for different layers.#####

    def forward(self, x: Array, training: Optional[bool] = None) -> Array:
        raise NotImplementedError

    def backward(self, dout: Array) -> Array:
        raise NotImplementedError

    def params(self) -> Dict[str, Array]:
        return {}

    def grads(self) -> Dict[str, Array]:
        return {}

    def reg_loss(self) -> float:
        """Return regularization loss contributed by this layer (default 0)."""
        return 0.0


def _he_init(fan_in: int, fan_out: int, rng: np.random.Generator) -> Array:
    # Good default for ReLU networks
    std = np.sqrt(2.0 / fan_in)
    return rng.normal(0.0, std, size=(fan_in, fan_out))


def _xavier_init(fan_in: int, fan_out: int, rng: np.random.Generator) -> Array:
    # Good default for tanh/sigmoid
    limit = np.sqrt(6.0 / (fan_in + fan_out))
    return rng.uniform(-limit, limit, size=(fan_in, fan_out))


@dataclass
class DenseConfig:
    in_dim: int
    out_dim: int
    init: str = "he"          # "he" or "xavier" or "normal"
    weight_scale: float = 0.01  # used if init == "normal"
    l2: float = 0.0           # L2 regularization strength (lambda)
    seed: Optional[int] = None


class Dense(Layer):
    """
    Fully-connected layer: y = xW + b

    Shapes:
      x:    (N, in_dim)
      W:    (in_dim, out_dim)
      b:    (out_dim,)
      y:    (N, out_dim)
    """
    def __init__(self, cfg: DenseConfig):
        super().__init__()
        self.cfg = cfg
        self.rng = np.random.default_rng(cfg.seed)

        if cfg.init == "he":
            W = _he_init(cfg.in_dim, cfg.out_dim, self.rng)
        elif cfg.init == "xavier":
            W = _xavier_init(cfg.in_dim, cfg.out_dim, self.rng)
        elif cfg.init == "normal":
            W = self.rng.normal(0.0, cfg.weight_scale, size=(cfg.in_dim, cfg.out_dim))
        else:
            raise ValueError(f"Unknown init '{cfg.init}'. Use: he, xavier, normal")

        b = np.zeros((cfg.out_dim,), dtype=float)

        self.W: Array = W.astype(float)
        self.b: Array = b

        # Cache for backward
        self._x: Optional[Array] = None

        # Gradients
        self.dW: Array = np.zeros_like(self.W)
        self.db: Array = np.zeros_like(self.b)

    def forward(self, x: Array, training: Optional[bool] = None) -> Array:
        if x.ndim != 2:
            raise ValueError(f"Dense expects 2D input (N, D). Got shape {x.shape}")
        if x.shape[1] != self.W.shape[0]:
            raise ValueError(
                f"Input dim mismatch: x has D={x.shape[1]}, expected {self.W.shape[0]}"
            )
        self._x = x
        return x @ self.W + self.b

    def backward(self, dout: Array) -> Array:
        if self._x is None:
            raise RuntimeError("Cannot call backward() before forward().")

        x = self._x
        if dout.ndim != 2:
            raise ValueError(f"Dense backward expects 2D dout. Got shape {dout.shape}")
        if dout.shape[1] != self.W.shape[1]:
            raise ValueError(
                f"dout dim mismatch: dout has {dout.shape[1]}, expected {self.W.shape[1]}"
            )

        # Gradients
        # dW = x^T dout
        self.dW = x.T @ dout
        # db = sum over batch
        self.db = np.sum(dout, axis=0)
        # dx = dout W^T
        dx = dout @ self.W.T

        # L2 regularization gradient term: lambda * W
        if self.cfg.l2 and self.cfg.l2 > 0.0:
            self.dW = self.dW + self.cfg.l2 * self.W

        return dx

    def params(self) -> Dict[str, Array]:
        return {"W": self.W, "b": self.b}

    def grads(self) -> Dict[str, Array]:
        return {"W": self.dW, "b": self.db}

    def reg_loss(self) -> float:
        # (lambda/2) * ||W||^2 is standard
        if self.cfg.l2 and self.cfg.l2 > 0.0:
            return 0.5 * self.cfg.l2 * float(np.sum(self.W * self.W))
        return 0.0


class Dropout(Layer):
    """
    Inverted dropout:
      training:  y = x * mask / (1 - p)
      eval:      y = x

    p = probability of dropping a unit (0.0 to <1.0)
    """
    def __init__(self, p: float = 0.5, seed: Optional[int] = None):
        super().__init__()
        if not (0.0 <= p < 1.0):
            raise ValueError("Dropout p must be in [0.0, 1.0).")
        self.p = float(p)
        self.rng = np.random.default_rng(seed)
        self._mask: Optional[Array] = None

    def forward(self, x: Array, training: Optional[bool] = None) -> Array:
        use_training = self._training if training is None else bool(training)
        if (not use_training) or self.p == 0.0:
            self._mask = None
            return x

        keep_prob = 1.0 - self.p
        mask = (self.rng.random(size=x.shape) < keep_prob).astype(float)
        self._mask = mask
        return (x * mask) / keep_prob

    def backward(self, dout: Array) -> Array:
        if self._mask is None:
            return dout
        keep_prob = 1.0 - self.p
        return (dout * self._mask) / keep_prob


class Flatten(Layer):
    """
    Flattens everything except batch dimension.
      (N, ...) -> (N, D)
    """
    def __init__(self):
        super().__init__()
        self._orig_shape: Optional[Tuple[int, ...]] = None

    def forward(self, x: Array, training: Optional[bool] = None) -> Array:
        self._orig_shape = x.shape
        return x.reshape(x.shape[0], -1)

    def backward(self, dout: Array) -> Array:
        if self._orig_shape is None:
            raise RuntimeError("Cannot call backward() before forward().")
        return dout.reshape(self._orig_shape)
