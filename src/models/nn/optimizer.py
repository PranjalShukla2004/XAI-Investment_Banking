# src/models/nn/optim.py
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Tuple
import numpy as np

from .sequential import ParamRef

Array = np.ndarray


@dataclass
class SGD:
    lr: float = 1e-2
    momentum: float = 0.0
    weight_decay: float = 0.0  # extra L2 (separate from layer l2)

    def __post_init__(self):
        self._vel: Dict[Tuple[int, str], Array] = {}

    def step(self, params_and_grads) -> None:
        for ref in params_and_grads:
            key = (ref.layer_idx, ref.name)
            grad = ref.grad

            if self.weight_decay > 0.0:
                grad = grad + self.weight_decay * ref.value

            if self.momentum > 0.0:
                v = self._vel.get(key)
                if v is None:
                    v = np.zeros_like(ref.value)
                v = self.momentum * v + grad
                self._vel[key] = v
                ref.value -= self.lr * v
            else:
                ref.value -= self.lr * grad


@dataclass
class Adam:
    lr: float = 1e-3
    beta1: float = 0.9
    beta2: float = 0.999
    eps: float = 1e-8
    weight_decay: float = 0.0

    def __post_init__(self):
        self._m: Dict[Tuple[int, str], Array] = {}
        self._v: Dict[Tuple[int, str], Array] = {}
        self._t: int = 0

    def step(self, params_and_grads) -> None:
        self._t += 1
        t = self._t

        for ref in params_and_grads:
            key = (ref.layer_idx, ref.name)
            g = ref.grad

            if self.weight_decay > 0.0:
                g = g + self.weight_decay * ref.value

            m = self._m.get(key)
            v = self._v.get(key)
            if m is None:
                m = np.zeros_like(ref.value)
            if v is None:
                v = np.zeros_like(ref.value)

            m = self.beta1 * m + (1 - self.beta1) * g
            v = self.beta2 * v + (1 - self.beta2) * (g * g)

            m_hat = m / (1 - (self.beta1 ** t))
            v_hat = v / (1 - (self.beta2 ** t))

            ref.value -= self.lr * m_hat / (np.sqrt(v_hat) + self.eps)

            self._m[key] = m
            self._v[key] = v
