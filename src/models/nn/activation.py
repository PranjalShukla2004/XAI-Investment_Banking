# src/models/nn/activations.py
from __future__ import annotations
from typing import Optional
import numpy as np

Array = np.ndarray

from .layers import Layer


class ReLU(Layer):
    def __init__(self):
        super().__init__()
        self._mask: Optional[Array] = None

    def forward(self, x: Array, training: Optional[bool] = None) -> Array:
        self._mask = (x > 0)
        return np.maximum(x, 0)

    def backward(self, dout: Array) -> Array:
        if self._mask is None:
            raise RuntimeError("Cannot call backward() before forward().")
        return dout * self._mask.astype(float)


class Tanh(Layer):
    def __init__(self):
        super().__init__()
        self._y: Optional[Array] = None

    def forward(self, x: Array, training: Optional[bool] = None) -> Array:
        y = np.tanh(x)
        self._y = y
        return y

    def backward(self, dout: Array) -> Array:
        if self._y is None:
            raise RuntimeError("Cannot call backward() before forward().")
        return dout * (1.0 - self._y * self._y)


class Sigmoid(Layer):
    def __init__(self):
        super().__init__()
        self._y: Optional[Array] = None

    def forward(self, x: Array, training: Optional[bool] = None) -> Array:
        x = np.clip(x, -50.0, 50.0)
        y = 1.0 / (1.0 + np.exp(-x))
        self._y = y
        return y

    def backward(self, dout: Array) -> Array:
        if self._y is None:
            raise RuntimeError("Cannot call backward() before forward().")
        return dout * (self._y * (1.0 - self._y))
