# src/models/nn/model.py
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple
import numpy as np

Array = np.ndarray

from .layers import Layer


@dataclass
class ParamRef:
    layer_idx: int
    name: str
    value: Array
    grad: Array


class Sequential:
    """
    Chains layers.

    - forward(x): applies each layer
    - backward(dout): reverse apply
    - params_and_grads(): yields ParamRef objects for optimizer
    - reg_loss(): sum of layer reg losses
    """
    def __init__(self, layers: List[Layer]):
        self.layers = layers

    def train(self) -> None:
        for l in self.layers:
            l.train()

    def eval(self) -> None:
        for l in self.layers:
            l.eval()

    def forward(self, x: Array, training: Optional[bool] = None) -> Array:
        out = x
        for l in self.layers:
            # Each layer can decide training from its internal flag,
            # but we also pass through the explicit argument if provided.
            out = l.forward(out, training=training)
        return out

    def backward(self, dout: Array) -> Array:
        dx = dout
        for l in reversed(self.layers):
            dx = l.backward(dx)
        return dx

    def reg_loss(self) -> float:
        return float(sum(l.reg_loss() for l in self.layers))

    def params_and_grads(self):
        """
        Yields ParamRef for all parameters, with stable names.
        """
        for i, l in enumerate(self.layers):
            p = l.params()
            g = l.grads()
            if not p:
                continue
            for name, value in p.items():
                if name not in g:
                    raise RuntimeError(f"Layer {i} param '{name}' missing gradient.")
                yield ParamRef(layer_idx=i, name=name, value=value, grad=g[name])

    def summary(self) -> str:
        lines = ["Sequential("]
        for i, l in enumerate(self.layers):
            lines.append(f"  [{i}] {l.__class__.__name__}")
        lines.append(")")
        return "\n".join(lines)
