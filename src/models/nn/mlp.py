# src/models/nn/mlp.py
from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, List, Optional, Sequence, Literal

import numpy as np

from layers import Dense, DenseConfig, Dropout, Layer
from activation import ReLU, Tanh, Sigmoid
from sequential import Sequential

ActName = Literal["relu", "tanh", "sigmoid"]


def _make_activation(name: ActName) -> Layer:
    name = name.lower()
    if name == "relu":
        return ReLU()
    if name == "tanh":
        return Tanh()
    if name == "sigmoid":
        return Sigmoid()
    raise ValueError(f"Unknown activation '{name}'. Use one of: relu, tanh, sigmoid")


@dataclass
class MLPConfig:
    input_dim: int
    hidden_dims: Sequence[int] = (128, 64)
    output_dim: int = 1

    activation: ActName = "relu"
    dropout: float = 0.0

    # weight init passed to DenseConfig
    init: str = "he"              
    weight_scale: float = 0.01     
    l2: float = 0.0              


class MLP(Sequential):
    """
    Convenience wrapper around Sequential that builds:
      Dense -> Act -> (Dropout) -> ... -> Dense(output)

    Outputs:
      - For regression: raw output (use MSELoss)
      - For binary classification: logits (use BCEWithLogitsLoss)
      - For multi-class classification: logits (use CrossEntropyLoss)
    """
    def __init__(self, cfg: MLPConfig):
        layers: List[Layer] = []

        prev = int(cfg.input_dim)
        act = cfg.activation

        # Hidden stack
        for h in cfg.hidden_dims:
            h = int(h)
            layers.append(
                Dense(
                    DenseConfig(
                        in_features=prev,
                        out_features=h,
                        init=cfg.init,
                        weight_scale=cfg.weight_scale,
                        l2=cfg.l2,
                    )
                )
            )
            layers.append(_make_activation(act))
            if cfg.dropout and cfg.dropout > 0.0:
                layers.append(Dropout(float(cfg.dropout)))
            prev = h

        # Output layer (no activation here by default we keep logits/raw output)
        layers.append(
            Dense(
                DenseConfig(
                    in_features=prev,
                    out_features=int(cfg.output_dim),
                    init="normal" if cfg.init == "he" else cfg.init,  # safe default
                    weight_scale=cfg.weight_scale,
                    l2=cfg.l2,
                )
            )
        )

        super().__init__(layers)

        self.cfg = cfg
        
    @classmethod
    def from_dims(
        cls,
        input_dim: int,
        hidden_dims: Sequence[int] = (128, 64),
        output_dim: int = 1,
        activation: ActName = "relu",
        dropout: float = 0.0,
        init: str = "he",
        weight_scale: float = 0.01,
        l2: float = 0.0,
    ) -> "MLP":
        return cls(
            MLPConfig(
                input_dim=input_dim,
                hidden_dims=hidden_dims,
                output_dim=output_dim,
                activation=activation,
                dropout=dropout,
                init=init,
                weight_scale=weight_scale,
                l2=l2,
            )
        )
