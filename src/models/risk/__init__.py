from __future__ import annotations

from .modeling import RiskXGBConfig, load_risk_dataset, train_and_eval_risk_xgb, train_risk_xgb

__all__ = [
    "RiskMLPConfig",
    "RiskXGBConfig",
    "load_risk_dataset",
    "train_risk_mlp",
    "train_risk_xgb",
    "train_and_eval_risk_mlp",
    "train_and_eval_risk_xgb",
]


def __getattr__(name: str):
    if name in {"RiskMLPConfig", "train_risk_mlp", "train_and_eval_risk_mlp"}:
        from .mlp_risk import RiskMLPConfig, train_and_eval_risk_mlp, train_risk_mlp

        mapping = {
            "RiskMLPConfig": RiskMLPConfig,
            "train_risk_mlp": train_risk_mlp,
            "train_and_eval_risk_mlp": train_and_eval_risk_mlp,
        }
        return mapping[name]
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
