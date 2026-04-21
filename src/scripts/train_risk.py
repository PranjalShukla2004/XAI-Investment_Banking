from __future__ import annotations

import argparse
import sys


def main() -> None:
    ap = argparse.ArgumentParser(add_help=False)
    ap.add_argument("--model", choices=("xgb", "mlp"), default="xgb")
    args, remaining = ap.parse_known_args()

    sys.argv = [sys.argv[0], *remaining]
    if args.model == "mlp":
        from src.models.risk.mlp_risk import main as entry
    else:
        from src.models.risk.xgb_risk import main as entry
    entry()


if __name__ == "__main__":
    main()
