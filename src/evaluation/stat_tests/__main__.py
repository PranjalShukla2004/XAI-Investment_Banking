from __future__ import annotations

import sys

from .risk import main as risk_main
from .valuation import main as valuation_main


def main() -> None:
    argv = sys.argv[1:]
    mode = "valuation"
    forwarded = argv
    if argv and argv[0] in {"valuation", "risk"}:
        mode = argv[0]
        forwarded = argv[1:]

    sys.argv = [sys.argv[0], *forwarded]
    if mode == "risk":
        risk_main()
        return
    valuation_main()


if __name__ == "__main__":
    main()
