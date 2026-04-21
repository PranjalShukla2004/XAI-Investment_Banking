from .risk import main as risk_main
from .valuation import main as valuation_main


def main() -> None:
    valuation_main()


__all__ = ["main", "valuation_main", "risk_main"]
