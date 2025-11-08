from __future__ import annotations

import argparse
import json

from .engine import SurveillanceEngine


def main() -> None:
    parser = argparse.ArgumentParser(description="Run synthetic crypto market surveillance pipeline")
    parser.add_argument(
        "--no-write",
        action="store_true",
        help="Skip writing csv artifacts to the data directory",
    )
    args = parser.parse_args()

    engine = SurveillanceEngine()
    result = engine.run(persist_artifacts=not args.no_write)

    summary = {
        "accounts": len(result.data.accounts),
        "trades": len(result.data.trades),
        "orders": len(result.data.orders),
        "positions": len(result.data.positions),
        "alerts_detected": len(result.alerts),
    }
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
