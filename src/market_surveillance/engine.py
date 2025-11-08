from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List

import pandas as pd

from .data_generator import GeneratedData, generate_dataset
from .ml_utils import compute_behavioral_anomalies
from .persistence import AlertStore
from .surveillance_rules import Alert, run_all_rules


@dataclass
class EngineResult:
    data: GeneratedData
    alerts: List[Alert]
    alert_count: int
    ml_scores: pd.DataFrame


class SurveillanceEngine:
    def __init__(self, output_dir: str | Path = "data", db_path: str | Path = "data/alerts.sqlite") -> None:
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.store = AlertStore(db_path)

    def run(self, persist_artifacts: bool = True) -> EngineResult:
        dataset = generate_dataset()
        ml_scores = compute_behavioral_anomalies(dataset.accounts, dataset.trades, dataset.orders, dataset.positions)
        alerts = run_all_rules(
            dataset.trades,
            dataset.orders,
            dataset.accounts,
            dataset.positions,
            ml_scores=ml_scores,
        )
        persisted = self.store.persist_alerts(alerts)
        if persist_artifacts:
            self._write_artifacts(dataset, alerts, persisted, ml_scores)
        return EngineResult(data=dataset, alerts=alerts, alert_count=persisted, ml_scores=ml_scores)

    def _write_artifacts(self, dataset: GeneratedData, alerts: List[Alert], persisted: int, ml_scores: pd.DataFrame) -> None:
        dataset.accounts.to_csv(self.output_dir / "accounts.csv", index=False)
        dataset.trades.to_csv(self.output_dir / "trades.csv", index=False)
        dataset.orders.to_csv(self.output_dir / "orders.csv", index=False)
        dataset.positions.to_csv(self.output_dir / "positions.csv", index=False)
        if not ml_scores.empty:
            ml_scores.to_csv(self.output_dir / "ml_scores.csv", index=False)
        alert_rows = [alert.as_record() for alert in alerts]
        if alert_rows:
            alerts_df = pd.DataFrame(alert_rows)
        else:
            alerts_df = pd.DataFrame(
                columns=["rule_name", "severity", "account_id", "related_accounts", "description", "metadata", "event_time"]
            )
        alerts_df.to_csv(self.output_dir / "alerts_snapshot.csv", index=False)


def run_pipeline() -> Dict[str, int]:
    engine = SurveillanceEngine()
    result = engine.run()
    return {
        "generated_accounts": len(result.data.accounts),
        "generated_trades": len(result.data.trades),
        "generated_orders": len(result.data.orders),
        "alerts_written": result.alert_count,
    }
