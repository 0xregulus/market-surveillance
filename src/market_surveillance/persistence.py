from __future__ import annotations

import json
import sqlite3
from pathlib import Path
from typing import Iterable, List

from .surveillance_rules import Alert


class AlertStore:
    def __init__(self, db_path: str | Path = "data/alerts.sqlite") -> None:
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self.conn = sqlite3.connect(self.db_path)
        self.conn.execute("PRAGMA journal_mode=WAL")
        self._ensure_schema()

    def _ensure_schema(self) -> None:
        self.conn.execute(
            """
            CREATE TABLE IF NOT EXISTS alerts (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                rule_name TEXT NOT NULL,
                severity TEXT NOT NULL,
                account_id TEXT,
                related_accounts TEXT,
                description TEXT NOT NULL,
                metadata TEXT,
                event_time TEXT NOT NULL,
                created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
            )
            """
        )
        self.conn.commit()

    def persist_alerts(self, alerts: Iterable[Alert]) -> int:
        rows = [
            (
                alert.rule_name,
                alert.severity,
                alert.account_id,
                ",".join(alert.related_accounts),
                alert.description,
                json.dumps(alert.metadata, default=str),
                alert.event_time.isoformat(),
            )
            for alert in alerts
        ]
        if not rows:
            return 0
        self.conn.executemany(
            """
            INSERT INTO alerts (
                rule_name, severity, account_id, related_accounts,
                description, metadata, event_time
            )
            VALUES (?, ?, ?, ?, ?, ?, ?)
            """,
            rows,
        )
        self.conn.commit()
        return len(rows)

    def fetch_recent(self, limit: int = 50) -> List[dict]:
        cursor = self.conn.execute(
            """
            SELECT id, rule_name, severity, account_id, related_accounts,
                   description, metadata, event_time, created_at
            FROM alerts
            ORDER BY event_time DESC
            LIMIT ?
            """,
            (limit,),
        )
        columns = [col[0] for col in cursor.description]
        return [dict(zip(columns, row)) for row in cursor.fetchall()]
