from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

from .ml_utils import compute_ml_scores


@dataclass
class Alert:
    rule_name: str
    severity: str
    account_id: Optional[str]
    related_accounts: Sequence[str]
    description: str
    metadata: Dict[str, Any]
    event_time: pd.Timestamp

    def as_record(self) -> Dict[str, Any]:
        return {
            "rule_name": self.rule_name,
            "severity": self.severity,
            "account_id": self.account_id,
            "related_accounts": ",".join(self.related_accounts),
            "description": self.description,
            "metadata": self.metadata,
            "event_time": self.event_time,
        }


def detect_self_trades(trades: pd.DataFrame) -> List[Alert]:
    suspicious = trades.loc[trades["buy_account_id"] == trades["sell_account_id"]]
    alerts: List[Alert] = []
    for _, row in suspicious.iterrows():
        alerts.append(
            Alert(
                rule_name="self_trade",
                severity="high",
                account_id=row["buy_account_id"],
                related_accounts=[],
                description=f"Self-trade on {row['asset']} trade_id={row['trade_id']}",
                metadata={
                    "price": row["price"],
                    "quantity": row["quantity"],
                },
                event_time=row["timestamp"],
            )
        )
    return alerts


def detect_ping_pong(
    trades: pd.DataFrame,
    min_trades: int = 6,
    window_seconds: int = 120,
) -> List[Alert]:
    alerts: List[Alert] = []
    tmp = trades.copy()
    tmp["pair_key"] = tmp.apply(
        lambda r: tuple(sorted([r["buy_account_id"], r["sell_account_id"]])),
        axis=1,
    )
    tmp = tmp.sort_values("timestamp")
    window = pd.Timedelta(seconds=window_seconds)

    for pair, group in tmp.groupby("pair_key"):
        group = group.reset_index(drop=True)
        if len(group) < min_trades:
            continue
        diff = group["timestamp"].diff(periods=min_trades - 1)
        hits = group.loc[diff <= window]
        if hits.empty:
            continue
        first_hit = hits.iloc[0]
        idx = group.index.get_loc(first_hit.name)
        start = max(0, idx - (min_trades - 1))
        window_rows = group.iloc[start : idx + 1]
        alerts.append(
            Alert(
                rule_name="ping_pong",
                severity="medium",
                account_id=pair[0],
                related_accounts=[pair[1]],
                description=f"High-frequency ping-pong trading between {pair[0]} and {pair[1]}",
                metadata={
                    "asset": window_rows["asset"].mode().iat[0],
                    "trade_count": len(window_rows),
                    "window_seconds": window_seconds,
                },
                event_time=window_rows["timestamp"].iloc[-1],
            )
        )
    return alerts


def detect_extreme_price_moves(
    trades: pd.DataFrame,
    window: int = 80,
    z_threshold: float = 4.5,
) -> List[Alert]:
    alerts: List[Alert] = []
    tmp = trades.sort_values("timestamp").copy()
    tmp["rolling_mean"] = (
        tmp.groupby("asset")["price"].transform(lambda s: s.rolling(window, min_periods=30).mean())
    )
    tmp["rolling_std"] = (
        tmp.groupby("asset")["price"].transform(lambda s: s.rolling(window, min_periods=30).std())
    )
    tmp = tmp.dropna(subset=["rolling_mean", "rolling_std"])
    tmp = tmp[tmp["rolling_std"] > 0]
    tmp["zscore"] = (tmp["price"] - tmp["rolling_mean"]) / tmp["rolling_std"]
    flagged = tmp.loc[tmp["zscore"].abs() >= z_threshold]

    for _, row in flagged.iterrows():
        direction = "up" if row["zscore"] > 0 else "down"
        alerts.append(
            Alert(
                rule_name="extreme_price_move",
                severity="medium",
                account_id=None,
                related_accounts=[],
                description=f"{row['asset']} price moved {direction} {row['zscore']:.1f}σ from rolling mean",
                metadata={
                    "price": row["price"],
                    "zscore": row["zscore"],
                    "rolling_mean": row["rolling_mean"],
                },
                event_time=row["timestamp"],
            )
        )
    return alerts


def detect_volume_spikes(
    trades: pd.DataFrame,
    window: str = "2h",
    spike_multiple: float = 4.0,
) -> List[Alert]:
    alerts: List[Alert] = []
    volume_records: List[Dict[str, Any]] = []
    for _, row in trades.iterrows():
        notional = row["price"] * row["quantity"]
        volume_records.append(
            {
                "account_id": row["buy_account_id"],
                "timestamp": row["timestamp"],
                "side": "buy",
                "quantity": row["quantity"],
                "notional": notional,
            }
        )
        volume_records.append(
            {
                "account_id": row["sell_account_id"],
                "timestamp": row["timestamp"],
                "side": "sell",
                "quantity": row["quantity"],
                "notional": notional,
            }
        )

    vol_df = pd.DataFrame(volume_records)
    if vol_df.empty:
        return alerts

    vol_df = vol_df.sort_values("timestamp")
    vol_df = vol_df.set_index("timestamp")
    rolling = (
        vol_df.groupby("account_id")["notional"].rolling(window).sum().rename("window_notional").reset_index()
    )
    baseline = (
        vol_df.groupby("account_id")["notional"]
        .rolling("12h")
        .mean()
        .rename("baseline_notional")
        .reset_index()
    )

    merged = rolling.merge(baseline, on=["account_id", "timestamp"], how="left")
    merged = merged.dropna()
    merged["multiple"] = merged["window_notional"] / (merged["baseline_notional"] + 1e-9)
    flagged = merged.loc[merged["multiple"] >= spike_multiple]

    for _, row in flagged.iterrows():
        alerts.append(
            Alert(
                rule_name="volume_spike",
                severity="medium",
                account_id=row["account_id"],
                related_accounts=[],
                description=f"Notional volume {row['multiple']:.1f}x baseline within {window}",
                metadata={
                    "window_notional": row["window_notional"],
                    "baseline_notional": row["baseline_notional"],
                },
                event_time=row["timestamp"],
            )
        )
    return alerts


def detect_account_profile_deviation(
    trades: pd.DataFrame,
    accounts: pd.DataFrame,
    window: str = "3h",
    deviation_multiple: float = 12.0,
    min_notional: float = 250_000.0,
) -> List[Alert]:
    alerts: List[Alert] = []
    if trades.empty or accounts.empty:
        return alerts

    trades = trades.copy()
    trades["timestamp"] = pd.to_datetime(trades["timestamp"], utc=True)
    trades["notional"] = trades["price"] * trades["quantity"]
    rolling = (
        trades.set_index("timestamp")
        .groupby("initiator_account_id")["notional"]
        .rolling(window)
        .sum()
        .reset_index()
        .rename(columns={"initiator_account_id": "account_id"})
    )

    baselines = accounts.set_index("account_id")["baseline_volume"]
    rolling["baseline"] = rolling["account_id"].map(baselines)
    rolling = rolling.dropna(subset=["baseline"])
    rolling = rolling.loc[rolling["baseline"] > 0]
    rolling["multiple"] = rolling["notional"] / (rolling["baseline"] + 1e-9)
    flagged = rolling.loc[
        (rolling["multiple"] >= deviation_multiple) & (rolling["notional"] >= min_notional)
    ]
    if flagged.empty:
        return alerts
    flagged = flagged.sort_values("timestamp").drop_duplicates("account_id", keep="last")

    for _, row in flagged.iterrows():
        alerts.append(
            Alert(
                rule_name="account_profile_deviation",
                severity="high",
                account_id=row["account_id"],
                related_accounts=[],
                description=f"Account exceeded baseline by {row['multiple']:.1f}x within {window}",
                metadata={
                    "window_notional": row["notional"],
                    "baseline_notional": row["baseline"],
                    "window": window,
                },
                event_time=row["timestamp"],
            )
        )
    return alerts


def detect_position_concentration(
    positions: pd.DataFrame,
    share_threshold: float = 0.2,
    notional_threshold: float = 1_000_000.0,
) -> List[Alert]:
    alerts: List[Alert] = []
    if positions.empty:
        return alerts
    positions = positions.copy()
    positions["abs_notional"] = positions["net_notional"].abs()
    flagged = positions.loc[
        (positions["float_share"] >= share_threshold) & (positions["abs_notional"] >= notional_threshold)
    ]
    if flagged.empty:
        return alerts
    for _, row in flagged.iterrows():
        severity = "high" if row["float_share"] >= 0.3 else "medium"
        alerts.append(
            Alert(
                rule_name="position_concentration",
                severity=severity,
                account_id=row["account_id"],
                related_accounts=[],
                description=f"Holds {row['float_share']*100:.1f}% of {row['asset']} float ({row['direction']})",
                metadata={
                    "asset": row["asset"],
                    "net_quantity": row["net_quantity"],
                    "net_notional": row["net_notional"],
                    "dominant_venue": row.get("dominant_venue"),
                },
                event_time=pd.Timestamp.now(tz="UTC"),
            )
        )
    return alerts


def detect_cross_market_divergence(
    trades: pd.DataFrame,
    window: str = "5min",
    pct_threshold: float = 0.02,
) -> List[Alert]:
    alerts: List[Alert] = []
    if trades.empty or "venue" not in trades.columns:
        return alerts
    tmp = trades.copy()
    tmp["timestamp"] = pd.to_datetime(tmp["timestamp"], utc=True)
    resampled = (
        tmp.set_index("timestamp")
        .groupby(["asset", "venue"])["price"]
        .resample(window)
        .mean()
        .reset_index()
    )
    grouped = resampled.groupby(["asset", "timestamp"])
    for (asset, ts), bucket in grouped:
        if bucket["price"].nunique() < 2:
            continue
        max_idx = bucket["price"].idxmax()
        min_idx = bucket["price"].idxmin()
        max_row = bucket.loc[max_idx]
        min_row = bucket.loc[min_idx]
        diff_pct = (max_row["price"] - min_row["price"]) / max(min_row["price"], 1e-9)
        if diff_pct < pct_threshold:
            continue
        alerts.append(
            Alert(
                rule_name="cross_market_divergence",
                severity="medium" if diff_pct < pct_threshold * 2 else "high",
                account_id=None,
                related_accounts=[],
                description=f"{asset} venues diverged {diff_pct*100:.2f}% within {window}",
                metadata={
                    "high_venue": max_row["venue"],
                    "high_price": max_row["price"],
                    "low_venue": min_row["venue"],
                    "low_price": min_row["price"],
                    "window": window,
                },
                event_time=ts,
            )
        )
    return alerts


def detect_network_collusion(
    trades: pd.DataFrame,
    accounts: pd.DataFrame,
    window_seconds: int = 60,
    min_accounts: int = 3,
    min_events: int = 6,
    min_same_side: float = 0.7,
) -> List[Alert]:
    alerts: List[Alert] = []
    if trades.empty or accounts.empty:
        return alerts
    meta = accounts[["account_id", "device_id", "ip_subnet"]].drop_duplicates()
    enriched = trades.copy()
    enriched["timestamp"] = pd.to_datetime(enriched["timestamp"], utc=True)
    enriched = enriched.merge(
        meta,
        left_on="initiator_account_id",
        right_on="account_id",
        how="left",
        suffixes=("", "_account"),
    )
    windows = f"{window_seconds}s"
    cluster_configs = [
        ("ip_subnet", "shared_ip"),
        ("device_id", "shared_device"),
    ]
    for column, label in cluster_configs:
        subset = enriched.dropna(subset=[column]).copy()
        if subset.empty:
            continue
        subset = subset.rename(columns={column: "cluster"})
        subset["window"] = subset["timestamp"].dt.floor(windows)
        agg = (
            subset.groupby(["cluster", "asset", "window"])
            .agg(
                trade_count=("trade_id", "count"),
                unique_accounts=("initiator_account_id", "nunique"),
            )
            .reset_index()
        )
        if agg.empty:
            continue
        agg = agg.loc[
            (agg["unique_accounts"] >= min_accounts)
            & (agg["trade_count"] >= min_events)
        ]
        if agg.empty:
            continue
        dominance = (
            subset.groupby(["cluster", "asset", "window"])["aggressor_side"]
            .agg(lambda s: s.value_counts(normalize=True).max())
            .rename("dominance")
            .reset_index()
        )
        agg = agg.merge(dominance, on=["cluster", "asset", "window"], how="left")
        agg = agg.loc[agg["dominance"] >= min_same_side]
        if agg.empty:
            continue
        for _, row in agg.iterrows():
            window_rows = subset.loc[
                (subset["cluster"] == row["cluster"])
                & (subset["asset"] == row["asset"])
                & (subset["window"] == row["window"])
            ]
            accounts_involved = sorted(window_rows["initiator_account_id"].unique().tolist())
            severity = "high" if row["dominance"] >= min_same_side * 1.2 else "medium"
            alerts.append(
                Alert(
                    rule_name="network_collusion",
                    severity=severity,
                    account_id=accounts_involved[0] if accounts_involved else None,
                    related_accounts=accounts_involved[1:5],
                    description=(
                        f"{label} cluster {row['cluster']} coordinated {row['trade_count']} "
                        f"{row['asset']} trades in {window_seconds}s window "
                        f"({row['unique_accounts']} accounts, {row['dominance']*100:.0f}% same side)"
                    ),
                    metadata={
                        "cluster_type": label,
                        "cluster_id": row["cluster"],
                        "accounts": accounts_involved,
                        "trade_count": row["trade_count"],
                        "dominance": row["dominance"],
                        "window_seconds": window_seconds,
                    },
                    event_time=row["window"],
                )
            )
    return alerts


def detect_ml_behavioral_anomalies(
    trades: pd.DataFrame,
    orders: pd.DataFrame,
    accounts: pd.DataFrame,
    positions: pd.DataFrame,
    precomputed: Optional[pd.DataFrame] = None,
    top_k: int = 5,
) -> List[Alert]:
    alerts: List[Alert] = []
    if precomputed is None:
        ml_scores = compute_ml_scores(accounts, trades, orders, positions)
    else:
        ml_scores = precomputed
    if ml_scores is None or ml_scores.empty:
        return alerts

    for detector, detector_scores in ml_scores.groupby("detector"):
        detector_scores = detector_scores.sort_values("anomaly_score", ascending=False)
        flagged = detector_scores.loc[detector_scores["ml_flag"]].head(top_k)
        if flagged.empty:
            continue
        high_threshold = detector_scores["anomaly_score"].quantile(0.9)
        for _, row in flagged.iterrows():
            severity = "high" if row["anomaly_score"] >= high_threshold else "medium"
            metadata_raw = row.drop(labels=["account_id", "ml_flag", "anomaly_score", "rank"]).to_dict()
            metadata = {}
            for key, value in metadata_raw.items():
                if isinstance(value, (np.floating, np.integer)):
                    metadata[key] = float(value)
                else:
                    metadata[key] = value
            metadata["anomaly_score"] = float(row["anomaly_score"])
            metadata["detector"] = detector
            alerts.append(
                Alert(
                    rule_name="ml_behavioral_anomaly",
                    severity=severity,
                    account_id=row["account_id"],
                    related_accounts=[],
                    description=f"{detector} flagged account with score {row['anomaly_score']:.2f}",
                    metadata=metadata,
                    event_time=pd.Timestamp.now(tz="UTC"),
                )
            )
    return alerts


def detect_spoofing(
    orders: pd.DataFrame,
    size_threshold: float = 350.0,
    cancel_seconds: int = 10,
) -> List[Alert]:
    alerts: List[Alert] = []
    if orders.empty:
        return alerts
    orders = orders.copy()
    orders["lifetime"] = (orders["completion_ts"] - orders["timestamp"]).dt.total_seconds()
    flagged = orders.loc[
        (orders["status"] == "canceled")
        & (orders["quantity"] >= size_threshold)
        & (orders["lifetime"] <= cancel_seconds)
    ]

    for _, row in flagged.iterrows():
        alerts.append(
            Alert(
                rule_name="spoofing_layering",
                severity="high",
                account_id=row["account_id"],
                related_accounts=[],
                description=f"Large {row['side']} order canceled after {row['lifetime']:.1f}s",
                metadata={
                    "asset": row["asset"],
                    "quantity": row["quantity"],
                    "price": row["price"],
                },
                event_time=row["timestamp"],
            )
        )
    return alerts


def detect_message_rate_anomalies(
    orders: pd.DataFrame,
    threshold_per_minute: int = 45,
    window: str = "1min",
) -> List[Alert]:
    alerts: List[Alert] = []
    if orders.empty:
        return alerts
    tmp = orders.copy()
    tmp["timestamp"] = pd.to_datetime(tmp["timestamp"], utc=True)
    counts = (
        tmp.groupby(["account_id", pd.Grouper(key="timestamp", freq=window)])
        .size()
        .rename("order_count")
        .reset_index()
    )
    counts = counts.loc[counts["order_count"] >= threshold_per_minute]
    counts = counts.sort_values("order_count", ascending=False).drop_duplicates("account_id", keep="first")
    for _, row in counts.iterrows():
        alerts.append(
            Alert(
                rule_name="message_rate_spike",
                severity="medium",
                account_id=row["account_id"],
                related_accounts=[],
                description=f"Order message velocity {row['order_count']} per {window}",
                metadata={
                    "window": window,
                    "order_count": row["order_count"],
                },
                event_time=row.get("timestamp", pd.Timestamp.now(tz="UTC")),
            )
        )
    return alerts


def detect_cancel_ratio_spikes(
    orders: pd.DataFrame,
    window: str = "30min",
    min_messages: int = 20,
    ratio_threshold: float = 6.0,
) -> List[Alert]:
    alerts: List[Alert] = []
    if orders.empty:
        return alerts
    tmp = orders.copy()
    tmp["timestamp"] = pd.to_datetime(tmp["timestamp"], utc=True)
    tmp["is_cancel"] = tmp["status"].eq("canceled").astype(int)
    tmp["is_fill"] = tmp["status"].eq("filled").astype(int)
    grouped = (
        tmp.set_index("timestamp")
        .groupby("account_id")[["is_cancel", "is_fill"]]
        .rolling(window)
        .sum()
        .reset_index()
    )
    grouped = grouped.loc[grouped["is_cancel"] >= min_messages]
    grouped["ratio"] = grouped["is_cancel"] / (grouped["is_fill"] + 1e-9)
    flagged = grouped.loc[grouped["ratio"] >= ratio_threshold]
    flagged = flagged.sort_values("timestamp").drop_duplicates("account_id", keep="last")

    for _, row in flagged.iterrows():
        alerts.append(
            Alert(
                rule_name="cancel_to_fill_spike",
                severity="medium",
                account_id=row["account_id"],
                related_accounts=[],
                description=f"Cancel/fill ratio {row['ratio']:.1f} within {window}",
                metadata={
                    "window": window,
                    "cancels": row["is_cancel"],
                    "fills": row["is_fill"],
                },
                event_time=row["timestamp"],
            )
        )
    return alerts


def detect_layering_sequences(
    orders: pd.DataFrame,
    min_levels: int = 4,
    window_seconds: int = 90,
    max_lifetime_seconds: int = 15,
) -> List[Alert]:
    alerts: List[Alert] = []
    if orders.empty:
        return alerts
    tmp = orders.loc[orders["status"] == "canceled"].copy()
    if tmp.empty:
        return alerts
    tmp["timestamp"] = pd.to_datetime(tmp["timestamp"], utc=True)
    tmp["completion_ts"] = pd.to_datetime(tmp["completion_ts"], utc=True)

    for (account_id, asset, side), group in tmp.groupby(["account_id", "asset", "side"]):
        group = group.sort_values("timestamp").reset_index(drop=True)
        n = len(group)
        if n < min_levels:
            continue
        for start in range(n - min_levels + 1):
            window = group.iloc[start : start + min_levels]
            time_delta = (window["timestamp"].iloc[-1] - window["timestamp"].iloc[0]).total_seconds()
            if time_delta > window_seconds:
                continue
            price_diff = np.diff(window["price"])
            if not ((price_diff > 0).all() or (price_diff < 0).all()):
                continue
            lifetimes = (window["completion_ts"] - window["timestamp"]).dt.total_seconds()
            if (lifetimes > max_lifetime_seconds).any():
                continue
            alerts.append(
                Alert(
                    rule_name="layering_multi_level",
                    severity="high",
                    account_id=account_id,
                    related_accounts=[],
                    description=f"{asset} {side} ladder of {len(window)} levels canceled within {time_delta:.0f}s",
                    metadata={
                        "asset": asset,
                        "side": side,
                        "levels": len(window),
                        "time_span_seconds": time_delta,
                    },
                    event_time=window["timestamp"].iloc[-1],
                )
            )
            break
    return alerts


def detect_front_running(
    trades: pd.DataFrame,
    large_quantity: float = 320.0,
    lookback_seconds: int = 45,
    unwind_seconds: int = 90,
    min_profit_bps: float = 25.0,
) -> List[Alert]:
    alerts: List[Alert] = []
    trades = trades.sort_values("timestamp").copy()
    large_trades = trades.loc[trades["quantity"] >= large_quantity]
    if large_trades.empty:
        return alerts

    for _, lt in large_trades.iterrows():
        asset = lt["asset"]
        lt_ts = lt["timestamp"]
        lt_side = lt["aggressor_side"]
        window_start = lt_ts - pd.Timedelta(seconds=lookback_seconds)
        pre = trades.loc[
            (trades["asset"] == asset)
            & (trades["timestamp"] >= window_start)
            & (trades["timestamp"] < lt_ts)
            & (trades["initiator_account_id"] != lt["initiator_account_id"])
        ]
        if pre.empty:
            continue
        post = trades.loc[
            (trades["asset"] == asset)
            & (trades["timestamp"] > lt_ts)
            & (trades["timestamp"] <= lt_ts + pd.Timedelta(seconds=unwind_seconds))
        ]
        if post.empty:
            continue

        for acct, entry_group in pre.groupby("initiator_account_id"):
            entry = entry_group.sort_values("timestamp").iloc[-1]
            if entry["aggressor_side"] != lt_side:
                continue
            exit_group = post.loc[
                post["initiator_account_id"] == acct
            ]
            exit_group = exit_group.sort_values("timestamp")
            exit_group = exit_group.loc[exit_group["aggressor_side"] != entry["aggressor_side"]]
            if exit_group.empty:
                continue
            exit_trade = exit_group.iloc[0]

            if entry["aggressor_side"] == "buy":
                profit_bps = (exit_trade["price"] - entry["price"]) / entry["price"] * 10000
            else:
                profit_bps = (entry["price"] - exit_trade["price"]) / entry["price"] * 10000

            if profit_bps < min_profit_bps:
                continue

            alerts.append(
                Alert(
                    rule_name="front_running",
                    severity="high",
                    account_id=acct,
                    related_accounts=[lt["initiator_account_id"]],
                    description=f"Front-running suspected ahead of large {lt_side} trade on {asset}",
                    metadata={
                        "profit_bps": profit_bps,
                        "entry_trade": entry["trade_id"],
                        "large_trade": lt["trade_id"],
                        "exit_trade": exit_trade["trade_id"],
                    },
                    event_time=lt_ts,
                )
            )
    return alerts


def run_all_rules(
    trades: pd.DataFrame,
    orders: pd.DataFrame,
    accounts: pd.DataFrame,
    positions: pd.DataFrame,
    ml_scores: Optional[pd.DataFrame] = None,
) -> List[Alert]:
    alerts: List[Alert] = []
    alerts.extend(detect_self_trades(trades))
    alerts.extend(detect_ping_pong(trades))
    alerts.extend(detect_extreme_price_moves(trades))
    alerts.extend(detect_volume_spikes(trades))
    alerts.extend(detect_account_profile_deviation(trades, accounts))
    alerts.extend(detect_position_concentration(positions))
    alerts.extend(detect_cross_market_divergence(trades))
    alerts.extend(detect_network_collusion(trades, accounts))
    alerts.extend(detect_ml_behavioral_anomalies(trades, orders, accounts, positions, precomputed=ml_scores))
    alerts.extend(detect_spoofing(orders))
    alerts.extend(detect_message_rate_anomalies(orders))
    alerts.extend(detect_cancel_ratio_spikes(orders))
    alerts.extend(detect_layering_sequences(orders))
    alerts.extend(detect_front_running(trades))
    return alerts
