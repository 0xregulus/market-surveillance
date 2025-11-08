from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional

import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler


def build_account_feature_matrix(
    accounts: pd.DataFrame,
    trades: pd.DataFrame,
    orders: pd.DataFrame,
    positions: pd.DataFrame,
) -> pd.DataFrame:
    if accounts.empty:
        return pd.DataFrame()

    feats = pd.DataFrame(index=accounts["account_id"])

    if not trades.empty:
        trades = trades.copy()
        trades["notional"] = trades["price"] * trades["quantity"]
        initiator = trades.groupby("initiator_account_id")
        feats.loc[initiator.size().index, "trade_count"] = initiator.size()
        feats.loc[initiator["asset"].nunique().index, "unique_assets"] = initiator["asset"].nunique()
        feats.loc[initiator["quantity"].mean().index, "avg_trade_qty"] = initiator["quantity"].mean()
        feats.loc[initiator["notional"].sum().index, "total_notional"] = initiator["notional"].sum()
        feats.loc[initiator["price"].std().index, "price_std"] = initiator["price"].std(ddof=0)
        venue_share = (
            trades.groupby(["initiator_account_id", "venue"])["quantity"]
            .sum()
            .groupby(level=0)
            .apply(lambda s: (s / s.sum()).max())
        )
        feats.loc[venue_share.index, "venue_concentration"] = venue_share

    if not orders.empty:
        orders = orders.copy()
        orders["is_cancel"] = orders["status"].eq("canceled").astype(int)
        orders["is_fill"] = orders["status"].eq("filled").astype(int)
        per_account = orders.groupby("account_id")
        feats.loc[per_account.size().index, "order_count"] = per_account.size()
        cancels = per_account["is_cancel"].sum()
        fills = per_account["is_fill"].sum().clip(lower=1)
        feats.loc[cancels.index, "cancel_ratio"] = cancels / fills
        orders["minute_bucket"] = pd.to_datetime(orders["timestamp"]).dt.floor("1min")
        bucket_counts = per_account["minute_bucket"].nunique().replace(0, 1)
        msg_rate = per_account.size() / bucket_counts
        feats.loc[msg_rate.index, "messages_per_minute"] = msg_rate

    if not positions.empty:
        pos = positions.groupby("account_id").agg(
            float_share=("float_share", "max"),
            gross_notional=("gross_notional", "sum"),
            net_notional=("net_notional", "sum"),
        )
        feats.loc[pos.index, ["float_share", "gross_notional", "net_notional"]] = pos

    if {"device_id", "ip_subnet"}.issubset(accounts.columns):
        device_counts = accounts.groupby("device_id")["account_id"].transform("count")
        ip_counts = accounts.groupby("ip_subnet")["account_id"].transform("count")
        feats["shared_device_size"] = device_counts.values
        feats["shared_ip_size"] = ip_counts.values

    feats = feats.fillna(0.0)
    feats = feats.replace([np.inf, -np.inf], 0.0)
    return feats


def compute_behavioral_anomalies(
    accounts: pd.DataFrame,
    trades: pd.DataFrame,
    orders: pd.DataFrame,
    positions: pd.DataFrame,
    contamination: float = 0.08,
    random_state: int = 42,
) -> pd.DataFrame:
    features = build_account_feature_matrix(accounts, trades, orders, positions)
    if features.empty:
        return pd.DataFrame(columns=["account_id", "anomaly_score", "ml_flag"])

    scaler = StandardScaler()
    X = scaler.fit_transform(features)
    model = IsolationForest(
        n_estimators=300,
        contamination=min(0.25, max(0.02, contamination)),
        random_state=random_state,
    )
    model.fit(X)
    anomaly_score = -model.decision_function(X)
    preds = model.predict(X)
    threshold = np.quantile(anomaly_score, 1 - contamination / 2)
    flagged = (preds == -1) | (anomaly_score >= threshold)

    result = features.reset_index().rename(columns={"index": "account_id"})
    result["anomaly_score"] = anomaly_score
    result["ml_flag"] = flagged
    result["rank"] = result["anomaly_score"].rank(method="dense", ascending=False)
    result = result.sort_values("anomaly_score", ascending=False).reset_index(drop=True)
    return result
