from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd

# Default assets within the synthetic exchange
ASSETS = ["BTC-USD", "ETH-USD", "SOL-USD", "XRP-USD", "ALT-USD"]
VENUES = ["AlphaX", "BetaY", "GammaZ"]
ASSET_FLOAT = {
    "BTC-USD": 9000,
    "ETH-USD": 60000,
    "SOL-USD": 250000,
    "XRP-USD": 5_000_000,
    "ALT-USD": 1_000_000,
}

START_TS = pd.Timestamp.now(tz="UTC").normalize()  # Today at midnight UTC


@dataclass
class GeneratedData:
    accounts: pd.DataFrame
    trades: pd.DataFrame
    orders: pd.DataFrame
    positions: pd.DataFrame


def generate_accounts(num_accounts: int = 60, seed: int = 7) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    account_ids = [f"ACC-{i:04d}" for i in range(num_accounts)]
    regions = ["US", "EU", "APAC", "LATAM", "MENA"]
    account_types = ["retail", "prop", "institutional", "market_maker"]
    risk_tiers = ["low", "medium", "high"]

    created_offsets = rng.integers(0, 120, size=num_accounts)
    baseline_volume = rng.lognormal(mean=11.6, sigma=0.4, size=num_accounts)

    device_ids = np.array([f"FP-{rng.integers(0, 400):04d}" for _ in range(num_accounts)])
    ip_subnets = np.array(
        [
            f"10.{rng.integers(0, 255)}.{rng.integers(0, 255)}.0/24"
            for _ in range(num_accounts)
        ]
    )
    behavior_cluster = np.array([f"BC-{rng.integers(0, 120):03d}" for _ in range(num_accounts)])

    cluster_count = max(4, num_accounts // 12)
    for cluster_idx in range(cluster_count):
        members = rng.choice(num_accounts, size=int(rng.integers(3, 5)), replace=False)
        shared_ip = f"172.{rng.integers(16, 32)}.{rng.integers(0, 255)}.0/24"
        shared_device = f"SHARED-{cluster_idx:02d}"
        ip_subnets[members] = shared_ip
        device_ids[members] = shared_device
        behavior_cluster[members] = f"BC-SH-{cluster_idx:02d}"

    df = pd.DataFrame(
        {
            "account_id": account_ids,
            "region": rng.choice(regions, size=num_accounts),
            "account_type": rng.choice(account_types, size=num_accounts, p=[0.55, 0.15, 0.2, 0.1]),
            "risk_tier": rng.choice(risk_tiers, size=num_accounts, p=[0.55, 0.35, 0.1]),
            "created_at": START_TS - pd.to_timedelta(created_offsets, unit="D"),
            "baseline_volume": baseline_volume,
            "device_id": device_ids,
            "ip_subnet": ip_subnets,
            "behavior_cluster": behavior_cluster,
        }
    )

    df["is_flagged"] = df["risk_tier"].eq("high") | df["account_type"].eq("prop")
    return df


def generate_trades(
    accounts: pd.DataFrame,
    num_trades: int = 3500,
    seed: int = 11,
) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    timestamps = np.sort(rng.integers(0, num_trades * 45, size=num_trades))
    timestamps = START_TS + pd.to_timedelta(timestamps, unit="s")

    asset_probs = np.array([0.35, 0.25, 0.2, 0.1, 0.1])
    base_prices = {
        "BTC-USD": 64000,
        "ETH-USD": 3200,
        "SOL-USD": 190,
        "XRP-USD": 0.6,
        "ALT-USD": 7.0,
    }
    vols = {
        "BTC-USD": 0.002,
        "ETH-USD": 0.003,
        "SOL-USD": 0.006,
        "XRP-USD": 0.01,
        "ALT-USD": 0.015,
    }

    account_ids = accounts["account_id"].to_numpy()
    asset_state = base_prices.copy()

    trade_rows: List[Dict] = []
    for idx, ts in enumerate(timestamps):
        asset = rng.choice(ASSETS, p=asset_probs)
        vol = vols[asset]
        drift = rng.normal(0, vol / 3)
        shock = rng.normal(0, vol)
        price = max(0.2, asset_state[asset] * (1 + drift + shock))

        quantity = float(np.round(np.exp(rng.normal(2.5, 0.6)), 4))
        buy_acct, sell_acct = rng.choice(account_ids, size=2, replace=False)
        aggressor_side = rng.choice(["buy", "sell"])
        initiator_acct = buy_acct if aggressor_side == "buy" else sell_acct
        venue = rng.choice(VENUES, p=[0.45, 0.35, 0.2])

        trade_rows.append(
            {
                "trade_id": f"T{idx:05d}",
                "timestamp": ts,
                "asset": asset,
                "price": price,
                "quantity": quantity,
                "venue": venue,
                "buy_account_id": buy_acct,
                "sell_account_id": sell_acct,
                "aggressor_side": aggressor_side,
                "initiator_account_id": initiator_acct,
            }
        )
        asset_state[asset] = price

    trades = pd.DataFrame(trade_rows)
    trades = trades.sort_values("timestamp").reset_index(drop=True)

    trades = _inject_wash_trades(trades, account_ids, rng)
    trades = _inject_ping_pong(trades, account_ids, rng)
    trades = _inject_pump_and_dump(trades, rng)
    trades = _inject_front_running(trades, account_ids, rng)
    trades = _inject_account_deviation_spikes(trades, accounts, account_ids, base_prices, rng)
    trades = _inject_cross_market_divergence(trades, rng, base_prices)
    trades = _inject_collusion_sequences(trades, accounts, rng, base_prices)

    trades["timestamp"] = pd.to_datetime(trades["timestamp"], utc=True)
    trades = trades.sort_values("timestamp").reset_index(drop=True)
    return trades


def generate_orders(
    accounts: pd.DataFrame,
    trades: pd.DataFrame,
    orders_per_account: Tuple[int, int] = (20, 40),
    seed: int = 99,
) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    order_rows: List[Dict] = []
    order_seq = 0

    for _, account in accounts.iterrows():
        count = rng.integers(orders_per_account[0], orders_per_account[1])
        offsets = np.sort(rng.integers(0, 60 * 60 * 24, size=count))
        timestamps = trades["timestamp"].min() + pd.to_timedelta(offsets, unit="s")

        for ts in timestamps:
            asset = rng.choice(ASSETS, p=[0.35, 0.3, 0.2, 0.1, 0.05])
            side = rng.choice(["buy", "sell"])
            ref_price = trades.loc[trades["asset"] == asset, "price"].median()
            price = float(np.round(ref_price * (1 + rng.normal(0, 0.002)), 2))
            quantity = float(np.round(np.exp(rng.normal(2.2, 0.5)), 4))
            lifespan = int(rng.integers(5, 300))
            completion_ts = ts + pd.Timedelta(seconds=lifespan)
            status = rng.choice(["filled", "partial_fill", "canceled"], p=[0.55, 0.15, 0.3])
            if status == "canceled":
                completion_ts = ts + pd.Timedelta(seconds=int(rng.integers(2, 30)))
            venue = rng.choice(VENUES, p=[0.45, 0.35, 0.2])
            order_rows.append(
                {
                    "order_id": f"O{order_seq:06d}",
                    "account_id": account["account_id"],
                    "timestamp": ts,
                    "completion_ts": completion_ts,
                    "asset": asset,
                    "side": side,
                    "price": price,
                    "quantity": quantity,
                    "status": status,
                    "order_type": rng.choice(["limit", "iceberg", "post_only"], p=[0.7, 0.2, 0.1]),
                    "visible_quantity": quantity * rng.uniform(0.2, 1.0),
                    "venue": venue,
                }
            )
            order_seq += 1

    extra_rows, order_seq = _inject_spoofing_orders(accounts, trades, rng, start_seq=order_seq)
    order_rows.extend(extra_rows)
    extra_rows, order_seq = _inject_order_message_bursts(accounts, rng, start_seq=order_seq)
    order_rows.extend(extra_rows)
    extra_rows, order_seq = _inject_layering_sequences(accounts, trades, rng, start_seq=order_seq)
    order_rows.extend(extra_rows)
    orders = pd.DataFrame(order_rows)
    orders["timestamp"] = pd.to_datetime(orders["timestamp"], utc=True)
    orders["completion_ts"] = pd.to_datetime(orders["completion_ts"], utc=True)
    return orders.sort_values("timestamp").reset_index(drop=True)


def generate_dataset() -> GeneratedData:
    accounts = generate_accounts()
    trades = generate_trades(accounts)
    orders = generate_orders(accounts, trades)
    positions = generate_positions(accounts, trades)
    return GeneratedData(accounts=accounts, trades=trades, orders=orders, positions=positions)


def generate_positions(accounts: pd.DataFrame, trades: pd.DataFrame) -> pd.DataFrame:
    if trades.empty:
        return pd.DataFrame(
            columns=[
                "account_id",
                "asset",
                "net_quantity",
                "gross_quantity",
                "net_notional",
                "gross_notional",
                "float_share",
                "direction",
                "dominant_venue",
            ]
        )

    records: List[Dict] = []
    for _, row in trades.iterrows():
        notional = row["price"] * row["quantity"]
        records.append(
            {
                "account_id": row["buy_account_id"],
                "asset": row["asset"],
                "quantity": row["quantity"],
                "abs_quantity": row["quantity"],
                "signed_notional": notional,
                "abs_notional": notional,
                "venue": row.get("venue", "AlphaX"),
            }
        )
        records.append(
            {
                "account_id": row["sell_account_id"],
                "asset": row["asset"],
                "quantity": -row["quantity"],
                "abs_quantity": row["quantity"],
                "signed_notional": -notional,
                "abs_notional": notional,
                "venue": row.get("venue", "AlphaX"),
            }
        )
    detail = pd.DataFrame(records)
    agg = (
        detail.groupby(["account_id", "asset"])
        .agg(
            net_quantity=("quantity", "sum"),
            gross_quantity=("abs_quantity", "sum"),
            net_notional=("signed_notional", "sum"),
            gross_notional=("abs_notional", "sum"),
        )
        .reset_index()
    )
    agg["float_supply"] = agg["asset"].map(ASSET_FLOAT).fillna(1.0)
    agg["float_share"] = (agg["net_quantity"].abs() / agg["float_supply"]).clip(upper=1.0)
    agg["direction"] = np.where(agg["net_quantity"] >= 0, "long", "short")

    # dominant venue by traded size
    detail["abs_quantity"] = detail["abs_quantity"].abs()
    venue_share = (
        detail.groupby(["account_id", "asset", "venue"])["abs_quantity"]
        .sum()
        .reset_index()
    )
    idx = venue_share.groupby(["account_id", "asset"])["abs_quantity"].idxmax()
    dominant = venue_share.loc[idx][["account_id", "asset", "venue"]].rename(columns={"venue": "dominant_venue"})

    positions = agg.merge(dominant, on=["account_id", "asset"], how="left")
    return positions


def _inject_wash_trades(trades: pd.DataFrame, account_ids: np.ndarray, rng: np.random.Generator) -> pd.DataFrame:
    trades = trades.copy()
    candidate_idx = rng.choice(trades.index, size=max(10, len(trades) // 150), replace=False)
    for idx in candidate_idx:
        acct = rng.choice(account_ids)
        trades.at[idx, "buy_account_id"] = acct
        trades.at[idx, "sell_account_id"] = acct
        trades.at[idx, "initiator_account_id"] = acct
        trades.at[idx, "aggressor_side"] = rng.choice(["buy", "sell"])
    return trades


def _inject_ping_pong(trades: pd.DataFrame, account_ids: np.ndarray, rng: np.random.Generator) -> pd.DataFrame:
    rows: List[Dict] = []
    seq_count = 6
    base_ts = trades["timestamp"].min()

    for seq_id in range(seq_count):
        a, b = rng.choice(account_ids, size=2, replace=False)
        asset = rng.choice(["BTC-USD", "ETH-USD", "SOL-USD"])
        start_offset = rng.integers(60 * 60, 60 * 60 * 24)
        start_ts = base_ts + pd.Timedelta(seconds=int(start_offset))
        price = trades.loc[trades["asset"] == asset, "price"].median()
        price = float(price if not np.isnan(price) else 100.0)
        venue = rng.choice(VENUES)

        for hop in range(10):
            ts = start_ts + pd.Timedelta(seconds=hop * rng.integers(3, 10))
            aggressor_side = "buy" if hop % 2 == 0 else "sell"
            initiator = a if aggressor_side == "buy" else b
            buy_account = initiator if aggressor_side == "buy" else b
            sell_account = b if aggressor_side == "buy" else initiator

            rows.append(
                {
                    "trade_id": f"PP{seq_id:02d}_{hop:02d}",
                    "timestamp": ts,
                    "asset": asset,
                    "price": price * (1 + rng.normal(0, 0.0005)),
                    "quantity": float(np.round(rng.uniform(20, 60), 4)),
                    "venue": venue,
                    "buy_account_id": buy_account,
                    "sell_account_id": sell_account,
                    "aggressor_side": aggressor_side,
                    "initiator_account_id": initiator,
                }
            )
    if rows:
        trades = pd.concat([trades, pd.DataFrame(rows)], ignore_index=True)
    return trades


def _inject_pump_and_dump(trades: pd.DataFrame, rng: np.random.Generator) -> pd.DataFrame:
    rows: List[Dict] = []
    base_ts = trades["timestamp"].max() - pd.Timedelta(hours=6)
    asset = "ALT-USD"
    pump_prices = np.linspace(7, 14, 25)
    dump_prices = np.linspace(14, 6, 20)
    sequence = list(pump_prices) + list(dump_prices)
    start_trade_idx = len(trades)

    for idx, price in enumerate(sequence):
        ts = base_ts + pd.Timedelta(minutes=idx * 2)
        quantity = float(np.round(rng.uniform(80, 200), 4))
        aggressor_side = "buy" if idx < len(pump_prices) else "sell"
        initiator = f"ACC-{rng.integers(0, 5):04d}"
        counterparty = f"ACC-{rng.integers(5, 20):04d}"
        buy_acct = initiator if aggressor_side == "buy" else counterparty
        sell_acct = counterparty if aggressor_side == "buy" else initiator

        rows.append(
            {
                "trade_id": f"PUMP{start_trade_idx + idx}",
                "timestamp": ts,
                "asset": asset,
                "price": price,
                "quantity": quantity,
                "venue": rng.choice(VENUES, p=[0.6, 0.3, 0.1]),
                "buy_account_id": buy_acct,
                "sell_account_id": sell_acct,
                "aggressor_side": aggressor_side,
                "initiator_account_id": initiator,
            }
        )
    if rows:
        trades = pd.concat([trades, pd.DataFrame(rows)], ignore_index=True)
    return trades


def _inject_front_running(trades: pd.DataFrame, account_ids: np.ndarray, rng: np.random.Generator) -> pd.DataFrame:
    rows: List[Dict] = []
    base_ts = trades["timestamp"].min() + pd.Timedelta(hours=2)
    suspicious_accounts = rng.choice(account_ids, size=4, replace=False)
    victim_accounts = rng.choice(account_ids, size=4, replace=False)

    for idx, (front_runner, victim) in enumerate(zip(suspicious_accounts, victim_accounts)):
        asset = rng.choice(["BTC-USD", "ETH-USD", "SOL-USD"])
        ts = base_ts + pd.Timedelta(minutes=idx * 15)
        entry_price = trades.loc[trades["asset"] == asset, "price"].median()
        entry_price = float(entry_price if not math.isnan(entry_price) else 100.0)
        entry_trade = {
            "trade_id": f"FR{idx}_ENTRY",
            "timestamp": ts - pd.Timedelta(seconds=25),
            "asset": asset,
            "price": entry_price * (1 - 0.001),
            "quantity": float(np.round(rng.uniform(30, 60), 4)),
            "venue": rng.choice(VENUES),
            "buy_account_id": front_runner,
            "sell_account_id": rng.choice(account_ids),
            "aggressor_side": "buy",
            "initiator_account_id": front_runner,
        }
        big_trade = {
            "trade_id": f"FR{idx}_BIG",
            "timestamp": ts,
            "asset": asset,
            "price": entry_price * (1 + 0.01),
            "quantity": float(np.round(rng.uniform(400, 650), 4)),
            "venue": rng.choice(VENUES),
            "buy_account_id": victim,
            "sell_account_id": rng.choice(account_ids),
            "aggressor_side": "buy",
            "initiator_account_id": victim,
        }
        exit_trade = {
            "trade_id": f"FR{idx}_EXIT",
            "timestamp": ts + pd.Timedelta(seconds=20),
            "asset": asset,
            "price": entry_price * (1 + 0.012),
            "quantity": entry_trade["quantity"],
            "venue": rng.choice(VENUES),
            "buy_account_id": rng.choice(account_ids),
            "sell_account_id": front_runner,
            "aggressor_side": "sell",
            "initiator_account_id": front_runner,
        }
        rows.extend([entry_trade, big_trade, exit_trade])

    if rows:
        trades = pd.concat([trades, pd.DataFrame(rows)], ignore_index=True)
    return trades


def _inject_account_deviation_spikes(
    trades: pd.DataFrame,
    accounts: pd.DataFrame,
    account_ids: np.ndarray,
    base_prices: Dict[str, float],
    rng: np.random.Generator,
) -> pd.DataFrame:
    rows: List[Dict] = []
    if trades.empty or accounts.empty:
        return trades

    candidate_accounts = accounts.nsmallest(6, "baseline_volume")
    base_ts = trades["timestamp"].max() - pd.Timedelta(hours=1)

    for idx, (_, acct_row) in enumerate(candidate_accounts.iterrows()):
        acct_id = acct_row["account_id"]
        asset = rng.choice(ASSETS, p=[0.45, 0.25, 0.15, 0.1, 0.05])
        ref_price = trades.loc[trades["asset"] == asset, "price"].median()
        if math.isnan(ref_price):
            ref_price = base_prices.get(asset, 100.0)
        baseline = float(acct_row["baseline_volume"])
        burst_notional = baseline * rng.uniform(10, 18)
        quantity = float(np.round(burst_notional / ref_price, 4))
        counterparty = rng.choice(account_ids)
        for hop in range(6):
            ts = base_ts + pd.Timedelta(minutes=idx * 6 + hop * 0.5)
            aggressor_side = rng.choice(["buy", "sell"])
            buy_acct = acct_id if aggressor_side == "buy" else counterparty
            sell_acct = counterparty if aggressor_side == "buy" else acct_id
            rows.append(
                {
                    "trade_id": f"DEV{idx:02d}_{hop:02d}",
                    "timestamp": ts,
                    "asset": asset,
                    "price": float(np.round(ref_price * (1 + rng.normal(0, 0.0008)), 2)),
                    "quantity": quantity,
                    "venue": rng.choice(VENUES),
                    "buy_account_id": buy_acct,
                    "sell_account_id": sell_acct,
                    "aggressor_side": aggressor_side,
                    "initiator_account_id": acct_id if aggressor_side == "buy" else counterparty,
                }
            )
    if rows:
        trades = pd.concat([trades, pd.DataFrame(rows)], ignore_index=True)
    return trades


def _inject_spoofing_orders(
    accounts: pd.DataFrame,
    trades: pd.DataFrame,
    rng: np.random.Generator,
    start_seq: int,
) -> tuple[List[Dict], int]:
    rows: List[Dict] = []
    high_risk_accounts = accounts.loc[accounts["is_flagged"], "account_id"]
    if high_risk_accounts.empty:
        high_risk_accounts = accounts["account_id"]

    base_ts = trades["timestamp"].max() - pd.Timedelta(hours=4)
    for idx in range(12):
        account = rng.choice(high_risk_accounts)
        ts = base_ts + pd.Timedelta(minutes=idx * 6)
        asset = rng.choice(["BTC-USD", "ETH-USD", "SOL-USD"])
        ref_price = trades.loc[trades["asset"] == asset, "price"].median()
        price = ref_price * (1 + rng.normal(0, 0.001))
        quantity = float(np.round(rng.uniform(400, 1200), 4))
        completion_ts = ts + pd.Timedelta(seconds=int(rng.integers(2, 6)))
        rows.append(
            {
                "order_id": f"O{start_seq + idx:06d}",
                "account_id": account,
                "timestamp": ts,
                "completion_ts": completion_ts,
                "asset": asset,
                "side": rng.choice(["buy", "sell"]),
                "price": float(np.round(price, 2)),
                "quantity": quantity,
                "status": "canceled",
                "order_type": "layered_spoof",
                "visible_quantity": quantity,
                "venue": rng.choice(VENUES, p=[0.5, 0.3, 0.2]),
            }
        )
    return rows, start_seq + len(rows)


def _inject_order_message_bursts(
    accounts: pd.DataFrame,
    rng: np.random.Generator,
    start_seq: int,
) -> tuple[List[Dict], int]:
    rows: List[Dict] = []
    if accounts.empty:
        return rows, start_seq

    burst_accounts = accounts.nlargest(3, "baseline_volume")["account_id"].to_numpy()
    base_ts = START_TS + pd.Timedelta(hours=6)
    for idx, acct in enumerate(burst_accounts):
        burst_start = base_ts + pd.Timedelta(minutes=idx * 5)
        msg_count = rng.integers(60, 110)
        for hop in range(msg_count):
            ts = burst_start + pd.Timedelta(seconds=int(rng.integers(0, 50)))
            side = rng.choice(["buy", "sell"])
            price = float(np.round(rng.uniform(50, 5000), 2))
            quantity = float(np.round(rng.uniform(0.5, 5.0), 4))
            status = rng.choice(["canceled", "filled"], p=[0.85, 0.15])
            completion_ts = ts + pd.Timedelta(seconds=int(rng.integers(1, 60)))
            rows.append(
                {
                    "order_id": f"O{start_seq:06d}",
                    "account_id": acct,
                    "timestamp": ts,
                    "completion_ts": completion_ts,
                    "asset": rng.choice(ASSETS, p=[0.5, 0.2, 0.15, 0.1, 0.05]),
                    "side": side,
                    "price": price,
                    "quantity": quantity,
                    "status": status,
                    "order_type": rng.choice(["limit", "iceberg", "post_only"], p=[0.8, 0.1, 0.1]),
                    "visible_quantity": quantity if status == "filled" else quantity * rng.uniform(0.2, 1.0),
                    "venue": rng.choice(VENUES, p=[0.5, 0.35, 0.15]),
                }
            )
            start_seq += 1
    return rows, start_seq


def _inject_cross_market_divergence(
    trades: pd.DataFrame,
    rng: np.random.Generator,
    base_prices: Dict[str, float],
) -> pd.DataFrame:
    if trades.empty:
        return trades
    rows: List[Dict] = []
    all_accounts = pd.unique(pd.concat([trades["buy_account_id"], trades["sell_account_id"]], ignore_index=True))
    base_ts = trades["timestamp"].max() - pd.Timedelta(hours=3)
    assets = rng.choice(["BTC-USD", "ETH-USD", "SOL-USD"], size=2, replace=False)

    for asset in assets:
        ref_price = trades.loc[trades["asset"] == asset, "price"].median()
        if math.isnan(ref_price):
            ref_price = base_prices.get(asset, 100.0)
        spike = ref_price * rng.uniform(0.015, 0.035)
        for venue_idx, venue in enumerate(VENUES[:2]):  # create divergence between first two venues
            price_shift = spike if venue_idx == 0 else -spike
            for hop in range(15):
                ts = base_ts + pd.Timedelta(minutes=int(hop * 2) + venue_idx)
                buyer = rng.choice(all_accounts)
                seller = rng.choice(all_accounts)
                while seller == buyer:
                    seller = rng.choice(all_accounts)
                aggressor_side = rng.choice(["buy", "sell"])
                initiator = buyer if aggressor_side == "buy" else seller
                rows.append(
                    {
                        "trade_id": f"XD{asset.replace('-', '')}{venue_idx:02d}_{hop:02d}",
                        "timestamp": ts,
                        "asset": asset,
                        "price": float(np.round(ref_price + price_shift, 2)),
                        "quantity": float(np.round(rng.uniform(40, 120), 4)),
                        "venue": venue,
                        "buy_account_id": buyer,
                        "sell_account_id": seller,
                        "aggressor_side": aggressor_side,
                        "initiator_account_id": initiator,
                    }
                )
    if rows:
        trades = pd.concat([trades, pd.DataFrame(rows)], ignore_index=True)
    return trades


def _inject_collusion_sequences(
    trades: pd.DataFrame,
    accounts: pd.DataFrame,
    rng: np.random.Generator,
    base_prices: Dict[str, float],
) -> pd.DataFrame:
    if trades.empty or accounts.empty:
        return trades
    rows: List[Dict] = []
    clusters = [
        group
        for _, group in accounts.groupby("ip_subnet")
        if len(group) >= 3
    ]
    if not clusters:
        return trades
    base_ts = trades["timestamp"].min() + pd.Timedelta(hours=1)
    for idx, group in enumerate(clusters[:5]):
        asset = rng.choice(["BTC-USD", "ETH-USD", "SOL-USD"])
        ref_price = trades.loc[trades["asset"] == asset, "price"].median()
        if math.isnan(ref_price):
            ref_price = base_prices.get(asset, 200.0)
        side = rng.choice(["buy", "sell"])
        seq_len = rng.integers(6, 10)
        venue = rng.choice(VENUES)
        for hop in range(seq_len):
            acct = rng.choice(group["account_id"])
            ts = base_ts + pd.Timedelta(minutes=idx * 15) + pd.Timedelta(seconds=hop * rng.integers(5, 20))
            price = ref_price * (1 + rng.normal(0, 0.0006))
            quantity = float(np.round(rng.uniform(25, 80), 4))
            buyer = acct if side == "buy" else rng.choice(accounts["account_id"])
            seller = rng.choice(accounts["account_id"]) if side == "buy" else acct
            rows.append(
                {
                    "trade_id": f"CL{idx:02d}_{hop:02d}",
                    "timestamp": ts,
                    "asset": asset,
                    "price": float(np.round(price, 2)),
                    "quantity": quantity,
                    "venue": venue,
                    "buy_account_id": buyer,
                    "sell_account_id": seller,
                    "aggressor_side": side,
                    "initiator_account_id": acct,
                }
            )
    if rows:
        trades = pd.concat([trades, pd.DataFrame(rows)], ignore_index=True)
    return trades


def _inject_layering_sequences(
    accounts: pd.DataFrame,
    trades: pd.DataFrame,
    rng: np.random.Generator,
    start_seq: int,
) -> tuple[List[Dict], int]:
    rows: List[Dict] = []
    if accounts.empty or trades.empty:
        return rows, start_seq

    candidate_accounts = accounts.loc[accounts["is_flagged"], "account_id"]
    if candidate_accounts.empty:
        candidate_accounts = accounts["account_id"]

    base_ts = trades["timestamp"].max() - pd.Timedelta(hours=2)
    sequences = rng.integers(3, 6)

    for seq_idx in range(sequences):
        account = rng.choice(candidate_accounts)
        asset = rng.choice(["BTC-USD", "ETH-USD", "SOL-USD"])
        side = rng.choice(["buy", "sell"])
        ref_price = trades.loc[trades["asset"] == asset, "price"].median()
        if math.isnan(ref_price):
            ref_price = 500.0
        direction = 1 if side == "buy" else -1
        start_price = ref_price * (1 + direction * rng.uniform(0.0005, 0.0015))
        seq_len = rng.integers(4, 7)
        start_ts = base_ts + pd.Timedelta(minutes=seq_idx * 7)
        for hop in range(seq_len):
            ts = start_ts + pd.Timedelta(seconds=hop * rng.integers(5, 15))
            price = start_price + direction * hop * ref_price * 0.0003
            quantity = float(np.round(rng.uniform(250, 600), 4))
            completion_ts = ts + pd.Timedelta(seconds=int(rng.integers(2, 10)))
            rows.append(
                {
                    "order_id": f"O{start_seq:06d}",
                    "account_id": account,
                    "timestamp": ts,
                    "completion_ts": completion_ts,
                    "asset": asset,
                    "side": side,
                    "price": float(np.round(price, 2)),
                    "quantity": quantity,
                    "status": "canceled",
                    "order_type": "layering_stack",
                    "visible_quantity": quantity,
                    "venue": rng.choice(VENUES, p=[0.5, 0.3, 0.2]),
                }
            )
            start_seq += 1
    return rows, start_seq
