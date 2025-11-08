# Crypto Market Surveillance Demo

Synthetic end-to-end workflow that simulates a cryptocurrency exchange, injects manipulation patterns, and runs baseline surveillance rules. Alerts are persisted to SQLite and summarized in a Jupyter notebook for monitoring/case management.

## Features
- Generates synthetic accounts, trades, and order book events with embedded scenarios: wash trading, ping-pong bursts, pump & dump, spoofing, front-running, account-message bursts, layering ladders, and cross-venue price shocks.
- Adds bespoke account-deviation bursts so profiles can drift sharply from historical norms.
- Trades and orders are tagged with venues while a position snapshot quantifies float share per account, enabling venue-aware and concentration rules.
- Embedded network metadata (device fingerprints + IP subnets) power behavioral clustering scenarios to mimic insider collaboration.
- Unsupervised ML (IsolationForest + feature engineering) ranks anomalous accounts and feeds those scores into the alert stream plus the monitoring notebook.
- Surveillance engine runs multiple rules: self-trades, bilateral ping-pong loops, extreme price moves vs rolling baseline, abnormal volume spikes, spoofing via order cancellations, account profile deviations, position concentration, message-rate spikes, cancel-to-fill surges, layering sequences, cross-market divergence, network collusion, ML behavioral anomalies, and front-running around large prints.
- Alerts persisted to `data/alerts.sqlite` plus CSV snapshot for offline review.
- Notebook (`notebooks/surveillance_dashboard.ipynb`) now plots price, aggressor volume, order-book imbalance, and includes an interactive case-management widget for alert triage.

## Surveillance Rules
- **Self-trade (wash trading)** – flags trades where the buyer and seller account ids match, signaling intentional volume inflation or manipulation.
- **Ping-pong loops** – detects rapid back-and-forth trading between two accounts within a short window, often used to manipulate prints or paint the tape.
- **Extreme price moves** – monitors each asset’s rolling mean/std; generates alerts when a trade price deviates >4.5σ, capturing pump/dump bursts.
- **Per-account volume spikes** – compares each account’s 2-hour notional sum against its trailing 12-hour baseline to surface abnormal participation.
- **Spoofing/layering** – identifies large orders canceled within seconds of submission, indicative of intent to mislead order-book liquidity.
- **Front-running** – looks for accounts entering just before a large aggressive trade, then exiting quickly for ≥25 bps profit, suggesting use of non-public flow.
- **Account profile deviation** – leverages stored account baselines to detect retail/low-activity accounts whose rolling notional suddenly exceeds their norm by 12× within a 3-hour window.
- **Position concentration** – flags accounts whose net size controls ≥20% of an asset’s float (with ≥$1M exposure), highlighting cornering attempts.
- **Message-rate spike** – monitors per-account order submissions per minute and alerts when a trader floods the venue with >45 instructions in 60 seconds.
- **Cancel-to-fill surge** – tracks rolling 30-minute cancel/fill ratios, flagging accounts that cancel six or more orders for every fill while sending at least 20 messages.
- **Multi-level layering** – spots monotonic stacks of ≥4 canceled orders at escalating/descending price levels within 90 seconds, indicative of layered spoofing.
- **Cross-market divergence** – resamples prices by venue and highlights windows where venue spreads exceed 2%, suggesting dislocations or manipulative prints.
- **Network collusion** – clusters accounts by IP/device fingerprints and flags windows where ≥3 accounts push the same asset/direction within 60s, hinting at coordinated manipulation.
- **ML behavioral anomalies** – IsolationForest ingests aggregated trade/order/position features to spot outlier accounts that deviate from learned norms even when they avoid individual rule triggers.

## Getting Started
1. Create a UV-managed virtual environment and install dependencies (Python 3.10+).
   ```bash
   uv python install 3.10          # once per machine
   uv venv --python 3.10 .venv
   source .venv/bin/activate
   uv pip install -e .
   ```
2. Run the surveillance pipeline (writes CSV artifacts and SQLite alerts under `data/`).
   ```bash
   python -m market_surveillance.main
   ```
3. Launch Jupyter to explore the monitoring notebook.
   ```bash
   jupyter notebook notebooks/surveillance_dashboard.ipynb
   ```

## Project Layout
```
.
├── data/                        # auto-generated artifacts (csv + sqlite)
│   ├── accounts.csv
│   ├── trades.csv
│   ├── orders.csv
│   ├── positions.csv
│   └── ml_scores.csv
├── notebooks/
│   └── surveillance_dashboard.ipynb
├── src/
│   └── market_surveillance/
│       ├── data_generator.py    # synthetic data + scenario injectors
│       ├── surveillance_rules.py# detection logic + Alert model
│       ├── persistence.py       # SQLite alert store helper
│       ├── engine.py            # orchestration + artifact writes
│       └── main.py              # CLI entrypoint
└── pyproject.toml
```

## Extending
- Add new rule functions under `surveillance_rules.py` and include them in `run_all_rules`.
- Expand generators with additional behaviors to test coverage.
- Extend the notebook with richer plots (e.g., order book imbalance) or workflow widgets for case tracking.
