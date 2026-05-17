# Entry Points — How AlphaEdge Actually Starts

The repo has 9 files at the root that look like entry points. Most are
either wrappers, dev-only, or vestigial. Use this as the source of truth.

## Production path (this is what runs in GitHub Actions)

```
.github/workflows/daily_scan.yml
    └── runs: python cloud_scan.py
              └── calls: main.run_daily_scan()
                  ├── PaperTrader.load_state()
                  ├── SectorRotation().analyze()
                  ├── RiskCircuitBreaker.check()
                  ├── StockScanner.fetch_earnings_calendar()
                  ├── StockScanner.scan_stocks()    ◄── most of the work
                  ├── StockScanner.scan_crypto()
                  ├── for symbol in signals:
                  │       PaperTrader.open_position(...)
                  ├── for symbol in trader.positions:
                  │       PaperTrader.update_position(...)
                  ├── atomic_json_write(dashboard data)
                  ├── TelegramBot.alert_daily_summary()
                  └── if weekly:
                         PerformanceAnalytics.send_report()
                         CriticAgent.run_weekly_review()
```

## Everything else

| File | Purpose | Status |
|---|---|---|
| `cloud_scan.py` | Production entry. Skips Saturdays. Calls main. | ✅ keep |
| `main.py` | The orchestrator. All real work routes here. | ✅ keep |
| `run.py` | Local dev wrapper that aggressively suppresses warnings before calling `main.run_daily_scan()`. Useful if your terminal is noisy. | ✅ keep |
| `live.py` | After V2 patch: loops `main.run_daily_scan()` with sleep. Used for local "long-running" sessions. | ✅ keep (patched) |
| `live_trading.py` | Variant of `live.py` that also calls Alpaca. Now mostly redundant since `live.py` covers the loop and Alpaca runs out of `cloud_scan.py` in prod. | ⚠️ consider deleting |
| `run_backtest.py` | Walk-forward backtest runner. Separate from live path. | ✅ keep |
| `run_dashboard.py` | Starts the Dash dashboard at localhost:8050. | ✅ keep |
| `generate_dashboard.py` | Generates static HTML dashboard for GitHub Pages. | ✅ keep |
| `deploy.py` / `deploy.sh` | Deployment helpers. | ✅ keep |

## What to delete

These files exist in the repo and provide no clear value:
- `12.0` (random file at root, likely a pip-install typo)
- `model_cache/*.joblib` (orphaned from a previous cache implementation;
  current cache lives at `cache/models/`)
- `patch_dashboard.py` (likely a one-off fix script)

## What "live trading" means in this codebase

There are two different "live" concepts:

1. **Live paper trading** — `cloud_scan.py` or `live.py` continually
   scanning markets and updating PaperTrader state. Money never moves.
   This is what your system does today.

2. **Live broker connection** — `execution/alpaca_broker.py` connecting
   to Alpaca's paper or live API. Currently used by `live_trading.py`
   and `execution/webhook_server.py`. Even when set to live URL, the
   patched `alpaca_broker.py` requires `ALPACA_LIVE_CONFIRM=I_UNDERSTAND`
   in env to actually connect — explicit opt-in to real money.

If you want to graduate from concept 1 to concept 2:
- Validate concept 1 with 3+ months of paper P&L
- Run concept 1 and concept 2 (against Alpaca paper) in parallel for
  another month to verify they make the same trades
- THEN flip the Alpaca URL to live and set the confirmation env var
