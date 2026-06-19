# run_backtest.py

"""
Walk Forward Backtest V5
With proper risk management: stop loss, take profit,
trailing stop, daily loss limit.

Fixes applied (v2):
- Features now engineered INSIDE each walk-forward window (Critical 1)
- Data validation before proceeding (Critical 2)
- All risk params defined in BACKTEST_CONFIG and passed explicitly (Critical 3)
- Results saved to timestamped JSON (Flaw 1)
- Feature names resolved per window inside backtester (Flaw 2)
- Sector-balanced watchlist to reduce selection bias (Flaw 3)
- Wrapped in main() with __name__ guard (Flaw 4)
- Random seed set for reproducibility (Risk 1)
- Timing added to every step (Risk 2)
- Graceful partial fetch failure handling (Risk 3)
"""

import os
import json
import time
import random
import logging
import numpy as np
from datetime import datetime

from data.stock_data import StockDataFetcher
from data.feature_engine import FeatureEngine
from backtest.walk_forward import WalkForwardBacktester

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s | %(levelname)s | %(message)s',
    datefmt='%H:%M:%S'
)
logger = logging.getLogger(__name__)

# ------------------------------------------------------------------ #
#  CONFIGURATION — single source of truth for all parameters         #
# ------------------------------------------------------------------ #

RANDOM_SEED = 42

# Fix 1.5: Survivorship-bias-aware watchlist.
# 'delisted' category includes stocks removed from major indices for poor
# performance. Excluding them would teach the model on winners only, inflating
# simulated returns by ~1-3% per year (Fama 1997, Jiang 2011).
# yfinance retains historical price data for delisted tickers.
WATCHLIST = {
    'tech'       : ['AAPL', 'MSFT', 'NVDA', 'AMD'],
    'consumer'   : ['AMZN', 'TSLA', 'NFLX', 'WMT'],
    'financials' : ['JPM', 'V', 'GS'],
    'healthcare' : ['JNJ', 'UNH'],
    'etf'        : ['SPY', 'QQQ'],
    'energy'     : ['XOM', 'CVX'],
    # Fix 1.5 — Delisted / distressed stocks (survivorship bias correction)
    # These were removed from indices due to poor performance or bankruptcy.
    # Include them so the model sees real failure cases during training.
    'delisted'   : [
        'SHLDQ',  # Sears — filed Ch.11 Oct 2018
        'BBBYQ',  # Bed Bath & Beyond — filed Ch.11 Apr 2023
        'HTZGQ',  # Hertz — filed Ch.11 May 2020
        'RVLCQ',  # Revlon — filed Ch.11 Jun 2022
        'FXIDF',  # Frontier Communications — filed Ch.11 Apr 2020
    ],
}

BACKTEST_CONFIG = {
    'train_window_days'      : 180,
    'retrain_frequency_days' : 30,
    'top_features'           : 20,
    'min_auc'                : 0.52,
    'stop_loss_pct'          : 0.015,   # -1.5%
    'take_profit_pct'        : 0.045,   # +4.5% (3:1 reward/risk)
    'trailing_stop_pct'      : 0.015,   # -1.5% from high
    'daily_loss_limit_pct'   : 0.02,   # -2% daily circuit breaker
    'random_seed'            : RANDOM_SEED,
}

MIN_ROWS_REQUIRED = (
    BACKTEST_CONFIG['train_window_days'] +
    BACKTEST_CONFIG['retrain_frequency_days']
)
MIN_SYMBOLS_REQUIRED = 5


# ------------------------------------------------------------------ #
#  MAIN                                                               #
# ------------------------------------------------------------------ #

def main():

    # Set random seeds for reproducibility
    random.seed(RANDOM_SEED)
    np.random.seed(RANDOM_SEED)

    flat_watchlist = [
        symbol
        for sector in WATCHLIST.values()
        for symbol in sector
    ]

    print("\n" + "🚀" * 25)
    print("  WALK FORWARD BACKTEST V5")
    print("  Risk Managed: Stop Loss + Take Profit + Trailing Stop")
    print("🚀" * 25)

    print(f"\n  Watchlist : {len(flat_watchlist)} stocks "
          f"across {len(WATCHLIST)} sectors")
    print(f"  Seed      : {RANDOM_SEED}")
    print(f"  Config    : {BACKTEST_CONFIG}")

    # ── Step 1: Fetch Data ─────────────────────────────────────────
    print("\n── 1. Fetching Data ──────────────────────────────────────")
    step_start = time.time()

    fetcher = StockDataFetcher(
        watchlist=flat_watchlist,
        lookback_days=1825
    )

    all_data = fetcher.get_combined_data()

    # Validate fetched data
    if all_data is None or all_data.empty:
        raise RuntimeError(
            "Data fetch returned empty dataframe. "
            "Check StockDataFetcher and yfinance connection."
        )

    if len(all_data) < MIN_ROWS_REQUIRED:
        raise RuntimeError(
            f"Insufficient data: got {len(all_data)} rows, "
            f"need at least {MIN_ROWS_REQUIRED}. "
            f"Increase lookback_days or reduce train_window_days."
        )

    # Check for partial fetch failures
    symbols_fetched = (
        all_data['symbol'].unique().tolist()
        if 'symbol' in all_data.columns
        else flat_watchlist
    )
    symbols_missing = [
        s for s in flat_watchlist
        if s not in symbols_fetched
    ]

    if symbols_missing:
        logger.warning(
            "Missing data for: %s — continuing with %d/%d symbols",
            symbols_missing,
            len(symbols_fetched),
            len(flat_watchlist)
        )

    if len(symbols_fetched) < MIN_SYMBOLS_REQUIRED:
        raise RuntimeError(
            f"Too few symbols ({len(symbols_fetched)}) to run a "
            f"meaningful backtest. Minimum required: {MIN_SYMBOLS_REQUIRED}."
        )

    logger.info(
        "Data loaded: %d rows | %d symbols | %s → %s | %.1fs",
        len(all_data),
        len(symbols_fetched),
        str(all_data.index.min())[:10],
        str(all_data.index.max())[:10],
        time.time() - step_start
    )

    # ── Step 2: Run Backtest ───────────────────────────────────────
    # NOTE: FeatureEngine is passed INTO the backtester so that
    # add_all_features() is called INSIDE each walk-forward window.
    # This prevents look-ahead bias from full-dataset normalization.
    print("\n── 2. Running Walk-Forward Backtest ──────────────────────")
    print("   (Features engineered per window to prevent look-ahead bias)")
    step_start = time.time()

    backtester = WalkForwardBacktester(
        **BACKTEST_CONFIG,
        feature_engine=FeatureEngine()
    )

    # Limit max simultaneous positions to reduce drawdown
    backtester.stock_selector.max_stocks = 3

    performance = backtester.run(all_data)

    backtest_duration = time.time() - step_start
    logger.info("Backtest complete: %.1fs", backtest_duration)

    # ── Step 3: Save Results ───────────────────────────────────────
    print("\n── 3. Saving Results ─────────────────────────────────────")

    os.makedirs('logs', exist_ok=True)
    timestamp    = datetime.now().strftime('%Y%m%d_%H%M%S')
    results_path = f"logs/backtest_{timestamp}.json"

    output = {
        'config'          : BACKTEST_CONFIG,
        'watchlist'       : WATCHLIST,
        'symbols_fetched' : symbols_fetched,
        'symbols_missing' : symbols_missing,
        'data_rows'       : len(all_data),
        'performance'     : performance,
        'duration_seconds': backtest_duration,
        'run_at'          : datetime.now().isoformat(),
    }

    # Atomic write — crash safe
    tmp_path = results_path + '.tmp'
    try:
        with open(tmp_path, 'w') as f:
            json.dump(output, f, indent=2, default=str)
        os.replace(tmp_path, results_path)
        logger.info("Results saved → %s", results_path)
    except Exception as e:
        logger.error("Failed to save results: %s", e)
        if os.path.exists(tmp_path):
            os.remove(tmp_path)

    # ── Step 4: Print Summary ──────────────────────────────────────
    print("\n" + "🎉" * 20)
    print("  BACKTEST V5 COMPLETE")
    print("🎉" * 20)

    print("\n  Risk Management Config Applied:")
    print(f"   Stop Loss:        -{BACKTEST_CONFIG['stop_loss_pct']:.0%}")
    print(f"   Take Profit:      +{BACKTEST_CONFIG['take_profit_pct']:.0%}")
    print(f"   Trailing Stop:    -{BACKTEST_CONFIG['trailing_stop_pct']:.0%} from high")
    print(f"   Daily Loss Limit: -{BACKTEST_CONFIG['daily_loss_limit_pct']:.0%}")
    print(f"\n  Results saved to: {results_path}")
    print(f"  Duration: {backtest_duration:.1f}s\n")

    return performance


if __name__ == '__main__':
    main()