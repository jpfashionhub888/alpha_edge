# live.py

"""
AlphaEdge Live System
Runs the scanner + dashboard together.
Scanner auto-updates every 30 minutes.
Dashboard auto-refreshes every 60 seconds.

Usage: python live.py
Then open http://localhost:8050
"""

import os
import sys
import json
import time
import logging
import warnings
import threading
from datetime import datetime

warnings.filterwarnings('ignore')

logging.basicConfig(
    level=logging.INFO,
    format='%(levelname)s: %(message)s'
)

logger = logging.getLogger(__name__)

# How often to re-scan markets (in seconds)
SCAN_INTERVAL = 1800  # 30 minutes


def run_scanner():
    """
    C5 FIX: Stale V2 skeleton replaced — delegates to production AlpacaLiveTrader.

    The old implementation was missing ALL V4 features:
      - No VetoAgent, circuit breaker, market regime, MTF, correlation filter
      - No earnings calendar, MetaLabeler, options flow, insider tracker
      - Hardcoded 15-stock watchlist (current system has 41 stocks)
      - Ran every 30 min instead of once-daily at 16:15 ET

    Direct usage: python alpaca_live.py  (preferred)
    This function exists only for backward compatibility with live.py main().
    """
    import warnings
    warnings.warn(
        "live.py run_scanner() is deprecated. Run: python alpaca_live.py",
        DeprecationWarning, stacklevel=2,
    )
    try:
        from alpaca_live import AlpacaLiveTrader
        trader = AlpacaLiveTrader()
        if trader.broker.connected:
            trader._run_scan()
        else:
            logger.error("Alpaca not connected — run_scanner() aborted.")
    except Exception as e:
        logger.error(f"run_scanner() delegation to AlpacaLiveTrader failed: {e}")
        raise



def scanner_loop():
    """Run scanner on a loop in background thread."""

    while True:
        try:
            run_scanner()
        except Exception as e:
            logger.error(f"Scanner error: {e}")

        next_scan = SCAN_INTERVAL // 60
        print(
            f"\n   ⏰ Next scan in {next_scan} minutes."
            f" Dashboard is live at http://localhost:8050"
        )
        time.sleep(SCAN_INTERVAL)


def start_dashboard():
    """Start the web dashboard."""

    from monitoring.dashboard import create_app

    app = create_app()
    app.run(
        debug=False,
        host='0.0.0.0',
        port=8050,
        use_reloader=False
    )


def main():
    print("\n" + "🌐" * 25)
    print("ALPHAEDGE LIVE SYSTEM")
    print("🌐" * 25)
    print("\nStarting scanner + dashboard...")
    print("Dashboard will be at: http://localhost:8050")
    print("Scanner runs every 30 minutes automatically")
    print("Press Ctrl+C to stop everything\n")

    # Run first scan immediately
    try:
        run_scanner()
    except Exception as e:
        logger.error(f"Initial scan failed: {e}")
        print("   ⚠️ Initial scan failed but dashboard starting anyway")

    # Start scanner in background thread
    scanner_thread = threading.Thread(
        target=scanner_loop,
        daemon=True
    )
    scanner_thread.start()

    # Start dashboard in main thread (blocks)
    start_dashboard()


if __name__ == "__main__":
    main()