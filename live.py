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
import signal
import threading
from datetime import datetime

warnings.filterwarnings('ignore')

from alpaca_live import AlpacaLiveTrader

logger = logging.getLogger(__name__)

SCAN_INTERVAL = int(os.getenv('SCAN_INTERVAL_SECONDS', str(30 * 60)))  # 30 min default

# ── Graceful shutdown event (SIGTERM / Ctrl-C) ────────────────────────────────
_stop_event = threading.Event()

def _handle_signal(signum, frame):
    """Wake the sleep event immediately so the process exits cleanly."""
    sig_name = signal.Signals(signum).name
    logger.info("Signal %s received — shutting down gracefully", sig_name)
    _stop_event.set()

signal.signal(signal.SIGTERM, _handle_signal)
signal.signal(signal.SIGINT,  _handle_signal)


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
    """Run scanner on a loop. Exits cleanly on SIGTERM (no 30-min block)."""
    while not _stop_event.is_set():
        try:
            run_scanner()
        except Exception as e:
            logger.error("Scanner error: %s", e)

        if _stop_event.is_set():
            break

        next_scan = SCAN_INTERVAL // 60
        logger.info("Next scan in %d minutes.", next_scan)
        # wait() wakes immediately when _stop_event.set() is called (SIGTERM)
        _stop_event.wait(timeout=SCAN_INTERVAL)


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
    logger.info("="*50)
    logger.info("ALPHAEDGE LIVE SYSTEM — starting scanner + dashboard")
    logger.info("Dashboard: http://localhost:8050  |  scan every %d min", SCAN_INTERVAL // 60)
    logger.info("="*50)

    # Run first scan immediately
    try:
        run_scanner()
    except Exception as e:
        logger.error("Initial scan failed: %s", e)
        logger.warning("Initial scan failed — dashboard starting anyway")

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