# live_trading.py

"""
AlphaEdge Live Trading System.
Combines scanner + Alpaca broker + webhooks + dashboard.
One command to run everything.
"""

import warnings
import os
import sys
import time
import threading
import logging
from datetime import datetime

warnings.filterwarnings('ignore')
os.environ['PYTHONWARNINGS'] = 'ignore'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

logging.basicConfig(
    level=logging.INFO,
    format='%(levelname)s: %(message)s'
)

logger = logging.getLogger(__name__)


def run_live_scan():
    """Run scan and execute on Alpaca."""

    from main import run_daily_scan
    from execution.alpaca_broker import AlpacaBroker

    # Run the scan
    run_daily_scan()

    # Show Alpaca status
    broker = AlpacaBroker()
    if broker.connected:
        broker.get_summary()


def scanner_loop():
    """Run scanner every 30 minutes."""

    SCAN_INTERVAL = 1800

    while True:
        try:
            run_live_scan()
        except Exception as e:
            logger.error(f"Scan error: {e}")

        minutes = SCAN_INTERVAL // 60
        print(
            f"\n   ⏰ Next scan in {minutes} minutes."
        )
        time.sleep(SCAN_INTERVAL)


def start_webhook():
    """Start webhook server in background."""

    try:
        from execution.webhook_server import app
        app.run(
            host='0.0.0.0',
            port=5000,
            debug=False,
            use_reloader=False
        )
    except Exception as e:
        logger.error(f"Webhook server error: {e}")


def start_dashboard():
    """Start dashboard."""

    from monitoring.dashboard import create_app
    app = create_app()
    app.run(
        debug=False,
        host='0.0.0.0',
        port=8050,
        use_reloader=False
    )


def main():
    print("\n" + "🚀" * 25)
    print("ALPHAEDGE LIVE TRADING SYSTEM")
    print("🚀" * 25)

    print("\nStarting all services:")
    print("   📊 Dashboard: http://localhost:8050")
    print("   📡 Webhooks:  http://localhost:5000/webhook")
    print("   🔄 Scanner:   Every 30 minutes")
    print("\nPress Ctrl+C to stop everything\n")

    # Check Alpaca connection
    from execution.alpaca_broker import AlpacaBroker
    broker = AlpacaBroker()

    if not broker.connected:
        print("\n   ⚠️ Alpaca not connected.")
        print("   System will use paper trading instead.")
        print("   Set API keys in alpaca_broker.py\n")

    # Run first scan
    try:
        run_live_scan()
    except Exception as e:
        logger.error(f"Initial scan failed: {e}")

    # Start webhook server in background
    webhook_thread = threading.Thread(
        target=start_webhook,
        daemon=True
    )
    webhook_thread.start()
    print("   ✅ Webhook server started on port 5000")

    # Start scanner loop in background
    scanner_thread = threading.Thread(
        target=scanner_loop,
        daemon=True
    )
    scanner_thread.start()
    print("   ✅ Scanner loop started (30 min interval)")

    # Start dashboard in main thread
    print("   ✅ Starting dashboard...")
    start_dashboard()


if __name__ == "__main__":
    main()