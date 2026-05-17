# live.py
"""
AlphaEdge Live System — V2 (thin wrapper)

Replaces the V1 fork that duplicated scan logic and silently drifted
from main.py. This version is 30 lines: it calls main.run_daily_scan()
in a loop with sleep. Any improvement to main.py propagates here for free.

Usage:
    python live.py             # one scan every 30 minutes
    SCAN_INTERVAL=600 python live.py   # override interval

Stop with Ctrl+C.
"""

import logging
import os
import sys
import time
import warnings
from datetime import datetime

warnings.filterwarnings('ignore')

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s | %(levelname)s | %(message)s',
    datefmt='%H:%M:%S',
)
logger = logging.getLogger(__name__)

SCAN_INTERVAL = int(os.environ.get('SCAN_INTERVAL', 1800))   # default 30 min


def main():
    print(f"AlphaEdge live — scanning every {SCAN_INTERVAL}s. Ctrl+C to stop.\n")

    while True:
        start = time.time()
        try:
            from main import run_daily_scan
            run_daily_scan()
        except KeyboardInterrupt:
            print("\nStopped by user.")
            sys.exit(0)
        except Exception as e:
            # main.run_daily_scan() already alerts Telegram on failure
            # and re-raises. Catching here so the loop continues.
            logger.error("Scan loop caught: %s", e, exc_info=True)

        elapsed = time.time() - start
        sleep_for = max(0, SCAN_INTERVAL - elapsed)
        next_at = datetime.now().strftime('%H:%M:%S')
        logger.info(
            "Scan took %.1fs. Sleeping %.0fs (next ~%s).",
            elapsed, sleep_for, next_at,
        )
        try:
            time.sleep(sleep_for)
        except KeyboardInterrupt:
            print("\nStopped by user.")
            sys.exit(0)


if __name__ == "__main__":
    main()
