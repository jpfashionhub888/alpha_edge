# cloud_scan.py

import os
import sys

os.environ['PYTHONWARNINGS'] = 'ignore'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ['TOKENIZERS_PARALLELISM'] = 'false'
os.environ['OMP_NUM_THREADS'] = '1'

import warnings
warnings.filterwarnings('ignore')

import logging
logging.basicConfig(
    level=logging.INFO,
    format='%(levelname)s: %(message)s'
)

from datetime import datetime

logger = logging.getLogger(__name__)

print("TELEGRAM TOKEN:", os.environ.get("TELEGRAM_BOT_TOKEN"))
print("TELEGRAM CHAT ID:", os.environ.get("TELEGRAM_CHAT_ID"))

def main():
    now = datetime.now()
    day = now.strftime('%A')

    if day in ['Saturday', 'Sunday']:
        print(f"{day} - Market closed.")
        return

    print(f"\nAlphaEdge Cloud Scan - {now}")

    try:
        from main import run_daily_scan
        run_daily_scan()
        print("\nScan complete.")
    except Exception as e:
        logger.error(f"Scan failed: {e}")
        print(f"\nScan failed: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
