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

def main():
    now = datetime.now()
    day = now.strftime('%A')

    if day == 'Saturday':
        print("Saturday - Market closed. Skipping scan.")
        return

    if day == 'Sunday':
        # Sunday: run only the Critic Agent weekly review, not the full ML scan.
        # Markets are closed — no new daily bars — so the full pipeline on Sunday
        # produces stale or meaningless signals from Friday's data.
        print(f"\nAlphaEdge Sunday Critic Review - {now}")
        try:
            from execution.paper_trader import PaperTrader
            from monitoring.telegram_bot import TelegramBot
            from critic_agent import CriticAgent
            from performance_analytics import PerformanceAnalytics

            trader   = PaperTrader(starting_capital=10000.0)
            trader.load_state()
            telegram = TelegramBot()

            total_value = trader.capital + sum(
                pos.get('shares', 0) * pos.get('current_price', pos.get('entry_price', 0))
                for pos in trader.positions.values()
            )

            analytics = PerformanceAnalytics()
            if analytics.should_run_today():
                analytics.send_report(telegram)

            critic = CriticAgent()
            critic.run_weekly_review(
                trade_history    = trader.trade_history,
                portfolio_value  = total_value,
                starting_capital = trader.starting_capital,
                telegram_bot     = telegram,
            )
            print("\nSunday critic review complete.")
        except Exception as e:
            logger.error(f"Sunday critic review failed: {e}")
            print(f"\nSunday critic review failed: {e}")
            sys.exit(1)
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
