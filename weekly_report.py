# weekly_report.py
"""
Weekly Risk Report

Runs every Saturday at 8 PM UTC via cron:
  0 20 * * 6 cd /root/alpha_edge && export $(grep -v '^#' .env | xargs) && /root/alpha_edge/venv/bin/python weekly_report.py >> /root/alpha_edge/logs/weekly_report.log 2>&1

Or run manually:
  python weekly_report.py

Sends to Telegram:
  - Portfolio performance (stocks)
  - Win rate, Sharpe, drawdown
  - Best and worst trades
  - Critic agent review with improvement suggestions
"""

import os
import json
import logging
from datetime import datetime

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s | %(levelname)s | %(message)s',
    datefmt='%H:%M:%S'
)
logger = logging.getLogger(__name__)

STOCK_TRADES_FILE   = 'logs/paper_trades_stocks_only.json'


def load_json(path, default=None):
    if default is None:
        default = {}
    if not os.path.exists(path):
        return default
    try:
        with open(path) as f:
            return json.load(f)
    except Exception as e:
        logger.warning(f'Failed to load {path}: {e}')
        return default


def run_weekly_report():
    print("\n" + "📊" * 25)
    print(f"ALPHAEDGE WEEKLY RISK REPORT — {datetime.now().strftime('%Y-%m-%d %H:%M')}")
    print("📊" * 25)

    # ── Load trade data ───────────────────────────────────────────
    stock_data = load_json(STOCK_TRADES_FILE)

    if not stock_data:
        print("  No trade data found — nothing to report yet")
        print("  Run the system for at least a few days before reporting")
        return

    trade_history    = stock_data.get('trade_history', [])
    positions        = stock_data.get('positions', {})
    capital          = stock_data.get('capital', 10000)
    starting_capital = stock_data.get('starting_capital', 10000)

    print(f"\n  Data loaded:")
    print(f"  Total trades: {len(trade_history)}")
    print(f"  Open positions: {len(positions)}")
    print(f"  Capital:  ${capital:,.2f}")

    # ── Performance Analytics ─────────────────────────────────────
    print("\n── Performance Analytics ────────────────────────────────")
    try:
        from performance_analytics import PerformanceAnalytics
        from monitoring.telegram_bot import TelegramBot

        analytics = PerformanceAnalytics()

        report = analytics.generate_report(days_back=7)
        print(report)

        # Send to Telegram
        telegram = TelegramBot()
        if telegram.enabled:
            analytics.send_report(telegram, days_back=7)
            print("  ✅ Performance report sent to Telegram")
        else:
            print("  ⚠️  Telegram not configured — report printed only")

    except Exception as e:
        logger.error(f"Performance analytics failed: {e}")

    # ── Critic Agent Review ───────────────────────────────────────
    print("\n── Critic Agent Review ──────────────────────────────────")
    try:
        from critic_agent import CriticAgent
        from monitoring.telegram_bot import TelegramBot

        critic   = CriticAgent()
        telegram = TelegramBot()

        critic.run_weekly_review(
            trade_history    = trade_history,
            portfolio_value  = capital,
            starting_capital = starting_capital,
            telegram_bot     = telegram,
        )
        print("  ✅ Critic review sent to Telegram")

    except Exception as e:
        logger.error(f"Critic agent failed: {e}")

    # ── Summary stats ─────────────────────────────────────────────
    print("\n── Summary ──────────────────────────────────────────────")
    total_pnl     = capital - starting_capital
    total_pnl_pct = (total_pnl / starting_capital) * 100 if starting_capital > 0 else 0

    sells = [t for t in trade_history if t.get('action') in ('SELL', 'PARTIAL_SELL')]
    wins  = [t for t in sells if t.get('pnl', 0) > 0]
    win_rate = len(wins) / len(sells) * 100 if sells else 0

    print(f"  Total P&L:    ${total_pnl:+,.2f} ({total_pnl_pct:+.2f}%)")
    print(f"  Closed trades: {len(sells)}")
    print(f"  Win rate:      {win_rate:.1f}%")
    print(f"  Open positions: {len(positions)}")

    print(f"\n✅ Weekly report complete")
    print(f"   Next report: Saturday {datetime.now().strftime('%Y-%m-%d')} at 20:00 UTC\n")


if __name__ == '__main__':
    run_weekly_report()
