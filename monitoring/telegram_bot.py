# monitoring/telegram_bot.py

"""
Telegram Bot for AlphaEdge.
Sends trading signals directly to your phone.
"""

import requests
import os
import logging
from datetime import datetime

logger = logging.getLogger(__name__)

# ── Secrets loader ────────────────────────────────────────────────────
# systemd services inject env vars via EnvironmentFile=/etc/alphaedge/secrets.
# When scripts are run directly (retrain.py, run_validation.py) that file
# is not loaded automatically, so we load it here as a fallback.
_SECRETS_PATHS = [
    '/etc/alphaedge/secrets',
    'config/secrets.env',
    '.env',
]

def _load_secrets_if_needed() -> None:
    """Load secrets file into os.environ if Telegram vars are missing."""
    if os.getenv('TELEGRAM_BOT_TOKEN'):
        return   # already in environment — nothing to do
    for path in _SECRETS_PATHS:
        try:
            with open(path) as f:
                for line in f:
                    line = line.strip()
                    if not line or line.startswith('#') or '=' not in line:
                        continue
                    key, _, val = line.partition('=')
                    os.environ.setdefault(key.strip(), val.strip())
            logger.debug(f'Loaded secrets from {path}')
            return
        except FileNotFoundError:
            continue
        except Exception as e:
            logger.warning(f'Could not load secrets from {path}: {e}')

_load_secrets_if_needed()

TELEGRAM_TOKEN   = os.getenv('TELEGRAM_BOT_TOKEN', '')
TELEGRAM_CHAT_ID = os.getenv('TELEGRAM_CHAT_ID', '')

class TelegramBot:

    def __init__(self, token=None, chat_id=None):
        self.token = token or TELEGRAM_TOKEN
        self.chat_id = chat_id or TELEGRAM_CHAT_ID
        self.base_url = (
            f"https://api.telegram.org/bot{self.token}"
        )
        self.enabled = (
            'YOUR_BOT_TOKEN_HERE' not in self.token
            and 'YOUR_CHAT_ID_HERE' not in self.chat_id
        )

        if not self.enabled:
            print("Telegram alerts disabled.")

    def send_message(self, text):
        if not self.enabled:
            print("Telegram not enabled, skipping.")
            return False

        url     = f"{self.base_url}/sendMessage"
        payload = {'chat_id': self.chat_id, 'text': text}

        # S7 FIX: retry up to 3× with 2s backoff (was single attempt — critical
        # alerts like circuit-breaker triggers could be silently dropped on transient errors)
        for attempt in range(1, 4):
            try:
                response = requests.post(url, json=payload, timeout=10)
                if response.status_code == 200:
                    return True
                logger.warning(
                    f"Telegram attempt {attempt}/3 failed: HTTP {response.status_code}"
                )
            except Exception as e:
                logger.warning(f"Telegram attempt {attempt}/3 exception: {e}")
            if attempt < 3:
                import time as _t; _t.sleep(2)

        logger.error("Telegram send_message failed after 3 attempts")
        return False

    def alert_buy_signal(self, symbol, price,
                         prediction, regime, sentiment):
        text = (
            f"BUY SIGNAL\n"
            f"\n"
            f"Symbol: {symbol}\n"
            f"Price: ${price:.2f}\n"
            f"Prediction: {prediction:.3f}\n"
            f"Regime: {regime}\n"
            f"Sentiment: {sentiment:+.2f}\n"
            f"\n"
            f"Time: {datetime.now().strftime('%Y-%m-%d %H:%M')}"
        )
        print(f"   Sending BUY alert for {symbol}...")
        return self.send_message(text)

    def alert_sell_signal(self, symbol, price,
                          pnl, pnl_pct, reason):
        direction = "PROFIT" if pnl > 0 else "LOSS"
        text = (
            f"POSITION CLOSED - {direction}\n"
            f"\n"
            f"Symbol: {symbol}\n"
            f"Price: ${price:.2f}\n"
            f"PnL: ${pnl:+.2f} ({pnl_pct:+.1%})\n"
            f"Reason: {reason}\n"
            f"\n"
            f"Time: {datetime.now().strftime('%Y-%m-%d %H:%M')}"
        )
        return self.send_message(text)

    def alert_daily_summary(self, portfolio_value,
                            total_pnl, total_pct,
                            positions, signals):

        # Build positions text
        pos_text = ""
        if positions:
            for sym, pos in positions.items():
                shares = pos.get('shares', 0)
                entry = pos.get('entry_price', 0)
                current = pos.get('current_price', entry)
                pnl = pos.get('pnl', 0.0)
                pnl_pct = pos.get('pnl_pct', 0.0)
                pnl_sign = "+" if pnl >= 0 else ""
                direction = "UP" if pnl >= 0 else "DOWN"
                pos_text += (
                    f"  {sym}: {shares} shares\n"
                    f"  Entry: ${entry:.2f} "
                    f"Now: ${current:.2f}\n"
                    f"  PnL: {direction} "
                    f"{pnl_sign}${pnl:.2f} "
                    f"({pnl_sign}{pnl_pct:.1%})\n\n"
                )
        else:
            pos_text = "  No open positions\n"

        # Build signals text
        buy_signals = []
        avoid_signals = []
        for sym, data in signals.items():
            sig = data.get('signal', 'HOLD')
            if sig == 'BUY':
                buy_signals.append(sym)
            elif sig == 'AVOID':
                avoid_signals.append(sym)

        buy_text = (
            ', '.join(buy_signals)
            if buy_signals else 'None'
        )
        avoid_text = (
            ', '.join(avoid_signals)
            if avoid_signals else 'None'
        )

        pnl_sign = "+" if total_pnl >= 0 else ""
        pnl_direction = "UP" if total_pnl >= 0 else "DOWN"

        text = (
            f"ALPHAEDGE DAILY SUMMARY\n"
            f"========================\n"
            f"Time: {datetime.now().strftime('%Y-%m-%d %H:%M')}\n"
            f"\n"
            f"Portfolio: ${portfolio_value:,.2f}\n"
            f"Total PnL: {pnl_direction} "
            f"{pnl_sign}${total_pnl:,.2f} "
            f"({pnl_sign}{total_pct:.1%})\n"
            f"\n"
            f"Open Positions:\n"
            f"{pos_text}"
            f"BUY signals: {buy_text}\n"
            f"AVOID signals: {avoid_text}\n"
            f"\n"
            f"AlphaEdge V3 Automated"
        )

        print(f"   Sending daily summary to Telegram...")
        result = self.send_message(text)
        print(f"   Telegram result: {result}")
        return result

    def alert_stop_loss(self, symbol, price, pnl):
        text = (
            f"STOP LOSS TRIGGERED\n"
            f"\n"
            f"Symbol: {symbol}\n"
            f"Exit Price: ${price:.2f}\n"
            f"Loss: ${pnl:.2f}\n"
            f"\n"
            f"Position closed automatically.\n"
            f"\n"
            f"Time: {datetime.now().strftime('%Y-%m-%d %H:%M')}"
        )
        return self.send_message(text)

    def alert_take_profit(self, symbol, price, pnl):
        text = (
            f"TAKE PROFIT HIT\n"
            f"\n"
            f"Symbol: {symbol}\n"
            f"Exit Price: ${price:.2f}\n"
            f"Profit: +${pnl:.2f}\n"
            f"\n"
            f"Time: {datetime.now().strftime('%Y-%m-%d %H:%M')}"
        )
        return self.send_message(text)

    def test(self):
        text = (
            f"AlphaEdge Bot Connected!\n"
            f"\n"
            f"Your trading bot is now linked.\n"
            f"You will receive:\n"
            f"  BUY signals\n"
            f"  SELL alerts\n"
            f"  Daily summaries\n"
            f"\n"
            f"Time: {datetime.now().strftime('%Y-%m-%d %H:%M')}"
        )
        result = self.send_message(text)
        if result:
            print("   Test message sent to Telegram!")
        else:
            print("   Test message failed")
        return result


if __name__ == "__main__":
    bot = TelegramBot()
    bot.test()
