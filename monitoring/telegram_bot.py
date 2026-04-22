# monitoring/telegram_bot.py

"""
Telegram Bot for AlphaEdge.
Sends trading signals directly to your phone.
"""

import requests
import json
import os
import logging
from datetime import datetime

logger = logging.getLogger(__name__)

# ==========================================
# PUT YOUR TELEGRAM CREDENTIALS HERE
# ==========================================
TELEGRAM_TOKEN = os.getenv(
    'TELEGRAM_BOT_TOKEN',
    '8483995149:AAGOkl-1hX2pwYwfCbcVNTOwLkEUyhSzekQ'
)
TELEGRAM_CHAT_ID = os.getenv(
    'TELEGRAM_CHAT_ID',
    '8616636381'
)


class TelegramBot:
    """
    Sends trading alerts to your Telegram.
    """

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
            print(
                "   ⚠️ Telegram alerts disabled."
                " Set token in telegram_bot.py"
            )

    def send_message(self, text):
        """Send a message to Telegram."""

        if not self.enabled:
            return False

        try:
            url = f"{self.base_url}/sendMessage"
            payload = {
                'chat_id': self.chat_id,
                'text': text,
                'parse_mode': 'HTML',
            }
            response = requests.post(
                url, json=payload, timeout=10
            )

            if response.status_code == 200:
                return True
            else:
                logger.warning(
                    f"Telegram error: {response.text}"
                )
                return False

        except Exception as e:
            logger.warning(f"Telegram send failed: {e}")
            return False

    def alert_buy_signal(self, symbol, price,
                         prediction, regime, sentiment):
        """Send BUY signal alert."""

        text = (
            f"🟢 <b>BUY SIGNAL</b>\n"
            f"\n"
            f"Symbol: <b>{symbol}</b>\n"
            f"Price: <b>${price:.2f}</b>\n"
            f"Prediction: {prediction:.3f}\n"
            f"Regime: {regime}\n"
            f"Sentiment: {sentiment:+.2f}\n"
            f"\n"
            f"🕐 {datetime.now().strftime('%Y-%m-%d %H:%M')}"
        )

        return self.send_message(text)

    def alert_sell_signal(self, symbol, price,
                          pnl, pnl_pct, reason):
        """Send SELL signal alert."""

        emoji = "🟢" if pnl > 0 else "🔴"

        text = (
            f"{emoji} <b>POSITION CLOSED</b>\n"
            f"\n"
            f"Symbol: <b>{symbol}</b>\n"
            f"Price: <b>${price:.2f}</b>\n"
            f"P&L: ${pnl:+.2f} ({pnl_pct:+.1%})\n"
            f"Reason: {reason}\n"
            f"\n"
            f"🕐 {datetime.now().strftime('%Y-%m-%d %H:%M')}"
        )

        return self.send_message(text)

    def alert_daily_summary(self, portfolio_value,
                            total_pnl, total_pct,
                            positions, signals):
        """Send daily portfolio summary."""

        # Build positions text
        # Build positions text

        pos_text = ""
        if positions:
            for sym, pos in positions.items():
                shares = pos.get('shares', 0)
                entry = pos.get('entry_price', 0)
                current = pos.get('current_price', entry)
                pnl = pos.get('pnl', 0.0)
                pnl_pct = pos.get('pnl_pct', 0.0)
                p_emoji = "🟢" if pnl >= 0 else "🔴"
                p_sign = "+" if pnl >= 0 else ""
                pos_text += (
                    f"   {p_emoji} {sym}: {shares}sh"
                    f" | entry ${entry:.2f}"
                    f" | now ${current:.2f}"
                    f" | {p_sign}${pnl:.2f}"
                    f" ({p_sign}{pnl_pct:.1%})\n"
                )
        else:
            pos_text = "   No open positions\n"

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
            if buy_signals
            else 'None'
        )
        avoid_text = (
            ', '.join(avoid_signals)
            if avoid_signals
            else 'None'
        )

        pnl_emoji = "UP" if total_pnl >= 0 else "DOWN"
        pnl_sign = "+" if total_pnl >= 0 else ""

        text = (
            f"<b>DAILY SUMMARY</b>\n"
            f"\n"
            f"Portfolio: <b>${portfolio_value:,.2f}</b>\n"
            f"P&amp;L: {pnl_emoji} <b>{pnl_sign}${total_pnl:,.2f}"
            f" ({pnl_sign}{total_pct:.1%})</b>\n"
            f"\n"
            f"<b>Open Positions:</b>\n"
            f"{pos_text}\n"
            f"<b>BUY signals:</b> {buy_text}\n"
            f"<b>AVOID signals:</b> {avoid_text}\n"
            f"\n"
            f"AlphaEdge V3 - Automated"
        )

        print(f"   Sending daily summary to Telegram...")
        result = self.send_message(text)
        print(f"   Telegram result: {result}")
        return result

    def alert_stop_loss(self, symbol, price, pnl):
        """Alert when stop loss triggers."""

        text = (
            f"🔴 <b>STOP LOSS TRIGGERED</b>\n"
            f"\n"
            f"Symbol: <b>{symbol}</b>\n"
            f"Exit Price: ${price:.2f}\n"
            f"Loss: ${pnl:.2f}\n"
            f"\n"
            f"Position closed automatically.\n"
            f"\n"
            f"🕐 {datetime.now().strftime('%Y-%m-%d %H:%M')}"
        )

        return self.send_message(text)

    def alert_take_profit(self, symbol, price, pnl):
        """Alert when take profit hits."""

        text = (
            f"🟢 <b>TAKE PROFIT HIT!</b>\n"
            f"\n"
            f"Symbol: <b>{symbol}</b>\n"
            f"Exit Price: ${price:.2f}\n"
            f"Profit: +${pnl:.2f}\n"
            f"\n"
            f"🕐 {datetime.now().strftime('%Y-%m-%d %H:%M')}"
        )

        return self.send_message(text)

    def test(self):
        """Send a test message."""

        text = (
            f"✅ <b>AlphaEdge Bot Connected!</b>\n"
            f"\n"
            f"Your trading bot is now linked.\n"
            f"You will receive:\n"
            f"   🟢 BUY signals\n"
            f"   🔴 SELL alerts\n"
            f"   📊 Daily summaries\n"
            f"\n"
            f"🕐 {datetime.now().strftime('%Y-%m-%d %H:%M')}"
        )

        result = self.send_message(text)

        if result:
            print("   ✅ Test message sent to Telegram!")
        else:
            print("   ❌ Test message failed")

        return result


if __name__ == "__main__":
    bot = TelegramBot()
    bot.test()
