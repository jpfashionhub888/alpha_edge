# monitoring/telegram_bot.py

"""
Telegram Bot for AlphaEdge.

Changes:
- M2 fix: send_message() now logs errors at ERROR level instead of
  silently printing. Telegram outages no longer get swallowed.
- Replaced print()-based diagnostics with proper logger calls.
- Added simple rate-limit handling: on HTTP 429, wait and retry once.
- response.text is logged truncated to 200 chars (avoid log spam from
  full Telegram error JSON).
"""

import logging
import os
import time
from datetime import datetime

import requests

logger = logging.getLogger(__name__)

TELEGRAM_TOKEN   = os.getenv('TELEGRAM_BOT_TOKEN', '')
TELEGRAM_CHAT_ID = os.getenv('TELEGRAM_CHAT_ID', '')

REQUEST_TIMEOUT_SEC = 10


class TelegramBot:
    """Wrapper over Telegram Bot API for alerts."""

    def __init__(self, token=None, chat_id=None):
        self.token    = token or TELEGRAM_TOKEN
        self.chat_id  = chat_id or TELEGRAM_CHAT_ID
        self.base_url = f"https://api.telegram.org/bot{self.token}"
        self.enabled  = (
            self.token
            and self.chat_id
            and 'YOUR_BOT_TOKEN_HERE' not in self.token
            and 'YOUR_CHAT_ID_HERE'  not in self.chat_id
        )
        if not self.enabled:
            logger.info("Telegram alerts disabled (no token/chat_id set)")

    def send_message(self, text: str) -> bool:
        """
        Send a message. Returns True on success.

        Failures are logged at ERROR level (M2 fix — was silent print()).
        Retries once on HTTP 429 (rate limit).
        """
        if not self.enabled:
            logger.debug("Telegram not enabled, skipping send")
            return False

        url = f"{self.base_url}/sendMessage"
        payload = {'chat_id': self.chat_id, 'text': text}

        for attempt in (1, 2):
            try:
                r = requests.post(url, json=payload, timeout=REQUEST_TIMEOUT_SEC)

                if r.status_code == 200:
                    logger.debug("Telegram message sent (%d chars)", len(text))
                    return True

                # Rate limit: wait and retry
                if r.status_code == 429 and attempt == 1:
                    try:
                        retry_after = int(r.json().get('parameters', {}).get('retry_after', 1))
                    except Exception:
                        retry_after = 1
                    logger.warning(
                        "Telegram rate-limited (429), retrying after %ds",
                        retry_after,
                    )
                    time.sleep(min(retry_after, 30))
                    continue

                # Anything else is a real failure
                logger.error(
                    "Telegram send failed: status=%d body=%s",
                    r.status_code, r.text[:200],
                )
                return False

            except requests.exceptions.Timeout:
                logger.error("Telegram send timed out after %ds", REQUEST_TIMEOUT_SEC)
                return False
            except requests.exceptions.RequestException as e:
                logger.error("Telegram send network error: %s", e)
                return False
            except Exception as e:
                # Catch-all so a Telegram failure NEVER crashes the trader.
                # M2: log at ERROR so it shows up; do not silently swallow.
                logger.error("Telegram send unexpected error: %s", e, exc_info=True)
                return False

        return False

    # ─────────────────────────────────────────────────────────────────
    #  Alert helpers
    # ─────────────────────────────────────────────────────────────────

    def alert_buy_signal(self, symbol, price, prediction, regime, sentiment):
        text = (
            f"BUY SIGNAL\n\n"
            f"Symbol: {symbol}\n"
            f"Price: ${price:.2f}\n"
            f"Prediction: {prediction:.3f}\n"
            f"Regime: {regime}\n"
            f"Sentiment: {sentiment:+.2f}\n\n"
            f"Time: {datetime.now().strftime('%Y-%m-%d %H:%M')}"
        )
        return self.send_message(text)

    def alert_sell_signal(self, symbol, price, pnl, pnl_pct, reason):
        direction = "PROFIT" if pnl > 0 else "LOSS"
        text = (
            f"POSITION CLOSED - {direction}\n\n"
            f"Symbol: {symbol}\n"
            f"Price: ${price:.2f}\n"
            f"PnL: ${pnl:+.2f} ({pnl_pct:+.1%})\n"
            f"Reason: {reason}\n\n"
            f"Time: {datetime.now().strftime('%Y-%m-%d %H:%M')}"
        )
        return self.send_message(text)

    def alert_stop_loss(self, symbol, price, pnl):
        text = (
            f"STOP LOSS TRIGGERED\n\n"
            f"Symbol: {symbol}\n"
            f"Exit Price: ${price:.2f}\n"
            f"Loss: ${pnl:.2f}\n\n"
            f"Position closed automatically.\n\n"
            f"Time: {datetime.now().strftime('%Y-%m-%d %H:%M')}"
        )
        return self.send_message(text)

    def alert_take_profit(self, symbol, price, pnl):
        text = (
            f"TAKE PROFIT HIT\n\n"
            f"Symbol: {symbol}\n"
            f"Exit Price: ${price:.2f}\n"
            f"Profit: +${pnl:.2f}\n\n"
            f"Time: {datetime.now().strftime('%Y-%m-%d %H:%M')}"
        )
        return self.send_message(text)

    def alert_daily_summary(self, portfolio_value, total_pnl,
                             total_pct, positions, signals):
        # Positions block
        if positions:
            pos_lines = []
            for sym, pos in positions.items():
                shares  = pos.get('shares', 0)
                entry   = pos.get('entry_price', 0)
                current = pos.get('current_price', entry)
                pnl     = pos.get('pnl', 0.0)
                pnl_pct = pos.get('pnl_pct', 0.0)
                sign    = "+" if pnl >= 0 else ""
                arrow   = "UP" if pnl >= 0 else "DOWN"
                pos_lines.append(
                    f"  {sym}: {shares} shares\n"
                    f"  Entry: ${entry:.2f}  Now: ${current:.2f}\n"
                    f"  PnL: {arrow} {sign}${pnl:.2f} "
                    f"({sign}{pnl_pct:.1%})"
                )
            pos_text = "\n\n".join(pos_lines)
        else:
            pos_text = "  No open positions"

        # Signals block
        buy_signals   = [s for s, d in signals.items() if d.get('signal') == 'BUY']
        avoid_signals = [s for s, d in signals.items() if d.get('signal') == 'AVOID']
        buy_text   = ', '.join(buy_signals)   if buy_signals   else 'None'
        avoid_text = ', '.join(avoid_signals) if avoid_signals else 'None'

        sign  = "+" if total_pnl >= 0 else ""
        arrow = "UP" if total_pnl >= 0 else "DOWN"

        text = (
            f"ALPHAEDGE DAILY SUMMARY\n"
            f"========================\n"
            f"Time: {datetime.now().strftime('%Y-%m-%d %H:%M')}\n\n"
            f"Portfolio: ${portfolio_value:,.2f}\n"
            f"Total PnL: {arrow} {sign}${total_pnl:,.2f} "
            f"({sign}{total_pct:.1%})\n\n"
            f"Open Positions:\n{pos_text}\n\n"
            f"BUY signals: {buy_text}\n"
            f"AVOID signals: {avoid_text}\n\n"
            f"AlphaEdge V5 Automated"
        )

        return self.send_message(text)

    def test(self):
        text = (
            f"AlphaEdge Bot Connected!\n\n"
            f"Your trading bot is now linked.\n"
            f"You will receive:\n"
            f"  BUY signals\n  SELL alerts\n  Daily summaries\n\n"
            f"Time: {datetime.now().strftime('%Y-%m-%d %H:%M')}"
        )
        result = self.send_message(text)
        if result:
            print("   Test message sent to Telegram!")
        else:
            print("   Test message failed (see logs)")
        return result


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    bot = TelegramBot()
    bot.test()
