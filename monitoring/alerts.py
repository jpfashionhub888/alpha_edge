# monitoring/alerts.py

import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from datetime import datetime
import logging
import json
import os

logger = logging.getLogger(__name__)


class AlertSystem:
    """
    Sends email alerts when important events happen.
    Uses Gmail SMTP (free).

    Setup:
    1. Go to Google Account settings
    2. Security -> 2-Step Verification -> ON
    3. Security -> App Passwords -> Generate
    4. Use that password below
    """

    def __init__(self, email=None, password=None):
        self.email = mayatra.girish@gmail.com
        self.password = Abundance$888
        self.enabled = email is not None

        if not self.enabled:
            print(
                "   ⚠️ Email alerts disabled."
                " Set email in config to enable."
            )

    def send_email(self, subject, body):
        """Send an email alert."""

        if not self.enabled:
            return False

        try:
            msg = MIMEMultipart()
            msg['From'] = self.email
            msg['To'] = self.email
            msg['Subject'] = f"🚀 AlphaEdge: {subject}"

            msg.attach(MIMEText(body, 'plain'))

            server = smtplib.SMTP('smtp.gmail.com', 587)
            server.starttls()
            server.login(self.email, self.password)
            server.send_message(msg)
            server.quit()

            logger.info(f"Alert sent: {subject}")
            return True

        except Exception as e:
            logger.warning(f"Email alert failed: {e}")
            return False

    def alert_buy_signal(self, symbol, price, prediction,
                         regime, sentiment):
        """Send alert when BUY signal triggers."""

        subject = f"BUY Signal: {symbol} at ${price:.2f}"

        body = f"""
AlphaEdge BUY Signal
=====================
Time: {datetime.now().strftime('%Y-%m-%d %H:%M')}
Symbol: {symbol}
Price: ${price:.2f}
Prediction: {prediction:.3f}
Regime: {regime}
Sentiment: {sentiment:+.2f}

Action Required: Review and confirm trade.
        """

        return self.send_email(subject, body)

    def alert_stop_loss(self, symbol, price, pnl):
        """Send alert when stop loss triggers."""

        subject = f"STOP LOSS: {symbol} at ${price:.2f}"

        body = f"""
AlphaEdge Stop Loss Triggered
==============================
Time: {datetime.now().strftime('%Y-%m-%d %H:%M')}
Symbol: {symbol}
Exit Price: ${price:.2f}
P&L: ${pnl:+.2f}

Position closed automatically.
        """

        return self.send_email(subject, body)

    def alert_take_profit(self, symbol, price, pnl):
        """Send alert when take profit triggers."""

        subject = f"PROFIT: {symbol} at ${price:.2f}"

        body = f"""
AlphaEdge Take Profit Hit!
===========================
Time: {datetime.now().strftime('%Y-%m-%d %H:%M')}
Symbol: {symbol}
Exit Price: ${price:.2f}
P&L: ${pnl:+.2f}

Position closed with profit!
        """

        return self.send_email(subject, body)

    def alert_daily_summary(self, portfolio_value,
                            total_pnl, positions,
                            signals):
        """Send daily portfolio summary."""

        subject = f"Daily Summary: ${portfolio_value:,.2f}"

        positions_text = ""
        if positions:
            for sym, pos in positions.items():
                positions_text += (
                    f"   {sym}: {pos.get('shares', 0)} shares"
                    f" @ ${pos.get('entry_price', 0):.2f}\n"
                )
        else:
            positions_text = "   No open positions\n"

        buy_signals = [
            s for s, d in signals.items()
            if d.get('signal') == 'BUY'
        ]
        buy_text = ', '.join(buy_signals) if buy_signals else 'None'

        body = f"""
AlphaEdge Daily Summary
========================
Time: {datetime.now().strftime('%Y-%m-%d %H:%M')}

Portfolio Value: ${portfolio_value:,.2f}
Total P&L: ${total_pnl:+,.2f}

Open Positions:
{positions_text}
BUY Signals Today: {buy_text}

Dashboard: http://localhost:8050
        """

        return self.send_email(subject, body)


def save_alert_log(alert_type, details):
    """Save alert to local log file."""

    log_file = 'logs/alerts.json'

    alerts = []
    if os.path.exists(log_file):
        with open(log_file, 'r') as f:
            try:
                alerts = json.load(f)
            except json.JSONDecodeError:
                alerts = []

    alerts.append({
        'type': alert_type,
        'time': datetime.now().isoformat(),
        'details': details,
    })

    # Keep last 100 alerts
    alerts = alerts[-100:]

    os.makedirs('logs', exist_ok=True)
    with open(log_file, 'w') as f:
        json.dump(alerts, f, indent=2)