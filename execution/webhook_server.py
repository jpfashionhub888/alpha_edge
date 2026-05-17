# execution/webhook_server.py

"""
Webhook server that receives signals from TradingView.

SECURITY FIX:
    WEBHOOK_SECRET moved to env var. Was hardcoded.
    Also: secret comparison now uses hmac.compare_digest to prevent
    timing attacks (yes, on a local webhook this is overkill, but the
    cost is zero so we do it right).

    Set WEBHOOK_SECRET in config/secrets.env or as an env var.
    A startup-time random fallback is provided ONLY for local dev,
    and the secret is printed to console so you can configure
    TradingView. In production, set it explicitly.
"""

import hmac
import json
import logging
import os
import secrets as _stdlib_secrets
from datetime import datetime

from flask import Flask, request, jsonify

logger = logging.getLogger(__name__)

app = Flask(__name__)

# Store received signals (memory only; flushed to disk on update)
received_signals = []

# ----- Webhook secret resolution -----------------------------------------
# 1. WEBHOOK_SECRET from env (production)
# 2. Random fallback at startup (dev only; user must read it from logs)
_WEBHOOK_SECRET = os.environ.get('WEBHOOK_SECRET', '').strip()
if not _WEBHOOK_SECRET:
    _WEBHOOK_SECRET = _stdlib_secrets.token_urlsafe(24)
    logger.warning(
        "WEBHOOK_SECRET not set. Generated random secret for this run: %s "
        "(set WEBHOOK_SECRET in config/secrets.env to make it persistent)",
        _WEBHOOK_SECRET,
    )


def _secret_ok(provided: str) -> bool:
    """Constant-time comparison to avoid timing oracle leaks."""
    if not provided:
        return False
    return hmac.compare_digest(str(provided), _WEBHOOK_SECRET)


# ----- Routes -------------------------------------------------------------

@app.route('/webhook', methods=['POST'])
def receive_webhook():
    """Receive webhook from TradingView."""
    try:
        data = request.get_json(silent=True)
        if data is None:
            data = request.form.to_dict()
        if not data:
            raw = request.data.decode('utf-8', errors='replace')
            try:
                data = json.loads(raw)
            except json.JSONDecodeError:
                data = {'message': raw}

        # SECURITY FIX: secret is now mandatory, not optional
        if not _secret_ok(data.get('secret', '')):
            logger.warning(
                "Webhook rejected: bad/missing secret from %s",
                request.remote_addr,
            )
            return jsonify({'status': 'error', 'message': 'invalid secret'}), 401

        signal = {
            'time'   : datetime.now().isoformat(),
            'symbol' : data.get('symbol', 'UNKNOWN'),
            'action' : data.get('action', 'UNKNOWN'),
            'price'  : data.get('price', 0),
            'message': data.get('message', ''),
            'source' : 'tradingview',
            # NOTE: do NOT echo raw payload back — it can contain the secret
        }

        received_signals.append(signal)
        if len(received_signals) > 100:
            received_signals.pop(0)

        save_signals()
        process_signal(signal)

        logger.info(
            "Webhook: %s %s @ $%s",
            signal['action'], signal['symbol'], signal['price'],
        )

        return jsonify({'status': 'success', 'signal': signal}), 200

    except Exception as e:
        logger.error("Webhook error: %s", e)
        return jsonify({'status': 'error', 'message': 'internal error'}), 500


@app.route('/signals', methods=['GET'])
def get_signals():
    return jsonify({
        'signals': received_signals[-20:],
        'total'  : len(received_signals),
    })


@app.route('/health', methods=['GET'])
def health_check():
    return jsonify({
        'status'          : 'running',
        'time'            : datetime.now().isoformat(),
        'signals_received': len(received_signals),
    })


# ----- Processing ---------------------------------------------------------

def process_signal(signal):
    """Act on a webhook signal."""
    action = signal.get('action', '').upper()
    symbol = signal.get('symbol', '')
    if not symbol:
        return

    try:
        from execution.alpaca_broker import AlpacaBroker
        from monitoring.telegram_bot import TelegramBot

        broker   = AlpacaBroker()
        telegram = TelegramBot()

        if action == 'BUY':
            account = broker.get_account()
            if account:
                amount = account['buying_power'] * 0.10
                # Use plain buy(); bracket order method may not exist
                success = broker.buy(symbol, amount_dollars=amount)
                if success:
                    telegram.send_message(
                        f"📡 WEBHOOK BUY\n"
                        f"Symbol: {symbol}\nSource: TradingView\n"
                        f"Amount: ${amount:.2f}"
                    )

        elif action == 'SELL':
            success = broker.sell(symbol)
            if success:
                telegram.send_message(
                    f"📡 WEBHOOK SELL\nSymbol: {symbol}\nSource: TradingView"
                )

    except Exception as e:
        logger.error("Process signal failed: %s", e)


def save_signals():
    os.makedirs('logs', exist_ok=True)
    with open('logs/webhook_signals.json', 'w') as f:
        json.dump(received_signals[-100:], f, indent=2)


def start_webhook_server(port=5000):
    print("\n📡 TRADINGVIEW WEBHOOK SERVER")
    print(f"Webhook URL : http://YOUR_IP:{port}/webhook")
    print(f"Health      : http://localhost:{port}/health")
    print(f"Signals     : http://localhost:{port}/signals")
    print("\nTradingView alert format:")
    print('   {')
    print(f'     "secret": "<configured server-side, do not paste here>",')
    print('     "symbol": "{{ticker}}",')
    print('     "action": "BUY",')
    print('     "price": {{close}}')
    print('   }')
    print(
        "\nSet WEBHOOK_SECRET in config/secrets.env to a long random "
        "string. The server uses that to validate every incoming request.\n"
    )

    # debug=False — do NOT enable Werkzeug debug, it leaks env on errors
    app.run(host='0.0.0.0', port=port, debug=False)


if __name__ == "__main__":
    start_webhook_server()
