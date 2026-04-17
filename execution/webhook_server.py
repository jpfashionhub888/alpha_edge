# execution/webhook_server.py

"""
Webhook server that receives signals from TradingView.
TradingView sends alerts → This server receives them →
Your system processes and trades.
"""

import json
import logging
from datetime import datetime
from flask import Flask, request, jsonify

logger = logging.getLogger(__name__)

app = Flask(__name__)

# Store received signals
received_signals = []

# Webhook secret (set this in TradingView too)
WEBHOOK_SECRET = 'alphaedge_secret_2026'


@app.route('/webhook', methods=['POST'])
def receive_webhook():
    """Receive webhook from TradingView."""

    try:
        data = request.get_json()

        if data is None:
            data = request.form.to_dict()

        if not data:
            raw = request.data.decode('utf-8')
            try:
                data = json.loads(raw)
            except json.JSONDecodeError:
                data = {'message': raw}

        # Verify secret if provided
        secret = data.get('secret', '')
        if secret and secret != WEBHOOK_SECRET:
            return jsonify({
                'status': 'error',
                'message': 'invalid secret'
            }), 401

        # Parse signal
        signal = {
            'time': datetime.now().isoformat(),
            'symbol': data.get('symbol', 'UNKNOWN'),
            'action': data.get('action', 'UNKNOWN'),
            'price': data.get('price', 0),
            'message': data.get('message', ''),
            'source': 'tradingview',
            'raw': data,
        }

        received_signals.append(signal)

        # Keep last 100 signals
        if len(received_signals) > 100:
            received_signals.pop(0)

        # Save to file
        save_signals()

        action = signal['action'].upper()
        symbol = signal['symbol']
        price = signal['price']

        print(
            f"\n   📡 WEBHOOK RECEIVED:"
            f" {action} {symbol} @ ${price}"
        )

        # Process the signal
        process_signal(signal)

        return jsonify({
            'status': 'success',
            'signal': signal
        }), 200

    except Exception as e:
        logger.error(f"Webhook error: {e}")
        return jsonify({
            'status': 'error',
            'message': str(e)
        }), 500


@app.route('/signals', methods=['GET'])
def get_signals():
    """View all received signals."""

    return jsonify({
        'signals': received_signals[-20:],
        'total': len(received_signals),
    })


@app.route('/health', methods=['GET'])
def health_check():
    """Health check endpoint."""

    return jsonify({
        'status': 'running',
        'time': datetime.now().isoformat(),
        'signals_received': len(received_signals),
    })


def process_signal(signal):
    """Process a received webhook signal."""

    action = signal.get('action', '').upper()
    symbol = signal.get('symbol', '')

    if not symbol:
        return

    try:
        from execution.alpaca_broker import AlpacaBroker
        from monitoring.telegram_bot import TelegramBot

        broker = AlpacaBroker()
        telegram = TelegramBot()

        if action == 'BUY':
            account = broker.get_account()
            if account:
                # Use 10% of buying power per trade
                amount = account['buying_power'] * 0.10

                success = broker.set_bracket_order(
                    symbol,
                    amount_dollars=amount,
                    stop_loss_pct=0.03,
                    take_profit_pct=0.08
                )

                if success:
                    telegram.send_message(
                        f"📡 <b>WEBHOOK BUY</b>\n\n"
                        f"Symbol: {symbol}\n"
                        f"Source: TradingView\n"
                        f"Amount: ${amount:.2f}\n"
                        f"Stop Loss: 3%\n"
                        f"Take Profit: 8%"
                    )

        elif action == 'SELL':
            success = broker.sell(symbol)

            if success:
                telegram.send_message(
                    f"📡 <b>WEBHOOK SELL</b>\n\n"
                    f"Symbol: {symbol}\n"
                    f"Source: TradingView"
                )

    except Exception as e:
        logger.error(f"Process signal failed: {e}")


def save_signals():
    """Save signals to file."""

    import os
    os.makedirs('logs', exist_ok=True)

    with open('logs/webhook_signals.json', 'w') as f:
        json.dump(received_signals[-100:], f, indent=2)


def start_webhook_server(port=5000):
    """Start the webhook server."""

    print("\n" + "📡" * 25)
    print("TRADINGVIEW WEBHOOK SERVER")
    print("📡" * 25)
    print(f"\nWebhook URL: http://YOUR_IP:{port}/webhook")
    print(f"Health check: http://localhost:{port}/health")
    print(f"View signals: http://localhost:{port}/signals")
    print("\nTradingView Alert Message Format:")
    print('   {')
    print(f'     "secret": "{WEBHOOK_SECRET}",')
    print('     "symbol": "{{ticker}}",')
    print('     "action": "BUY",')
    print('     "price": {{close}}')
    print('   }')
    print("\nPress Ctrl+C to stop\n")

    app.run(
        host='0.0.0.0',
        port=port,
        debug=False
    )


if __name__ == "__main__":
    start_webhook_server()