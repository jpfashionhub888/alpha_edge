# execution/webhook_server.py

"""
Webhook server that receives signals from TradingView.
TradingView sends alerts → This server receives them →
Your system processes and trades.
"""

import os
import json
import logging
import time
from collections import defaultdict
from datetime import datetime
from flask import Flask, request, jsonify

logger = logging.getLogger(__name__)

app = Flask(__name__)

# Store received signals
received_signals = []

# Realized exit action types
REALIZED_ACTIONS = {'SELL', 'PARTIAL_SELL'}

# All valid action types
VALID_ACTIONS = {'BUY', 'SELL', 'PARTIAL_SELL', 'CLOSE', 'HOLD'}

# Webhook secret — read from env, never hardcode
WEBHOOK_SECRET = os.getenv('WEBHOOK_SECRET', 'alphaedge_secret_2026')

# ── Rate limiter: max 10 requests/min per IP ───────────────────────────────────
_rate_counters: dict = defaultdict(list)  # ip -> [timestamps]
RATE_LIMIT_MAX    = 10    # requests
RATE_LIMIT_WINDOW = 60   # seconds


def _check_rate_limit(ip: str) -> bool:
    """Return True if request is allowed, False if rate limited."""
    now = time.time()
    window_start = now - RATE_LIMIT_WINDOW
    # Prune old timestamps
    _rate_counters[ip] = [t for t in _rate_counters[ip] if t > window_start]
    if len(_rate_counters[ip]) >= RATE_LIMIT_MAX:
        return False
    _rate_counters[ip].append(now)
    return True


def _validate_payload(data: dict) -> tuple[bool, str]:
    """
    Validate webhook payload schema.
    Returns (is_valid, error_message).
    """
    if not isinstance(data, dict):
        return False, 'Payload must be a JSON object'
    action = data.get('action', '')
    if not action:
        return False, 'Missing required field: action'
    if action.upper() not in VALID_ACTIONS:
        return False, f'Unknown action: {action}. Valid: {sorted(VALID_ACTIONS)}'
    symbol = data.get('symbol', '')
    if not symbol or not isinstance(symbol, str):
        return False, 'Missing or invalid field: symbol'
    price = data.get('price', None)
    if price is not None and not isinstance(price, (int, float)):
        return False, 'Field price must be a number'
    return True, ''


@app.route('/webhook', methods=['POST'])
def receive_webhook():
    """Receive and validate webhook signal from TradingView."""

    # ── Rate limit ─────────────────────────────────────────────────────────────
    client_ip = request.headers.get('X-Real-IP', request.remote_addr)
    if not _check_rate_limit(client_ip):
        logger.warning(f'Rate limit exceeded for IP {client_ip}')
        return jsonify({
            'status' : 'error',
            'message': 'Rate limit exceeded — max 10 signals per minute'
        }), 429
    # ── Kill switch ────────────────────────────────────────────────────────────
    if _kill_switch_active:
        logger.warning(f'Signal rejected — kill switch active: {_kill_switch_reason}')
        return jsonify({
            'status' : 'halted',
            'message': f'Kill switch active: {_kill_switch_reason}',
            'resume' : 'POST /kill-switch/reset to resume',
        }), 503

    try:
        data = request.get_json(force=True, silent=True)
        if data is None:
            # Fallback: try raw body
            raw = request.data.decode('utf-8', errors='replace')
            try:
                data = json.loads(raw)
            except (json.JSONDecodeError, ValueError):
                return jsonify({
                    'status' : 'error',
                    'message': 'Invalid JSON body'
                }), 400

        if not data:
            return jsonify({'status': 'error', 'message': 'Empty payload'}), 400

        # ── Authenticate ───────────────────────────────────────────────────────
        secret = data.get('secret', '')
        if not secret:
            logger.warning(f'Webhook from {client_ip}: missing secret field')
            return jsonify({
                'status' : 'error',
                'message': 'Missing required field: secret'
            }), 401
        if secret != WEBHOOK_SECRET:
            logger.warning(f'Webhook from {client_ip}: invalid secret')
            return jsonify({
                'status' : 'error',
                'message': 'Invalid secret'
            }), 401

        # ── Validate schema ────────────────────────────────────────────────────
        ok, err = _validate_payload(data)
        if not ok:
            logger.warning(f'Webhook from {client_ip}: invalid payload — {err}')
            return jsonify({'status': 'error', 'message': err}), 422

        # ── Build signal ───────────────────────────────────────────────────────
        signal = {
            'time'  : datetime.now().isoformat(),
            'symbol': str(data.get('symbol', 'UNKNOWN')).upper(),
            'action': str(data.get('action', 'UNKNOWN')).upper(),
            'price' : float(data.get('price', 0) or 0),
            'score' : float(data.get('score', 0.0) or 0.0),
            'source': 'tradingview',
            'ip'    : client_ip,
        }

        received_signals.append(signal)
        if len(received_signals) > 100:
            received_signals.pop(0)

        save_signals()

        logger.info(
            f'Webhook: {signal["action"]} {signal["symbol"]} '
            f'@ ${signal["price"]} from {client_ip}'
        )
        print(
            f'\n   📡 WEBHOOK RECEIVED:'
            f' {signal["action"]} {signal["symbol"]} @ ${signal["price"]}'
        )

        process_signal(signal)

        return jsonify({'status': 'success', 'signal': signal}), 200

    except Exception as e:
        logger.error(f'Webhook unhandled error from {client_ip}: {e}')
        return jsonify({'status': 'error', 'message': str(e)}), 500


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
        'status'          : 'halted' if _kill_switch_active else 'running',
        'kill_switch'     : _kill_switch_active,
        'time'            : datetime.now().isoformat(),
        'signals_received': len(received_signals),
    })


# ── Kill Switch ────────────────────────────────────────────────────────────────
_kill_switch_active: bool = False
_kill_switch_reason: str  = ''


@app.route('/kill-switch', methods=['POST'])
def activate_kill_switch():
    """
    Emergency halt — immediately blocks all new signal processing.
    POST body: {"secret": "<WEBHOOK_SECRET>", "reason": "why"}
    """
    global _kill_switch_active, _kill_switch_reason
    try:
        data   = request.get_json(force=True, silent=True) or {}
        secret = data.get('secret', '')
        if not secret or secret != WEBHOOK_SECRET:
            logger.warning('Kill switch attempt with invalid secret')
            return jsonify({'status': 'error', 'message': 'Invalid secret'}), 401

        reason = str(data.get('reason', 'Manual halt'))[:200]
        _kill_switch_active = True
        _kill_switch_reason = reason
        logger.critical(f'KILL SWITCH ACTIVATED — reason: {reason}')

        try:
            from monitoring.telegram_bot import TelegramBot
            bot = TelegramBot()
            if bot.enabled:
                bot.send_message(
                    f'EMERGENCY HALT\n\nKill switch activated.\n'
                    f'Reason: {reason}\nTime: {datetime.now().isoformat()}\n\n'
                    f'All new signals are BLOCKED.\nResume: POST /kill-switch/reset'
                )
        except Exception as e:
            logger.debug(f'Kill switch Telegram failed: {e}')

        return jsonify({'status': 'halted', 'reason': reason}), 200

    except Exception as e:
        return jsonify({'status': 'error', 'message': str(e)}), 500


@app.route('/kill-switch/reset', methods=['POST'])
def reset_kill_switch():
    """Resume trading after emergency halt."""
    global _kill_switch_active, _kill_switch_reason
    try:
        data   = request.get_json(force=True, silent=True) or {}
        secret = data.get('secret', '')
        if not secret or secret != WEBHOOK_SECRET:
            return jsonify({'status': 'error', 'message': 'Invalid secret'}), 401

        _kill_switch_active = False
        _kill_switch_reason = ''
        logger.info('Kill switch RESET — trading resumed')

        try:
            from monitoring.telegram_bot import TelegramBot
            TelegramBot().send_message(
                f'Trading Resumed\nKill switch reset.\nTime: {datetime.now().isoformat()}'
            )
        except Exception:
            pass

        return jsonify({'status': 'running', 'message': 'Kill switch reset'}), 200
    except Exception as e:
        return jsonify({'status': 'error', 'message': str(e)}), 500


@app.route('/kill-switch', methods=['GET'])
def kill_switch_status():
    """Return current kill switch state."""
    return jsonify({
        'active': _kill_switch_active,
        'reason': _kill_switch_reason,
        'time'  : datetime.now().isoformat(),
    }), 200


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

        elif action in REALIZED_ACTIONS:
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