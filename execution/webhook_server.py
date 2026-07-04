# execution/webhook_server.py

"""
Webhook server that receives signals from TradingView.
TradingView sends alerts → This server receives them →
Your system processes and trades.

Fix 4.1: HMAC-SHA256 request signature verification.
Callers must include X-AlphaEdge-Signature: sha256=<hex> header.
Invalid or missing signatures are rejected with HTTP 403.
"""

import os
import hmac
import hashlib
import json
import logging
import time
from collections import defaultdict
from datetime import datetime
from flask import Flask, request, jsonify, abort

logger = logging.getLogger(__name__)

app = Flask(__name__)

# Store received signals
received_signals = []

# Realized exit action types
REALIZED_ACTIONS = {'SELL', 'PARTIAL_SELL'}

# All valid action types
VALID_ACTIONS = {'BUY', 'SELL', 'PARTIAL_SELL', 'CLOSE', 'HOLD'}

# Webhook order sizing/limits — kept in sync with alpaca_live.py's own
# RISK_PER_TRADE_PCT / MAX_POSITIONS so this entry point doesn't run under
# a different risk regime than the primary strategy.
WEBHOOK_RISK_PER_TRADE_PCT = 0.015
MAX_WEBHOOK_POSITIONS      = 5

# Webhook secret — read from env, never hardcode
# Fix 4.1: Use ALPHAEDGE_WEBHOOK_SECRET env var (stronger name, required by audit)
#
# No hardcoded fallback: this file is in a public GitHub repo, so any
# fallback string here is a known value to anyone who reads the source.
# A silent fallback means a forgotten env var on a fresh deploy degrades
# auth to "anyone who read the repo" with no error and no log line.
# Fail loudly instead — refuse to import rather than run unauthenticated.
WEBHOOK_SECRET = os.getenv('ALPHAEDGE_WEBHOOK_SECRET') or os.getenv('WEBHOOK_SECRET')
if not WEBHOOK_SECRET:
    raise RuntimeError(
        'ALPHAEDGE_WEBHOOK_SECRET is not set. Refusing to start the webhook '
        'server without a real secret — there is no safe hardcoded default.'
    )

# ── Fix 4.1: HMAC-SHA256 signature verification ───────────────────────────────

def verify_signature(payload: bytes, signature_header: str) -> bool:
    """
    Fix 4.1: Verify HMAC-SHA256 request signature.
    Header format: 'sha256=<hex_digest>'

    TradingView / calling system must compute:
        signature = hmac.new(ALPHAEDGE_WEBHOOK_SECRET.encode(), body_bytes, sha256).hexdigest()
    and send: X-AlphaEdge-Signature: sha256=<signature>

    Returns True if valid, False otherwise.
    Always use hmac.compare_digest() to prevent timing attacks.
    """
    if not signature_header or not signature_header.startswith('sha256='):
        return False
    expected = hmac.new(
        WEBHOOK_SECRET.encode('utf-8'),
        payload,
        hashlib.sha256,
    ).hexdigest()
    received = signature_header[len('sha256='):]
    return hmac.compare_digest(expected, received)

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

    # ── Fix 4.1: HMAC-SHA256 signature verification ───────────────────────────
    # Check before rate limit — invalid signature = free 403 reject, no counter.
    # Header: X-AlphaEdge-Signature: sha256=<hex>
    raw_body  = request.get_data()   # read body once
    signature = request.headers.get('X-AlphaEdge-Signature', '')
    if not verify_signature(raw_body, signature):
        logger.warning(
            f'Webhook from {request.remote_addr}: invalid/missing HMAC signature'
        )
        return jsonify({
            'status' : 'error',
            'message': 'Invalid signature — request rejected',
        }), 403

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
        if not hmac.compare_digest(secret, WEBHOOK_SECRET):
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
# Persisted to disk: an in-memory-only flag means a crash or systemd restart
# while the kill switch is active would silently re-enable trading with no
# record it ever happened. That defeats the point of an emergency halt.
KILL_SWITCH_FILE = 'logs/kill_switch.json'


def _load_kill_switch_state() -> tuple[bool, str]:
    try:
        if os.path.exists(KILL_SWITCH_FILE):
            with open(KILL_SWITCH_FILE, 'r') as f:
                data = json.load(f)
            return bool(data.get('active', False)), str(data.get('reason', ''))
    except Exception as e:
        logger.error(f'Failed to load kill switch state: {e}')
    return False, ''


def _save_kill_switch_state(active: bool, reason: str) -> None:
    try:
        os.makedirs('logs', exist_ok=True)
        tmp_path = KILL_SWITCH_FILE + '.tmp'
        with open(tmp_path, 'w') as f:
            json.dump({
                'active': active,
                'reason': reason,
                'updated': datetime.now().isoformat(),
            }, f, indent=2)
        os.replace(tmp_path, KILL_SWITCH_FILE)
    except Exception as e:
        logger.error(f'Failed to persist kill switch state: {e}')


_kill_switch_active, _kill_switch_reason = _load_kill_switch_state()
if _kill_switch_active:
    logger.critical(
        f'Kill switch loaded as ACTIVE from disk on startup — '
        f'reason: {_kill_switch_reason}'
    )


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
        if not secret or not hmac.compare_digest(secret, WEBHOOK_SECRET):
            logger.warning('Kill switch attempt with invalid secret')
            return jsonify({'status': 'error', 'message': 'Invalid secret'}), 401

        reason = str(data.get('reason', 'Manual halt'))[:200]
        _kill_switch_active = True
        _kill_switch_reason = reason
        _save_kill_switch_state(_kill_switch_active, _kill_switch_reason)
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
        if not secret or not hmac.compare_digest(secret, WEBHOOK_SECRET):
            return jsonify({'status': 'error', 'message': 'Invalid secret'}), 401

        _kill_switch_active = False
        _kill_switch_reason = ''
        _save_kill_switch_state(_kill_switch_active, _kill_switch_reason)
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
        from risk_circuit_breaker import RiskCircuitBreaker

        broker = AlpacaBroker()
        telegram = TelegramBot()

        if action == 'BUY':
            # ── Circuit breaker gate ────────────────────────────────────
            # The primary strategy (alpaca_live.py) checks this before
            # every trade. The webhook path must not be a side door that
            # bypasses daily-loss / drawdown / total-loss protection.
            account = broker.get_account()
            if not account:
                logger.error('Webhook BUY blocked — could not fetch account')
                return

            circuit_breaker = RiskCircuitBreaker()
            starting_capital = circuit_breaker.state.get('starting_capital') \
                or account['portfolio_value']
            if circuit_breaker.check(
                current_value=account['portfolio_value'],
                starting_capital=starting_capital,
                telegram=telegram,
            ):
                logger.warning(
                    f'Webhook BUY {symbol} blocked — circuit breaker active'
                )
                return

            # ── Max open positions gate ─────────────────────────────────
            open_positions = broker.get_positions() if hasattr(broker, 'get_positions') else None
            if open_positions and len(open_positions) >= MAX_WEBHOOK_POSITIONS:
                logger.warning(
                    f'Webhook BUY {symbol} blocked — max positions '
                    f'({MAX_WEBHOOK_POSITIONS}) already open'
                )
                return

            # ── Risk-based sizing (matches alpaca_live.py RISK_PER_TRADE_PCT) ──
            # NOT a flat % of buying power — buying_power on a margin account
            # is not equity, and 10% of it can be a much larger fraction of
            # actual capital than intended.
            stop_loss_pct   = 0.03
            take_profit_pct = 0.08
            portfolio       = account['portfolio_value']
            dollar_risk     = portfolio * WEBHOOK_RISK_PER_TRADE_PCT
            amount          = min(dollar_risk / stop_loss_pct, portfolio * 0.15)

            success = broker.set_bracket_order(
                symbol,
                amount_dollars=amount,
                stop_loss_pct=stop_loss_pct,
                take_profit_pct=take_profit_pct,
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
    """Save signals to file atomically (temp + rename).
    Prevents a truncated/corrupted JSON if the process is killed mid-write.
    """
    import tempfile
    target = 'logs/webhook_signals.json'
    os.makedirs('logs', exist_ok=True)
    tmp_fd, tmp_path = tempfile.mkstemp(dir='logs', suffix='.tmp')
    try:
        with os.fdopen(tmp_fd, 'w') as f:
            json.dump(received_signals[-100:], f, indent=2)
        os.replace(tmp_path, target)   # atomic on POSIX + Windows
    except Exception:
        if os.path.exists(tmp_path):
            os.unlink(tmp_path)
        raise


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

    # Fix 7.2: Bind to localhost only — closes external attack surface.
    # If TradingView external access is needed, place nginx in front with
    # SSL termination and IP allowlisting. Do NOT re-expose 0.0.0.0.
    app.run(
        host='127.0.0.1',
        port=port,
        debug=False,
        use_reloader=False,
    )


if __name__ == "__main__":
    start_webhook_server()