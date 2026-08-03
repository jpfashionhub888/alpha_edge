# execution/bybit_stream.py
"""
Bybit Realtime WebSocket Price Stream

Subscribes to Bybit's public WebSocket and maintains a live
price + OHLCV cache. Calls registered callbacks on each
candle close so the signal engine can react in realtime.

Usage:
    stream = BybitStream(symbols=['BTCUSDT', 'ETHUSDT'], interval='15')
    stream.on_candle_close(my_callback)   # called with (symbol, candle_dict)
    stream.start()                        # non-blocking, runs in background thread
    ...
    stream.stop()

Callback receives:
    symbol: str  — e.g. 'BTCUSDT'
    candle: dict — {open, high, low, close, volume, timestamp, confirm}
"""

import json
import time
import logging
import threading
from typing import Callable, Optional
import websocket   # pip install websocket-client

logger = logging.getLogger(__name__)

# Bybit public WebSocket endpoints
WS_MAINNET = 'wss://stream.bybit.com/v5/public/linear'
WS_TESTNET = 'wss://stream-testnet.bybit.com/v5/public/linear'

# Reconnect config
RECONNECT_DELAY    = 5     # seconds before first reconnect attempt
MAX_RECONNECT_WAIT = 60    # cap exponential backoff at 60s
PING_INTERVAL      = 20    # seconds between keepalive pings


class BybitStream:
    """
    Bybit public WebSocket stream.
    Subscribes to kline topics and fires callbacks on confirmed candle closes.
    """

    def __init__(
        self,
        symbols:   list[str],
        interval:  str  = '15',     # '1','3','5','15','30','60','240','D'
        testnet:   bool = False,
        on_candle: Optional[Callable] = None,
    ):
        self.symbols      = [s.upper() for s in symbols]
        self.interval     = interval
        self.ws_url       = WS_TESTNET if testnet else WS_MAINNET
        self._callbacks   = []
        self._ws          = None
        self._thread      = None
        self._ping_thread = None
        self._running     = False
        self._reconnect_wait = RECONNECT_DELAY

        # Live price cache: symbol → latest price float
        self.prices: dict[str, float] = {}

        # Candle cache: symbol → latest confirmed candle dict
        self.candles: dict[str, dict] = {}

        if on_candle:
            self._callbacks.append(on_candle)

    # ── Public API ───────────────────────────────────────────────────

    def on_candle_close(self, callback: Callable):
        """Register a callback fired on each confirmed candle close."""
        self._callbacks.append(callback)

    def start(self):
        """Start stream in background thread. Non-blocking."""
        self._running = True
        self._thread  = threading.Thread(target=self._run_loop, daemon=True)
        self._thread.start()
        logger.info(f'BybitStream started | {self.symbols} | {self.interval}m')

    def stop(self):
        """Stop the stream gracefully."""
        self._running = False
        if self._ws:
            self._ws.close()
        logger.info('BybitStream stopped')

    def get_price(self, symbol: str) -> Optional[float]:
        """Get latest cached price for a symbol."""
        return self.prices.get(symbol.upper())

    def get_candle(self, symbol: str) -> Optional[dict]:
        """Get latest confirmed candle for a symbol."""
        return self.candles.get(symbol.upper())

    def wait_for_prices(self, timeout: int = 30) -> bool:
        """
        Block until at least one price is received or timeout.
        Returns True if prices received, False if timed out.
        """
        deadline = time.time() + timeout
        while time.time() < deadline:
            if self.prices:
                return True
            time.sleep(0.5)
        return False

    # ── WebSocket internals ──────────────────────────────────────────

    def _build_topics(self) -> list[str]:
        """Build Bybit kline subscription topics."""
        return [f'kline.{self.interval}.{symbol}' for symbol in self.symbols]

    def _subscribe(self, ws):
        """Send subscription message after connection opens."""
        msg = {
            'op'    : 'subscribe',
            'args'  : self._build_topics(),
        }
        ws.send(json.dumps(msg))
        logger.info(f'Subscribed to: {self._build_topics()}')

    def _on_open(self, ws):
        logger.info(f'WebSocket connected: {self.ws_url}')
        self._reconnect_wait = RECONNECT_DELAY   # reset backoff on success
        self._subscribe(ws)
        self._start_ping(ws)

    def _on_message(self, ws, message: str):
        try:
            data = json.loads(message)
        except json.JSONDecodeError:
            return

        # Handle subscription confirmation
        if data.get('op') == 'subscribe':
            if data.get('success'):
                logger.info('Subscription confirmed')
            else:
                logger.warning(f'Subscription failed: {data.get("ret_msg")}')
            return

        # Handle pong
        if data.get('op') == 'pong':
            return

        # Handle kline data
        topic = data.get('topic', '')
        if not topic.startswith('kline.'):
            return

        symbol = topic.split('.')[-1]
        candles = data.get('data', [])

        for candle in candles:
            self._process_candle(symbol, candle)

    def _process_candle(self, symbol: str, candle: dict):
        """
        Process incoming candle data.
        Updates price cache always.
        Fires callbacks only on confirmed (closed) candles.
        """
        try:
            close  = float(candle.get('close', 0))
            is_confirmed = candle.get('confirm', False)

            # Always update live price
            if close > 0:
                self.prices[symbol] = close

            # Build structured candle dict
            c = {
                'symbol'   : symbol,
                'timestamp': int(candle.get('start', 0)),
                'open'     : float(candle.get('open', 0)),
                'high'     : float(candle.get('high', 0)),
                'low'      : float(candle.get('low', 0)),
                'close'    : close,
                'volume'   : float(candle.get('volume', 0)),
                'confirm'  : is_confirmed,
            }

            if is_confirmed:
                # Update candle cache
                self.candles[symbol] = c
                logger.debug(f'Candle closed: {symbol} | close={close:.4f} | vol={c["volume"]:.2f}')

                # Fire all callbacks
                for callback in self._callbacks:
                    try:
                        callback(symbol, c)
                    except Exception as e:
                        logger.error(f'Callback error for {symbol}: {e}')

        except (ValueError, TypeError) as e:
            logger.warning(f'Candle parse error ({symbol}): {e}')

    def _on_error(self, ws, error):
        logger.error(f'WebSocket error: {error}')

    def _on_close(self, ws, close_status_code, close_msg):
        logger.warning(f'WebSocket closed: {close_status_code} {close_msg}')
        self._stop_ping()

    # ── Reconnect loop ───────────────────────────────────────────────

    def _run_loop(self):
        """Main loop — reconnects automatically on disconnect."""
        while self._running:
            try:
                self._ws = websocket.WebSocketApp(
                    self.ws_url,
                    on_open    = self._on_open,
                    on_message = self._on_message,
                    on_error   = self._on_error,
                    on_close   = self._on_close,
                )
                self._ws.run_forever(ping_interval=0)   # we handle pings manually

            except Exception as e:
                logger.error(f'WebSocket run error: {e}')

            if not self._running:
                break

            logger.info(f'Reconnecting in {self._reconnect_wait}s...')
            time.sleep(self._reconnect_wait)
            # Exponential backoff capped at MAX_RECONNECT_WAIT
            self._reconnect_wait = min(self._reconnect_wait * 2, MAX_RECONNECT_WAIT)

    # ── Keepalive ping ───────────────────────────────────────────────

    def _start_ping(self, ws):
        """Send periodic pings to keep connection alive."""
        self._stop_ping()
        self._ping_thread = threading.Thread(
            target=self._ping_loop, args=(ws,), daemon=True
        )
        self._ping_thread.start()

    def _stop_ping(self):
        self._ping_thread = None   # daemon thread will die on its own

    def _ping_loop(self, ws):
        while self._running:
            time.sleep(PING_INTERVAL)
            try:
                if ws.sock and ws.sock.connected:
                    ws.send(json.dumps({'op': 'ping'}))
            except Exception:
                break
