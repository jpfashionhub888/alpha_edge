# execution/gateio_stream.py
"""
Gate.io Realtime WebSocket Price Stream

Subscribes to Gate.io public WebSocket and maintains a live
price + OHLCV cache. Fires callbacks on each confirmed candle close.

Usage:
    stream = GateioStream(symbols=['BTC_USDT', 'ETH_USDT'], interval='15m')
    stream.on_candle_close(my_callback)
    stream.start()
    ...
    stream.stop()

Callback receives:
    symbol: str  — e.g. 'BTC_USDT'
    candle: dict — {open, high, low, close, volume, timestamp, confirm}
"""

import json
import time
import logging
import threading
from typing import Callable, Optional
import websocket

logger = logging.getLogger(__name__)

WS_MAINNET = 'wss://api.gateio.ws/ws/v4/'
WS_TESTNET = 'wss://fx-ws-testnet.gateio.ws/v4/ws/usdt'

RECONNECT_DELAY    = 5
MAX_RECONNECT_WAIT = 60
PING_INTERVAL      = 20

# Gate.io interval map: our format → Gate.io format
INTERVAL_MAP = {
    '1'  : '1m',
    '5'  : '5m',
    '15' : '15m',
    '30' : '30m',
    '60' : '1h',
    '240': '4h',
    'D'  : '1d',
    # Pass-through if already in Gate.io format
    '1m' : '1m',
    '5m' : '5m',
    '15m': '15m',
    '30m': '30m',
    '1h' : '1h',
    '4h' : '4h',
    '1d' : '1d',
}


class GateioStream:
    """
    Gate.io public WebSocket stream.
    Subscribes to candlestick topics and fires callbacks on confirmed closes.
    """

    def __init__(
        self,
        symbols:   list[str],
        interval:  str  = '15m',
        testnet:   bool = False,
        on_candle: Optional[Callable] = None,
    ):
        self.symbols    = [s.upper() for s in symbols]
        self.interval   = INTERVAL_MAP.get(interval, interval)
        self.ws_url     = WS_TESTNET if testnet else WS_MAINNET
        self._callbacks = []
        self._ws        = None
        self._thread    = None
        self._running   = False
        self._reconnect_wait = RECONNECT_DELAY

        # Live price cache: symbol → float
        self.prices:  dict[str, float] = {}
        # Candle cache: symbol → latest confirmed candle dict
        self.candles: dict[str, dict]  = {}

        if on_candle:
            self._callbacks.append(on_candle)

    # ── Public API ───────────────────────────────────────────────────

    def on_candle_close(self, callback: Callable):
        self._callbacks.append(callback)

    def start(self):
        self._running = True
        self._thread  = threading.Thread(target=self._run_loop, daemon=True)
        self._thread.start()
        logger.info(f'GateioStream started | {self.symbols} | {self.interval}')

    def stop(self):
        self._running = False
        if self._ws:
            self._ws.close()
        logger.info('GateioStream stopped')

    def get_price(self, symbol: str) -> Optional[float]:
        return self.prices.get(symbol.upper())

    def get_candle(self, symbol: str) -> Optional[dict]:
        return self.candles.get(symbol.upper())

    def wait_for_prices(self, timeout: int = 30) -> bool:
        deadline = time.time() + timeout
        while time.time() < deadline:
            if self.prices:
                return True
            time.sleep(0.5)
        return False

    # ── WebSocket internals ──────────────────────────────────────────

    def _subscribe(self, ws):
        """
        Gate.io candlestick subscription format:
        {"time": ts, "channel": "spot.candlesticks",
         "event": "subscribe", "payload": ["15m", "BTC_USDT"]}
        One subscription per symbol.
        """
        ts = int(time.time())
        for symbol in self.symbols:
            msg = {
                'time'   : ts,
                'channel': 'spot.candlesticks',
                'event'  : 'subscribe',
                'payload': [self.interval, symbol],
            }
            ws.send(json.dumps(msg))
        logger.info(f'Subscribed: {self.symbols} @ {self.interval}')

    def _on_open(self, ws):
        logger.info(f'GateioStream connected: {self.ws_url}')
        self._reconnect_wait = RECONNECT_DELAY
        self._subscribe(ws)
        self._start_ping(ws)

    def _on_message(self, ws, message: str):
        try:
            data = json.loads(message)
        except json.JSONDecodeError:
            return

        # Subscription confirmation
        event = data.get('event', '')
        if event == 'subscribe':
            err = data.get('error')
            if err:
                logger.warning(f'Subscription error: {err}')
            else:
                logger.info(f'Subscription confirmed: {data.get("channel")}')
            return

        # Pong
        if event == 'pong':
            return

        # Candlestick update
        channel = data.get('channel', '')
        if channel != 'spot.candlesticks':
            return

        result = data.get('result', {})
        if not result:
            return

        self._process_candle(result)

    def _process_candle(self, result: dict):
        """
        Gate.io candlestick result format:
        {
          "t": "1596566400",   # timestamp
          "v": "97151",        # volume
          "c": "11785.6",      # close
          "h": "11801.5",      # high
          "l": "11768.5",      # low
          "o": "11778.5",      # open
          "n": "15m_BTC_USDT", # interval_symbol
          "a": "1",            # 1 = confirmed/closed candle
        }
        """
        try:
            n       = result.get('n', '')       # e.g. "15m_BTC_USDT"
            parts   = n.split('_', 1)
            if len(parts) < 2:
                return
            symbol  = parts[1]                  # e.g. "BTC_USDT"
            close   = float(result.get('c', 0))
            is_confirmed = str(result.get('a', '0')) == '1'

            if close > 0:
                self.prices[symbol] = close

            c = {
                'symbol'   : symbol,
                'timestamp': int(result.get('t', 0)) * 1000,
                'open'     : float(result.get('o', 0)),
                'high'     : float(result.get('h', 0)),
                'low'      : float(result.get('l', 0)),
                'close'    : close,
                'volume'   : float(result.get('v', 0)),
                'confirm'  : is_confirmed,
            }

            if is_confirmed:
                self.candles[symbol] = c
                logger.debug(f'Candle closed: {symbol} | close={close:.4f}')
                for callback in self._callbacks:
                    try:
                        callback(symbol, c)
                    except Exception as e:
                        logger.error(f'Callback error ({symbol}): {e}')

        except (ValueError, TypeError, KeyError) as e:
            logger.warning(f'Candle parse error: {e} | data={result}')

    def _on_error(self, ws, error):
        logger.error(f'WebSocket error: {error}')

    def _on_close(self, ws, code, msg):
        logger.warning(f'WebSocket closed: {code} {msg}')
        self._stop_ping()

    # ── Reconnect loop ───────────────────────────────────────────────

    def _run_loop(self):
        while self._running:
            try:
                self._ws = websocket.WebSocketApp(
                    self.ws_url,
                    on_open    = self._on_open,
                    on_message = self._on_message,
                    on_error   = self._on_error,
                    on_close   = self._on_close,
                )
                self._ws.run_forever(ping_interval=0)
            except Exception as e:
                logger.error(f'WebSocket run error: {e}')

            if not self._running:
                break

            logger.info(f'Reconnecting in {self._reconnect_wait}s...')
            time.sleep(self._reconnect_wait)
            self._reconnect_wait = min(self._reconnect_wait * 2, MAX_RECONNECT_WAIT)

    # ── Keepalive ping ───────────────────────────────────────────────

    def _start_ping(self, ws):
        self._stop_ping()
        threading.Thread(
            target=self._ping_loop, args=(ws,), daemon=True
        ).start()

    def _stop_ping(self):
        pass  # daemon threads self-terminate

    def _ping_loop(self, ws):
        while self._running:
            time.sleep(PING_INTERVAL)
            try:
                if ws.sock and ws.sock.connected:
                    ws.send(json.dumps({
                        'time'   : int(time.time()),
                        'channel': 'spot.candlesticks',
                        'event'  : 'ping',
                    }))
            except Exception:
                break
