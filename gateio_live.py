# gateio_live.py
"""
AlphaEdge — Gate.io Realtime Trading Loop

Mirrors bybit_live.py exactly but uses Gate.io API.
Same signal engine, same filters, same risk rules.
Only the exchange layer differs.

How to run:
  export GATEIO_API_KEY=your_key
  export GATEIO_API_SECRET=your_secret
  export GATEIO_TESTNET=true    # start here
  pip install websocket-client
  python gateio_live.py

Symbols: Gate.io format BTC_USDT (configured below)
"""

import os
import time
import logging
import threading
import pandas as pd
import numpy as np
from datetime import datetime
from typing import Optional

from execution.gateio_client import GateioClient
from execution.gateio_stream  import GateioStream
from execution.symbol_map     import SymbolMap
from market_regime            import MarketRegimeFilter
from multi_timeframe          import MultiTimeframeAnalyzer
from monitoring.telegram_bot  import TelegramBot
from risk_circuit_breaker     import RiskCircuitBreaker

logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
logger = logging.getLogger(__name__)

# ── Config ───────────────────────────────────────────────────────────
# Gate.io symbol format: BTC_USDT (underscore, not no-separator)
CRYPTO_SYMBOLS  = ['BTC_USDT', 'ETH_USDT', 'SOL_USDT']
CANDLE_INTERVAL = '15m'
KLINE_LOOKBACK  = 200

BUY_THRESHOLD    = 0.63
VOLUME_SPIKE_MIN = 1.3
MIN_RR_RATIO     = 2.0
ATR_STOP_MULT    = 1.0
ATR_TARGET_MULT  = 2.5
MONITOR_INTERVAL = 60
SIGNAL_COOLDOWN  = 60 * 30
# ─────────────────────────────────────────────────────────────────────


class GateioLiveTrader:
    """Realtime crypto trading system for Gate.io."""

    def __init__(self):
        self.client          = GateioClient()
        self.stream          = GateioStream(symbols=CRYPTO_SYMBOLS, interval=CANDLE_INTERVAL)
        self.telegram        = TelegramBot()
        self.mtf             = MultiTimeframeAnalyzer()
        self.regime_filter   = MarketRegimeFilter()
        self.circuit_breaker = RiskCircuitBreaker()

        self.positions:        dict[str, dict] = {}
        self.history:          dict[str, list] = {}
        self.last_signal_time: dict[str, float] = {}

    # ── Startup ──────────────────────────────────────────────────────

    def start(self):
        env = 'TESTNET' if os.getenv('GATEIO_TESTNET', '').lower() == 'true' else 'LIVE'
        print('\n' + '🚀' * 25)
        print(f'ALPHAEDGE GATE.IO LIVE  —  {datetime.now().strftime("%Y-%m-%d %H:%M")}')
        print(f'Symbols: {CRYPTO_SYMBOLS}  |  Interval: {CANDLE_INTERVAL}  |  Mode: {env}')
        print('🚀' * 25)

        if not self.client.connected:
            print('\n⚠️  No API keys — running in SIGNAL-ONLY mode (no execution)\n')

        # Seed history
        print('\nLoading historical candles...')
        for symbol in CRYPTO_SYMBOLS:
            self._load_history(symbol)
            print(f'  {symbol}: {len(self.history.get(symbol, []))} bars')

        # Register candle callback
        self.stream.on_candle_close(self._on_candle_close)
        self.stream.start()

        print('\nWaiting for live prices...')
        if self.stream.wait_for_prices(timeout=30):
            prices_str = ' | '.join(
                f"{SymbolMap.to_base(s) or s}=${self.stream.get_price(s):.2f}"
                for s in CRYPTO_SYMBOLS if self.stream.get_price(s)
            )
            print(f'  {prices_str}')
        else:
            print('  Warning: no prices within 30s — check connection')

        # Position monitor
        threading.Thread(target=self._position_monitor_loop, daemon=True).start()
        print(f'  Position monitor started ({MONITOR_INTERVAL}s interval)')

        if self.client.connected:
            self.client.get_summary()

        print('\n✅ System live. Waiting for candle closes...\n')

        try:
            while True:
                time.sleep(60)
                self._print_status()
        except KeyboardInterrupt:
            print('\n\nShutting down...')
            self.stream.stop()

    # ── History loading ───────────────────────────────────────────────

    def _load_history(self, symbol: str):
        """Seed candle cache from Gate.io REST before stream starts."""
        klines = self.client.get_klines(symbol, interval=CANDLE_INTERVAL, limit=KLINE_LOOKBACK)
        if not klines:
            logger.warning(f'History load failed: {symbol}')
            self.history[symbol] = []
            return

        # Gate.io returns oldest first: [ts, vol, close, high, low, open]
        candles = []
        for k in klines:
            try:
                candles.append({
                    'symbol'   : symbol,
                    'timestamp': int(k[0]) * 1000,
                    'open'     : float(k[5]),
                    'high'     : float(k[3]),
                    'low'      : float(k[4]),
                    'close'    : float(k[2]),
                    'volume'   : float(k[1]),
                    'confirm'  : True,
                })
            except (IndexError, ValueError, TypeError):
                continue
        self.history[symbol] = candles

    # ── Candle close handler ──────────────────────────────────────────

    def _on_candle_close(self, symbol: str, candle: dict):
        self.history.setdefault(symbol, []).append(candle)
        if len(self.history[symbol]) > KLINE_LOOKBACK + 50:
            self.history[symbol] = self.history[symbol][-KLINE_LOOKBACK:]

        ts = datetime.fromtimestamp(candle['timestamp'] / 1000).strftime('%H:%M')
        logger.info(
            f"Candle: {symbol} {ts} | "
            f"O={candle['open']:.2f} H={candle['high']:.2f} "
            f"L={candle['low']:.2f} C={candle['close']:.2f} V={candle['volume']:.0f}"
        )

        if symbol in self.positions:
            return
        if time.time() - self.last_signal_time.get(symbol, 0) < SIGNAL_COOLDOWN:
            return

        self._evaluate_signal(symbol, candle)

    # ── Signal evaluation ─────────────────────────────────────────────

    def _evaluate_signal(self, symbol: str, latest_candle: dict):
        history = self.history.get(symbol, [])
        if len(history) < 50:
            return

        df    = self._history_to_df(history)
        price = float(latest_candle['close'])

        # 1. Market regime
        try:
            market_regime = self.regime_filter.analyze()
            if not market_regime.get('can_trade', True):
                logger.info(f'{symbol}: SKIP — market regime: {market_regime.get("reason")}')
                return
        except Exception as e:
            logger.warning(f'Regime filter error: {e}')

        # 2. Circuit breaker
        try:
            balance = self.client.get_usdt_balance() if self.client.connected else 10000.0
            if self.circuit_breaker.check(
                current_value    = balance,
                starting_capital = balance,
                telegram         = self.telegram,
            ):
                logger.info(f'{symbol}: SKIP — circuit breaker')
                return
        except Exception:
            pass

        # 3. Technical indicators
        indicators = self._calculate_indicators(df)
        if indicators is None:
            return

        regime = indicators['regime']
        score  = indicators['score']
        atr    = indicators['atr']

        # 4. Regime gate
        if regime == 'downtrend':
            logger.info(f'{symbol}: SKIP — local downtrend')
            return
        if regime == 'volatile' and score < 0.70:
            logger.info(f'{symbol}: SKIP — volatile + low score ({score:.2f})')
            return

        # 5. Score threshold
        if score < BUY_THRESHOLD:
            logger.info(f'{symbol}: SKIP — score {score:.2f} < {BUY_THRESHOLD}')
            return

        # 6. Volume confirmation
        vol_ok, vol_ratio = self._check_volume(df)
        if not vol_ok:
            logger.info(f'{symbol}: SKIP — volume {vol_ratio:.1f}x < {VOLUME_SPIKE_MIN}x')
            return

        # 7. MTF — pass live DataFrame directly
        try:
            mtf_score = self.mtf.get_mtf_score(symbol, daily_df=df)
            if mtf_score < 0.0:
                logger.info(f'{symbol}: SKIP — MTF bearish ({mtf_score:+.2f})')
                return
        except Exception as e:
            logger.warning(f'MTF error for {symbol}: {e}')
            mtf_score = 0.0

        # 8. Risk/reward
        stop_price   = price - (atr * ATR_STOP_MULT)
        target_price = price + (atr * ATR_TARGET_MULT)
        stop_dist    = price - stop_price
        rr_ratio     = (target_price - price) / stop_dist if stop_dist > 0 else 0

        if rr_ratio < MIN_RR_RATIO:
            logger.info(f'{symbol}: SKIP — R:R {rr_ratio:.1f} < {MIN_RR_RATIO}')
            return

        # ALL FILTERS PASSED
        display = SymbolMap.to_base(symbol) or symbol
        logger.info(
            f'✅ BUY {display} | score={score:.2f} | vol={vol_ratio:.1f}x'
            f' | mtf={mtf_score:+.2f} | R:R={rr_ratio:.1f}'
            f' | entry={price:.2f} stop={stop_price:.2f} target={target_price:.2f}'
        )

        self._execute_buy(symbol, price, stop_price, target_price, score, rr_ratio, vol_ratio)

    # ── Execution ─────────────────────────────────────────────────────

    def _execute_buy(
        self,
        symbol:       str,
        price:        float,
        stop_price:   float,
        target_price: float,
        score:        float,
        rr_ratio:     float,
        vol_ratio:    float,
    ):
        """Place spot buy order on Gate.io."""
        usdt_amount = 0.0
        base        = SymbolMap.to_base(symbol) or symbol.split('_')[0]
        display     = base

        if self.client.connected:
            usdt_amount = self.client.calculate_position_size(symbol, price, stop_price)
            if usdt_amount <= 0:
                logger.warning(f'{symbol}: size calc returned 0 — skipping')
                return

            result = self.client.place_order(
                currency_pair = symbol,
                side          = 'buy',
                amount        = usdt_amount,
                comment       = f'score={score:.2f}',
            )
            if result is None:
                logger.warning(f'{symbol}: order failed')
                return
        else:
            logger.info(f'{symbol}: SIGNAL-ONLY — would buy ${usdt_amount:.2f} USDT @ {price:.2f}')

        qty_base = usdt_amount / price if price > 0 else 0

        self.positions[symbol] = {
            'entry_price' : price,
            'qty'         : qty_base,
            'usdt_spent'  : usdt_amount,
            'base'        : base,
            'stop'        : stop_price,
            'target'      : target_price,
            'open_time'   : time.time(),
            'score'       : score,
        }
        self.last_signal_time[symbol] = time.time()

        try:
            self.telegram.alert_buy_signal(
                display, price, score,
                f'crypto | R:R={rr_ratio:.1f} | vol={vol_ratio:.1f}x', 0.0
            )
        except Exception:
            pass

        print(
            f'\n🟢 BUY {display} @ ${price:.2f}'
            f' | stop=${stop_price:.2f} | target=${target_price:.2f}'
            f' | score={score:.2f} | R:R={rr_ratio:.1f}\n'
        )

    # ── Position monitor ──────────────────────────────────────────────

    def _position_monitor_loop(self):
        while True:
            time.sleep(MONITOR_INTERVAL)
            self._check_positions()

    def _check_positions(self):
        for symbol in list(self.positions.keys()):
            pos   = self.positions[symbol]
            price = self.stream.get_price(symbol)
            if price is None:
                continue

            entry      = pos['entry_price']
            hit_stop   = price <= pos['stop']
            hit_target = price >= pos['target']
            pnl        = ((price - entry) / entry) * 100

            if not (hit_stop or hit_target):
                continue

            reason  = 'STOP LOSS' if hit_stop else 'TAKE PROFIT'
            emoji   = '🔴' if hit_stop else '✅'
            display = SymbolMap.to_base(symbol) or symbol

            logger.info(f'{emoji} {display}: {reason} @ ${price:.2f} | PnL={pnl:.1f}%')

            if self.client.connected:
                self.client.close_position(symbol, pos['base'])

            pnl_usd = (price - entry) * pos.get('qty', 0)
            try:
                if hit_stop:
                    self.telegram.alert_stop_loss(display, price, pnl_usd)
                else:
                    self.telegram.alert_take_profit(display, price, pnl_usd)
            except Exception:
                pass

            del self.positions[symbol]
            self.last_signal_time[symbol] = time.time()
            print(f'\n{emoji} {display}: {reason} @ ${price:.2f} | PnL={pnl:.1f}%\n')

    # ── Indicators ────────────────────────────────────────────────────

    def _calculate_indicators(self, df: pd.DataFrame) -> Optional[dict]:
        try:
            close  = df['close']
            high   = df['high']
            low    = df['low']
            volume = df['volume']

            if len(close) < 50:
                return None

            ema20  = close.ewm(span=20,  adjust=False).mean()
            ema50  = close.ewm(span=50,  adjust=False).mean()
            ema200 = close.ewm(span=200, adjust=False).mean() if len(close) >= 200 else None

            price          = float(close.iloc[-1])
            above_ema20    = price > float(ema20.iloc[-1])
            above_ema50    = price > float(ema50.iloc[-1])
            ema20_above_50 = float(ema20.iloc[-1]) > float(ema50.iloc[-1])
            ema200_ok      = price > float(ema200.iloc[-1]) if ema200 is not None else True

            delta  = close.diff()
            gain   = delta.clip(lower=0).rolling(14).mean()
            loss   = (-delta.clip(upper=0)).rolling(14).mean()
            rs     = gain / loss.replace(0, np.nan)
            rsi    = float(100 - (100 / (1 + rs.iloc[-1])))
            rsi_ok = 50 < rsi < 75

            tr1 = high - low
            tr2 = abs(high - close.shift(1))
            tr3 = abs(low  - close.shift(1))
            atr = float(pd.concat([tr1, tr2, tr3], axis=1).max(axis=1).rolling(14).mean().iloc[-1])

            macd    = close.ewm(span=12, adjust=False).mean() - close.ewm(span=26, adjust=False).mean()
            macd_ok = float(macd.iloc[-1]) > float(macd.ewm(span=9, adjust=False).mean().iloc[-1])

            vol_avg   = float(volume.rolling(20).mean().iloc[-1])
            vol_ratio = float(volume.iloc[-1]) / vol_avg if vol_avg > 0 else 1.0

            recent_highs = high.iloc[-10:]
            if recent_highs.is_monotonic_increasing:
                regime = 'uptrend'
            elif high.iloc[-10:].max() < high.iloc[-20:-10].max():
                regime = 'downtrend'
            else:
                regime = 'sideways'

            conditions = [above_ema20, above_ema50, ema20_above_50,
                          ema200_ok, rsi_ok, macd_ok, vol_ratio >= 1.0]
            score = sum(conditions) / len(conditions)

            return {
                'score'    : round(score, 3),
                'regime'   : regime,
                'atr'      : round(atr, 4),
                'rsi'      : round(rsi, 1),
                'vol_ratio': round(vol_ratio, 2),
            }
        except Exception as e:
            logger.warning(f'Indicator calc failed: {e}')
            return None

    def _check_volume(self, df: pd.DataFrame) -> tuple[bool, float]:
        try:
            current = float(df['volume'].iloc[-1])
            avg20   = float(df['volume'].iloc[-21:-1].mean())
            if avg20 <= 0:
                return False, 0.0
            ratio = current / avg20
            return ratio >= VOLUME_SPIKE_MIN, round(ratio, 2)
        except Exception:
            return False, 0.0

    def _history_to_df(self, history: list) -> pd.DataFrame:
        df = pd.DataFrame(history)
        return df.sort_values('timestamp').reset_index(drop=True)

    def _print_status(self):
        now = datetime.now().strftime('%H:%M:%S')
        prices_str = ' | '.join(
            f"{SymbolMap.to_base(s) or s}=${self.stream.get_price(s):.2f}"
            for s in CRYPTO_SYMBOLS if self.stream.get_price(s)
        )
        positions_str = ', '.join(
            SymbolMap.to_base(s) or s for s in self.positions
        ) or 'none'
        print(f'[{now}] {prices_str} | positions: {positions_str}')


if __name__ == '__main__':
    trader = GateioLiveTrader()
    trader.start()
