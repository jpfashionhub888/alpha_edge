# gateio_live.py
"""
AlphaEdge — Gate.io Realtime Trading Loop (V2)
Paper trading mode wired in via PaperTrader.

Set GATEIO_PAPER_TRADE=true  → uses PaperTrader (default, safe)
Set GATEIO_PAPER_TRADE=false → uses live Gate.io execution

How to run:
  set GATEIO_API_KEY=your_key
  set GATEIO_API_SECRET=your_secret
  set GATEIO_PAPER_TRADE=true
  python gateio_live.py
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
from execution.paper_trader   import PaperTrader
from market_regime import MarketRegimeDetector as MarketRegimeDetector
from multi_timeframe          import MultiTimeframeAnalyzer
from monitoring.telegram_bot  import TelegramBot
from risk_circuit_breaker     import RiskCircuitBreaker

logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
logger = logging.getLogger(__name__)

# ── Config ───────────────────────────────────────────────────────────
CRYPTO_SYMBOLS   = ['BTC_USDT', 'ETH_USDT', 'SOL_USDT']
CANDLE_INTERVAL  = '15m'
KLINE_LOOKBACK   = 200
PAPER_TRADE      = os.getenv('GATEIO_PAPER_TRADE', 'true').lower() != 'false'
PAPER_CAPITAL    = float(os.getenv('GATEIO_PAPER_CAPITAL', '10000'))

BUY_THRESHOLD    = 0.63
VOLUME_SPIKE_MIN = 1.3
MIN_RR_RATIO     = 2.0
ATR_STOP_MULT    = 1.0
ATR_TARGET_MULT  = 2.5
MONITOR_INTERVAL = 60
SIGNAL_COOLDOWN  = 60 * 30
# ─────────────────────────────────────────────────────────────────────


class GateioLiveTrader:

    def __init__(self):
        self.client          = GateioClient()
        self.stream          = GateioStream(symbols=CRYPTO_SYMBOLS, interval=CANDLE_INTERVAL)
        self.telegram        = TelegramBot()
        self.mtf             = MultiTimeframeAnalyzer()
        self.circuit_breaker = RiskCircuitBreaker()

        # Regime detector — use correct class name from your repo
        try:
            self.regime_detector = MarketRegimeDetector()
        except Exception:
            self.regime_detector = None

        # Paper trader — always initialized, used when PAPER_TRADE=true
        self.paper = PaperTrader(
            starting_capital = PAPER_CAPITAL,
            log_file         = 'logs/gateio_paper_trades.json',
        )
        self.paper.load_state()

        self.history:          dict[str, list]  = {}
        self.last_signal_time: dict[str, float] = {}

        # Live positions (paper or real)
        # symbol → {entry_price, qty, stop, target, open_time, paper}
        self.positions: dict[str, dict] = {}

    # ── Startup ──────────────────────────────────────────────────────

    def start(self):
        mode = 'PAPER TRADING' if PAPER_TRADE else ('LIVE' if self.client.connected else 'SIGNAL-ONLY')
        print('\n' + '🚀' * 25)
        print(f'ALPHAEDGE GATE.IO  —  {datetime.now().strftime("%Y-%m-%d %H:%M")}')
        print(f'Symbols: {CRYPTO_SYMBOLS}  |  Interval: {CANDLE_INTERVAL}')
        print(f'Mode: {mode}')
        if PAPER_TRADE:
            print(f'Paper capital: ${PAPER_CAPITAL:,.0f}  |  Available: ${self.paper.capital:,.2f}')
        print('🚀' * 25)

        print('\nLoading historical candles...')
        for symbol in CRYPTO_SYMBOLS:
            self._load_history(symbol)
            print(f'  {symbol}: {len(self.history.get(symbol, []))} bars')

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
            print('  Warning: no prices within 30s')

        threading.Thread(target=self._monitor_loop, daemon=True).start()
        print(f'  Position monitor started ({MONITOR_INTERVAL}s)')

        if not PAPER_TRADE and self.client.connected:
            self.client.get_summary()

        print('\n✅ System live. Waiting for candle closes...\n')

        try:
            while True:
                time.sleep(60)
                self._print_status()
        except KeyboardInterrupt:
            print('\n\nShutting down...')
            self.paper.save_state()
            self.stream.stop()

    # ── History ───────────────────────────────────────────────────────

    def _load_history(self, symbol: str):
        klines = self.client.get_klines(symbol, interval=CANDLE_INTERVAL, limit=KLINE_LOOKBACK)
        if not klines:
            logger.warning(f'History load failed: {symbol}')
            self.history[symbol] = []
            return
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

    # ── Candle handler ────────────────────────────────────────────────

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

        df    = self._to_df(history)
        price = float(latest_candle['close'])

        # 1. Circuit breaker
        try:
            capital = self.paper.capital if PAPER_TRADE else (
                self.client.get_usdt_balance() if self.client.connected else 10000.0
            )
            if self.circuit_breaker.check(
                current_value    = capital,
                starting_capital = PAPER_CAPITAL if PAPER_TRADE else capital,
                telegram         = self.telegram,
            ):
                logger.info(f'{symbol}: SKIP — circuit breaker')
                return
        except Exception:
            pass

        # 2. Indicators
        ind = self._calculate_indicators(df)
        if ind is None:
            return

        regime = ind['regime']
        score  = ind['score']
        atr    = ind['atr']

        # 3. Regime gate
        if regime == 'downtrend':
            logger.info(f'{symbol}: SKIP — downtrend')
            return
        if regime == 'volatile' and score < 0.70:
            logger.info(f'{symbol}: SKIP — volatile + score {score:.2f}')
            return

        # 4. Score threshold
        if score < BUY_THRESHOLD:
            logger.info(f'{symbol}: SKIP — score {score:.2f} < {BUY_THRESHOLD}')
            return

        # 5. Volume confirmation
        vol_ok, vol_ratio = self._check_volume(df)
        if not vol_ok:
            logger.info(f'{symbol}: SKIP — volume {vol_ratio:.1f}x < {VOLUME_SPIKE_MIN}x')
            return

        # 6. MTF
        try:
            mtf_score = self.mtf.get_mtf_score(symbol, daily_df=df)
            if mtf_score < 0.0:
                logger.info(f'{symbol}: SKIP — MTF bearish ({mtf_score:+.2f})')
                return
        except Exception as e:
            logger.warning(f'MTF error: {e}')
            mtf_score = 0.0

        # 7. Risk/reward
        stop_price   = price - (atr * ATR_STOP_MULT)
        target_price = price + (atr * ATR_TARGET_MULT)
        stop_dist    = price - stop_price
        rr_ratio     = (target_price - price) / stop_dist if stop_dist > 0 else 0

        if rr_ratio < MIN_RR_RATIO:
            logger.info(f'{symbol}: SKIP — R:R {rr_ratio:.1f} < {MIN_RR_RATIO}')
            return

        # ALL PASSED
        display = SymbolMap.to_base(symbol) or symbol
        logger.info(
            f'✅ BUY {display} | score={score:.2f} | vol={vol_ratio:.1f}x'
            f' | mtf={mtf_score:+.2f} | R:R={rr_ratio:.1f}'
            f' | entry={price:.2f} stop={stop_price:.2f} target={target_price:.2f}'
        )

        self._execute_buy(symbol, price, stop_price, target_price, score, atr, rr_ratio, vol_ratio)

    # ── Execution ─────────────────────────────────────────────────────

    def _execute_buy(
        self,
        symbol:       str,
        price:        float,
        stop_price:   float,
        target_price: float,
        score:        float,
        atr:          float,
        rr_ratio:     float,
        vol_ratio:    float,
    ):
        display = SymbolMap.to_base(symbol) or symbol

        if PAPER_TRADE:
            # ── Paper trade ───────────────────────────────────────
            opened = self.paper.open_position(
                symbol  = display,
                price   = price,
                signal_strength = score,
                reason  = 'gateio_live',
                atr     = atr,
            )
            if not opened:
                logger.info(f'{symbol}: paper open_position returned False (max positions or daily loss limit)')
                return

            self.positions[symbol] = {
                'entry_price': price,
                'stop'       : stop_price,
                'target'     : target_price,
                'open_time'  : time.time(),
                'score'      : score,
                'paper'      : True,
                'display'    : display,
            }
            print(
                f'\n📝 PAPER BUY {display} @ ${price:.2f}'
                f' | stop=${stop_price:.2f} | target=${target_price:.2f}'
                f' | score={score:.2f} | R:R={rr_ratio:.1f}'
                f' | capital=${self.paper.capital:,.2f}\n'
            )

        elif self.client.connected:
            # ── Live trade ────────────────────────────────────────
            usdt_amount = self.client.calculate_position_size(symbol, price, stop_price)
            if usdt_amount <= 0:
                return
            result = self.client.place_order(
                currency_pair = symbol,
                side          = 'buy',
                amount        = usdt_amount,
                comment       = f'score={score:.2f}',
            )
            if result is None:
                return
            self.positions[symbol] = {
                'entry_price': price,
                'stop'       : stop_price,
                'target'     : target_price,
                'open_time'  : time.time(),
                'score'      : score,
                'paper'      : False,
                'display'    : display,
                'base'       : SymbolMap.to_base(symbol),
            }
            print(f'\n🟢 LIVE BUY {display} @ ${price:.2f} | ${usdt_amount:.2f} USDT\n')

        else:
            # ── Signal only ───────────────────────────────────────
            print(f'\n📡 SIGNAL {display} @ ${price:.2f} | score={score:.2f} | R:R={rr_ratio:.1f}\n')
            return

        self.last_signal_time[symbol] = time.time()
        self.paper.save_state()

        try:
            self.telegram.alert_buy_signal(
                display, price, score,
                f'{"PAPER" if PAPER_TRADE else "LIVE"} | R:R={rr_ratio:.1f} | vol={vol_ratio:.1f}x',
                0.0,
            )
        except Exception:
            pass

    # ── Position monitor ──────────────────────────────────────────────

    def _monitor_loop(self):
        while True:
            time.sleep(MONITOR_INTERVAL)
            self._check_positions()
            if PAPER_TRADE:
                self._update_paper_positions()

    def _update_paper_positions(self):
        """Push live prices into PaperTrader so trailing stops work."""
        current_prices = {}
        for symbol, pos in list(self.positions.items()):
            price = self.stream.get_price(symbol)
            if price:
                current_prices[pos['display']] = price

        for display_sym, price in current_prices.items():
            self.paper.update_position(display_sym, price)
            # Check if PaperTrader closed it
            if display_sym not in self.paper.positions:
                # Find and remove from our positions dict
                for sym, pos in list(self.positions.items()):
                    if pos.get('display') == display_sym:
                        pnl = (price - pos['entry_price']) / pos['entry_price'] * 100
                        emoji = '✅' if price >= pos['entry_price'] else '🔴'
                        print(f'\n{emoji} PAPER CLOSE {display_sym} @ ${price:.2f} | PnL={pnl:.1f}% | capital=${self.paper.capital:,.2f}\n')
                        del self.positions[sym]
                        self.last_signal_time[sym] = time.time()
                        self.paper.save_state()

    def _check_positions(self):
        """Check manual stop/target for live positions."""
        for symbol in list(self.positions.keys()):
            pos   = self.positions[symbol]
            if pos.get('paper'):
                continue   # handled by _update_paper_positions

            price = self.stream.get_price(symbol)
            if price is None:
                continue

            hit_stop   = price <= pos['stop']
            hit_target = price >= pos['target']
            if not (hit_stop or hit_target):
                continue

            reason  = 'STOP LOSS' if hit_stop else 'TAKE PROFIT'
            emoji   = '🔴' if hit_stop else '✅'
            display = pos.get('display', symbol)
            pnl     = ((price - pos['entry_price']) / pos['entry_price']) * 100

            if self.client.connected:
                self.client.close_position(symbol, pos.get('base', ''))

            print(f'\n{emoji} {display}: {reason} @ ${price:.2f} | PnL={pnl:.1f}%\n')
            del self.positions[symbol]
            self.last_signal_time[symbol] = time.time()

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
            return {
                'score'    : round(sum(conditions) / len(conditions), 3),
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

    def _to_df(self, history: list) -> pd.DataFrame:
        return pd.DataFrame(history).sort_values('timestamp').reset_index(drop=True)

    def _print_status(self):
        now = datetime.now().strftime('%H:%M:%S')
        prices_str = ' | '.join(
            f"{SymbolMap.to_base(s) or s}=${self.stream.get_price(s):.2f}"
            for s in CRYPTO_SYMBOLS if self.stream.get_price(s)
        )
        pos_str = ', '.join(pos.get('display', s) for s, pos in self.positions.items()) or 'none'
        capital = f' | paper=${self.paper.capital:,.2f}' if PAPER_TRADE else ''
        print(f'[{now}] {prices_str} | positions: {pos_str}{capital}')


if __name__ == '__main__':
    trader = GateioLiveTrader()
    trader.start()
