# backtest/event_engine.py
"""
AlphaEdge — Event-Driven Backtesting Engine
Inspired by: QuantConnect LEAN Engine
             https://github.com/QuantConnect/Lean

Why event-driven vs vectorised:
    Vectorised backtests (pandas apply, pct_change) are fast but leak.
    Applying a function to a column at index i can still see row i+1 during
    the pandas operation. Event-driven is the ONLY way to guarantee
    zero look-ahead.

Architecture (LEAN-inspired):
    DataFeed  → feeds one bar at a time in chronological order
    EventQueue → each bar fires a MarketDataEvent
    Strategy  → receives event, generates OrderEvent if signal triggered
    ExecutionHandler → simulates fill with RealisticFillModel
    Portfolio → tracks cash, positions, equity curve

Usage:
    from backtest.event_engine import EventDrivenBacktest
    from backtest.event_engine import SimpleAlphaEdgeStrategy

    engine = EventDrivenBacktest(
        strategy=SimpleAlphaEdgeStrategy(buy_threshold=0.63),
        initial_capital=10000,
        commission=0.005,
        slippage_pct=0.0005,
    )
    result = engine.run(
        data={'AAPL': aapl_df, 'NVDA': nvda_df},
        start='2023-01-01',
        end='2024-12-31',
    )
    print(result.summary())
    result.plot_equity_curve()
"""

import logging
import math
from collections import defaultdict, deque
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


# ── Event Types ───────────────────────────────────────────────────────────────

class EventType(Enum):
    MARKET  = 'MARKET'
    SIGNAL  = 'SIGNAL'
    ORDER   = 'ORDER'
    FILL    = 'FILL'


@dataclass
class MarketEvent:
    """Fired once per bar per symbol."""
    type:   EventType = EventType.MARKET
    symbol: str       = ''
    date:   datetime  = None
    open:   float     = 0.0
    high:   float     = 0.0
    low:    float     = 0.0
    close:  float     = 0.0
    volume: float     = 0.0


@dataclass
class SignalEvent:
    """Strategy decision."""
    type:      EventType = EventType.SIGNAL
    symbol:    str       = ''
    date:      datetime  = None
    direction: str       = 'LONG'   # 'LONG' or 'EXIT'
    strength:  float     = 0.5      # 0-1 confidence
    stop:      float     = 0.0
    target:    float     = 0.0


@dataclass
class OrderEvent:
    """Order submitted to execution handler."""
    type:      EventType = EventType.ORDER
    symbol:    str       = ''
    date:      datetime  = None
    order_type: str      = 'MKT'    # 'MKT' or 'LMT'
    direction: str       = 'BUY'    # 'BUY' or 'SELL'
    quantity:  float     = 0.0
    price:     float     = 0.0


@dataclass
class FillEvent:
    """Confirmed fill with realistic costs applied."""
    type:       EventType = EventType.FILL
    symbol:     str       = ''
    date:       datetime  = None
    direction:  str       = 'BUY'
    quantity:   float     = 0.0
    fill_price: float     = 0.0
    commission: float     = 0.0
    slippage:   float     = 0.0


# ── Fill Model ────────────────────────────────────────────────────────────────

class RealisticFillModel:
    """
    Simulates realistic order execution with:
    - Commission: $0.005/share (Interactive Brokers rate)
    - Slippage:   0.05% market impact on entry/exit
    - Partial fills: not modelled (assumes sufficient liquidity for paper sizes)

    These costs typically reduce backtest returns by 0.1-0.2% per round trip.
    """

    def __init__(
        self,
        commission_per_share: float = 0.005,
        slippage_pct:         float = 0.0005,
        min_commission:       float = 1.0,
    ):
        self.commission_per_share = commission_per_share
        self.slippage_pct         = slippage_pct
        self.min_commission       = min_commission

    def fill(self, order: OrderEvent, bar: MarketEvent) -> FillEvent:
        """
        Simulate fill for a market order on the NEXT bar's open.
        This is the correct LEAN approach: signal at close → fill at next open.
        """
        # Fill at next open + slippage
        direction_mult = 1 if order.direction == 'BUY' else -1
        fill_price = bar.open * (1 + direction_mult * self.slippage_pct)

        commission = max(
            order.quantity * self.commission_per_share,
            self.min_commission,
        )
        slippage_cost = abs(fill_price - bar.open) * order.quantity

        return FillEvent(
            symbol     = order.symbol,
            date       = bar.date,
            direction  = order.direction,
            quantity   = order.quantity,
            fill_price = fill_price,
            commission = commission,
            slippage   = slippage_cost,
        )


# ── Portfolio ─────────────────────────────────────────────────────────────────

@dataclass
class Position:
    symbol:       str
    shares:       float
    entry_price:  float
    entry_date:   datetime
    stop:         float = 0.0
    target:       float = 0.0
    bars_open:    int   = 0


class Portfolio:
    """
    Tracks cash, positions, equity curve.
    No look-ahead — updated only via FillEvents.
    """

    def __init__(self, initial_capital: float = 10_000.0):
        self.initial_capital = initial_capital
        self.cash            = initial_capital
        self.positions: Dict[str, Position] = {}
        self.equity_curve: List[Tuple[datetime, float]] = []
        self.trades: List[Dict] = []

    @property
    def total_value(self) -> float:
        return self.cash + sum(
            pos.shares * pos.entry_price for pos in self.positions.values()
        )

    def update_market(self, event: MarketEvent) -> None:
        """Update current price of any open position (for equity curve)."""
        if event.symbol in self.positions:
            pos = self.positions[event.symbol]
            pos.bars_open += 1
            # Equity snapshot with mark-to-market
            mtm_equity = self.cash + sum(
                (event.close if sym == event.symbol else p.entry_price) * p.shares
                for sym, p in self.positions.items()
            )
            self.equity_curve.append((event.date, mtm_equity))

    def process_fill(self, fill: FillEvent) -> None:
        """Apply fill to portfolio."""
        cost = fill.fill_price * fill.quantity + fill.commission + fill.slippage

        if fill.direction == 'BUY':
            if cost > self.cash:
                logger.debug(f'Portfolio: insufficient cash for {fill.symbol} — skipping')
                return
            self.cash -= cost
            self.positions[fill.symbol] = Position(
                symbol      = fill.symbol,
                shares      = fill.quantity,
                entry_price = fill.fill_price,
                entry_date  = fill.date,
            )

        elif fill.direction == 'SELL' and fill.symbol in self.positions:
            pos      = self.positions.pop(fill.symbol)
            proceeds = fill.fill_price * fill.quantity - fill.commission - fill.slippage
            self.cash += proceeds
            pnl      = proceeds - (pos.entry_price * pos.shares + fill.commission + fill.slippage)
            pnl_pct  = pnl / (pos.entry_price * pos.shares)
            self.trades.append({
                'symbol'     : fill.symbol,
                'entry_date' : pos.entry_date.isoformat(),
                'exit_date'  : fill.date.isoformat(),
                'entry_price': round(pos.entry_price, 4),
                'exit_price' : round(fill.fill_price, 4),
                'shares'     : pos.shares,
                'pnl'        : round(pnl, 4),
                'pnl_pct'   : round(pnl_pct, 4),
                'bars_open'  : pos.bars_open,
                'commission' : round(fill.commission, 4),
                'slippage'   : round(fill.slippage, 4),
            })


# ── Strategy Interface ────────────────────────────────────────────────────────

class BaseStrategy:
    """
    Base class for event-driven strategies.
    Subclass and implement on_bar().
    """

    def __init__(self):
        self._bar_history: Dict[str, deque] = defaultdict(lambda: deque(maxlen=300))

    def on_bar(self, event: MarketEvent, portfolio: Portfolio) -> Optional[SignalEvent]:
        """
        Called on each new bar. Return a SignalEvent to trigger a trade,
        or None to do nothing.
        Subclasses MUST only use self._bar_history[symbol] — no future data.
        """
        raise NotImplementedError

    def _record_bar(self, event: MarketEvent) -> None:
        self._bar_history[event.symbol].append({
            'date': event.date, 'open': event.open,
            'high': event.high, 'low': event.low,
            'close': event.close, 'volume': event.volume,
        })

    def _history_df(self, symbol: str) -> pd.DataFrame:
        """Get bar history as DataFrame (strictly past data only)."""
        bars = list(self._bar_history[symbol])
        if not bars:
            return pd.DataFrame()
        return pd.DataFrame(bars).set_index('date')


class SimpleAlphaEdgeStrategy(BaseStrategy):
    """
    Simplified AlphaEdge strategy for backtesting parameter validation.
    Mirrors the logic in scanner.py using only event-driven bar history.

    Parameters
    ----------
    buy_threshold   : Momentum score threshold (default 0.63)
    volume_spike    : Minimum volume ratio vs 20d MA (default 1.3)
    atr_stop_mult   : ATR multiplier for stop loss (default 1.2)
    atr_target_mult : ATR multiplier for take profit (default 2.5)
    max_pos_pct     : Max position size as fraction of portfolio (default 0.10)
    """

    def __init__(
        self,
        buy_threshold:   float = 0.63,
        volume_spike:    float = 1.3,
        atr_stop_mult:   float = 1.2,
        atr_target_mult: float = 2.5,
        max_pos_pct:     float = 0.10,
    ):
        super().__init__()
        self.buy_threshold   = buy_threshold
        self.volume_spike    = volume_spike
        self.atr_stop_mult   = atr_stop_mult
        self.atr_target_mult = atr_target_mult
        self.max_pos_pct     = max_pos_pct

    def on_bar(self, event: MarketEvent, portfolio: Portfolio) -> Optional[SignalEvent]:
        self._record_bar(event)
        hist = self._history_df(event.symbol)

        if len(hist) < 60:
            return None   # not enough history

        close  = hist['close']
        volume = hist['volume']
        high   = hist['high']
        low    = hist['low']

        # Exit: check stop/target on open positions
        if event.symbol in portfolio.positions:
            pos = portfolio.positions[event.symbol]
            if event.close <= pos.stop or event.close >= pos.target:
                return SignalEvent(
                    symbol    = event.symbol,
                    date      = event.date,
                    direction = 'EXIT',
                    strength  = 1.0,
                    stop      = pos.stop,
                    target    = pos.target,
                )
            return None   # hold

        # Entry: compute momentum score
        ma5  = close.rolling(5).mean().iloc[-1]
        ma20 = close.rolling(20).mean().iloc[-1]
        if pd.isna(ma5) or pd.isna(ma20) or ma20 == 0:
            return None

        mom   = (ma5 - ma20) / ma20
        mom60 = close.pct_change(1).rolling(60)
        score = (mom - mom60.min().iloc[-1]) / (mom60.max().iloc[-1] - mom60.min().iloc[-1] + 1e-9)

        # Volume spike check
        vol_ma  = volume.rolling(20).mean().iloc[-1]
        vol_now = volume.iloc[-1]
        vol_ratio = vol_now / (vol_ma + 1e-9)

        if score < self.buy_threshold or vol_ratio < self.volume_spike:
            return None

        # ATR-based stop/target
        atr = (high - low).rolling(14).mean().iloc[-1]
        if pd.isna(atr):
            return None

        stop   = event.close - self.atr_stop_mult   * atr
        target = event.close + self.atr_target_mult * atr

        return SignalEvent(
            symbol    = event.symbol,
            date      = event.date,
            direction = 'LONG',
            strength  = float(score),
            stop      = round(stop, 4),
            target    = round(target, 4),
        )


# ── Event Engine ──────────────────────────────────────────────────────────────

class BacktestResult:
    """Holds results of a completed backtest."""

    def __init__(self, portfolio: Portfolio, params: Dict):
        self.portfolio   = portfolio
        self.params      = params
        self.trades      = pd.DataFrame(portfolio.trades)
        self.equity_df   = (
            pd.DataFrame(portfolio.equity_curve, columns=['date', 'equity'])
            .set_index('date')
        ) if portfolio.equity_curve else pd.DataFrame()

    def summary(self) -> Dict:
        if self.trades.empty:
            return {'error': 'No trades executed'}

        wins   = self.trades[self.trades['pnl'] > 0]
        losses = self.trades[self.trades['pnl'] <= 0]
        rets   = self.trades['pnl_pct'].values

        sharpe = (rets.mean() / (rets.std(ddof=1) + 1e-9)) * math.sqrt(252) if len(rets) > 1 else 0

        total_commission = self.trades['commission'].sum()
        total_slippage   = self.trades['slippage'].sum()
        final_equity     = self.portfolio.total_value
        total_return     = (final_equity - self.portfolio.initial_capital) / self.portfolio.initial_capital

        return {
            'initial_capital'  : self.portfolio.initial_capital,
            'final_equity'     : round(final_equity, 2),
            'total_return_pct' : round(total_return * 100, 2),
            'total_trades'     : len(self.trades),
            'win_rate'         : round(len(wins) / len(self.trades) * 100, 1),
            'avg_win'          : round(wins['pnl'].mean(), 2) if not wins.empty else 0,
            'avg_loss'         : round(losses['pnl'].mean(), 2) if not losses.empty else 0,
            'profit_factor'    : round(wins['pnl'].sum() / (abs(losses['pnl'].sum()) + 1e-9), 2),
            'sharpe_ratio'     : round(sharpe, 3),
            'total_commission' : round(total_commission, 2),
            'total_slippage'   : round(total_slippage, 2),
            'avg_bars_held'    : round(self.trades['bars_open'].mean(), 1) if 'bars_open' in self.trades else 0,
            'params'           : self.params,
        }

    def plot_equity_curve(self) -> None:
        """Plot equity curve if matplotlib available."""
        if self.equity_df.empty:
            print('No equity data to plot')
            return
        try:
            import matplotlib.pyplot as plt
            fig, ax = plt.subplots(figsize=(12, 4))
            ax.plot(self.equity_df.index, self.equity_df['equity'], color='#10b981', linewidth=1.5)
            ax.axhline(self.portfolio.initial_capital, color='gray', linestyle='--', alpha=0.5)
            ax.set_title('AlphaEdge — Event-Driven Backtest Equity Curve', fontsize=13)
            ax.set_ylabel('Portfolio Value ($)')
            ax.grid(alpha=0.2)
            plt.tight_layout()
            plt.show()
        except ImportError:
            print('matplotlib not installed — cannot plot')


class EventDrivenBacktest:
    """
    LEAN-inspired event-driven backtesting engine.

    Key guarantee: on_bar() is called STRICTLY in chronological order.
    The strategy NEVER sees any bar it hasn't received as an event.
    This makes look-ahead bias structurally impossible.

    Parameters
    ----------
    strategy        : BaseStrategy subclass instance
    initial_capital : Starting portfolio value
    commission      : Commission per share in $
    slippage_pct    : Market impact as fraction (0.0005 = 0.05%)
    max_pos_pct     : Max position size as fraction of portfolio
    """

    def __init__(
        self,
        strategy:        BaseStrategy,
        initial_capital: float = 10_000.0,
        commission:      float = 0.005,
        slippage_pct:    float = 0.0005,
        max_pos_pct:     float = 0.10,
    ):
        self.strategy        = strategy
        self.initial_capital = initial_capital
        self.fill_model      = RealisticFillModel(commission, slippage_pct)
        self.max_pos_pct     = max_pos_pct

    def run(
        self,
        data:  Dict[str, pd.DataFrame],
        start: Optional[str] = None,
        end:   Optional[str] = None,
    ) -> BacktestResult:
        """
        Run event-driven backtest.

        Parameters
        ----------
        data  : {symbol: OHLCV_DataFrame} — must have open/high/low/close/volume
        start : Optional start date filter
        end   : Optional end date filter
        """
        portfolio = Portfolio(self.initial_capital)
        event_q   = deque()

        # Merge all symbols into chronological event stream
        all_events = self._build_event_stream(data, start, end)
        total      = len(all_events)
        logger.info(f'EventDrivenBacktest: {total} market events | {len(data)} symbols')

        # ── Main event loop ───────────────────────────────────────────────
        pending_orders: Dict[str, Tuple[OrderEvent, MarketEvent]] = {}

        for i, market_event in enumerate(all_events):
            sym = market_event.symbol

            # 1. Fill any pending order on this bar's open
            if sym in pending_orders:
                order, _ = pending_orders.pop(sym)
                fill = self.fill_model.fill(order, market_event)
                portfolio.process_fill(fill)

            # 2. Update portfolio MTM
            portfolio.update_market(market_event)

            # 3. Strategy generates signal
            signal = self.strategy.on_bar(market_event, portfolio)
            if signal is None:
                continue

            # 4. Convert signal to order
            order = self._signal_to_order(signal, market_event, portfolio)
            if order is None:
                continue

            # 5. Queue order for next bar (signal at close → fill at next open)
            pending_orders[sym] = (order, market_event)

        # Close any open positions at final bar price
        self._close_all_positions(portfolio, all_events)

        params = {
            'initial_capital': self.initial_capital,
            'commission'     : self.fill_model.commission_per_share,
            'slippage_pct'   : self.fill_model.slippage_pct,
        }
        if hasattr(self.strategy, '__dict__'):
            params.update({
                k: v for k, v in self.strategy.__dict__.items()
                if not k.startswith('_') and isinstance(v, (int, float, str))
            })

        result = BacktestResult(portfolio, params)
        summary = result.summary()
        logger.info(
            f'Backtest complete: return={summary.get("total_return_pct")}% | '
            f'trades={summary.get("total_trades")} | '
            f'win_rate={summary.get("win_rate")}% | '
            f'Sharpe={summary.get("sharpe_ratio")} | '
            f'commission+slippage=${summary.get("total_commission", 0)+summary.get("total_slippage", 0):.2f}'
        )
        return result

    def _build_event_stream(
        self,
        data:  Dict[str, pd.DataFrame],
        start: Optional[str],
        end:   Optional[str],
    ) -> List[MarketEvent]:
        """Build sorted list of MarketEvents from all symbols."""
        events = []
        for sym, df in data.items():
            df = df.copy()
            df.columns = [c.lower() for c in df.columns]
            if hasattr(df.index, 'tz') and df.index.tz is not None:
                df.index = df.index.tz_localize(None)
            if start:
                df = df[df.index >= pd.Timestamp(start)]
            if end:
                df = df[df.index <= pd.Timestamp(end)]

            for date, row in df.iterrows():
                events.append(MarketEvent(
                    symbol = sym,
                    date   = date,
                    open   = float(row.get('open', row.get('Open', 0))),
                    high   = float(row.get('high', row.get('High', 0))),
                    low    = float(row.get('low',  row.get('Low',  0))),
                    close  = float(row.get('close', row.get('Close', 0))),
                    volume = float(row.get('volume', row.get('Volume', 0))),
                ))

        # Sort strictly by date (ties: alphabetical by symbol for determinism)
        events.sort(key=lambda e: (e.date, e.symbol))
        return events

    def _signal_to_order(
        self,
        signal:       SignalEvent,
        bar:          MarketEvent,
        portfolio:    Portfolio,
    ) -> Optional[OrderEvent]:
        """Convert signal to sized order."""
        if signal.direction == 'EXIT':
            pos = portfolio.positions.get(signal.symbol)
            if pos is None:
                return None
            return OrderEvent(
                symbol     = signal.symbol,
                date       = signal.date,
                direction  = 'SELL',
                quantity   = pos.shares,
                price      = bar.close,
            )

        elif signal.direction == 'LONG':
            if signal.symbol in portfolio.positions:
                return None   # already long
            if portfolio.cash < 100:
                return None   # no cash

            # Position sizing: max_pos_pct of total portfolio
            alloc   = portfolio.total_value * self.max_pos_pct * signal.strength
            alloc   = min(alloc, portfolio.cash * 0.95)
            shares  = math.floor(alloc / (bar.close + 1e-9))
            if shares < 1:
                return None

            order = OrderEvent(
                symbol    = signal.symbol,
                date      = signal.date,
                direction = 'BUY',
                quantity  = shares,
                price     = bar.close,
            )

            # Attach stop/target to portfolio position after fill
            # (stored on SignalEvent for later retrieval)
            order._stop   = signal.stop
            order._target = signal.target
            return order

        return None

    def _close_all_positions(
        self,
        portfolio: Portfolio,
        events:    List[MarketEvent],
    ) -> None:
        """Close remaining positions at last available price."""
        if not portfolio.positions or not events:
            return
        last_prices = {e.symbol: e.close for e in events}
        for sym, pos in list(portfolio.positions.items()):
            price = last_prices.get(sym, pos.entry_price)
            proceeds = price * pos.shares
            portfolio.cash += proceeds
            portfolio.trades.append({
                'symbol'     : sym,
                'entry_date' : pos.entry_date.isoformat(),
                'exit_date'  : events[-1].date.isoformat(),
                'entry_price': round(pos.entry_price, 4),
                'exit_price' : round(price, 4),
                'shares'     : pos.shares,
                'pnl'        : round(proceeds - pos.entry_price * pos.shares, 4),
                'pnl_pct'   : round((price - pos.entry_price) / pos.entry_price, 4),
                'bars_open'  : pos.bars_open,
                'commission' : 0.0,
                'slippage'   : 0.0,
            })
        portfolio.positions.clear()
