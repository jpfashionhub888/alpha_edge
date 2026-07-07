# risk/position_sizer.py
"""
AlphaEdge — Institutional-Grade Position Sizer

Replaces the fixed RISK_PER_TRADE_PCT constant with a dynamic,
multi-factor position sizing engine used by professional quant funds.

Sizing layers (applied in order, final = minimum of all):
  1. Fractional Kelly Criterion   — size proportional to measured edge
  2. Daily VaR Cap                — no position risks >2% of portfolio in 1 day
  3. Correlation Adjustment       — reduce size when correlated to open positions
  4. Regime Scaling               — multiply by market regime confidence
  5. Hard Portfolio Cap           — never exceed 15% portfolio in one position

Usage:
    from risk.position_sizer import PositionSizer

    sizer = PositionSizer(portfolio_value=10_000)
    dollar_amount = sizer.calculate(
        symbol          = 'AAPL',
        price           = 185.0,
        atr             = 3.2,
        signal_score    = 0.73,
        win_rate        = 0.58,    # from trade_tracker
        avg_win         = 0.045,   # avg winning trade return
        avg_loss        = 0.028,   # avg losing trade return (positive number)
        regime_conf     = 0.85,    # from MarketRegimeDetector.confidence
        open_positions  = {'MSFT': {...}, 'GOOGL': {...}},
    )
"""

from __future__ import annotations

import json
import logging
import os
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)

# ── Hard limits ───────────────────────────────────────────────────────────────
MAX_POSITION_PCT    = 0.15     # Never exceed 15% of portfolio in one position
MIN_POSITION_USD    = 100.0    # Minimum meaningful position size
KELLY_FRACTION      = 0.25     # Quarter-Kelly: institutional standard for live trading
VAR_DAILY_LIMIT     = 0.02     # 2% of portfolio max 1-day 95th-percentile loss
CORR_PENALTY_COEFF  = 0.60     # If correlation >= this, reduce size proportionally

# Fallback values when no trade history available yet
FALLBACK_WIN_RATE   = 0.55
FALLBACK_AVG_WIN    = 0.04     # 4% avg win
FALLBACK_AVG_LOSS   = 0.025    # 2.5% avg loss


class PositionSizer:
    """
    Multi-layer institutional position sizing.

    Parameters
    ----------
    portfolio_value : float
        Current total portfolio value in USD
    base_risk_pct : float
        Fallback fixed-% risk when Kelly cannot be computed
        (used only when trade_tracker has < 10 trades)
    """

    def __init__(self, portfolio_value: float, base_risk_pct: float = 0.015):
        self.portfolio_value = portfolio_value
        self.base_risk_pct   = base_risk_pct

    def calculate(
        self,
        symbol:         str,
        price:          float,
        atr:            float,
        signal_score:   float = 0.65,
        win_rate:       Optional[float] = None,
        avg_win:        Optional[float] = None,
        avg_loss:       Optional[float] = None,
        regime_conf:    float = 1.0,
        open_positions: dict  = None,
        n_trades:       int   = 0,
    ) -> float:
        """
        Calculate dollar amount to invest in this position.

        Returns
        -------
        float
            Dollar amount to invest (>= MIN_POSITION_USD, or 0 if too small)
        """
        if open_positions is None:
            open_positions = {}

        pv = self.portfolio_value
        if pv <= 0:
            logger.warning('Portfolio value is 0 — returning minimum position')
            return MIN_POSITION_USD

        # ── Layer 1: Fractional Kelly ─────────────────────────────────────
        kelly_size = self._kelly_size(
            pv, win_rate, avg_win, avg_loss, n_trades
        )

        # ── Layer 2: VaR Cap ──────────────────────────────────────────────
        var_size = self._var_size(pv, price, atr)

        # ── Layer 3: Correlation Adjustment ──────────────────────────────
        corr_mult = self._correlation_mult(symbol, open_positions)

        # ── Layer 4: Regime Scaling ───────────────────────────────────────
        regime_mult = max(0.3, min(1.0, regime_conf))

        # ── Layer 5: Signal Strength Scaling (bonus, not penalty) ─────────
        signal_mult = max(0.5, min(1.2, signal_score / 0.65))

        # ── Combine: take minimum of Kelly and VaR, then apply multipliers ─
        base_size    = min(kelly_size, var_size)
        final_size   = base_size * corr_mult * regime_mult * signal_mult

        # Hard portfolio cap
        cap          = pv * MAX_POSITION_PCT
        final_size   = min(final_size, cap)

        # Log breakdown
        logger.info(
            'PositionSizer [%s]: kelly=$%.0f var=$%.0f corr=%.2f '
            'regime=%.2f signal=%.2f → $%.0f',
            symbol, kelly_size, var_size, corr_mult,
            regime_mult, signal_mult, final_size
        )

        if final_size < MIN_POSITION_USD:
            logger.info(
                'PositionSizer [%s]: $%.0f below minimum — skipping', symbol, final_size
            )
            return 0.0

        return round(final_size, 2)

    # ── Private helpers ───────────────────────────────────────────────────────

    def _kelly_size(
        self,
        portfolio_value: float,
        win_rate:  Optional[float],
        avg_win:   Optional[float],
        avg_loss:  Optional[float],
        n_trades:  int,
    ) -> float:
        """
        Fractional Kelly position size.

        f* = (W * avg_win - L * avg_loss) / avg_win
        Kelly fraction = f* × KELLY_FRACTION (quarter-Kelly)

        Falls back to fixed base_risk_pct when < 10 trades are recorded
        (not enough data for reliable Kelly estimate).
        """
        if n_trades < 10 or win_rate is None or avg_win is None or avg_loss is None:
            # Not enough history — use conservative fixed risk
            logger.debug(
                'Kelly: insufficient history (%d trades) — using base_risk_pct=%.1f%%',
                n_trades, self.base_risk_pct * 100
            )
            return portfolio_value * self.base_risk_pct

        wr = max(0.01, min(0.99, win_rate))
        lr = 1.0 - wr
        w  = max(0.001, avg_win)
        l  = max(0.001, avg_loss)

        # Full Kelly fraction
        full_kelly = (wr * w - lr * l) / w

        if full_kelly <= 0:
            # Negative edge — size down to minimum
            logger.info('Kelly: negative edge (%.3f) — minimum sizing', full_kelly)
            return portfolio_value * 0.005   # 0.5% fallback

        # Quarter-Kelly for safety
        frac_kelly = full_kelly * KELLY_FRACTION
        frac_kelly = min(frac_kelly, MAX_POSITION_PCT)   # cap at hard limit
        size       = portfolio_value * frac_kelly

        logger.debug(
            'Kelly: wr=%.2f avgW=%.3f avgL=%.3f full_kelly=%.3f '
            'frac(%.0f%%)=%.3f → $%.0f',
            wr, w, l, full_kelly, KELLY_FRACTION * 100, frac_kelly, size
        )
        return size

    def _var_size(self, portfolio_value: float, price: float, atr: float) -> float:
        """
        Value-at-Risk cap: position size such that 1-day 95th-pct loss
        does not exceed VAR_DAILY_LIMIT × portfolio.

        We use ATR as a proxy for 1-day 95th-percentile move
        (ATR ≈ 1.25σ for roughly normally distributed returns).

        max_loss_usd = portfolio_value * VAR_DAILY_LIMIT
        atr_pct      = atr / price
        units        = max_loss_usd / (price * atr_pct)
        dollar_size  = units * price = max_loss_usd / atr_pct
        """
        if price <= 0 or atr <= 0:
            return portfolio_value * MAX_POSITION_PCT

        atr_pct    = atr / price
        max_loss   = portfolio_value * VAR_DAILY_LIMIT
        var_size   = max_loss / atr_pct

        logger.debug(
            'VaR: atr_pct=%.3f max_loss=$%.0f → $%.0f',
            atr_pct, max_loss, var_size
        )
        return var_size

    def _correlation_mult(self, symbol: str, open_positions: dict) -> float:
        """
        Reduce size when the new symbol is highly correlated to
        existing positions (same sector / similar ETF).

        Uses a simple lookup table of known high-correlation pairs.
        A proper implementation would use a rolling correlation matrix
        built from daily returns — add that once 30+ days of data exist.
        """
        if not open_positions:
            return 1.0

        # High-correlation peer groups (same sector/theme)
        PEER_GROUPS = {
            # Big Tech
            'AAPL':  {'MSFT', 'GOOGL', 'META', 'AMZN'},
            'MSFT':  {'AAPL', 'GOOGL', 'META', 'AMZN'},
            'GOOGL': {'AAPL', 'MSFT', 'META'},
            'META':  {'AAPL', 'MSFT', 'GOOGL'},
            'AMZN':  {'AAPL', 'MSFT'},
            # Semis
            'NVDA':  {'AMD', 'INTC', 'TSM', 'AVGO', 'QCOM'},
            'AMD':   {'NVDA', 'INTC', 'QCOM'},
            'INTC':  {'NVDA', 'AMD', 'QCOM'},
            'TSM':   {'NVDA', 'AMD'},
            # Financials
            'JPM':   {'GS', 'BAC', 'MS', 'WFC'},
            'GS':    {'JPM', 'MS', 'BAC'},
            'BAC':   {'JPM', 'GS', 'WFC'},
            # Energy
            'XOM':   {'CVX', 'COP', 'SLB'},
            'CVX':   {'XOM', 'COP'},
            # Healthcare
            'JNJ':   {'PFE', 'ABBV', 'MRK'},
            'PFE':   {'JNJ', 'ABBV', 'MRK'},
            # ETFs (instant full correlation)
            'SPY':   {'QQQ', 'IVV', 'VOO'},
            'QQQ':   {'SPY', 'IVV', 'TQQQ'},
        }

        peers = PEER_GROUPS.get(symbol, set())
        n_correlated = sum(1 for pos in open_positions if pos in peers)

        if n_correlated == 0:
            return 1.0
        elif n_correlated == 1:
            mult = 0.70   # 30% reduction for 1 correlated position
        elif n_correlated == 2:
            mult = 0.45   # 55% reduction for 2 correlated positions
        else:
            mult = 0.25   # 75% reduction for 3+ — very concentrated

        logger.info(
            'PositionSizer [%s]: %d correlated positions → size ×%.2f',
            symbol, n_correlated, mult
        )
        return mult


def get_trade_stats_for_sizing(trades_file: str = 'logs/closed_trades.json') -> dict:
    """
    Load win rate, avg win, avg loss, n_trades from TradeTracker output.
    Returns safe fallback values if file not found or too few trades.
    """
    try:
        with open(trades_file) as f:
            data = json.load(f)
        s = data.get('summary', {})
        n = s.get('total', 0)
        if n < 10:
            raise ValueError(f'Only {n} trades — insufficient for Kelly')
        wr       = s.get('win_rate', FALLBACK_WIN_RATE)
        avg_win  = abs(s.get('avg_win',  FALLBACK_AVG_WIN))
        avg_loss = abs(s.get('avg_loss', FALLBACK_AVG_LOSS))
        return {
            'n_trades' : n,
            'win_rate' : wr,
            'avg_win'  : avg_win,
            'avg_loss' : avg_loss,
        }
    except FileNotFoundError:
        logger.debug('No trade history file — using fallback sizing values')
    except Exception as e:
        logger.warning('Could not load trade stats for sizing: %s', e)

    return {
        'n_trades' : 0,
        'win_rate' : FALLBACK_WIN_RATE,
        'avg_win'  : FALLBACK_AVG_WIN,
        'avg_loss' : FALLBACK_AVG_LOSS,
    }
