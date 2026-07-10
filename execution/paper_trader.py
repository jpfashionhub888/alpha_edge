# execution/paper_trader.py

import pandas as pd
import numpy as np
from datetime import datetime
import json
import os
import tempfile
import logging
from config import settings
from risk.position_sizer import PositionSizer, get_trade_stats_for_sizing

logger = logging.getLogger(__name__)

# ML Score Exit threshold — exit a held position when confidence drops below this
ML_EXIT_THRESHOLD = getattr(settings, 'ML_EXIT_THRESHOLD', 0.40)

# Fix 2.2 — Regulatory fee constants (applied on sell side only)
# SEC fee rate: $0.0000278 per dollar of proceeds (current 2025 rate)
SEC_FEE_RATE         = getattr(settings, 'SEC_FEE_RATE',         0.0000278)
# FINRA Trading Activity Fee: $0.000166 per share sold (current 2025 rate)
FINRA_TAF_PER_SHARE  = getattr(settings, 'FINRA_TAF_PER_SHARE',  0.000166)

# Fix 3.4 — Max portfolio exposure (never more than 80% of capital deployed)
MAX_PORTFOLIO_EXPOSURE_PCT = getattr(settings, 'MAX_PORTFOLIO_EXPOSURE_PCT', 0.80)


class PaperTrader:
    """
    Simulated trading engine. Tracks all trades,
    positions and P&L as if trading real money.
    Saves state to disk so you can stop and resume.
    Includes slippage and commission for realistic results.

    Fixes applied (v2):
    - ATR-based volatility position sizing (Bug 1 & 2)
    - Partial exit cost basis corrected (Bug 3)
    - Atomic state save prevents corruption (Bug 4)
    - load_state() has full error handling (Bug 5)
    - Trailing stop only fires after min profit (Bug 6)
    - update_position() check order corrected (Flaw 1)
    - Signal strength uses discrete tier system (Flaw 2)
    - Partial exit now deducts commission+slippage (Flaw 3)
    - Missing prices use entry as fallback (Flaw 4)
    - Per-trade max loss cap added (Risk 1)
    - Daily loss limit at trader level (Risk 2)
    - Auto-save after every open and close (Risk 3)

    Additions (v3):
    - ML Score Exit: closes position when live ML confidence < ML_EXIT_THRESHOLD
    """

    def __init__(self,
                 starting_capital   = 10000.0,
                 max_position_pct   = None,
                 max_positions      = None,
                 slippage_pct       = None,
                 commission         = None,
                 risk_per_trade_pct = None,
                 daily_loss_limit_pct = None,
                 log_file           = 'logs/paper_trades_stocks_only.json'):

        self.starting_capital     = starting_capital
        self.capital              = starting_capital
        self.max_position_pct     = max_position_pct if max_position_pct is not None else settings.MAX_POSITION_SIZE
        self.max_positions        = max_positions if max_positions is not None else settings.MAX_OPEN_POSITIONS
        self.slippage_pct         = slippage_pct if slippage_pct is not None else settings.SLIPPAGE_PCT
        self.commission           = commission if commission is not None else settings.COMMISSION
        self.risk_per_trade_pct   = risk_per_trade_pct if risk_per_trade_pct is not None else settings.MAX_RISK_PER_TRADE
        self.log_file             = log_file

        # Kelly Sizing Properties
        self.kelly_position_sizing = settings.KELLY_POSITION_SIZING
        self.kelly_multiplier      = settings.KELLY_MULTIPLIER
        self.kelly_reward_risk_ratio = settings.KELLY_REWARD_RISK_RATIO

        self.positions            = {}
        self.trade_history        = []
        self.daily_pnl            = []

        # --- Daily loss limit tracking (Risk 2) ---
        limit_pct = daily_loss_limit_pct if daily_loss_limit_pct is not None else settings.MAX_DAILY_LOSS
        self.daily_loss_limit     = starting_capital * limit_pct
        self.daily_realized_pnl   = 0.0
        self.last_reset_date      = datetime.now().date()
        self._halt_trading        = False
        # P2-4: time stop threshold from settings (configurable without code change)
        self.time_stop_days       = getattr(settings, 'TIME_STOP_DAYS', 5)

    # ------------------------------------------------------------------ #
    #  INTERNAL HELPERS                                                    #
    # ------------------------------------------------------------------ #

    def _check_daily_reset(self):
        """
        Reset daily P&L tracker at the start of each new trading day.
        Also lifts the halt flag so trading resumes next day.
        """
        today = datetime.now().date()
        if today != self.last_reset_date:
            logger.info(
                f"New trading day. Resetting daily P&L "
                f"(was {self.daily_realized_pnl:+.2f})"
            )
            self.daily_realized_pnl = 0.0
            self._halt_trading      = False
            self.last_reset_date    = today

    def _is_trading_halted(self):
        """
        Returns True if the daily loss limit has been hit.
        Always checks for a new day first so halt auto-lifts next morning.
        """
        self._check_daily_reset()
        if self._halt_trading:
            logger.warning("Trading halted: daily loss limit reached.")
            return True
        return False

    @staticmethod
    def _signal_to_size_multiplier(signal_strength):
        """
        Convert a raw signal score to a discrete position-size tier.
        Replaces the old linear scaling which had no documented rationale.

        Tier table:
            >= 0.80  →  1.00  (full position, high conviction)
            >= 0.70  →  0.75  (three-quarter position)
            >= 0.60  →  0.50  (half position)
            <  0.60  →  0.25  (quarter position, minimum conviction)
        """
        if signal_strength >= 0.80:
            return 1.00
        elif signal_strength >= 0.70:
            return 0.75
        elif signal_strength >= 0.60:
            return 0.50
        else:
            return 0.25

    # ------------------------------------------------------------------ #
    #  PORTFOLIO EXPOSURE                                                  #
    # ------------------------------------------------------------------ #

    def _get_open_positions_value(self) -> float:
        """Total market value of all open positions at entry price (conservative)."""
        return sum(
            pos.get('shares', 0) * pos.get('entry_price', 0)
            for pos in self.positions.values()
        )

    def _get_current_exposure_pct(self) -> float:
        """Fix 3.4: Fraction of starting capital currently deployed in open positions."""
        return self._get_open_positions_value() / self.starting_capital if self.starting_capital else 0.0

    # ------------------------------------------------------------------ #
    #  POSITION SIZING                                                     #
    # ------------------------------------------------------------------ #

    def get_position_size(self, price, signal_strength=1.0, atr=None, symbol=None, regime_conf=1.0):
        """
        Position sizing using optional Kelly Criterion.

        KELLY PATH:
            Calculate Kelly fraction using calibrated prediction score (p)
            and reward_risk_ratio (b): f* = (p * (b + 1) - 1) / b.
            Apply Half-Kelly fractional multiplier.
            Capped at max_position_pct.

        STANDARD ATR PATH (ATR available):
            Risk exactly risk_per_trade_pct of capital per trade.
            Stop distance = 2 * ATR.
            shares = dollar_risk / stop_distance
            Hard-capped at max_position_pct of capital.

        FALLBACK PATH (no ATR):
            Fixed-fractional sizing using max_position_pct,
            scaled by discrete signal-strength tier.
        """
        if getattr(self, 'kelly_position_sizing', False):
            trade_stats = get_trade_stats_for_sizing()
            sizer = PositionSizer(
                portfolio_value=self.capital,
                base_risk_pct=self.risk_per_trade_pct,
            )
            dollar_allocation = sizer.calculate(
                symbol         = symbol or 'UNKNOWN',
                price          = price,
                atr            = atr if atr else price * 0.03,
                signal_score   = float(signal_strength),
                win_rate       = trade_stats['win_rate'],
                avg_win        = trade_stats['avg_win'],
                avg_loss       = trade_stats['avg_loss'],
                regime_conf    = regime_conf,
                open_positions = self.positions,
                n_trades       = trade_stats['n_trades'],
            )
            if price > 0 and dollar_allocation > 0:
                shares = int(dollar_allocation / price)
                if shares == 0:
                    # PositionSizer already judged this trade worth taking
                    # (it cleared MIN_POSITION_USD internally) — truncating
                    # to 0 whole shares here would silently discard that
                    # decision. Round up to 1 share, but only if 1 share
                    # doesn't itself blow through the max-concentration cap
                    # — for a stock priced high enough relative to this
                    # account that even 1 share exceeds that limit, the
                    # honest answer is this account can't size this trade
                    # at all yet, not to force an oversized position.
                    one_share_cost = price
                    if one_share_cost <= self.capital * self.max_position_pct:
                        shares = 1
                        logger.info(
                            f"PositionSizer [{symbol}]: ${dollar_allocation:.0f} "
                            f"rounds to 0 shares at ${price:.2f} — using 1 share floor"
                        )
                    else:
                        logger.info(
                            f"PositionSizer [{symbol}]: ${dollar_allocation:.0f} "
                            f"rounds to 0 shares at ${price:.2f}, and even 1 share "
                            f"(${one_share_cost:.0f}) exceeds max_position_pct "
                            f"({self.max_position_pct:.0%}) — skipping, not forcing"
                        )
            else:
                shares = 0
            logger.info(
                f"PositionSizer (via PaperTrader): symbol={symbol} "
                f"alloc=${dollar_allocation:.0f} shares={shares} "
                f"n_trades={trade_stats['n_trades']}"
            )
            return max(shares, 0)

        size_multiplier = self._signal_to_size_multiplier(signal_strength)

        if atr and atr > 0:
            # Primary: risk-based sizing
            dollar_risk   = self.capital * self.risk_per_trade_pct
            stop_distance = 2.0 * atr
            shares        = int(dollar_risk / stop_distance)

            # Apply conviction multiplier
            shares = int(shares * size_multiplier)

            # Hard cap: never exceed max_position_pct of capital
            max_shares = int((self.capital * self.max_position_pct) / price)
            shares     = min(shares, max_shares)

        else:
            # Fallback: fixed-fractional
            max_dollars = self.capital * self.max_position_pct
            adjusted    = max_dollars * size_multiplier
            shares      = int(adjusted / price)

        return max(shares, 0)

    # ------------------------------------------------------------------ #
    #  OPEN POSITION                                                       #
    # ------------------------------------------------------------------ #

    def open_position(self, symbol, price, signal,
                      reason='signal', atr=None, ml_score=None,
                      next_open_price=None, regime_conf=1.0):
        """
        Open a new long position.

        Fix 2.1: If next_open_price is provided, fills at next-bar open
        (+ slippage) instead of the signal-bar close. This eliminates the
        'buy at close of signal bar' look-ahead bias in backtesting.

        next_open_price: float or None. Pass the following bar's open price
          from the walk-forward loop: next_open = df.iloc[split+1]['open']

        Guards (in order):
            1. Daily loss limit halt
            2. Max concurrent positions
            3. Duplicate symbol
            3b. Portfolio exposure cap (Fix 3.4)
            4. Zero shares after sizing
            5. Insufficient capital
            6. Per-trade max loss cap
        """

        # Guard 1: daily loss limit
        if self._is_trading_halted():
            return False

        # Guard 2: max positions
        if len(self.positions) >= self.max_positions:
            logger.info("Max positions reached, skipping %s", symbol)
            return False

        # Guard 3: duplicate
        if symbol in self.positions:
            logger.info("Already in %s, skipping", symbol)
            return False

        # Fix 3.4: portfolio exposure cap — never more than 80% deployed
        if self._get_current_exposure_pct() >= MAX_PORTFOLIO_EXPOSURE_PCT:
            logger.info(
                "Max portfolio exposure reached (%.0f%%), blocking %s",
                MAX_PORTFOLIO_EXPOSURE_PCT * 100, symbol
            )
            return False

        # FIX Bug 1 & 2: pass atr into get_position_size
        shares = self.get_position_size(price, signal, atr=atr, symbol=symbol, regime_conf=regime_conf)

        # Guard 4: zero shares
        if shares == 0:
            logger.info("Position size is 0 for %s, skipping", symbol)
            return False

        # Fix 2.1: Use next-bar open as fill price when available.
        # This is the correct simulation — the order fires at next day's open,
        # not at the close of the signal bar (which would be lookahead).
        signal_price = price
        if next_open_price and next_open_price > 0:
            fill_base  = next_open_price
            fill_type  = 'next_open'
        else:
            fill_base  = price
            fill_type  = 'close'  # legacy fallback

        # Apply slippage on the buy side
        fill_price = fill_base * (1 + self.slippage_pct)
        cost       = shares * fill_price + self.commission

        # Guard 5: insufficient capital — scale down instead of rejecting
        if cost > self.capital:
            shares = int(
                (self.capital * 0.95 - self.commission) / fill_price
            )
            if shares <= 0:
                logger.info("Insufficient capital for %s", symbol)
                return False
            cost = shares * fill_price + self.commission

        # Guard 6: per-trade max loss cap (Risk 1)
        if atr and atr > 0:
            potential_loss    = shares * (2.0 * atr)
            max_loss_dollars  = self.capital * self.risk_per_trade_pct
            if potential_loss > max_loss_dollars:
                shares = int(max_loss_dollars / (2.0 * atr))
                if shares <= 0:
                    logger.info(
                        "Per-trade loss cap: 0 shares allowed for %s", symbol
                    )
                    return False
                cost = shares * fill_price + self.commission

        # Deduct cost from cash
        self.capital -= cost

        # ATR-based stop loss
        if atr and atr > 0:
            atr_stop_pct  = (2.0 * atr) / price
            stop_loss_pct = max(0.02, min(0.08, atr_stop_pct))
        else:
            stop_loss_pct = 0.03  # default 3% stop

        self.positions[symbol] = {
            'shares'             : shares,
            'entry_price'        : fill_price,
            'market_price'       : signal_price,
            'entry_date'         : datetime.now().isoformat(),
            'highest_price'      : fill_price,
            'signal'             : signal,
            'cost'               : cost,
            'reason'             : reason,
            'stop_loss_pct'      : stop_loss_pct,
            'atr'                : atr or 0,
            'partial_exit_done'  : False,
            'entry_ml_score'     : ml_score,
            'fill_type'          : fill_type,   # Fix 2.1: audit trail
        }

        trade = {
            'action'         : 'BUY',
            'symbol'         : symbol,
            'shares'         : shares,
            'price'          : signal_price,
            'fill_price'     : fill_price,
            'fill_type'      : fill_type,   # Fix 2.1: 'next_open' or 'close'
            'slippage_pct'   : self.slippage_pct,
            'commission'     : self.commission,
            'cost'           : cost,
            'date'           : datetime.now().isoformat(),
            'reason'         : reason,
            'atr'            : atr or 0,
            'stop_loss_pct'  : stop_loss_pct,
            'entry_ml_score' : ml_score,
        }
        self.trade_history.append(trade)

        # P1-2 fix: ATR conditional moved out of f-string
        atr_str = f" | ATR {atr:.2f}" if atr else " | ATR n/a"
        print(
            f"   BUY  {shares:>5} {symbol:<6}"
            f" @ ${price:>8.2f}"
            f" fill ${fill_price:>8.2f}"
            f" cost ${cost:>9.2f}"
            f" | Stop {stop_loss_pct:.1%}"
            f"{atr_str}"
        )

        # Auto-save after every trade (Risk 3)
        self.save_state()
        return True

    # ------------------------------------------------------------------ #
    #  CLOSE POSITION                                                      #
    # ------------------------------------------------------------------ #

    def close_position(self, symbol, price, reason='signal'):
        """
        Close an existing position in full.
        Updates daily P&L and triggers halt if limit is hit.
        Auto-saves state after every close.
        """

        if symbol not in self.positions:
            return False

        pos    = self.positions[symbol]
        shares = pos['shares']
        entry  = pos['entry_price']

        # Slippage on sell side
        fill_price = price * (1 - self.slippage_pct)

        # Fix 2.2 — Apply regulatory fees on sell side
        sec_fee   = shares * fill_price * SEC_FEE_RATE
        finra_taf = shares * FINRA_TAF_PER_SHARE
        revenue   = shares * fill_price - self.commission - sec_fee - finra_taf
        pnl        = revenue - pos['cost']
        pnl_pct    = (fill_price - entry) / entry

        self.capital += revenue

        # Daily P&L tracking (Risk 2)
        self._check_daily_reset()
        self.daily_realized_pnl += pnl
        if self.daily_realized_pnl < -self.daily_loss_limit:
            logger.critical(
                "Daily loss limit hit (%.2f). Halting trading for today.",
                self.daily_realized_pnl
            )
            self._halt_trading = True

        trade = {
            'action'      : 'SELL',
            'symbol'      : symbol,
            'shares'      : shares,
            'price'       : price,
            'fill_price'  : fill_price,
            'slippage_pct': self.slippage_pct,
            'commission'  : self.commission,
            'revenue'     : revenue,
            'pnl'         : pnl,
            'pnl_pct'     : pnl_pct,
            'date'        : datetime.now().isoformat(),
            'reason'      : reason,
        }
        self.trade_history.append(trade)

        emoji = "🟢" if pnl > 0 else "🔴"
        print(
            f"   {emoji} SELL {shares:>5} {symbol:<6}"
            f" @ ${price:>8.2f}"
            f" PnL ${pnl:>+9.2f} ({pnl_pct:>+6.1%})"
            f" [{reason}]"
        )

        del self.positions[symbol]

        # Auto-save after every trade (Risk 3)
        self.save_state()
        return True

    # ------------------------------------------------------------------ #
    #  UPDATE POSITION (called every scan cycle)                           #
    # ------------------------------------------------------------------ #

    def update_position(self, symbol, current_price,
                        stop_loss     = 0.03,
                        take_profit   = 0.08,
                        trailing_stop = 0.035,
                        ml_score      = None):
        """
        Check exit conditions for an open position.

        Check order (v3):
            1. Stop loss          — most urgent, always first
            2. Full take profit   — 8% wins before partial 5% is checked
            3. Partial exit       — 50% size reduction at 5% gain
            4. ML Score Exit      — exits when live ML confidence < 0.40
            5. Trailing stop      — only fires after min profit threshold
            6. Time stop          — last resort for flat positions

        Trailing stop activation threshold (Bug 6 fix):
            Only fires after the position has reached +2% at its peak.
            Prevents premature exit on day-one slippage dips.

        ML Score Exit (v3):
            If ml_score is provided and drops below ML_EXIT_THRESHOLD,
            the position is exited regardless of P&L direction.
            Skipped if the position is already profitable enough for
            trailing stop (max_gain >= 5%) to avoid cutting winners.
        """

        if symbol not in self.positions:
            return

        pos   = self.positions[symbol]
        entry = pos['entry_price']

        # Use ATR-derived stop if stored, else use parameter default
        stop_loss = pos.get('stop_loss_pct', stop_loss)

        # Update highest price seen
        if current_price > pos['highest_price']:
            pos['highest_price'] = current_price

        pnl_pct  = (current_price - entry) / entry
        max_gain = (pos['highest_price'] - entry) / entry

        # ── 1. STOP LOSS ──────────────────────────────────────────────
        if pnl_pct <= -stop_loss:
            print(f"   🛑 STOP LOSS:    {symbol} down {pnl_pct:.1%}")
            self.close_position(symbol, current_price, 'stop_loss')
            return

        # ── 2. FULL TAKE PROFIT (checked BEFORE partial) ──────────────
        if pnl_pct >= take_profit:
            print(f"   🎯 TAKE PROFIT:  {symbol} up {pnl_pct:.1%}")
            self.close_position(symbol, current_price, 'take_profit')
            return

        # ── 3. PARTIAL EXIT at 5% ─────────────────────────────────────
        if pnl_pct >= 0.05 and not pos.get('partial_exit_done'):
            shares_to_sell = pos['shares'] // 2
            if shares_to_sell > 0:

                # FIX Flaw 3: apply slippage + commission to partial exit
                partial_fill    = current_price * (1 - self.slippage_pct)
                partial_revenue = shares_to_sell * partial_fill - self.commission
                partial_pnl     = shares_to_sell * (partial_fill - entry)

                self.capital   += partial_revenue

                # FIX Bug 3: scale cost proportionally, preserve commission
                original_shares   = pos['shares'] + shares_to_sell
                cost_per_share    = pos['cost'] / original_shares
                pos['shares']    -= shares_to_sell
                pos['cost']       = pos['shares'] * cost_per_share

                pos['partial_exit_done'] = True

                self.trade_history.append({
                    'action'      : 'PARTIAL_SELL',
                    'symbol'      : symbol,
                    'shares'      : shares_to_sell,
                    'price'       : current_price,
                    'fill_price'  : partial_fill,
                    'slippage_pct': self.slippage_pct,
                    'commission'  : self.commission,
                    'revenue'     : partial_revenue,
                    'pnl'         : partial_pnl,
                    'pnl_pct'     : pnl_pct,
                    'date'        : datetime.now().isoformat(),
                    'reason'      : 'partial_exit_5pct',
                })

                print(
                    f"   📤 PARTIAL EXIT: {symbol} up {pnl_pct:.1%}"
                    f" — sold {shares_to_sell} shares"
                    f" locked +${partial_pnl:.2f}"
                )

                # Auto-save after partial exit
                self.save_state()
            return

        # ── 4. ML SCORE EXIT ──────────────────────────────────────────
        # v3: if the model's current confidence has dropped below threshold
        # AND the position hasn't already surged (max_gain < 5%),
        # exit now before the loss deepens.
        if ml_score is not None and ml_score < ML_EXIT_THRESHOLD:
            if max_gain < 0.05:   # don't cut a winner that's already at 5%+
                print(
                    f"   🤖 ML EXIT:      {symbol}"
                    f" ML score dropped to {ml_score:.2f}"
                    f" (threshold {ML_EXIT_THRESHOLD:.2f})"
                    f" | P&L {pnl_pct:+.1%}"
                )
                self.close_position(symbol, current_price, 'ml_score_exit')
                return
            else:
                logger.info(
                    f"{symbol}: ML score low ({ml_score:.2f}) but position"
                    f" at +{max_gain:.1%} — letting trailing stop handle it"
                )

        # ── 5. TRAILING STOP ──────────────────────────────────────────
        # FIX Bug 6: only activates after position has reached +2% peak
        TRAILING_ACTIVATION = 0.02
        drop = (
            (pos['highest_price'] - current_price) / pos['highest_price']
        )
        if drop >= trailing_stop and max_gain >= TRAILING_ACTIVATION:
            print(
                f"   📉 TRAILING STOP: {symbol}"
                f" dropped {drop:.1%} from high"
            )
            self.close_position(symbol, current_price, 'trailing_stop')
            return

        # ── 6. TIME-BASED STOP ────────────────────────────────────────
        try:
            entry_date = datetime.fromisoformat(
                pos.get('entry_date', '')
            )
            days_held = (datetime.now() - entry_date).days
            # P2-4 fix: threshold from settings, not hardcoded 5
            if days_held >= self.time_stop_days and abs(pnl_pct) < 0.02:
                print(
                    f"   ⏱️  TIME STOP:    {symbol}"
                    f" flat after {days_held} days"
                )
                self.close_position(symbol, current_price, 'time_stop')
                return
        except Exception as e:
            logger.debug(f'Time-stop date parse failed for {symbol}: {e}')

    # ------------------------------------------------------------------ #
    #  PORTFOLIO VALUATION                                                 #
    # ------------------------------------------------------------------ #

    def get_portfolio_value(self, current_prices):
        """
        Calculate total portfolio value (cash + positions).

        FIX Flaw 4: positions missing from current_prices now use
        entry_price as a conservative fallback instead of being
        silently excluded (which was understating total value and
        could trigger the circuit breaker incorrectly).
        """
        position_value = 0.0

        for symbol, pos in self.positions.items():
            if symbol in current_prices:
                price = current_prices[symbol]
            else:
                price = pos['entry_price']
                logger.warning(
                    "No current price for %s — using entry price as fallback",
                    symbol
                )
            position_value += pos['shares'] * price

        return self.capital + position_value

    # ------------------------------------------------------------------ #
    #  PORTFOLIO SUMMARY                                                   #
    # ------------------------------------------------------------------ #

    def get_summary(self, current_prices=None):
        """Print a formatted portfolio summary to console."""

        if current_prices is None:
            current_prices = {}

        position_value = 0.0

        print("\n" + "=" * 60)
        print("  PAPER TRADING PORTFOLIO")
        print("=" * 60)
        print(f"  Cash:              ${self.capital:>12,.2f}")
        print(f"  Daily P&L:         ${self.daily_realized_pnl:>+12,.2f}")
        print(
            f"  Daily limit:       ${self.daily_loss_limit:>12,.2f}"
            f"  {'🛑 HALTED' if self._halt_trading else '✅ Active'}"
        )

        if self.positions:
            print("\n  Open Positions:")
            for symbol, pos in self.positions.items():
                shares = pos['shares']
                entry  = pos['entry_price']

                if symbol in current_prices:
                    curr    = current_prices[symbol]
                    val     = shares * curr
                    pnl     = val - pos['cost']
                    pnl_pct = (curr - entry) / entry
                    position_value += val
                    emoji = "🟢" if pnl > 0 else "🔴"
                    print(
                        f"    {emoji} {symbol:<6}"
                        f" {shares:>5} shares"
                        f" entry ${entry:>8.2f}"
                        f" now ${curr:>8.2f}"
                        f" PnL ${pnl:>+9.2f} ({pnl_pct:>+6.1%})"
                    )
                else:
                    val = shares * entry
                    position_value += val
                    print(
                        f"    ⚪ {symbol:<6}"
                        f" {shares:>5} shares"
                        f" entry ${entry:>8.2f}"
                        f" (no current price)"
                    )

        total     = self.capital + position_value
        total_pnl = total - self.starting_capital
        total_pct = total_pnl / self.starting_capital

        print(f"\n  Position Value:    ${position_value:>12,.2f}")
        print(f"  Total Value:       ${total:>12,.2f}")
        print(
            f"  Total PnL:         ${total_pnl:>+12,.2f}"
            f" ({total_pct:>+.1%})"
        )
        print(f"  Total Trades:      {len(self.trade_history):>12}")
        print("=" * 60)

        return total

    # ------------------------------------------------------------------ #
    #  STATE PERSISTENCE                                                   #
    # ------------------------------------------------------------------ #

    def save_state(self):
        """
        Save current state to disk atomically.

        FIX Bug 4: writes to a .tmp file first, then os.replace()
        which is atomic on all platforms. A mid-write crash can no
        longer corrupt the live state file.
        """
        state = {
            'capital'            : self.capital,
            'starting_capital'   : self.starting_capital,
            'positions'          : self.positions,
            'trade_history'      : self.trade_history,
            'daily_realized_pnl' : self.daily_realized_pnl,
            'daily_loss_limit'   : self.daily_loss_limit,
            'last_reset_date'    : str(self.last_reset_date),
            '_halt_trading'      : self._halt_trading,
            'saved_at'           : datetime.now().isoformat(),
        }

        os.makedirs(os.path.dirname(self.log_file), exist_ok=True)

        tmp_path = self.log_file + '.tmp'
        try:
            with open(tmp_path, 'w') as f:
                json.dump(state, f, indent=2)
            os.replace(tmp_path, self.log_file)
            logger.debug("State saved to %s", self.log_file)

        except Exception as e:
            logger.error("Failed to save state: %s", e)
            if os.path.exists(tmp_path):
                os.remove(tmp_path)
            raise

    def load_state(self):
        """
        Load state from disk with full error handling.

        FIX Bug 5:
        - Validates all required keys before applying any values
        - Catches JSON decode errors and key errors gracefully
        - Backs up the corrupted file instead of deleting it
        - Never crashes the trading system at startup
        """
        if not os.path.exists(self.log_file):
            print("   No saved state found — starting fresh")
            return

        try:
            with open(self.log_file, 'r') as f:
                state = json.load(f)

            # Validate required keys before touching any instance state
            required = [
                'capital', 'starting_capital',
                'positions', 'trade_history'
            ]
            missing = [k for k in required if k not in state]
            if missing:
                raise ValueError(
                    f"Corrupted state file — missing keys: {missing}"
                )

            # Apply loaded state
            self.capital          = float(state['capital'])
            self.starting_capital = float(state['starting_capital'])
            self.positions        = state['positions']
            self.trade_history    = state['trade_history']

            # Restore daily tracking state if present
            self.daily_realized_pnl = float(
                state.get('daily_realized_pnl', 0.0)
            )
            self.daily_loss_limit   = float(
                state.get('daily_loss_limit',
                          self.starting_capital * 0.05)
            )
            self._halt_trading      = bool(
                state.get('_halt_trading', False)
            )

            # Restore last reset date
            raw_date = state.get('last_reset_date', '')
            try:
                self.last_reset_date = datetime.strptime(
                    raw_date, '%Y-%m-%d'
                ).date()
            except Exception as e:
                logger.debug(f'Could not parse last_reset_date "{raw_date}": {e}')
                self.last_reset_date = datetime.now().date()

            print(
                f"   📂 State loaded: "
                f"${self.capital:,.2f} cash | "
                f"{len(self.positions)} open positions | "
                f"{len(self.trade_history)} total trades"
            )

            if self._halt_trading:
                print("   ⚠️  Trading was halted when state was saved.")

        except (json.JSONDecodeError, ValueError, KeyError) as e:
            logger.error("Failed to load state: %s", e)
            print(f"   ⚠️  Corrupted state file ({e})")
            print("   Backing up and starting fresh.")

            backup = self.log_file + '.corrupted'
            try:
                os.replace(self.log_file, backup)
                print(f"   Backup saved → {backup}")
            except Exception as backup_err:
                logger.error(
                    "Could not back up corrupted file: %s", backup_err
                )