# execution/paper_trader.py

"""
PaperTrader V3

Changes from V2 (documented in AUDIT.md):
- reconcile() invariant check: capital + position_costs - total_realized_pnl
  must equal starting_capital. Called on every save_state(). Crashes
  loudly if state is corrupt.
- ATR fallback raised from price*0.02 to price*0.03 (handled at caller,
  but the stop_loss_pct floor here also raised from 0.02 to 0.025).
- _signal_to_size_multiplier left as discrete tiers — see AUDIT.md M3.
- Type hints added on public methods.
- Format strings cleaned up to remove the if-inside-format bug.
"""

import json
import logging
import os
import tempfile
from datetime import datetime
from typing import Optional

logger = logging.getLogger(__name__)


class PaperTrader:
    """
    Simulated trading engine. Tracks all trades, positions, and P&L
    as if real money. Saves state atomically so you can stop/resume.
    Includes slippage and commission for realistic results.
    """

    def __init__(self,
                 starting_capital     : float = 10000.0,
                 max_position_pct     : float = 0.15,
                 max_positions        : int   = 5,
                 slippage_pct         : float = 0.0005,
                 commission           : float = 1.0,
                 risk_per_trade_pct   : float = 0.02,
                 daily_loss_limit_pct : float = 0.05,
                 log_file             : str   = 'logs/paper_trades.json'):

        self.starting_capital     = float(starting_capital)
        self.capital              = float(starting_capital)
        self.max_position_pct     = max_position_pct
        self.max_positions        = max_positions
        self.slippage_pct         = slippage_pct
        self.commission           = commission
        self.risk_per_trade_pct   = risk_per_trade_pct
        self.log_file             = log_file

        self.positions     : dict = {}
        self.trade_history : list = []
        self.daily_pnl     : list = []

        # Daily loss limit tracking
        self.daily_loss_limit   = starting_capital * daily_loss_limit_pct
        self.daily_realized_pnl = 0.0
        self.last_reset_date    = datetime.now().date()
        self._halt_trading      = False

    # ------------------------------------------------------------------ #
    #  RECONCILIATION INVARIANT — NEW                                      #
    # ------------------------------------------------------------------ #

    def reconcile(self, tolerance: float = 0.01) -> None:
        """
        Verify the accounting invariant:

            starting_capital + total_realized_pnl
                = capital + sum(position_costs)

        Realized P&L includes commission and slippage already (because
        revenue and cost both already include them). If this ever drifts
        beyond `tolerance` dollars, state is corrupt — we crash loudly
        rather than continue trading on bad numbers.
        """
        position_costs = sum(p.get('cost', 0.0) for p in self.positions.values())

        realized_pnl = 0.0
        for trade in self.trade_history:
            if trade.get('action') in ('SELL', 'PARTIAL_SELL'):
                realized_pnl += trade.get('pnl', 0.0)

        expected = self.starting_capital + realized_pnl
        actual   = self.capital + position_costs
        drift    = actual - expected

        if abs(drift) > tolerance:
            msg = (
                f"PaperTrader reconciliation FAILED.\n"
                f"  expected (start + realized_pnl) = ${expected:,.4f}\n"
                f"  actual   (cash    + costs)      = ${actual:,.4f}\n"
                f"  drift                           = ${drift:+,.4f}\n"
                f"  starting_capital                = ${self.starting_capital:,.4f}\n"
                f"  capital (cash)                  = ${self.capital:,.4f}\n"
                f"  sum(position_costs)             = ${position_costs:,.4f}\n"
                f"  realized_pnl from trade_history = ${realized_pnl:,.4f}\n"
                f"  n_positions = {len(self.positions)}, "
                f"n_trades = {len(self.trade_history)}"
            )
            logger.critical(msg)
            raise AssertionError(msg)

        logger.debug(
            "Reconcile OK: drift=$%.4f, realized=$%.2f, "
            "cash=$%.2f, costs=$%.2f",
            drift, realized_pnl, self.capital, position_costs,
        )

    # ------------------------------------------------------------------ #
    #  INTERNAL HELPERS                                                    #
    # ------------------------------------------------------------------ #

    def _check_daily_reset(self) -> None:
        today = datetime.now().date()
        if today != self.last_reset_date:
            logger.info(
                "New trading day. Resetting daily P&L (was %+.2f)",
                self.daily_realized_pnl,
            )
            self.daily_realized_pnl = 0.0
            self._halt_trading      = False
            self.last_reset_date    = today

    def _is_trading_halted(self) -> bool:
        self._check_daily_reset()
        if self._halt_trading:
            logger.warning("Trading halted: daily loss limit reached.")
            return True
        return False

    @staticmethod
    def _signal_to_size_multiplier(signal_strength: float) -> float:
        """
        Discrete tier table. See AUDIT.md M3 for stability discussion.
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
    #  POSITION SIZING                                                     #
    # ------------------------------------------------------------------ #

    def get_position_size(self,
                          price: float,
                          signal_strength: float = 1.0,
                          atr: Optional[float] = None) -> int:
        size_multiplier = self._signal_to_size_multiplier(signal_strength)

        if atr and atr > 0:
            dollar_risk   = self.capital * self.risk_per_trade_pct
            stop_distance = 2.0 * atr
            shares        = int(dollar_risk / stop_distance)
            shares        = int(shares * size_multiplier)
            max_shares    = int((self.capital * self.max_position_pct) / price)
            shares        = min(shares, max_shares)
        else:
            max_dollars = self.capital * self.max_position_pct
            adjusted    = max_dollars * size_multiplier
            shares      = int(adjusted / price)

        return max(shares, 0)

    # ------------------------------------------------------------------ #
    #  OPEN POSITION                                                       #
    # ------------------------------------------------------------------ #

    def open_position(self,
                      symbol: str,
                      price: float,
                      signal: float,
                      reason: str = 'signal',
                      atr: Optional[float] = None) -> bool:

        if self._is_trading_halted():
            return False
        if len(self.positions) >= self.max_positions:
            logger.info("Max positions reached, skipping %s", symbol)
            return False
        if symbol in self.positions:
            logger.info("Already in %s, skipping", symbol)
            return False

        shares = self.get_position_size(price, signal, atr=atr)
        if shares == 0:
            logger.info("Position size is 0 for %s, skipping", symbol)
            return False

        fill_price = price * (1 + self.slippage_pct)
        cost       = shares * fill_price + self.commission

        # Scale down on insufficient capital
        if cost > self.capital:
            shares = int((self.capital * 0.95 - self.commission) / fill_price)
            if shares <= 0:
                logger.info("Insufficient capital for %s", symbol)
                return False
            cost = shares * fill_price + self.commission

        # Per-trade max loss cap
        if atr and atr > 0:
            potential_loss   = shares * (2.0 * atr)
            max_loss_dollars = self.capital * self.risk_per_trade_pct
            if potential_loss > max_loss_dollars:
                shares = int(max_loss_dollars / (2.0 * atr))
                if shares <= 0:
                    logger.info(
                        "Per-trade loss cap: 0 shares allowed for %s", symbol,
                    )
                    return False
                cost = shares * fill_price + self.commission

        self.capital -= cost

        # ATR-based stop loss; floor raised from 0.02 to 0.025
        if atr and atr > 0:
            atr_stop_pct  = (2.0 * atr) / price
            stop_loss_pct = max(0.025, min(0.08, atr_stop_pct))
        else:
            stop_loss_pct = 0.03

        self.positions[symbol] = {
            'shares'            : shares,
            'entry_price'       : fill_price,
            'market_price'      : price,
            'entry_date'        : datetime.now().isoformat(),
            'highest_price'     : fill_price,
            'signal'            : signal,
            'cost'              : cost,
            'reason'            : reason,
            'stop_loss_pct'     : stop_loss_pct,
            'atr'               : atr or 0,
            'partial_exit_done' : False,
        }

        self.trade_history.append({
            'action'       : 'BUY',
            'symbol'       : symbol,
            'shares'       : shares,
            'price'        : price,
            'fill_price'   : fill_price,
            'slippage_pct' : self.slippage_pct,
            'commission'   : self.commission,
            'cost'         : cost,
            'date'         : datetime.now().isoformat(),
            'reason'       : reason,
            'atr'          : atr or 0,
            'stop_loss_pct': stop_loss_pct,
        })

        # FIX: removed if-inside-format expression which silently dropped
        # the ATR string entirely when atr was None.
        atr_str = f" | ATR {atr:.2f}" if atr else ""
        print(
            f"   BUY  {shares:>5} {symbol:<6} @ ${price:>8.2f}"
            f" fill ${fill_price:>8.2f} cost ${cost:>9.2f}"
            f" | Stop {stop_loss_pct:.1%}{atr_str}"
        )

        self.save_state()
        return True

    # ------------------------------------------------------------------ #
    #  CLOSE POSITION                                                      #
    # ------------------------------------------------------------------ #

    def close_position(self,
                       symbol: str,
                       price: float,
                       reason: str = 'signal') -> bool:
        if symbol not in self.positions:
            return False

        pos    = self.positions[symbol]
        shares = pos['shares']
        entry  = pos['entry_price']

        fill_price = price * (1 - self.slippage_pct)
        revenue    = shares * fill_price - self.commission
        pnl        = revenue - pos['cost']
        pnl_pct    = (fill_price - entry) / entry if entry > 0 else 0.0

        self.capital += revenue

        self._check_daily_reset()
        self.daily_realized_pnl += pnl
        if self.daily_realized_pnl < -self.daily_loss_limit:
            logger.critical(
                "Daily loss limit hit (%.2f). Halting trading for today.",
                self.daily_realized_pnl,
            )
            self._halt_trading = True

        self.trade_history.append({
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
        })

        emoji = "🟢" if pnl > 0 else "🔴"
        print(
            f"   {emoji} SELL {shares:>5} {symbol:<6} @ ${price:>8.2f}"
            f" PnL ${pnl:>+9.2f} ({pnl_pct:>+6.1%}) [{reason}]"
        )

        del self.positions[symbol]
        self.save_state()
        return True

    # ------------------------------------------------------------------ #
    #  UPDATE POSITION                                                     #
    # ------------------------------------------------------------------ #

    def update_position(self,
                        symbol: str,
                        current_price: float,
                        stop_loss     : float = 0.03,
                        take_profit   : float = 0.08,
                        trailing_stop : float = 0.035) -> None:
        if symbol not in self.positions:
            return

        pos   = self.positions[symbol]
        entry = pos['entry_price']
        stop_loss = pos.get('stop_loss_pct', stop_loss)

        if current_price > pos['highest_price']:
            pos['highest_price'] = current_price

        pnl_pct  = (current_price - entry) / entry if entry > 0 else 0.0
        max_gain = (pos['highest_price'] - entry) / entry if entry > 0 else 0.0

        # 1. Stop loss
        if pnl_pct <= -stop_loss:
            print(f"   🛑 STOP LOSS:    {symbol} down {pnl_pct:.1%}")
            self.close_position(symbol, current_price, 'stop_loss')
            return

        # 2. Full take profit
        if pnl_pct >= take_profit:
            print(f"   🎯 TAKE PROFIT:  {symbol} up {pnl_pct:.1%}")
            self.close_position(symbol, current_price, 'take_profit')
            return

        # 3. Partial exit at 5%
        if pnl_pct >= 0.05 and not pos.get('partial_exit_done'):
            shares_to_sell = pos['shares'] // 2
            if shares_to_sell > 0:
                partial_fill    = current_price * (1 - self.slippage_pct)
                partial_revenue = shares_to_sell * partial_fill - self.commission

                self.capital += partial_revenue

                # ── Cost basis scaling (REAL FIX) ─────────────────────
                # V2 had: `original_shares = pos['shares'] + shares_to_sell`
                # but pos['shares'] was still pre-sale, so this produced
                # 1.5x the real total → under-reported cost-per-share →
                # inflated future P&L reporting. AUDIT.md H5.
                #
                # Correct: pos['shares'] IS the pre-sale total here.
                cost_per_share  = pos['cost'] / pos['shares']
                cost_of_sold    = shares_to_sell * cost_per_share
                pos['shares']  -= shares_to_sell
                pos['cost']     = pos['shares'] * cost_per_share
                pos['partial_exit_done'] = True

                # Reported PnL must include commission of this partial
                # exit AND the proportional share of original entry
                # commission. Without this, reconcile() drifts by ~$1
                # per partial exit (entry comm not amortised + exit comm
                # not deducted). H5 part 2.
                partial_pnl = partial_revenue - cost_of_sold

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
                self.save_state()
            return

        # 4. Trailing stop (requires 2% peak first)
        TRAILING_ACTIVATION = 0.02
        drop = ((pos['highest_price'] - current_price)
                / pos['highest_price']) if pos['highest_price'] > 0 else 0
        if drop >= trailing_stop and max_gain >= TRAILING_ACTIVATION:
            print(
                f"   📉 TRAILING STOP: {symbol} dropped {drop:.1%} from high"
            )
            self.close_position(symbol, current_price, 'trailing_stop')
            return

        # 5. Time stop
        try:
            entry_date = datetime.fromisoformat(pos.get('entry_date', ''))
            days_held = (datetime.now() - entry_date).days
            if days_held >= 5 and abs(pnl_pct) < 0.02:
                print(
                    f"   ⏱️  TIME STOP:    {symbol} flat after {days_held} days"
                )
                self.close_position(symbol, current_price, 'time_stop')
                return
        except (ValueError, TypeError):
            pass

    # ------------------------------------------------------------------ #
    #  PORTFOLIO VALUATION                                                 #
    # ------------------------------------------------------------------ #

    def get_portfolio_value(self, current_prices: dict) -> float:
        position_value = 0.0
        for symbol, pos in self.positions.items():
            price = current_prices.get(symbol, pos['entry_price'])
            if symbol not in current_prices:
                logger.warning(
                    "No current price for %s — using entry price as fallback",
                    symbol,
                )
            position_value += pos['shares'] * price
        return self.capital + position_value

    def get_summary(self, current_prices=None) -> float:
        if current_prices is None:
            current_prices = {}

        position_value = 0.0

        print("\n" + "=" * 60)
        print("  PAPER TRADING PORTFOLIO")
        print("=" * 60)
        print(f"  Cash:              ${self.capital:>12,.2f}")
        print(f"  Daily P&L:         ${self.daily_realized_pnl:>+12,.2f}")
        halt_str = '🛑 HALTED' if self._halt_trading else '✅ Active'
        print(
            f"  Daily limit:       ${self.daily_loss_limit:>12,.2f}  {halt_str}"
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
                    pnl_pct = (curr - entry) / entry if entry > 0 else 0.0
                    position_value += val
                    emoji = "🟢" if pnl > 0 else "🔴"
                    print(
                        f"    {emoji} {symbol:<6} {shares:>5} shares"
                        f" entry ${entry:>8.2f} now ${curr:>8.2f}"
                        f" PnL ${pnl:>+9.2f} ({pnl_pct:>+6.1%})"
                    )
                else:
                    val = shares * entry
                    position_value += val
                    print(
                        f"    ⚪ {symbol:<6} {shares:>5} shares"
                        f" entry ${entry:>8.2f} (no current price)"
                    )

        total     = self.capital + position_value
        total_pnl = total - self.starting_capital
        total_pct = total_pnl / self.starting_capital if self.starting_capital else 0.0

        print(f"\n  Position Value:    ${position_value:>12,.2f}")
        print(f"  Total Value:       ${total:>12,.2f}")
        print(f"  Total PnL:         ${total_pnl:>+12,.2f} ({total_pct:>+.1%})")
        print(f"  Total Trades:      {len(self.trade_history):>12}")
        print("=" * 60)
        return total

    # ------------------------------------------------------------------ #
    #  STATE PERSISTENCE                                                   #
    # ------------------------------------------------------------------ #

    def save_state(self) -> None:
        """Atomic save. Runs reconcile() first; refuses to save corrupt state."""
        # Run invariant check before persisting. If this throws, we want
        # the system to halt before writing bad state to disk.
        self.reconcile()

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
                try:
                    os.remove(tmp_path)
                except OSError:
                    pass
            raise

    def load_state(self) -> None:
        if not os.path.exists(self.log_file):
            print("   No saved state found — starting fresh")
            return

        try:
            with open(self.log_file, 'r') as f:
                state = json.load(f)

            required = ['capital', 'starting_capital', 'positions', 'trade_history']
            missing  = [k for k in required if k not in state]
            if missing:
                raise ValueError(f"Corrupted state file — missing keys: {missing}")

            self.capital            = float(state['capital'])
            self.starting_capital   = float(state['starting_capital'])
            self.positions          = state['positions']
            self.trade_history      = state['trade_history']
            self.daily_realized_pnl = float(state.get('daily_realized_pnl', 0.0))
            self.daily_loss_limit   = float(
                state.get('daily_loss_limit', self.starting_capital * 0.05)
            )
            self._halt_trading      = bool(state.get('_halt_trading', False))

            raw_date = state.get('last_reset_date', '')
            try:
                self.last_reset_date = datetime.strptime(raw_date, '%Y-%m-%d').date()
            except (ValueError, TypeError):
                self.last_reset_date = datetime.now().date()

            # Run invariant check on loaded state
            try:
                self.reconcile()
            except AssertionError as e:
                # Corrupt state on load — back up and start fresh
                logger.critical("Loaded state failed reconciliation: %s", e)
                self._backup_corrupt_state()
                self._reset_to_fresh()
                return

            print(
                f"   📂 State loaded: ${self.capital:,.2f} cash | "
                f"{len(self.positions)} open positions | "
                f"{len(self.trade_history)} total trades"
            )
            if self._halt_trading:
                print("   ⚠️  Trading was halted when state was saved.")

        except (json.JSONDecodeError, ValueError, KeyError) as e:
            logger.error("Failed to load state: %s", e)
            print(f"   ⚠️  Corrupted state file ({e})")
            print("   Backing up and starting fresh.")
            self._backup_corrupt_state()

    def _backup_corrupt_state(self) -> None:
        backup = self.log_file + '.corrupted'
        try:
            os.replace(self.log_file, backup)
            print(f"   Backup saved → {backup}")
        except OSError as e:
            logger.error("Could not back up corrupted file: %s", e)

    def _reset_to_fresh(self) -> None:
        self.capital            = self.starting_capital
        self.positions          = {}
        self.trade_history      = []
        self.daily_realized_pnl = 0.0
        self._halt_trading      = False
