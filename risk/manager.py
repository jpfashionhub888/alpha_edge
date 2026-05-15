# risk/manager.py

"""
RiskManager V2

Fixes applied:
- ATR now calculated at entry time per trade, not last row (Bug 1)
- df.at[idx] replaces df.iloc[i, get_loc()] — safe with any index (Bug 2)
- Daily loss reset now clears all trade state (Bug 3)
- Trailing stop only activates after 2% gain (Bug 4)
- _get_atr_value() tries multiple column names with warning (Flaw 1)
- max_portfolio_risk now implemented (Flaw 2)
- Stop loss uses actual pnl not theoretical stop level (Flaw 3)
- Print statements replaced with logger (Flaw 4)
- Required columns validated at entry (Risk 1)
- Division by zero guard on entry_price (Risk 2)
- Output validated before return (Risk 3)
"""

import logging

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

# Trailing stop only activates after this minimum gain
TRAILING_ACTIVATION_THRESHOLD = 0.02


class RiskManager:
    """
    Dynamic risk management.

    Stop loss and take profit adjust based on stock
    volatility (ATR) at trade entry time — not end of period.

    Parameters
    ----------
    base_stop_loss           : fallback stop if ATR unavailable
    reward_risk_ratio        : take_profit = stop * ratio
    trailing_stop_multiplier : trailing = stop * multiplier
    max_daily_loss           : halt trading after this daily loss
    max_portfolio_risk       : close all positions at this drawdown
    """

    def __init__(self,
                 base_stop_loss           = 0.03,
                 reward_risk_ratio        = 2.5,
                 trailing_stop_multiplier = 0.8,
                 max_daily_loss           = 0.02,
                 max_portfolio_risk       = 0.06,
                 cooloff_days             = 5):
        self.cooloff_days = cooloff_days
        self.base_stop_loss      = base_stop_loss
        self.reward_risk_ratio   = reward_risk_ratio
        self.trailing_multiplier = trailing_stop_multiplier
        self.max_daily_loss      = max_daily_loss
        self.max_portfolio_risk  = max_portfolio_risk

    # ---------------------------------------------------------------- #
    #  PUBLIC                                                            #
    # ---------------------------------------------------------------- #

    def calculate_dynamic_stops(self,
                                 df: pd.DataFrame,
                                 row_idx: int = -1) -> dict:
        """
        Calculate stop loss, take profit, and trailing stop
        based on ATR at a specific row (trade entry point).

        Parameters
        ----------
        df      : full fold dataframe
        row_idx : integer position of the entry row.
                  Defaults to -1 (last row) for legacy calls.

        Returns
        -------
        dict with stop_loss, take_profit, trailing_stop
        """
        atr = self._get_atr_value(df, row_idx)

        # Dynamic stop: 1.5x ATR, clipped to [2%, 8%]
        stop_loss   = float(np.clip(atr * 1.5, 0.02, 0.08))
        take_profit = stop_loss * self.reward_risk_ratio
        trailing    = stop_loss * self.trailing_multiplier

        return {
            'stop_loss'    : stop_loss,
            'take_profit'  : take_profit,
            'trailing_stop': trailing,
        }

    def apply(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Apply dynamic risk management to a signal dataframe.

        Iterates row by row to simulate realistic trade
        entry, exit, and daily loss limits.

        Returns df with added columns:
            managed_signal : signal after risk filter
            managed_return : return after stops and costs
            exit_reason    : why each position was closed
        """
        # ── Validate required columns ─────────────────────────────
        required = ['close', 'signal', 'returns']
        missing  = [c for c in required if c not in df.columns]
        if missing:
            raise ValueError(
                f"RiskManager.apply() missing required "
                f"columns: {missing}"
            )

        df = df.copy()

        # Initialise output columns
        df['managed_signal'] = 0.0
        df['managed_return'] = 0.0
        df['exit_reason']    = 'none'

        # Trade state
        in_trade       = False
        entry_price    = 0.0
        highest        = 0.0
        trade_signal   = 0.0
        stop_loss      = self.base_stop_loss
        take_profit    = self.base_stop_loss * self.reward_risk_ratio
        trailing_stop  = self.base_stop_loss * self.trailing_multiplier

        # Daily loss tracking
        daily_loss     = 0.0
        current_date   = None

        # Portfolio-level cumulative return tracking
        portfolio_return = 0.0
        halted           = False

        for i in range(len(df)):
            idx = df.index[i]
            row = df.iloc[i]

            price      = row['close']
            raw_signal = row['signal']
            daily_ret  = row['returns']

            # ── Daily loss reset ──────────────────────────────────
            td = idx.date() if hasattr(idx, 'date') else idx
            if td != current_date:
                daily_loss   = 0.0
                current_date = td

            # ── Portfolio risk halt ───────────────────────────────
            if portfolio_return < -self.max_portfolio_risk:
                if not halted:
                    logger.warning(
                        "Portfolio risk limit hit (%.2f) — "
                        "halting all new trades",
                        portfolio_return,
                    )
                    halted = True

                if in_trade:
                    # Close open position at portfolio limit
                    if entry_price > 0:
                        pnl = (price - entry_price) / entry_price
                    else:
                        pnl = 0.0
                    ret = pnl
                    df.at[idx, 'managed_signal'] = 0.0
                    df.at[idx, 'managed_return'] = ret
                    df.at[idx, 'exit_reason']    = 'portfolio_risk_limit'
                    daily_loss      += ret
                    portfolio_return += ret
                    # Reset ALL trade state
                    in_trade     = False
                    entry_price  = 0.0
                    highest      = 0.0
                    trade_signal = 0.0
                else:
                    df.at[idx, 'exit_reason'] = 'portfolio_halted'
                continue

            # ── Daily loss limit ──────────────────────────────────
            if daily_loss < -self.max_daily_loss:
                df.at[idx, 'managed_signal'] = 0.0
                df.at[idx, 'exit_reason']    = 'daily_limit'

                if in_trade:
                    # Fix Bug 3: properly close trade and reset state
                    if entry_price > 0:
                        pnl = (price - entry_price) / entry_price
                    else:
                        pnl = 0.0
                    ret = pnl
                    df.at[idx, 'managed_return'] = ret
                    daily_loss      += ret
                    portfolio_return += ret

                # Reset ALL trade state — not just in_trade
                in_trade     = False
                entry_price  = 0.0
                highest      = 0.0
                trade_signal = 0.0
                continue

            # ── In-trade exit checks ──────────────────────────────
            if in_trade:

                # Division by zero guard
                if entry_price <= 0:
                    logger.error(
                        "entry_price=%.2f at %s — "
                        "closing to prevent division by zero",
                        entry_price, idx,
                    )
                    in_trade    = False
                    entry_price = 0.0
                    highest     = 0.0
                    trade_signal= 0.0
                    continue

                if price > highest:
                    highest = price

                pnl      = (price - entry_price) / entry_price
                max_gain = (highest - entry_price) / entry_price

                # Stop loss — use ACTUAL pnl not theoretical level
                # Fix Flaw 3: simulates gap-down risk correctly
                if pnl <= -stop_loss:
                    ret = pnl
                    df.at[idx, 'managed_signal'] = 0.0
                    df.at[idx, 'managed_return'] = ret
                    df.at[idx, 'exit_reason']    = 'stop_loss'
                    daily_loss      += ret
                    portfolio_return += ret
                    in_trade     = False
                    entry_price  = 0.0
                    highest      = 0.0
                    trade_signal = 0.0
                    self._last_stop_bar = i
                    continue

                # Take profit
                if pnl >= take_profit:
                    ret = pnl
                    df.at[idx, 'managed_signal'] = 0.0
                    df.at[idx, 'managed_return'] = ret
                    df.at[idx, 'exit_reason']    = 'take_profit'
                    daily_loss      += ret
                    portfolio_return += ret
                    in_trade     = False
                    entry_price  = 0.0
                    highest      = 0.0
                    trade_signal = 0.0
                    continue

                # Trailing stop — only after 2% activation threshold
                # Fix Bug 4: prevents firing on day-one dips
                drop = (highest - price) / highest if highest > 0 else 0

                if (drop >= trailing_stop
                        and max_gain >= TRAILING_ACTIVATION_THRESHOLD):
                    ret = pnl
                    df.at[idx, 'managed_signal'] = 0.0
                    df.at[idx, 'managed_return'] = ret
                    df.at[idx, 'exit_reason']    = 'trailing_stop'
                    daily_loss      += ret
                    portfolio_return += ret
                    in_trade     = False
                    entry_price  = 0.0
                    highest      = 0.0
                    trade_signal = 0.0
                    continue

                # Hold: record daily return
                # Use binary (in/out) for return calculation
                # Fractional sizing is for position sizing only
                ret = daily_ret

                df.at[idx, 'managed_signal'] = trade_signal
                df.at[idx, 'managed_return'] = ret
                daily_loss      += ret
                portfolio_return += ret

            # ── Entry ─────────────────────────────────────────────
            else:
                # Check cooling off period after stop loss
                if not hasattr(self, '_last_stop_bar'):
                    self._last_stop_bar = -999
                bars_since_stop = i - self._last_stop_bar

                if raw_signal > 0 and not halted and bars_since_stop >= self.cooloff_days:

                    # Fix Bug 1: calculate stops at entry time
                    # not from last row of fold
                    entry_stops  = self.calculate_dynamic_stops(df, i)
                    stop_loss    = entry_stops['stop_loss']
                    take_profit  = entry_stops['take_profit']
                    trailing_stop= entry_stops['trailing_stop']

                    in_trade     = True
                    entry_price  = price
                    highest      = price
                    trade_signal = raw_signal

                    ret = daily_ret
                    df.at[idx, 'managed_signal'] = raw_signal
                    df.at[idx, 'managed_return'] = ret
                    df.at[idx, 'exit_reason']    = 'entry'
                    daily_loss      += ret
                    portfolio_return += ret

        # ── Output validation ─────────────────────────────────────
        n_trades = int((df['managed_signal'] > 0).sum())
        if n_trades == 0:
            logger.warning(
                "RiskManager produced zero active trades. "
                "Check signal column and daily loss limit."
            )

        n_nan = int(df['managed_return'].isna().sum())
        if n_nan > 0:
            logger.error(
                "%d NaN values in managed_return — filling 0.0",
                n_nan,
            )
            df['managed_return'] = df['managed_return'].fillna(0.0)

        # ── Stats logging ─────────────────────────────────────────
        exits   = df[df['exit_reason'] != 'none']
        ecounts = exits['exit_reason'].value_counts()

        logger.info(
            "Risk mgmt stops — "
            "stop=%.1f%% tp=%.1f%% trail=%.1f%%",
            stop_loss   * 100,
            take_profit * 100,
            trailing_stop * 100,
        )

        for reason, count in ecounts.items():
            logger.debug("  exit %-25s %d", reason, count)

        for exit_type, label in [
            ('stop_loss',     'Avg stop loss  '),
            ('take_profit',   'Avg take profit'),
            ('trailing_stop', 'Avg trailing   '),
        ]:
            subset = exits[exits['exit_reason'] == exit_type]
            if len(subset) > 0:
                avg = subset['managed_return'].mean()
                logger.debug("  %s: %.2f%%", label, avg * 100)

        return df

    # ---------------------------------------------------------------- #
    #  PRIVATE                                                           #
    # ---------------------------------------------------------------- #

    def _get_atr_value(self,
                       df: pd.DataFrame,
                       row_idx: int) -> float:
        """
        Extract ATR as a percentage of price from the dataframe
        at a specific row position.

        Tries multiple common ATR column names in priority order.
        Normalises raw-price ATR columns by close price.
        Warns clearly if no valid ATR found so dynamic stops
        being disabled is never silent.

        Parameters
        ----------
        df      : fold dataframe
        row_idx : integer row position (entry point)

        Returns
        -------
        float ATR as a fraction (e.g. 0.025 = 2.5%)
        """
        # Columns to try, in priority order
        atr_candidates = [
            ('atr_ratio', False),   # already a ratio — use as-is
            ('atr_pct',   False),   # same
            ('atr_14',    True),    # raw price units — normalise
            ('atr',       True),    # raw price units — normalise
        ]

        window = df.iloc[:row_idx + 1]

        for col, needs_normalise in atr_candidates:
            if col not in df.columns:
                continue

            val = window[col].iloc[-1]

            if pd.isna(val) or val <= 0:
                continue

            if needs_normalise:
                if 'close' in df.columns:
                    close = window['close'].iloc[-1]
                    if close > 0:
                        val = val / close
                    else:
                        continue
                else:
                    continue

            logger.debug(
                "ATR from '%s' at row %d: %.4f", col, row_idx, val
            )
            return float(val)

        logger.warning(
            "No valid ATR column found at row %d — "
            "using base_stop_loss=%.3f. "
            "Dynamic stops are DISABLED. "
            "Ensure feature_engine produces one of: %s",
            row_idx,
            self.base_stop_loss,
            [c for c, _ in atr_candidates],
        )
        return self.base_stop_loss